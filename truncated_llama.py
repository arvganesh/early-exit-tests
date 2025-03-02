import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput
from typing import List, Tuple

def masked_kl_loss(logits_log_prob, og_log_prob, kl_loss_mask):
    # Apply loss mask (attention mask => B, T)
    # Define a KLDivLoss function without reduction to get elementwise losses.
    kl_loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)

    # Compute the per-element loss (B, T, V)
    loss_tensor = kl_loss_fct(logits_log_prob, og_log_prob)

    # Sum over the vocabulary dimension to get a per-token loss (B, T)
    loss_per_token = loss_tensor.sum(dim=-1)

    # Apply the attention mask.
    # (Convert the mask to float so that 0's zero out the loss and 1's leave it unchanged.)
    mask = kl_loss_mask.float()
    masked_loss = loss_per_token * mask

    # Normalize by batch size (B) to match "batchmean" behavior.
    batch_size = logits_log_prob.size(0)
    loss = masked_loss.sum() / batch_size
    return loss

class TruncatedLlama(nn.Module):
    def __init__(self, model_path: str, early_exit_idx: int, lm_head_random_init: bool = True, use_flash_attn: bool = False, use_lora = False, ft_last_transformer = False, inference_mode=False):
        super().__init__()
        if use_flash_attn:
            model = LlamaForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2")
        else:
            model = LlamaForCausalLM.from_pretrained(model_path)
        
        self.inference_mode = inference_mode

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Break up model
        self.headless_model = model.model
        self.og_lm_head = model.lm_head

        # Register hook to save early exit activations
        self.early_exit_activations = None
        def save_activations(module, input, output):
            self.early_exit_activations = output[0]

        if not self.inference_mode:
            self.headless_model.layers[early_exit_idx].register_forward_hook(save_activations)
        else:
            self.headless_model.layers = self.headless_model.layers[:early_exit_idx]

        # Optionally, randomly initialize the early-exiting LM head.
        self.new_lm_head = nn.Linear(self.og_lm_head.in_features, self.og_lm_head.out_features, bias=False)
        if not lm_head_random_init:
            self.new_lm_head.load_state_dict(self.og_lm_head.state_dict())

        if not use_lora:
            for param in self.new_lm_head.parameters():
                param.requires_grad = True
            if ft_last_transformer:
                for param in self.headless_model.layers[early_exit_idx].parameters():
                    param.required_grad = True
   
    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None, loss_type=None, keep_og_logits=False, kl_loss_mask=None):
        # Apply RMSNorm to early exit activations (embeddings already applied).
        if not self.inference_mode:
            final_layer_activations = self.headless_model(input_ids, attention_mask=attention_mask)
            assert self.early_exit_activations != None
            self.early_exit_activations = self.headless_model.norm(self.early_exit_activations)
            logits = self.new_lm_head(self.early_exit_activations)
        else:
            activations = self.headless_model(input_ids, attention_mask=attention_mask)
            activations = self.headless_model.norm(activations.last_hidden_state)
            logits = self.new_lm_head(activations)

        loss = None
        og_lm_logits = None
        if loss_type == "cross_entropy":
            assert labels is not None
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        elif loss_type == "kl_divergence":
            og_lm_logits = self.og_lm_head(final_layer_activations.last_hidden_state) 
            og_log_prob = F.log_softmax(og_lm_logits, dim=-1) # b, t, vocab size
            logits_log_prob = F.log_softmax(logits, dim=-1) # b, t, vocab size
            loss = masked_kl_loss(logits_log_prob, og_log_prob, kl_loss_mask) 

        if keep_og_logits and loss_type != "kl_divergence":
            og_lm_logits = self.og_lm_head(final_layer_activations.last_hidden_state)

        return {
            "loss": loss,
            "logits": logits,
            "og_lm_logits": og_lm_logits if keep_og_logits else None,
        }
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and p.requires_grad]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or not p.requires_grad]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        use_fused = device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def generate(self, input_ids: torch.Tensor, max_length: int, eos_token_id: int):
        for i in range(max_length):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self(input_ids)
            logits = outputs["logits"]
            next_token = torch.multinomial(torch.softmax(logits[:, -1, :], dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token == eos_token_id:
                break
        return input_ids

    def print_trainable_parameters(self):
        # Print the number of parameters in the model
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Number of trainable parameters in the model: {num_trainable} | {num_trainable * 100 / total:.3f}")
        print(f"Number of parameters in the model: {total}")
