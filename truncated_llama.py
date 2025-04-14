import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput
from typing import List, Tuple

def masked_kl_loss(logits_log_prob, og_log_prob, kl_loss_mask, reduction='batchmean'):
    # Apply loss mask (attention mask => B, T)
    # Define a KLDivLoss function without reduction to get elementwise losses.
    kl_loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)

    # Compute the per-element loss (B, T, V)
    loss_tensor = kl_loss_fct(logits_log_prob, og_log_prob)

    # Sum over the vocabulary dimension to get a per-token loss (B, T)
    loss_per_token = loss_tensor.sum(dim=-1)

    # Apply the attention mask.
    # (Convert the mask to float so that 0's zero out the loss and 1's leave it unchanged.)
    if kl_loss_mask is not None:
        mask = kl_loss_mask.float()
        masked_loss = loss_per_token * mask
    else:
        masked_loss = loss_per_token

    # Normalize by batch size (B) to match "batchmean" behavior.
    if reduction == 'batchmean':
        batch_size = logits_log_prob.size(0)
        loss = masked_loss.sum() / batch_size
    elif reduction == 'mean':
        loss = masked_loss.mean()
    else:
        raise Exception("Invalid reduction type.")

    return loss

class TruncatedLlama(nn.Module):
    def __init__(self, model_path: str, early_exit_idx: int, use_flash_attn: bool = False, ft_last_transformer = False, ft_head = False, lm_head_random_init = True):
        super().__init__()
        # Load reference model (always kept frozen)
        if use_flash_attn:
            self.reference_model = LlamaForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2")
        else:
            self.reference_model = LlamaForCausalLM.from_pretrained(model_path)

        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
            
        # Load truncated model (for early exit)
        if use_flash_attn:
            self.truncated_model = LlamaForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2")
        else:
            self.truncated_model = LlamaForCausalLM.from_pretrained(model_path)
            
        # Freeze all parameters in truncated model by default
        for param in self.truncated_model.parameters():
            param.requires_grad = False
            
        # We'll only keep layers up to early_exit_idx in the truncated model
        self.early_exit_idx = early_exit_idx
        
        # Actually truncate the model by keeping only layers up to early_exit_idx
        # Store a reference to the last layer for potential fine-tuning
        self.early_exit_layer = self.truncated_model.model.layers[early_exit_idx]
        
        # Truncate the layers list to only include layers up to early_exit_idx (inclusive)
        self.truncated_model.model.layers = self.truncated_model.model.layers[:early_exit_idx + 1]
        
        # If requested, make the last transformer layer trainable
        if ft_last_transformer:
            for param in self.early_exit_layer.parameters():
                param.requires_grad = True
            print(f"Enabled fine-tuning for the last transformer layer (idx: {early_exit_idx})")
        
        if ft_head:
            for param in self.truncated_model.lm_head.parameters():
                param.requires_grad = True
        
    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None, loss_type=None, keep_og_logits=False):
        # Process with reference model when needed
        og_lm_logits = None
        if loss_type == "kl_divergence" or keep_og_logits:
            with torch.no_grad():
                ref_outputs = self.reference_model(input_ids, attention_mask=attention_mask)
                og_lm_logits = ref_outputs.logits
        
        # Process with truncated model for early exit
        # Since we've actually truncated the model's layers, we can use the model directly
        logits = self.truncated_model(input_ids, attention_mask=attention_mask).logits
        
        # Calculate loss if needed
        loss = None
        if loss_type == "cross_entropy":
            assert labels is not None
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        elif loss_type == "kl_divergence":
            logits_log_prob = F.log_softmax(logits, dim=-1)
            og_log_prob = F.log_softmax(og_lm_logits, dim=-1)
            loss = masked_kl_loss(logits_log_prob, og_log_prob, attention_mask, reduction='mean')

        return {
            "loss": loss,
            "logits": logits,
            "og_lm_logits": og_lm_logits if (keep_og_logits or loss_type == "kl_divergence") else None,
        }
    
    def load_from_checkpoint(self, lm_head, last_transformer):
        if lm_head:
            self.truncated_model.lm_head.load_state_dict(lm_head)
        if last_transformer:
            self.early_exit_layer.load_state_dict(last_transformer)
    
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