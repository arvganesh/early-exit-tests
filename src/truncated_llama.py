import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput
from typing import List, Tuple

class TruncatedLlama(nn.Module):
    def __init__(self, model_path: str, early_exit_idx: int, lm_head_random_init: bool = True, use_flash_attn: bool = False):
        super().__init__()
        if use_flash_attn:
            model = LlamaForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2")
        else:
            model = LlamaForCausalLM.from_pretrained(model_path)

        print(model)

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
        self.headless_model.layers[early_exit_idx].register_forward_hook(save_activations)

        # Optionally, randomly initialize the early-exiting LM head.
        self.new_lm_head = nn.Linear(self.og_lm_head.in_features, self.og_lm_head.out_features, bias=False)
        if not lm_head_random_init:
            self.new_lm_head.load_state_dict(self.og_lm_head.state_dict())

        for param in self.new_lm_head.parameters():
            param.requires_grad = True

        # Print the number of parameters in the model
        num_trainable = sum(p.numel() for p in self.new_lm_head.parameters() if p.requires_grad)
        total = num_trainable + sum(p.numel() for p in self.headless_model.parameters())
        print(f"Number of trainable parameters in the model: {num_trainable}")
        print(f"Number of parameters in the model: {total}")

    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None, loss_type=None):
        # Apply RMSNorm to early exit activations (embeddings already applied).
        final_layer_activations = self.headless_model(input_ids, attention_mask=attention_mask)
        assert self.early_exit_activations != None
        self.early_exit_activations = self.headless_model.norm(self.early_exit_activations)
        logits = self.new_lm_head(self.early_exit_activations)

        loss = None
        if loss_type == "cross_entropy":
            # Compute normal cross entropy against the labels.
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        elif loss_type == "kl_divergence":
            # Get logits of OG LM.
            truth_logits = self.og_lm_head(final_layer_activations.last_hidden_state)
            kl_loss = nn.KLDivLoss(log_target=True, reduction="batchmean")
            loss = kl_loss(logits, truth_logits)

        return MaskedLMOutput(
                loss=loss,
                logits=logits
            )
    
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
            logits = outputs.logits
            next_token = torch.multinomial(torch.softmax(logits[:, -1, :], dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token == eos_token_id:
                break
        return input_ids

# import lm_eval
# from lm_eval.api.instance import Instance
# import torch.nn.functional as F

# class TruncatedLlamaWrapper(lm_eval.api.model.LM):
#     def __init__(self, model, tokenizer):
#         super().__init__()
#         self.model = model
#         self.tokenizer = tokenizer
#         self.tokenizer.pad_token = tokenizer.eos_token
#         self.device = next(model.parameters()).device
        
#     @property
#     def eot_token_id(self):
#         # Return the EOT token ID for your model
#         return self.tokenizer.eos_token_id
        
#     @property
#     def max_length(self):
#         # Return the maximum sequence length
#         return 4096  # Adjust based on your model's configuration
        
#     @property
#     def max_gen_toks(self):
#         # Return the maximum number of tokens to generate
#         return 256  # Adjust as needed
        
#     def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
#         results = []
#         for request in requests:
#             # Tokenize context and continuation
#             context, continuation = request.args()
#             context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
#             continuation_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)
            
#             # Combine context and continuation
#             input_ids = torch.tensor([context_tokens + continuation_tokens], device=self.device)
            
#             # Create labels (-100 for context, actual tokens for continuation)
#             labels = torch.tensor(
#                 [[-100] * len(context_tokens) + continuation_tokens], 
#                 device=self.device
#             )
        
#             with torch.no_grad():
#                 outputs = self.model(input_ids, labels=labels)
#                 logits = outputs.logits
                
#                 # Get logits for continuation tokens only
#                 cont_logits = logits[0, len(context_tokens)-1:-1] # removes last token legits
#                 cont_tokens = input_ids[0, len(context_tokens):]
                
#                 # Calculate log likelihood
#                 log_probs = F.log_softmax(cont_logits, dim=-1)
#                 token_log_probs = log_probs[range(len(cont_tokens)), cont_tokens]
#                 total_log_prob = token_log_probs.sum().item()
                
#                 # Check if the continuation is greedy
#                 is_greedy = True
#                 for logits_i, token_i in zip(cont_logits, cont_tokens):
#                     if torch.argmax(logits_i).item() != token_i.item():
#                         is_greedy = False
#                         break
                
#             results.append((total_log_prob, is_greedy))
        
#         return results
    
#     def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
#         results = []
#         for request in requests:
#             # Tokenize the full sequence
#             text = request.args()
#             tokens = self.tokenizer.encode(text, add_special_tokens=True)
#             tokens = torch.tensor([tokens], device=self.device)

#             # how this all works (illustrated on a causal decoder-only setup):
#                 #          CTX      CONT
#                 # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
#                 # model  \               \
#                 # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
#                 # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice # noqa: E501
            
#             # Calculate log likelihood token by token
#             total_log_prob = 0.0
#             with torch.no_grad():
#                 outputs = self.model(tokens)
#                 logits = outputs.logits[:, :-1] # Remove last logit
#                 tokens = tokens[1:] # Remove first token (b/c we don't care about the probability of the start token being generated)
#                 log_probs = F.log_softmax(logits, dim=-1) # 1, seq length, vocab size (normalized)
#                 for i in range(len(tokens)):
#                     total_log_prob += log_probs[0, tokens[0, i]].item() # 

#             # Check if the continuation is greedy
#             is_greedy = True
#             log_probs = log_probs[0]
#             for logits_i, token_i in zip(log_probs, tokens):
#                 if torch.argmax(logits_i).item() != token_i.item():
#                     is_greedy = False
#                     break
        
#             results.append((total_log_prob, is_greedy))
        
#         return results
    
#     def generate_until(self, requests: List[Instance]) -> List[str]:
#         results = []
#         for request in requests:
#             # Tokenize context
#             input_str, kwargs = request.args()
#             input_ids = self.tokenizer.encode(input_str, return_tensors="pt").to(self.device)
            
#             generated_tokens = []
#             max_new_tokens = self.max_gen_toks
            
#             with torch.no_grad():
#                 for _ in range(max_new_tokens):
#                     outputs = self.model(input_ids)
#                     next_token_logits = outputs.logits[:, -1, :]
#                     next_token = torch.argmax(next_token_logits, dim=-1)
                    
#                     # Append the new token
#                     input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
#                     generated_tokens.append(next_token.item())
                    
#                     # Check if we've generated any of the until sequences
#                     generated_text = self.tokenizer.decode(generated_tokens)
#                     if any(stop_seq in generated_text for stop_seq in until):
#                         break
                    
#                     # Check if we've hit the EOT token
#                     if next_token.item() == self.eot_token_id:
#                         break
            
#             # Decode the generated tokens
#             generated_text = self.tokenizer.decode(generated_tokens)
            
#             # Truncate at the first occurrence of any until sequence
#             for stop_seq in until:
#                 if stop_seq in generated_text:
#                     generated_text = generated_text[:generated_text.index(stop_seq)]
            
#             results.append(generated_text)
        
#         return results
    

# if __name__ == "__main__":
#     model_path = "meta-llama/llama-2-7B-hf"
#     mdl = TruncatedLlamaWrapper(TruncatedLlama(model_path), AutoTokenizer.from_pretrained(model_path))
#     inst = Instance()
#     inst.arguments = "Hi! What's the probability of this?"
#     print(mdl.loglikelihood([inst]))