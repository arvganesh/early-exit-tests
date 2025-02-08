import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, Trainer, TrainingArguments
from transformers.modeling_outputs import MaskedLMOutput

class TruncatedLlama(nn.Module):
    def __init__(self, model_path: str, num_transformer_layers: int, use_flash_attn: bool = False):
        super().__init__()
        # self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        if use_flash_attn:
            self.model = LlamaForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2")
        else:
            self.model = LlamaForCausalLM.from_pretrained(model_path)
        self.model.model.layers = self.model.model.layers[:num_transformer_layers]
        # self.model.lm_head = nn.Linear(self.model.lm_head.in_features, self.model.lm_head.out_features, bias=False, dtype=torch.bfloat16)
        self.model.lm_head = nn.Linear(self.model.lm_head.in_features, self.model.lm_head.out_features, bias=False)

        # Freeze all parameters except the new LM head
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        # Print the number of parameters in the model
        print(f"Number of trainable parameters in the model: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        print(f"Number of parameters in the model: {sum(p.numel() for p in self.model.parameters())}")


    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None):
        # Get logits from model
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits # shape: (batch_size, seq_len, vocab_size)

        # Get loss
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

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

class TruncatedLlamaWrapper(LM):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    @property
    def eot_token_id(self):
        # Return the EOT token ID for your model
        return self.tokenizer.eos_token_id
        
    @property
    def max_length(self):
        # Return the maximum sequence length
        return 4096  # Adjust based on your model's configuration
        
    @property
    def max_gen_toks(self):
        # Return the maximum number of tokens to generate
        return 256  # Adjust as needed
        
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        results = []
        for context, continuation in requests:
            # Tokenize context and continuation
            context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
            continuation_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)
            
            # Combine context and continuation
            input_ids = torch.tensor([context_tokens + continuation_tokens], device=self.device)
            
            # Create labels (-100 for context, actual tokens for continuation)
            labels = torch.tensor(
                [[-100] * len(context_tokens) + continuation_tokens], 
                device=self.device
            )
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels)
                logits = outputs.logits
                
                # Get logits for continuation tokens only
                cont_logits = logits[0, len(context_tokens)-1:-1]
                cont_tokens = input_ids[0, len(context_tokens):]
                
                # Calculate log likelihood
                log_probs = F.log_softmax(cont_logits, dim=-1)
                token_log_probs = log_probs[range(len(cont_tokens)), cont_tokens]
                total_log_prob = token_log_probs.sum().item()
                
                # Check if the continuation is greedy
                is_greedy = True
                for logits_i, token_i in zip(cont_logits, cont_tokens):
                    if torch.argmax(logits_i).item() != token_i.item():
                        is_greedy = False
                        break
                
            results.append((total_log_prob, is_greedy))
        
        return results
    
    def loglikelihood_rolling(self, requests) -> List[float]:
        results = []
        for context, continuation in requests:
            # Tokenize the full sequence
            tokens = self.tokenizer.encode(context + continuation, add_special_tokens=False)
            tokens = torch.tensor([tokens], device=self.device)
            
            # Calculate log likelihood token by token
            total_log_prob = 0.0
            for i in range(len(context), len(context) + len(continuation)):
                input_ids = tokens[:, :i+1]
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits[:, -1]
                    log_probs = F.log_softmax(logits, dim=-1)
                    total_log_prob += log_probs[0, tokens[0, i]].item()
            
            results.append(total_log_prob)
        
        return results
    
    def generate_until(self, requests) -> List[str]:
        results = []
        for context, until in requests:
            # Tokenize context
            input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
            
            generated_tokens = []
            max_new_tokens = self.max_gen_toks
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = self.model(input_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    
                    # Append the new token
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                    generated_tokens.append(next_token.item())
                    
                    # Check if we've generated any of the until sequences
                    generated_text = self.tokenizer.decode(generated_tokens)
                    if any(stop_seq in generated_text for stop_seq in until):
                        break
                    
                    # Check if we've hit the EOT token
                    if next_token.item() == self.eot_token_id:
                        break
            
            # Decode the generated tokens
            generated_text = self.tokenizer.decode(generated_tokens)
            
            # Truncate at the first occurrence of any until sequence
            for stop_seq in until:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.index(stop_seq)]
            
            results.append(generated_text)
        
        return results