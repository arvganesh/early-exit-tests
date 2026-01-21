import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import List, Tuple, Optional, Literal

def masked_kl_loss(
    logits_log_prob: torch.Tensor,
    og_log_prob: torch.Tensor,
    kl_loss_mask: Optional[torch.Tensor],
    reduction: str = "tokenmean",
    eps: float = 1e-8,
) -> torch.Tensor:
    kl_loss_fct = torch.nn.KLDivLoss(reduction="none", log_target=True)

    # Per-element loss (B, T, V) where target is the reference distribution.
    loss_tensor = kl_loss_fct(logits_log_prob, og_log_prob)
    loss_per_token = loss_tensor.sum(dim=-1)  # (B, T)

    if kl_loss_mask is not None:
        mask = kl_loss_mask.to(dtype=loss_per_token.dtype)
        masked_loss = loss_per_token * mask
        denom = mask.sum().clamp_min(eps)
    else:
        masked_loss = loss_per_token
        denom = torch.tensor(loss_per_token.numel(), device=loss_per_token.device, dtype=loss_per_token.dtype).clamp_min(eps)

    if reduction == "tokenmean":
        return masked_loss.sum() / denom
    if reduction == "batchmean":
        return masked_loss.sum() / logits_log_prob.size(0)
    raise ValueError(f"Invalid reduction type: {reduction}")

AttnImplementation = Optional[Literal["eager", "sdpa", "flash_attention_2"]]

class TruncatedLlama(nn.Module):
    def __init__(
        self,
        model_path: str,
        early_exit_idx: int,
        *,
        reference_model: Optional[nn.Module] = None,
        attn_implementation: AttnImplementation = None,
        torch_dtype: Optional[torch.dtype] = None,
        low_cpu_mem_usage: bool = True,
        use_cache: bool = False,
        use_flash_attn: bool = False,
        ft_last_transformer: bool = False,
        ft_head: bool = False,
        lm_head_random_init: bool = True,
    ):
        super().__init__()
        if attn_implementation is None and use_flash_attn:
            attn_implementation = "flash_attention_2"

        from_pretrained_kwargs = {}
        if attn_implementation is not None:
            from_pretrained_kwargs["attn_implementation"] = attn_implementation
        if torch_dtype is not None:
            # transformers >= 4.57 prefers `dtype` over `torch_dtype` (deprecated)
            from_pretrained_kwargs["dtype"] = torch_dtype
        from_pretrained_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

        def _load_model():
            try:
                return AutoModelForCausalLM.from_pretrained(model_path, **from_pretrained_kwargs)
            except TypeError:
                if "dtype" in from_pretrained_kwargs:
                    fallback_kwargs = dict(from_pretrained_kwargs)
                    fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
                    return AutoModelForCausalLM.from_pretrained(model_path, **fallback_kwargs)
                raise

        # Load / attach reference model (always kept frozen)
        self.reference_model = _load_model() if reference_model is None else reference_model

        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()
        if hasattr(self.reference_model, "config"):
            self.reference_model.config.use_cache = bool(use_cache)
            
        # Load truncated model (for early exit)
        self.truncated_model = _load_model()
        if hasattr(self.truncated_model, "config"):
            self.truncated_model.config.use_cache = bool(use_cache)

        if not (hasattr(self.truncated_model, "model") and hasattr(self.truncated_model.model, "layers") and hasattr(self.truncated_model, "lm_head")):
            raise TypeError(
                "TruncatedLlama expects a LLaMA-like causal LM with `.model.layers` and `.lm_head`. "
                f"Got type={type(self.truncated_model)} from model_path={model_path}."
            )
            
        # Freeze all parameters in truncated model by default
        for param in self.truncated_model.parameters():
            param.requires_grad = False
            
        # We'll only keep layers up to early_exit_idx in the truncated model
        self.early_exit_idx = early_exit_idx
        self.ft_last_transformer = bool(ft_last_transformer)
        self.ft_head = bool(ft_head)
        
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

        # Legacy attribute names used by evaluation scripts
        self.new_lm_head = self.truncated_model.lm_head
        self.headless_model = self.truncated_model.model

    def _shift_attention_mask(self, attention_mask: torch.Tensor):
        if attention_mask is None:
            return None
        shifted = torch.zeros_like(attention_mask)
        if attention_mask.size(1) > 1:
            shifted[:, :-1] = attention_mask[:, 1:]
        return shifted
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        labels=None,
        loss_type=None,
        keep_og_logits: bool = False,
        kl_temperature: float = 1.0,
    ):
        # Teacher pass (reference model) when needed.
        og_lm_logits = None
        og_log_prob = None
        if loss_type in ("kl_divergence", "combined") or keep_og_logits:
            with torch.inference_mode():
                ref_logits = self.reference_model(input_ids, attention_mask=attention_mask, use_cache=False).logits
            if keep_og_logits:
                og_lm_logits = ref_logits
            if loss_type in ("kl_divergence", "combined"):
                og_log_prob = F.log_softmax(ref_logits / kl_temperature, dim=-1)
        
        # Process with truncated model for early exit
        # If we are only tuning the LM head, avoid tracking grads through the frozen transformer
        # to reduce activation memory.
        if not self.ft_last_transformer:
            with torch.no_grad():
                hidden_states = self.truncated_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state
            logits = self.truncated_model.lm_head(hidden_states)
        else:
            # Since we've actually truncated the model's layers, we can use the model directly.
            logits = self.truncated_model(input_ids, attention_mask=attention_mask, use_cache=False).logits
        
        # Calculate loss if needed
        loss = None
        if loss_type == "cross_entropy":
            assert labels is not None
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        elif loss_type == "kl_divergence":
            assert og_log_prob is not None
            logits_log_prob = F.log_softmax(logits / kl_temperature, dim=-1)
            kl_mask = self._shift_attention_mask(attention_mask)
            loss = (kl_temperature**2) * masked_kl_loss(logits_log_prob, og_log_prob, kl_mask, reduction="tokenmean")
        elif loss_type == "combined":
            assert labels is not None
            assert og_log_prob is not None
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            logits_log_prob = F.log_softmax(logits / kl_temperature, dim=-1)
            kl_mask = self._shift_attention_mask(attention_mask)
            kl_loss = (kl_temperature**2) * masked_kl_loss(logits_log_prob, og_log_prob, kl_mask, reduction="tokenmean")
            loss = ce_loss + kl_loss

        return {
            "loss": loss,
            "logits": logits,
            "og_lm_logits": og_lm_logits if keep_og_logits else None,
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
        for _ in range(max_length):
            if input_ids.device.type == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                autocast_ctx = contextlib.nullcontext()
            with autocast_ctx:
                outputs = self(input_ids)
            logits = outputs["logits"]
            next_token = torch.multinomial(torch.softmax(logits[:, -1, :], dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if torch.all(next_token.squeeze(-1) == eos_token_id).item():
                break
        return input_ids

    def print_trainable_parameters(self):
        # Print the number of parameters in the model
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Number of trainable parameters in the model: {num_trainable} | {num_trainable * 100 / total:.3f}")
        print(f"Number of parameters in the model: {total}")
