import torch.nn as nn
import torch
import contextlib
import collections
import math
from transformers import AutoModelForCausalLM

LLAMA32_CONFIG_1B = {
    # High level input parameters
    "vocab_size": 128_256,           # Vocabulary size
    "context_length": 8192,          # Maximum context length to use (reduced to save memory)
    "orig_context_length": 131_072,  # Context length that was used to train the model
    "max_position_embeddings": 131_072,

    # Lower-level model parameters
    "intermediate_size": 8192,              # Size of the intermediate dimension in FeedForward
    "hidden_size": 2048,                 # Embedding dimension
    "n_heads": 32,                   # Number of attention heads
    "n_layers": 16,                  # Number of layers
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rms_norm_eps": 1e-5,
    "mlp_bias": False,
    "attn_bias": False,
    "attn_dropout": 0.0,
    "is_causal": True,
    "pad_token_id": 128001,

    # Position embedding parameters
    "rope_base": 500_000.0,          # The base in RoPE's "theta"
    "dtype": torch.float32,         # Lower-precision dtype to reduce memory usage
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    "rope_theta": 500000.0,
}

def robust_copy_weights(custom_model, hf_model):
    custom_state = dict(custom_model.named_parameters())
    hf_state = dict(hf_model.named_parameters())
    for name, param in custom_state.items():
        if name in hf_state:
            if param.shape == hf_state[name].shape:
                param.data.copy_(hf_state[name].data)
            else:
                print(f"Shape mismatch for {name}: custom {param.shape}, HF {hf_state[name].shape}")
        else:
            print(f"Parameter {name} missing in HF model")
    for name in hf_state:
        if name not in custom_state:
            print(f"Parameter {name} missing in custom model")

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Convert to float32 for numerical stability.
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute variance.
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        # Scale and return.
        return self.weight.to(input_dtype) * x.to(input_dtype)

class EarlyExitConfig:
    def __init__(self, cfg, layer_idx: int, train_transformer: bool = True):
        self.layer_idx = layer_idx
        self.norm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        if train_transformer:
            self.last_decoder_layer = LlamaDecoderLayer(cfg, layer_idx)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def load_lm_head(self, lm_head):
        self.lm_head.weight = lm_head.weight

    def load_last_decoder_layer(self, last_decoder_layer):
        self.last_decoder_layer.load_state_dict(last_decoder_layer.state_dict())

class Llama3Model(nn.Module):
    def __init__(self, cfg, ee_cfg: EarlyExitConfig = None):
        super().__init__()
        self.padding_idx = cfg["pad_token_id"]
        self.vocab_size = cfg["vocab_size"]

        # Main model parameters
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], self.padding_idx, dtype=cfg["dtype"])
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(cfg, layer_idx) for layer_idx in range(cfg["n_layers"])]
        )

        self.norm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.rotary_emb = LlamaRotaryEmbedding(cfg) 

        # Early Exit Modules
        self.ee_cfg = ee_cfg

    def forward(self, 
        in_idx: torch.LongTensor,
        attention_mask: torch.Tensor):
        # Token embeddings
        tok_embeds = self.embed_tokens(in_idx)
        B, T, C = tok_embeds.shape

        # For position embeddings.
        position_ids = torch.arange(0, T, device=in_idx.device).unsqueeze(0)
        
        # Causal Mask
        causal_mask = self._update_causal_mask(attention_mask, tok_embeds)

        # Position Embeddings

        # Iterate through transformer layers
        hidden_states = tok_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Prefix Llama
        for i in range(self.ee_cfg.layer_idx):
            decoder_layer = self.layers[i]
            hidden_states = decoder_layer(hidden_states, attention_mask=causal_mask, position_embeddings=position_embeddings) 

        # Use frozen decoder layer or my decoder layer depending on the config
        if self.ee_cfg and self.ee_cfg.train_transformer:
            next_decoder_layer = self.ee_cfg.last_decoder_layer
        else:
            next_decoder_layer = self.layers[self.ee_cfg.layer_idx]
        
        hidden_states = next_decoder_layer(hidden_states, attention_mask=causal_mask, position_embeddings=position_embeddings)
        ee_hidden_states = self.ee_cfg.norm(hidden_states)

        # Post-early exit layers
        with torch.no_grad():
            for i in range(self.ee_cfg.layer_idx + 1, len(self.layers)):
                decoder_layer = self.layers[i]
                hidden_states = decoder_layer(hidden_states, attention_mask=causal_mask, position_embeddings=position_embeddings) 
            hidden_states = self.norm(hidden_states)

        return ee_hidden_states, hidden_states # <-- Return the hidden states, not logits

    def _update_causal_mask(self, attention_mask, tok_embeds):
        dtype, device = tok_embeds.dtype, tok_embeds.device
        sequence_length = tok_embeds.shape[1]
        target_length = attention_mask.shape[-1]

        # Take attention mask from (B, KV-Length) -> (B, 1, Query Length, KV-Length)
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            batch_size=tok_embeds.shape[0],
        )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length=None,
        target_length=None,
        dtype=None,
        device=None,
        batch_size=None):
        # Apply upper triangular mask
        min_dtype = torch.finfo(dtype).min # -infinity
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                causal_mask.device
            )
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
        return causal_mask


class LlamaDecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.self_attn = LlamaAttention(cfg, layer_idx=layer_idx)
        self.mlp = LlamaMLP(cfg)
        self.input_layernorm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def forward(self, x, attention_mask=None, position_embeddings=None):
        # Self-attention
        shortcut = x
        x = self.input_layernorm(x)
        x, attn_weights = self.self_attn(x, attention_mask, position_embeddings)
        x = x + shortcut

        # MLP
        shortcut = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + shortcut

        return x

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def _eager_attention(module, query_states, key_states, value_states, attention_mask, scaling, dropout = 0.0):
    key_states = repeat_kv(key_states, module.num_kv_groups)
    value_states = repeat_kv(value_states, module.num_kv_groups)

    # (b, n_heads, seq, head_dim) x (b, n_heads, head_dim,seq) -> (b, n_heads, seq, seq)
    lhs = query_states
    rhs = key_states.transpose(-2, -1)
    attn_weights = torch.matmul(lhs, rhs) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

# Multi-head attention
class LlamaAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.config = cfg
        self.layer_idx = layer_idx
        self.head_dim = cfg["hidden_size"] // cfg["n_heads"]
        self.num_kv_groups = cfg["n_heads"] // cfg["n_kv_groups"]
        self.scaling = self.head_dim ** -0.5
        self.attn_dropout = cfg["attn_dropout"] if "attn_dropout" in cfg else 0.0
        self.is_causal = cfg["is_causal"] if "is_causal" in cfg else True

        assert cfg["hidden_size"] % cfg["n_heads"] == 0, "d_out must be divisible by num_heads"
        assert cfg["n_heads"] % self.num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.q_proj = nn.Linear(
            cfg["hidden_size"], cfg["n_heads"] * self.head_dim, bias=cfg["attn_bias"]
        )
        self.k_proj = nn.Linear(
            cfg["hidden_size"], cfg["n_kv_groups"] * self.head_dim, bias=cfg["attn_bias"]
        )
        self.v_proj = nn.Linear(
            cfg["hidden_size"], cfg["n_kv_groups"] * self.head_dim, bias=cfg["attn_bias"]
        )
        self.o_proj = nn.Linear(
            cfg["n_heads"] * self.head_dim, cfg["hidden_size"], bias=cfg["attn_bias"]
        )

    def forward(self, hidden_states, attention_mask, position_embeddings):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings 
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        dropout = self.attn_dropout if self.training else 0.0
        attn_output, attn_weights = _eager_attention(self, query_states, key_states, value_states, attention_mask, self.scaling, dropout=dropout)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous() 
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=cfg["mlp_bias"])
        self.up_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=cfg["mlp_bias"])
        self.down_proj = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], bias=cfg["mlp_bias"])

    def forward(self, x):
        x_fc1 = self.gate_proj(x)
        x_fc2 = self.up_proj(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.down_proj(x)

def _compute_default_rope_parameters(
    config,
    device,
    seq_len,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config["rope_theta"]
        partial_rotary_factor = config["partial_rotary_factor"] if "partial_rotary_factor" in config else 1.0
        head_dim = config["head_dim"] if "head_dim" in config else config["hidden_size"] // config["n_heads"]
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor

def _compute_llama3_parameters(
    config, device, seq_len=None, **rope_kwargs
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)
    
    assert("rope_scaling" in config)
    assert("factor" in config["rope_scaling"])
    assert("low_freq_factor" in config["rope_scaling"])
    assert("high_freq_factor" in config["rope_scaling"])
    assert("original_max_position_embeddings" in config["rope_scaling"])

    factor = config["rope_scaling"]["factor"]  # `8` in the original implementation
    low_freq_factor = config["rope_scaling"]["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = config["rope_scaling"]["high_freq_factor"]  # `4` in the original implementation
    old_context_len = config["rope_scaling"]["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor

ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "llama3": _compute_llama3_parameters,
}

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        if "rope_scaling" in config and config["rope_scaling"] is not None:
            self.rope_type = config["rope_scaling"].get("rope_type", config["rope_scaling"].get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config["max_position_embeddings"]
        self.original_max_seq_len = config["max_position_embeddings"]

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

class MutantLlama(nn.Module):
    def __init__(self, cfg, ee_cfg: EarlyExitConfig = None):
        super().__init__()
        self.model = Llama3Model(cfg, ee_cfg=ee_cfg)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.ee_cfg = ee_cfg
        
        # Weight tying.
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, in_idx, attention_mask):
        ee_hidden_states, hidden_states = self.model(in_idx, attention_mask)
        ee_logits = self.ee_cfg.lm_head(ee_hidden_states)

        # Post-early exit layers    
        with torch.no_grad():
            logits = self.lm_head(hidden_states)

        return {"llama_logits": logits, "ee_logits": ee_logits}
    
    def load_from_hf(self, hf_model):
        robust_copy_weights(self, hf_model)

def compare_models(custom_model, hf_model, input_ids=None, attention_mask=None, verbose=True, atol=1e-4):
    """
    Compares the parameters and (optionally) outputs of two models.
    Prints a summary of the largest differences.
    """
    print("=== Comparing model parameters ===")
    custom_state = dict(custom_model.named_parameters())
    hf_state = dict(hf_model.named_parameters())
    all_keys = set(custom_state.keys()) | set(hf_state.keys())
    diffs = []
    for key in sorted(all_keys):
        if key not in custom_state:
            print(f"Parameter {key} missing in custom model.")
            continue
        if key not in hf_state:
            print(f"Parameter {key} missing in HF model.")
            continue
        c = custom_state[key].detach().cpu()
        h = hf_state[key].detach().cpu()
        if c.shape != h.shape:
            print(f"Shape mismatch for {key}: custom {c.shape}, HF {h.shape}")
            continue
        diff = (c - h).abs().max().item()
        diffs.append((diff, key))
        if verbose and diff > atol:
            print(f"Param {key}: max abs diff = {diff:.3e}")
    if diffs:
        max_diff, max_key = max(diffs)
        print(f"\nLargest parameter diff: {max_diff:.3e} in {max_key}")
    else:
        print("No parameters to compare.")

    if input_ids is not None:
        print("\n=== Comparing model outputs ===")
        with torch.no_grad():
            custom_out = custom_model(input_ids, attention_mask)
            hf_out = hf_model(input_ids, attention_mask).logits
        out_diff = (custom_out - hf_out).abs()
        max_out_diff = out_diff.max().item()
        print(f"Max output diff: {max_out_diff:.3e}")
        if verbose and max_out_diff > atol:
            idx = (out_diff == max_out_diff).nonzero(as_tuple=True)
            print(f"Output diff at index {idx}: custom={custom_out[idx]}, hf={hf_out[idx]}")
    print("=== Comparison complete ===")

# Usage example:
if __name__ == "__main__":
    model = MutantLlama(LLAMA32_CONFIG_1B)  # <-- Use the wrapper class
    
    # # Load HF implementation (requires transformers installed)
    # print("Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

    # # Copy weights from HF model to our custom model
    print("Copying weights to custom model...")
    robust_copy_weights(model, hf_model)
    
    # Test with dummy input
    input_ids = torch.randint(0, 128_256, (4, 10))
    attention_mask = torch.ones_like(input_ids)

    output = model(input_ids, attention_mask)
    hf_output = hf_model(input_ids, attention_mask)
    
    compare_models(model, hf_model, input_ids=input_ids, attention_mask=attention_mask, verbose=True, atol=1e-4)