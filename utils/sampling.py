import torch


def safe_normalize(dist: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    z = dist.sum(dim=-1, keepdim=True).clamp_min(eps)
    return dist / z


def sampling_probs_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    Common production policy: temperature -> (top_k/top_p) -> sample.
    Returns a full (B, V) probability vector (masked tokens have prob 0).
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for sampling.")

    scaled = logits / temperature

    if top_k and top_k > 0:
        topk_vals, topk_idx = torch.topk(scaled, k=top_k, dim=-1)
        masked = torch.full_like(scaled, float("-inf"))
        scaled = masked.scatter(-1, topk_idx, topk_vals)

    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1, dtype=torch.float32)
        cumprobs = sorted_probs.cumsum(dim=-1)

        # Mask tokens beyond nucleus; keep at least 1 token.
        sorted_mask = cumprobs > top_p
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

        # Unsort back to vocab order.
        scaled = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(scaled, dim=-1, dtype=torch.float32)
    return safe_normalize(probs)

