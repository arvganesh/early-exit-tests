"""
Standalone CPU sanity checks (no pytest) for:
  - sampling policy (temperature + top-p/top-k) normalization/support
  - speculative accept math: E[min(1,p/q)] == sum(min(p,q)) == 1 - TV
  - KL masking: appended padding doesn't change masked loss

Run: `python sanity_checks.py`
"""

import math
import torch
import torch.nn.functional as F

from sampling_utils import sampling_probs_from_logits
from truncated_llama import masked_kl_loss


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (p - q).abs().sum(dim=-1)


def overlap(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.minimum(p, q).sum(dim=-1)


def estimate_accept_prob(p: torch.Tensor, q: torch.Tensor, n: int = 50_000) -> float:
    xs = torch.multinomial(q, num_samples=n, replacement=True)  # (1, n)
    p_x = p.gather(-1, xs)
    q_x = q.gather(-1, xs).clamp_min(1e-12)
    a = torch.minimum(torch.ones_like(p_x), p_x / q_x)
    return a.mean().item()


def check_sampling_probs():
    torch.manual_seed(0)
    logits = torch.randn(4, 97)
    probs = sampling_probs_from_logits(logits, temperature=0.8, top_p=1.0, top_k=0)
    assert probs.shape == logits.shape
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.size(0)), atol=1e-5)

    logits = torch.randn(2, 50)
    k = 7
    probs = sampling_probs_from_logits(logits, temperature=1.0, top_p=1.0, top_k=k)
    support = (probs > 0).sum(dim=-1)
    assert torch.all(support <= k)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.size(0)), atol=1e-5)

    logits = torch.randn(2, 80)
    probs = sampling_probs_from_logits(logits, temperature=1.0, top_p=0.9, top_k=0)
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.size(0)), atol=1e-5)


def check_spec_accept_math():
    torch.manual_seed(0)
    logits_p = torch.randn(1, 101)
    logits_q = torch.randn(1, 101)
    p = sampling_probs_from_logits(logits_p, temperature=0.9, top_p=0.95, top_k=0)
    q = sampling_probs_from_logits(logits_q, temperature=0.9, top_p=0.95, top_k=0)
    ov = overlap(p, q)
    tv = tv_distance(p, q)
    assert torch.allclose(ov, 1.0 - tv, atol=1e-6)

    logits_p = torch.randn(1, 73)
    logits_q = torch.randn(1, 73)
    p = sampling_probs_from_logits(logits_p, temperature=1.1, top_p=1.0, top_k=20)
    q = sampling_probs_from_logits(logits_q, temperature=1.1, top_p=1.0, top_k=20)
    ov = overlap(p, q).item()
    est = estimate_accept_prob(p, q, n=80_000)
    assert abs(est - ov) < 0.02

    logits = torch.randn(1, 64)
    p = sampling_probs_from_logits(logits, temperature=0.7, top_p=0.9, top_k=0)
    q = p.clone()
    ov = overlap(p, q).item()
    est = estimate_accept_prob(p, q, n=30_000)
    assert abs(ov - 1.0) < 1e-6
    assert abs(est - 1.0) < 0.02


def check_masked_kl_padding_invariance():
    torch.manual_seed(0)
    B, T1, V = 2, 5, 11
    T2 = 9

    logits_a = torch.randn(B, T1, V)
    logits_b = torch.randn(B, T1, V)

    a_lp = F.log_softmax(logits_a, dim=-1)
    b_lp = F.log_softmax(logits_b, dim=-1)
    mask1 = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float32)
    loss1 = masked_kl_loss(a_lp, b_lp, mask1, reduction="tokenmean")

    logits_a2 = torch.cat([logits_a, torch.randn(B, T2 - T1, V)], dim=1)
    logits_b2 = torch.cat([logits_b, torch.randn(B, T2 - T1, V)], dim=1)
    a2_lp = F.log_softmax(logits_a2, dim=-1)
    b2_lp = F.log_softmax(logits_b2, dim=-1)
    mask2 = torch.cat([mask1, torch.zeros(B, T2 - T1)], dim=1)
    loss2 = masked_kl_loss(a2_lp, b2_lp, mask2, reduction="tokenmean")

    assert torch.allclose(loss1, loss2, atol=1e-6)


def main():
    check_sampling_probs()
    check_spec_accept_math()
    check_masked_kl_padding_invariance()
    print("sanity_checks.py: OK")


if __name__ == "__main__":
    main()

