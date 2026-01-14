import torch

from sampling_utils import sampling_probs_from_logits


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (p - q).abs().sum(dim=-1)


def overlap(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.minimum(p, q).sum(dim=-1)


def estimate_accept_prob(p: torch.Tensor, q: torch.Tensor, n: int = 200_000) -> float:
    """
    Monte Carlo estimate of E_{x~q}[min(1, p(x)/q(x))].
    """
    assert p.dim() == 2 and q.dim() == 2 and p.size(0) == 1 and q.size(0) == 1
    V = p.size(-1)
    xs = torch.multinomial(q, num_samples=n, replacement=True)  # (1, n)
    p_x = p.gather(-1, xs)  # (1, n)
    q_x = q.gather(-1, xs).clamp_min(1e-12)
    a = torch.minimum(torch.ones_like(p_x), p_x / q_x)
    return a.mean().item()


def test_overlap_equals_one_minus_tv():
    torch.manual_seed(0)
    logits_p = torch.randn(1, 101)
    logits_q = torch.randn(1, 101)
    p = sampling_probs_from_logits(logits_p, temperature=0.9, top_p=0.95, top_k=0)
    q = sampling_probs_from_logits(logits_q, temperature=0.9, top_p=0.95, top_k=0)
    ov = overlap(p, q)
    tv = tv_distance(p, q)
    assert torch.allclose(ov, 1.0 - tv, atol=1e-6)


def test_expected_accept_prob_matches_overlap():
    torch.manual_seed(0)
    logits_p = torch.randn(1, 73)
    logits_q = torch.randn(1, 73)
    p = sampling_probs_from_logits(logits_p, temperature=1.1, top_p=1.0, top_k=20)
    q = sampling_probs_from_logits(logits_q, temperature=1.1, top_p=1.0, top_k=20)
    ov = overlap(p, q).item()
    est = estimate_accept_prob(p, q, n=100_000)
    assert abs(est - ov) < 0.01


def test_accept_prob_is_one_when_p_equals_q():
    torch.manual_seed(0)
    logits = torch.randn(1, 64)
    p = sampling_probs_from_logits(logits, temperature=0.7, top_p=0.9, top_k=0)
    q = p.clone()
    ov = overlap(p, q).item()
    est = estimate_accept_prob(p, q, n=50_000)
    assert abs(ov - 1.0) < 1e-6
    assert abs(est - 1.0) < 0.01

