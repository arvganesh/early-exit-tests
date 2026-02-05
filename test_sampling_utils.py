import torch

from sampling_utils import sampling_probs_from_logits


def test_sampling_probs_normalized_and_nonnegative():
    torch.manual_seed(0)
    logits = torch.randn(4, 97)
    probs = sampling_probs_from_logits(logits, temperature=0.8, top_p=1.0, top_k=0)
    assert probs.shape == logits.shape
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.size(0)), atol=1e-5)


def test_top_k_limits_support():
    torch.manual_seed(0)
    logits = torch.randn(2, 50)
    k = 7
    probs = sampling_probs_from_logits(logits, temperature=1.0, top_p=1.0, top_k=k)
    support = (probs > 0).sum(dim=-1)
    assert torch.all(support <= k)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.size(0)), atol=1e-5)


def test_top_p_limits_support():
    torch.manual_seed(0)
    logits = torch.randn(2, 80)
    probs = sampling_probs_from_logits(logits, temperature=1.0, top_p=0.9, top_k=0)
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.size(0)), atol=1e-5)

