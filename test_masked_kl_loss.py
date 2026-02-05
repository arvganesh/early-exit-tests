import torch
import torch.nn.functional as F

from truncated_llama import masked_kl_loss


def test_masked_kl_loss_ignores_appended_padding():
    torch.manual_seed(0)

    B, T1, V = 2, 5, 11
    T2 = 9

    logits_a = torch.randn(B, T1, V)
    logits_b = torch.randn(B, T1, V)

    a_lp = F.log_softmax(logits_a, dim=-1)
    b_lp = F.log_softmax(logits_b, dim=-1)
    mask1 = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float32)

    loss1 = masked_kl_loss(a_lp, b_lp, mask1, reduction="tokenmean")

    # Append extra "padding" timesteps with mask=0; loss should be unchanged.
    logits_a2 = torch.cat([logits_a, torch.randn(B, T2 - T1, V)], dim=1)
    logits_b2 = torch.cat([logits_b, torch.randn(B, T2 - T1, V)], dim=1)
    a2_lp = F.log_softmax(logits_a2, dim=-1)
    b2_lp = F.log_softmax(logits_b2, dim=-1)
    mask2 = torch.cat([mask1, torch.zeros(B, T2 - T1)], dim=1)

    loss2 = masked_kl_loss(a2_lp, b2_lp, mask2, reduction="tokenmean")
    assert torch.allclose(loss1, loss2, atol=1e-6)

