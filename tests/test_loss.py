import torch

from speaker_verification.loss.multi_task import MultiTaskLoss


def test_multi_task_loss_returns_scalar_and_details():
    loss_fn = MultiTaskLoss(pit_pos_weight=1.5, lambda_smooth=0.05)

    diar_logits = torch.randn(2, 16, 4, requires_grad=True)

    target_matrix = torch.zeros(2, 16, 4)
    target_matrix[:, :8, 0] = 1.0
    target_matrix[:, 8:, 1] = 1.0
    valid_mask = torch.ones(2, 16, dtype=torch.bool)

    loss_dict = loss_fn(
        diar_logits=diar_logits,
        target_matrix=target_matrix,
        valid_mask=valid_mask,
        return_detail=True,
    )

    assert set(loss_dict) == {"total", "diar_loss", "smooth_loss"}
    assert torch.isfinite(loss_dict["total"])

    loss_dict["total"].backward()

    assert diar_logits.grad is not None
