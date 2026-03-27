import torch

from speaker_verification.loss.multi_task import MultiTaskLoss


def test_multi_task_loss_returns_scalar_and_details():
    loss_fn = MultiTaskLoss(max_spk=4)

    frame_embeds = torch.randn(2, 16, 32, requires_grad=True)
    slot_logits = torch.randn(2, 16, 5, requires_grad=True)
    pred_activity = torch.randn(2, 16, requires_grad=True)
    pred_count = torch.randn(2, 5, requires_grad=True)

    target_matrix = torch.zeros(2, 16, 4)
    target_matrix[:, :8, 0] = 1.0
    target_matrix[:, 8:, 1] = 1.0
    target_activity = torch.ones(2, 16)
    target_count = torch.tensor([2, 2])
    valid_mask = torch.ones(2, 16, dtype=torch.bool)

    loss_dict = loss_fn(
        frame_embeds=frame_embeds,
        slot_logits=slot_logits,
        pred_activity=pred_activity,
        pred_count=pred_count,
        target_matrix=target_matrix,
        target_activity=target_activity,
        target_count=target_count,
        valid_mask=valid_mask,
        return_detail=True,
    )

    assert set(loss_dict) == {"total", "pit_loss", "act_loss", "cnt_loss", "frm_loss"}
    assert torch.isfinite(loss_dict["total"])

    loss_dict["total"].backward()

    assert frame_embeds.grad is not None
    assert slot_logits.grad is not None
    assert pred_activity.grad is not None
    assert pred_count.grad is not None
