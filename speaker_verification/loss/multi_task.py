import torch
import torch.nn as nn

from speaker_verification.loss.pit import PITLoss


class MultiTaskLoss(nn.Module):
    """
    Simplified end-to-end diarization loss.

    The primary objective is PIT BCE on frame-wise diarization logits.
    An optional temporal smoothness term stabilizes slot activity transitions.
    """

    def __init__(
        self,
        pit_pos_weight=1.5,
        lambda_smooth=0.05,
    ):
        super().__init__()
        self.pit_loss = PITLoss(pos_weight=pit_pos_weight)
        self.lambda_smooth = float(lambda_smooth)

    @staticmethod
    def _smoothness_loss(diar_logits, valid_mask=None):
        if diar_logits.size(1) <= 1:
            return diar_logits.new_tensor(0.0)

        diff = diar_logits[:, 1:] - diar_logits[:, :-1]
        if valid_mask is None:
            return diff.abs().mean()

        pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        if not pair_mask.any():
            return diar_logits.new_tensor(0.0)
        return diff.abs()[pair_mask].mean()

    def forward(
        self,
        diar_logits,
        target_matrix,
        valid_mask=None,
        return_detail=False,
        **_,
    ):
        diar_loss = self.pit_loss(diar_logits, target_matrix, valid_mask=valid_mask)
        smooth_loss = self._smoothness_loss(diar_logits, valid_mask=valid_mask)
        total = diar_loss + self.lambda_smooth * smooth_loss

        if return_detail:
            return {
                "total": total,
                "diar_loss": diar_loss.detach(),
                "smooth_loss": smooth_loss.detach(),
            }

        return total
