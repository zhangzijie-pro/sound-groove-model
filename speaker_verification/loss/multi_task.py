import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.loss.diar_pit import PITDiarizationLoss


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        pit_pos_weight: float = 1.2,
        exist_pos_weight: float | None = None,
        lambda_exist: float = 1.0,
    ):
        super().__init__()
        self.diar_loss = PITDiarizationLoss(pos_weight=pit_pos_weight)
        self.exist_pos_weight = exist_pos_weight
        self.lambda_exist = float(lambda_exist)

    def forward(
        self,
        exist_logits: torch.Tensor,
        diar_logits: torch.Tensor,
        target_matrix: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_detail: bool = False,
    ):
        pit_loss, aligned_targets, perm_info = self.diar_loss(
            diar_logits,
            target_matrix,
            valid_mask=valid_mask,
            return_perm=True,
        )

        exist_targets = perm_info["exist_targets"]
        if self.exist_pos_weight is not None:
            pos_weight = torch.tensor(self.exist_pos_weight, device=exist_logits.device)
            exist_loss = F.binary_cross_entropy_with_logits(
                exist_logits,
                exist_targets,
                pos_weight=pos_weight,
            )
        else:
            exist_loss = F.binary_cross_entropy_with_logits(
                exist_logits,
                exist_targets,
            )

        total = pit_loss + self.lambda_exist * exist_loss

        if return_detail:
            return {
                "total": total,
                "pit_loss": pit_loss.detach(),
                "exist_loss": exist_loss.detach(),
                "aligned_targets": aligned_targets.detach(),
                "exist_targets": exist_targets.detach(),
            }

        return total
