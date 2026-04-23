import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.loss.diar_pit import PITDiarizationLoss


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().to(logits.device)

        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=logits.device)
            bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
                pos_weight=pos_weight,
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        return (bce * (1.0 - p_t).pow(self.gamma)).mean()


class ActivityLoss(nn.Module):
    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        activity_logits: torch.Tensor,
        activity_targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=activity_logits.device)
            loss = F.binary_cross_entropy_with_logits(
                activity_logits,
                activity_targets,
                reduction="none",
                pos_weight=pos_weight,
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                activity_logits,
                activity_targets,
                reduction="none",
            )

        if valid_mask is not None:
            loss = loss[valid_mask.bool()]
        if loss.numel() == 0:
            return activity_logits.new_tensor(0.0)
        return loss.mean()


class AttractorDiversityLoss(nn.Module):
    def forward(self, attractors: torch.Tensor, exist_targets: torch.Tensor) -> torch.Tensor:
        attractors = F.normalize(attractors, dim=-1)
        losses = []
        for batch_idx in range(attractors.size(0)):
            active_idx = torch.where(exist_targets[batch_idx] > 0.5)[0]
            if active_idx.numel() < 2:
                continue
            active_attr = attractors[batch_idx, active_idx]
            sim = torch.matmul(active_attr, active_attr.transpose(0, 1))
            eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
            off_diag = sim.masked_fill(eye, 0.0)
            losses.append(off_diag.pow(2).sum() / max(1, active_idx.numel() * (active_idx.numel() - 1)))
        if not losses:
            return attractors.new_tensor(0.0)
        return torch.stack(losses).mean()


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        pit_pos_weight: float = 1.2,
        activity_pos_weight: float | None = None,
        exist_pos_weight: float | None = None,
        exist_focal_gamma: float = 2.0,
        lambda_exist: float = 0.1,
        lambda_activity: float = 0.3,
        lambda_diversity: float = 0.02,
        overlap_weight_2spk: float = 2.0,
        overlap_weight_3spk: float = 3.0,
    ):
        super().__init__()
        self.diar_loss = PITDiarizationLoss(pos_weight=pit_pos_weight)
        self.activity_loss = ActivityLoss(pos_weight=activity_pos_weight)
        self.exist_loss = BinaryFocalLoss(gamma=exist_focal_gamma, pos_weight=exist_pos_weight)
        self.diversity_loss = AttractorDiversityLoss()
        self.lambda_exist = float(lambda_exist)
        self.lambda_activity = float(lambda_activity)
        self.lambda_diversity = float(lambda_diversity)
        self.overlap_weight_2spk = float(overlap_weight_2spk)
        self.overlap_weight_3spk = float(overlap_weight_3spk)

    def _build_frame_weights(self, target_matrix: torch.Tensor) -> torch.Tensor:
        overlap_count = (target_matrix > 0.5).float().sum(dim=-1)
        weights = torch.ones_like(overlap_count)
        weights = torch.where(overlap_count >= 2.0, torch.full_like(weights, self.overlap_weight_2spk), weights)
        weights = torch.where(overlap_count >= 3.0, torch.full_like(weights, self.overlap_weight_3spk), weights)
        return weights

    def forward(
        self,
        frame_embeds: torch.Tensor,
        attractors: torch.Tensor,
        exist_logits: torch.Tensor,
        diar_logits: torch.Tensor,
        activity_logits: torch.Tensor,
        target_matrix: torch.Tensor,
        target_count: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        return_detail: bool = False,
    ):
        del frame_embeds
        del target_count
        target_activity = target_matrix.max(dim=-1).values.clamp(0.0, 1.0)
        frame_weights = self._build_frame_weights(target_matrix)

        pit_loss, aligned_targets, perm_info = self.diar_loss(
            diar_logits,
            target_matrix,
            valid_mask=valid_mask,
            frame_weights=frame_weights,
            return_perm=True,
        )

        exist_targets = perm_info["exist_targets"]
        exist_loss = self.exist_loss(exist_logits, exist_targets)
        activity_loss = self.activity_loss(
            activity_logits,
            target_activity,
            valid_mask=valid_mask,
        )
        diversity_loss = self.diversity_loss(attractors, exist_targets)

        total = (
            pit_loss
            + self.lambda_exist * exist_loss
            + self.lambda_activity * activity_loss
            + self.lambda_diversity * diversity_loss
        )

        if return_detail:
            return {
                "total": total,
                "pit_loss": pit_loss.detach(),
                "exist_loss": exist_loss.detach(),
                "activity_loss": activity_loss.detach(),
                "diversity_loss": diversity_loss.detach(),
                "aligned_targets": aligned_targets.detach(),
                "exist_targets": exist_targets.detach(),
                "target_activity": target_activity.detach(),
            }

        return total
