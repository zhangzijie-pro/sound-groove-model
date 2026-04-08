import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.loss.diar_pit import PITDiarizationLoss


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = logits.float()
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
            bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )

        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        loss = bce * focal_weight

        if valid_mask is not None:
            valid_mask = valid_mask.to(logits.device)
            while valid_mask.dim() < loss.dim():
                valid_mask = valid_mask.unsqueeze(-1)
            loss = loss[valid_mask.expand_as(loss).bool()]

        if loss.numel() == 0:
            return logits.new_tensor(0.0)
        return loss.mean()


class MaskDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float().to(probs.device)

        if valid_mask is not None:
            mask = valid_mask.to(probs.device).unsqueeze(-1).float()
            probs = probs * mask
            targets = targets * mask

        intersection = (probs * targets).sum(dim=(1, 2))
        denom = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


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
            loss_raw = F.binary_cross_entropy_with_logits(
                activity_logits,
                activity_targets,
                reduction="none",
                pos_weight=pos_weight,
            )
        else:
            loss_raw = F.binary_cross_entropy_with_logits(
                activity_logits,
                activity_targets,
                reduction="none",
            )

        if valid_mask is not None:
            valid_mask = valid_mask.bool().to(activity_logits.device)
            loss_raw = loss_raw[valid_mask]

        if loss_raw.numel() == 0:
            return activity_logits.new_tensor(0.0)
        return loss_raw.mean()


class ConsistencyLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        diar_logits: torch.Tensor,
        activity_targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diar_prob = torch.sigmoid(diar_logits).clamp(min=self.eps, max=1.0 - self.eps)
        activity_from_diar = 1.0 - torch.prod(1.0 - diar_prob, dim=-1)
        activity_targets = activity_targets.float().to(diar_logits.device)
        loss_raw = F.binary_cross_entropy(activity_from_diar, activity_targets, reduction="none")

        if valid_mask is not None:
            valid_mask = valid_mask.bool().to(loss_raw.device)
            loss_raw = loss_raw[valid_mask]

        if loss_raw.numel() == 0:
            return diar_logits.new_tensor(0.0)
        return loss_raw.mean()


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        pit_pos_weight: float = 1.2,
        activity_pos_weight: float | None = None,
        exist_pos_weight: float | None = None,
        exist_focal_gamma: float = 2.0,
        lambda_activity: float = 0.5,
        lambda_dice: float = 0.5,
        lambda_consistency: float = 0.2,
        lambda_exist: float = 1.0,
        lambda_smooth: float = 0.03,
    ):
        super().__init__()
        self.diar_loss = PITDiarizationLoss(pos_weight=pit_pos_weight)
        self.activity_loss = ActivityLoss(pos_weight=activity_pos_weight)
        self.exist_loss = BinaryFocalLoss(gamma=exist_focal_gamma, pos_weight=exist_pos_weight)
        self.dice_loss = MaskDiceLoss()
        self.consistency_loss = ConsistencyLoss()
        self.lambda_activity = float(lambda_activity)
        self.lambda_dice = float(lambda_dice)
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_exist = float(lambda_exist)
        self.lambda_smooth = float(lambda_smooth)

    @staticmethod
    def boundary_aware_smoothness(
        diar_logits: torch.Tensor,
        aligned_targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if diar_logits.size(1) <= 1:
            return diar_logits.new_tensor(0.0)

        diff = (diar_logits[:, 1:] - diar_logits[:, :-1]).abs()
        stable_pair_mask = (aligned_targets[:, 1:] - aligned_targets[:, :-1]).abs().sum(dim=-1) < 0.5

        if valid_mask is not None:
            pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
            stable_pair_mask = stable_pair_mask & pair_mask.bool()

        if stable_pair_mask.any():
            return diff[stable_pair_mask].mean()
        return diar_logits.new_tensor(0.0)

    def forward(
        self,
        frame_embeds,
        attractors,
        exist_logits,
        diar_logits,
        activity_logits,
        target_matrix,
        target_count,
        valid_mask=None,
        return_detail=False,
    ):
        del frame_embeds
        del attractors
        del target_count

        target_activity = (target_matrix.sum(dim=-1) > 0).float()

        pit_loss, aligned_targets, perm_info = self.diar_loss(
            diar_logits,
            target_matrix,
            valid_mask=valid_mask,
            return_perm=True,
        )

        exist_targets = perm_info["exist_targets"]
        activity_loss = self.activity_loss(
            activity_logits,
            target_activity,
            valid_mask=valid_mask,
        )
        exist_loss = self.exist_loss(
            exist_logits,
            exist_targets,
        )
        dice_loss = self.dice_loss(
            diar_logits,
            aligned_targets,
            valid_mask=valid_mask,
        )
        consistency_loss = self.consistency_loss(
            diar_logits,
            target_activity,
            valid_mask=valid_mask,
        )
        smooth_loss = self.boundary_aware_smoothness(
            diar_logits,
            aligned_targets,
            valid_mask=valid_mask,
        )

        total = (
            pit_loss
            + self.lambda_dice * dice_loss
            + self.lambda_activity * activity_loss
            + self.lambda_exist * exist_loss
            + self.lambda_consistency * consistency_loss
            + self.lambda_smooth * smooth_loss
        )

        if return_detail:
            zero = total.new_tensor(0.0)
            return {
                "total": total,
                "pit_loss": pit_loss.detach(),
                "dice_loss": dice_loss.detach(),
                "activity_loss": activity_loss.detach(),
                "exist_loss": exist_loss.detach(),
                "consistency_loss": consistency_loss.detach(),
                "pull_loss": zero,
                "sep_loss": zero,
                "smooth_loss": smooth_loss.detach(),
                "aligned_targets": aligned_targets.detach(),
                "exist_targets": exist_targets.detach(),
                "target_activity": target_activity.detach(),
            }

        return total
