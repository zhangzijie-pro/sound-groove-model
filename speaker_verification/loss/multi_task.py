import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.loss.diar_pit import PITDiarizationLoss


class ExistenceLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, exist_logits: torch.Tensor, exist_targets: torch.Tensor):
        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=exist_logits.device)
            return F.binary_cross_entropy_with_logits(
                exist_logits,
                exist_targets,
                pos_weight=pos_weight,
            )
        return F.binary_cross_entropy_with_logits(exist_logits, exist_targets)


class AttractorMetricLoss(nn.Module):
    def __init__(self, pull_weight=1.0, sep_margin=0.3):
        super().__init__()
        self.pull_weight = pull_weight
        self.sep_margin = sep_margin

    def forward(self, frame_embeds, attractors, aligned_targets, exist_targets=None, valid_mask=None):
        """
        frame_embeds:   [B,T,D]
        attractors:     [B,N,D]
        aligned_targets:[B,T,N]
        valid_mask:     [B,T]
        """
        frame_embeds = F.normalize(frame_embeds, dim=-1)
        attractors = F.normalize(attractors, dim=-1)

        sim = torch.matmul(frame_embeds, attractors.transpose(1, 2))  # [B,T,N]
        pull_mask = aligned_targets > 0.5

        if valid_mask is not None:
            pull_mask = pull_mask & valid_mask.unsqueeze(-1).bool()

        if pull_mask.any():
            pull_loss = (1.0 - sim[pull_mask]).mean()
        else:
            pull_loss = sim.new_tensor(0.0)

        # separation only among active attractors selected by PIT
        B, N, _ = attractors.shape
        if exist_targets is None:
            exist_targets = (aligned_targets.sum(dim=1) > 0).float()

        sep_terms = []
        for b in range(B):
            active_idx = torch.where(exist_targets[b] > 0.5)[0]
            if active_idx.numel() < 2:
                continue

            active_attr = attractors[b, active_idx]                     # [M, D]
            aa = torch.matmul(active_attr, active_attr.transpose(0, 1)) # [M, M]
            eye = torch.eye(aa.size(0), device=aa.device, dtype=torch.bool)
            sep_vals = torch.relu(aa.masked_fill(eye, -1.0) - self.sep_margin)
            valid_sep = sep_vals > 0
            if valid_sep.any():
                sep_terms.append(sep_vals[valid_sep].mean())

        if sep_terms:
            sep_loss = torch.stack(sep_terms).mean()
        else:
            sep_loss = attractors.new_tensor(0.0)

        return pull_loss * self.pull_weight, sep_loss


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        pit_pos_weight=1.5,
        exist_pos_weight=None,
        lambda_exist=1.0,
        lambda_pull=0.2,
        lambda_sep=0.1,
        lambda_smooth=0.01,
    ):
        super().__init__()
        self.diar_loss = PITDiarizationLoss(pos_weight=pit_pos_weight)
        self.exist_loss = ExistenceLoss(pos_weight=exist_pos_weight)
        self.metric_loss = AttractorMetricLoss()
        self.lambda_exist = float(lambda_exist)
        self.lambda_pull = float(lambda_pull)
        self.lambda_sep = float(lambda_sep)
        self.lambda_smooth = float(lambda_smooth)

    @staticmethod
    def smoothness_loss(diar_logits, valid_mask=None):
        if diar_logits.size(1) <= 1:
            return diar_logits.new_tensor(0.0)

        diff = diar_logits[:, 1:] - diar_logits[:, :-1]
        if valid_mask is None:
            return diff.abs().mean()

        pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        if pair_mask.any():
            return diff[pair_mask].abs().mean()
        return diar_logits.new_tensor(0.0)

    def forward(
        self,
        frame_embeds,
        attractors,
        exist_logits,
        diar_logits,
        target_matrix,
        target_count,
        valid_mask=None,
        return_detail=False,
    ):
        del target_count  # now existence target comes from PIT matching, not naive count prefix

        pit_loss, aligned_targets, perm_info = self.diar_loss(
            diar_logits,
            target_matrix,
            valid_mask=valid_mask,
            return_perm=True,
        )

        exist_targets = perm_info["exist_targets"]  # [B,N_pred]
        exist_loss = self.exist_loss(exist_logits, exist_targets)

        pull_loss, sep_loss = self.metric_loss(
            frame_embeds,
            attractors,
            aligned_targets,
            exist_targets=exist_targets,
            valid_mask=valid_mask,
        )

        smooth_loss = self.smoothness_loss(diar_logits, valid_mask=valid_mask)

        total = (
            pit_loss
            + self.lambda_exist * exist_loss
            + self.lambda_pull * pull_loss
            + self.lambda_sep * sep_loss
            + self.lambda_smooth * smooth_loss
        )

        if return_detail:
            return {
                "total": total,
                "pit_loss": pit_loss.detach(),
                "exist_loss": exist_loss.detach(),
                "pull_loss": pull_loss.detach(),
                "sep_loss": sep_loss.detach(),
                "smooth_loss": smooth_loss.detach(),
                "aligned_targets": aligned_targets.detach(),
                "exist_targets": exist_targets.detach(),
            }

        return total
