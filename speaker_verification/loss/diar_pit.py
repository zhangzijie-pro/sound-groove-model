from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class PITDiarizationLoss(nn.Module):
    """
    Hungarian-aligned PIT BCE for EEND-style diarization.

    The prediction speaker axis is an unordered query/attractor axis. This loss
    aligns active target speakers to prediction slots per sample and returns the
    aligned speaker-existence targets used by MultiTaskLoss.
    """

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)

    def _pairwise_bce_cost(
        self,
        diar_logits: torch.Tensor,
        target_matrix: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if target_matrix.size(1) == 0:
            return diar_logits.new_zeros(diar_logits.size(1), 0)

        pred = diar_logits.unsqueeze(-1)
        tgt = target_matrix.unsqueeze(-2)
        pred = pred.expand(-1, -1, target_matrix.size(1))
        tgt = tgt.expand(-1, diar_logits.size(1), -1)

        pos_weight = torch.tensor(self.pos_weight, device=diar_logits.device)
        bce = F.binary_cross_entropy_with_logits(
            pred,
            tgt,
            reduction="none",
            pos_weight=pos_weight,
        )

        mask = valid_mask.unsqueeze(-1).unsqueeze(-1).float()
        denom = valid_mask.sum().clamp(min=1).float()
        return (bce * mask).sum(dim=0) / denom

    def forward(
        self,
        diar_logits: torch.Tensor,
        target_matrix: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_perm: bool = False,
    ):
        device = diar_logits.device
        target_matrix = target_matrix.float().to(device)

        batch_size, time_steps, num_pred = diar_logits.shape
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, time_steps, dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.bool().to(device)

        aligned_targets = torch.zeros(
            batch_size,
            time_steps,
            num_pred,
            dtype=target_matrix.dtype,
            device=device,
        )
        exist_targets = torch.zeros(
            batch_size,
            num_pred,
            dtype=target_matrix.dtype,
            device=device,
        )
        assignments: List[List[Tuple[int, int]]] = []
        sample_losses = []

        pos_weight = torch.tensor(self.pos_weight, device=device)

        for batch_idx in range(batch_size):
            vm = valid_mask[batch_idx]
            logits_b = diar_logits[batch_idx]
            target_b = target_matrix[batch_idx]

            if not bool(vm.any().item()):
                assignments.append([])
                sample_losses.append(logits_b.new_tensor(0.0))
                continue

            active_target_idx = torch.where(target_b[vm].sum(dim=0) > 0)[0]
            aligned_b = torch.zeros(time_steps, num_pred, dtype=target_b.dtype, device=device)
            matched_pairs: List[Tuple[int, int]] = []

            if active_target_idx.numel() > 0 and num_pred > 0:
                active_targets = target_b[:, active_target_idx]
                cost = self._pairwise_bce_cost(
                    diar_logits=logits_b,
                    target_matrix=active_targets,
                    valid_mask=vm,
                )

                row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
                for pred_idx, active_col in zip(row_ind.tolist(), col_ind.tolist()):
                    target_idx = int(active_target_idx[active_col].item())
                    aligned_b[:, pred_idx] = target_b[:, target_idx]
                    exist_targets[batch_idx, pred_idx] = 1.0
                    matched_pairs.append((int(pred_idx), target_idx))

            loss_raw = F.binary_cross_entropy_with_logits(
                logits_b,
                aligned_b,
                reduction="none",
                pos_weight=pos_weight,
            )
            sample_losses.append(loss_raw[vm].mean())
            aligned_targets[batch_idx] = aligned_b
            assignments.append(matched_pairs)

        loss = torch.stack(sample_losses).mean() if sample_losses else diar_logits.new_tensor(0.0)

        if return_perm:
            return loss, aligned_targets, {
                "exist_targets": exist_targets,
                "assignments": assignments,
            }
        return loss

