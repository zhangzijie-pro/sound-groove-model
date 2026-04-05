import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class PITDiarizationLoss(nn.Module):
    """
    PIT diarization loss for query/attractor-based diarization.

    Returns:
        - loss
        - aligned_targets: target aligned to predicted speaker-query order
        - best_perm_info:
            {
                "matched_pred_indices": List[List[int]],
                "matched_tgt_indices": List[List[int]],
                "exist_targets": Tensor[B, N_pred],
            }
    """

    def __init__(self, pos_weight=1.5):
        super().__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, diar_logits, targets, valid_mask=None, return_perm=False):
        """
        Args:
            diar_logits: [B, T, N_pred]
            targets:     [B, T, N_tgt]
            valid_mask:  [B, T]

        Returns:
            if return_perm=False:
                loss
            else:
                loss, aligned_targets, perm_info
        """
        device = diar_logits.device
        targets = targets.float().to(device)

        B, T, N_pred = diar_logits.shape
        _, _, N_tgt = targets.shape

        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.bool().to(device)

        aligned_targets = torch.zeros_like(diar_logits)
        exist_targets = torch.zeros(B, N_pred, dtype=torch.float32, device=device)

        best_losses = []
        matched_pred_indices_all = []
        matched_tgt_indices_all = []

        # speaker presence in reference target
        # which target speakers actually appear in this chunk?
        target_presence = (targets.sum(dim=1) > 0)  # [B, N_tgt]

        for b in range(B):
            tgt_present_idx = torch.where(target_presence[b])[0].tolist()
            num_present = len(tgt_present_idx)
            pos_weight = torch.tensor(self.pos_weight, device=device)

            if num_present == 0:
                # no active speaker: all queries should stay silent
                zero_targets = torch.zeros_like(diar_logits[b])
                loss_raw = F.binary_cross_entropy_with_logits(
                    diar_logits[b],
                    zero_targets,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                loss_raw = loss_raw[valid_mask[b]]
                if loss_raw.numel() == 0:
                    best_losses.append(diar_logits.new_tensor(0.0))
                else:
                    best_losses.append(loss_raw.mean())
                matched_pred_indices_all.append([])
                matched_tgt_indices_all.append([])
                continue

            # predicted query candidates
            pred_idx = list(range(N_pred))
            tgt_idx = tgt_present_idx

            # choose num_present predicted queries out of N_pred
            # and permute them to target speakers
            # cost may be high if N_pred is large; keep N_pred moderate
            best_loss_b = None
            best_pred_subset = None
            best_tgt_perm = None

            for pred_subset in itertools.combinations(pred_idx, num_present):
                pred_subset = list(pred_subset)

                targets_sel_all = targets[b, :, tgt_idx]            # [T, S]

                for tgt_perm in itertools.permutations(range(num_present)):
                    candidate_targets = torch.zeros_like(diar_logits[b])    # [T, N_pred]
                    for pred_q, local_tgt_idx in zip(pred_subset, tgt_perm):
                        candidate_targets[:, pred_q] = targets_sel_all[:, local_tgt_idx]

                    loss_raw = F.binary_cross_entropy_with_logits(
                        diar_logits[b],
                        candidate_targets,
                        reduction="none",
                        pos_weight=pos_weight,
                    )  # [T, N_pred]

                    loss_raw = loss_raw[valid_mask[b]]
                    if loss_raw.numel() == 0:
                        loss_val = diar_logits.new_tensor(0.0)
                    else:
                        loss_val = loss_raw.mean()

                    if best_loss_b is None or loss_val < best_loss_b:
                        best_loss_b = loss_val
                        best_pred_subset = pred_subset
                        best_tgt_perm = [tgt_idx[i] for i in tgt_perm]

            assert best_pred_subset is not None
            assert best_tgt_perm is not None

            best_losses.append(best_loss_b)

            # build aligned targets in predicted query space
            for pred_q, tgt_q in zip(best_pred_subset, best_tgt_perm):
                aligned_targets[b, :, pred_q] = targets[b, :, tgt_q]
                exist_targets[b, pred_q] = 1.0

            matched_pred_indices_all.append(best_pred_subset)
            matched_tgt_indices_all.append(best_tgt_perm)

        loss = torch.stack(best_losses).mean()

        if not return_perm:
            return loss

        perm_info = {
            "matched_pred_indices": matched_pred_indices_all,
            "matched_tgt_indices": matched_tgt_indices_all,
            "exist_targets": exist_targets,
        }
        return loss, aligned_targets, perm_info
