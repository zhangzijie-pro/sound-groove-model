import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


class PITLoss(nn.Module):
    """
    Permutation Invariant Training (PIT) Loss for speaker diarization.
    Handles mismatched number of speakers between logits and targets.
    """
    def __init__(self, pos_weight=1.5):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets, valid_mask=None):
        """
        logits:   [B, T, K_logit]
        targets:  [B, T, K_tgt]
        valid_mask: [B, T] or None
        """
        B, T, K_logit = logits.shape
        _, _, K_tgt   = targets.shape

        K = min(K_logit, K_tgt)

        device = logits.device

        if valid_mask is None:
            valid_mask = torch.ones((B, T), dtype=torch.float32, device=device)
        else:
            valid_mask = valid_mask.float().to(device)

        # Select only first K speakers
        logits_sel  = logits[..., :K]     # [B, T, K]
        targets_sel = targets[..., :K]    # [B, T, K]

        # Prepare pairwise BCE
        # Goal: [B, T, K_pred, K_gt] where each (i,j) = BCE(logits[:,i], targets[:,j])
        logits_exp  = logits_sel.unsqueeze(-1)   # [B, T, K, 1]
        targets_exp = targets_sel.unsqueeze(-2)  # [B, T, 1, K]

        all_bce = F.binary_cross_entropy_with_logits(
            logits_exp.expand(B, T, K, K),          # explicitly expand to [B,T,K,K]
            targets_exp.expand(B, T, K, K),
            reduction="none",
            pos_weight=torch.tensor(self.pos_weight, device=device)
        )  # → [B, T, K, K]

        mask_4d = valid_mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
        masked_bce = all_bce * mask_4d

        # Sum over time dimension → cost matrix [B, K, K]
        frame_sum = masked_bce.sum(dim=1)                # [B, K, K]
        denom = valid_mask.sum(dim=1).clamp(min=1.0)     # [B]
        denom = denom.view(-1, 1, 1)                     # [B, 1, 1]
        cost_matrix = frame_sum / denom                  # [B, K, K]

        p_list = list(permutations(range(K)))

        perm_losses = []
        for perm in p_list:
            loss_perm = 0.0
            for i in range(K):
                loss_perm += cost_matrix[:, i, perm[i]]
            perm_losses.append(loss_perm / K)

        stacked = torch.stack(perm_losses, dim=0)       # [num_perms, B]
        min_loss, _ = torch.min(stacked, dim=0)         # [B]
        return min_loss.mean()