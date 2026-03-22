import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


class PITLoss(nn.Module):
    """
    Permutation Invariant Training (PIT) Loss for speaker diarization.

    约定：
    - 如果 logits 最后一维比 targets 多 1，则认为 logits[..., 0] 是 silence slot，
      speaker slots 为 logits[..., 1:].
    - 否则默认 logits 全部都是 speaker slots.
    """

    def __init__(self, pos_weight=1.5):
        super().__init__()
        self.pos_weight = float(pos_weight)

    def _select_speaker_logits(self, logits, targets):
        """
        logits:  [B, T, K_logit]
        targets: [B, T, K_tgt]
        return:
            logits_spk: [B, T, K_spk]
        """
        k_logit = logits.size(-1)
        k_tgt = targets.size(-1)

        # logits = silence + K speakers
        if k_logit == k_tgt + 1:
            return logits[..., 1:]

        if k_logit > k_tgt:
            return logits[..., 1:]

        return logits

    def forward(self, logits, targets, valid_mask=None):
        """
        logits:     [B, T, K_logit]
        targets:    [B, T, K_tgt]
        valid_mask: [B, T] or None
        """
        device = logits.device
        targets = targets.float().to(device)

        logits_spk = self._select_speaker_logits(logits, targets)

        B, T, K_logit_spk = logits_spk.shape
        _, _, K_tgt = targets.shape
        K = min(K_logit_spk, K_tgt)

        if K == 0:
            return logits.new_tensor(0.0)

        if valid_mask is None:
            valid_mask = torch.ones((B, T), dtype=torch.float32, device=device)
        else:
            valid_mask = valid_mask.float().to(device)

        logits_sel = logits_spk[..., :K]   # [B, T, K]
        targets_sel = targets[..., :K]     # [B, T, K]

        logits_exp = logits_sel.unsqueeze(-1)   # [B, T, K, 1]
        targets_exp = targets_sel.unsqueeze(-2) # [B, T, 1, K]

        pos_weight = torch.tensor(self.pos_weight, device=device)

        all_bce = F.binary_cross_entropy_with_logits(
            logits_exp.expand(B, T, K, K),
            targets_exp.expand(B, T, K, K),
            reduction="none",
            pos_weight=pos_weight,
        )  # [B, T, K_pred, K_gt]

        mask_4d = valid_mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
        masked_bce = all_bce * mask_4d

        frame_sum = masked_bce.sum(dim=1)                 # [B, K, K]
        denom = valid_mask.sum(dim=1).clamp(min=1.0)      # [B]
        denom = denom.view(-1, 1, 1)                      # [B,1,1]
        cost_matrix = frame_sum / denom                   # [B, K, K]

        p_list = list(permutations(range(K)))
        perm_losses = []

        for perm in p_list:
            loss_perm = 0.0
            for i in range(K):
                loss_perm = loss_perm + cost_matrix[:, i, perm[i]]
            perm_losses.append(loss_perm / K)

        stacked = torch.stack(perm_losses, dim=0)  # [num_perm, B]
        min_loss, _ = torch.min(stacked, dim=0)    # [B]
        return min_loss.mean()