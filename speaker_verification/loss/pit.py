import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


class PITLoss(nn.Module):
    def __init__(self, pos_weight=1.5):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets, valid_mask=None):
        """
        logits:  [B,T,K]
        targets: [B,T,K] 0/1
        valid_mask: [B,T]
        """
        B, T, K = logits.shape
        device = logits.device

        if valid_mask is None:
            valid_mask = torch.ones((B, T), device=device, dtype=torch.float32)
        else:
            valid_mask = valid_mask.float().to(device)

        valid_mask = valid_mask.unsqueeze(-1)  # [B,T,1]

        p_list = list(permutations(range(K)))

        # [B,T,K,1] vs [B,T,1,K] -> [B,T,K,K]
        l_exp = logits.unsqueeze(-1).expand(B, T, K, K)
        t_exp = targets.unsqueeze(-2).expand(B, T, K, K)

        all_bce = F.binary_cross_entropy_with_logits(
            l_exp,
            t_exp,
            reduction="none",
            pos_weight=torch.tensor(self.pos_weight, device=device)
        )

        # [B,K,K]
        denom = valid_mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # [B,1,1]
        cost_matrix = (all_bce * valid_mask.unsqueeze(-1)).sum(dim=1) / denom

        perm_losses = []
        for p in p_list:
            curr = 0.0
            for i, j in enumerate(p):
                curr = curr + cost_matrix[:, i, j]
            perm_losses.append(curr / K)

        stacked = torch.stack(perm_losses, dim=0)  # [K!, B]
        min_loss, _ = torch.min(stacked, dim=0)    # [B]
        return min_loss.mean()