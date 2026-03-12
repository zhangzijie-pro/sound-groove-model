import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def hungarian_match_logits(slot_logits, target_matrix, valid_mask=None):
    B, T, K = slot_logits.shape
    device = slot_logits.device

    pred_prob = torch.sigmoid(slot_logits)
    pred_bin = (pred_prob >= 0.5).float()

    if valid_mask is None:
        valid_mask = torch.ones((B, T), device=device, dtype=torch.bool)

    reordered = torch.zeros_like(pred_bin)

    for b in range(B):
        vm = valid_mask[b].float().unsqueeze(-1)  # [T,1]

        # cost[i,j] = BCE(pred_slot_i, target_slot_j)
        cost = torch.zeros((K, K), device=device)
        for i in range(K):
            for j in range(K):
                bce = F.binary_cross_entropy(
                    pred_prob[b, :, i],
                    target_matrix[b, :, j],
                    reduction="none"
                )
                cost[i, j] = (bce * vm.squeeze(-1)).sum() / vm.sum().clamp(min=1.0)

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        for i, j in zip(row_ind, col_ind):
            reordered[b, :, j] = pred_bin[b, :, i]

    return reordered