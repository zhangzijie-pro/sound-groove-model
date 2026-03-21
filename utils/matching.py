import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

@torch.no_grad()
def hungarian_match_logits(slot_logits, target_matrix, valid_mask=None):
    B, T, K_logit = slot_logits.shape
    _, _, K_tgt = target_matrix.shape
    K = min(K_logit, K_tgt)

    device = slot_logits.device

    if valid_mask is None:
        valid_mask = torch.ones((B, T), dtype=torch.bool, device=device)

    pred_prob = torch.sigmoid(slot_logits[:, :, :K])          # [B,T,K]
    target_sel = target_matrix[:, :, :K]                      # [B,T,K]

    # Vectorized BCE cost matrix [B, K, K]
    pred_exp = pred_prob.unsqueeze(2)                         # [B,T,K,1]
    tgt_exp  = target_sel.unsqueeze(1)                        # [B,T,1,K]
    bce = F.binary_cross_entropy(pred_exp, tgt_exp, reduction='none')  # [B,T,K,K]

    vm_4d = valid_mask.unsqueeze(1).unsqueeze(-1)             # [B,1,T,1] → broadcasts
    cost_per_frame = bce * vm_4d
    cost = cost_per_frame.sum(dim=2) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    cost = cost.squeeze(1)                                    # [B, K, K]

    reordered = torch.zeros((B, T, K_logit), device=device)

    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost[b].cpu().numpy())
        for r, c in zip(row_ind, col_ind):
            reordered[b, :, c] = (pred_prob[b, :, r] >= 0.5).float()

    return reordered