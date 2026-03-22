import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def hungarian_match_logits(slot_logits, target_matrix, valid_mask=None):
    """
    将预测 slot 与 target speaker slots 做 Hungarian 对齐，并输出重排后的二值预测。

    约定：
    - 若 slot_logits.shape[-1] == target_matrix.shape[-1] + 1，
      则 slot_logits[..., 0] 为 silence slot，speaker slots 为 1:.
    - 返回：
        pred_bin_full: [B, T, K_full]
      其中：
        - 若输入含 silence slot，则 pred_bin_full[..., 0] 是 silence 的二值结果
        - 对齐后的 speaker 结果放在 pred_bin_full[..., 1:1+K]
    """
    device = slot_logits.device
    target_matrix = target_matrix.float().to(device)

    B, T, K_logit = slot_logits.shape
    _, _, K_tgt = target_matrix.shape

    has_silence = (K_logit == K_tgt + 1)

    if has_silence:
        silence_logits = slot_logits[..., :1]   # [B, T, 1]
        speaker_logits = slot_logits[..., 1:]   # [B, T, K_logit-1]
    else:
        silence_logits = None
        speaker_logits = slot_logits

    K_spk_logit = speaker_logits.size(-1)
    K = min(K_spk_logit, K_tgt)

    if valid_mask is None:
        valid_mask = torch.ones((B, T), dtype=torch.bool, device=device)
    else:
        valid_mask = valid_mask.bool().to(device)

    pred_bin_full = torch.zeros((B, T, K_logit), dtype=torch.float32, device=device)

    # 先填 silence
    if has_silence:
        pred_bin_full[..., 0] = (silence_logits[..., 0] >= 0.0).float()

    if K == 0:
        return pred_bin_full

    logits_sel = speaker_logits[..., :K]  # [B, T, K]
    target_sel = target_matrix[..., :K]   # [B, T, K]

    logits_exp = logits_sel.unsqueeze(-1)  # [B, T, K, 1]
    target_exp = target_sel.unsqueeze(-2)  # [B, T, 1, K]

    logits_exp = logits_exp.expand(-1, -1, -1, K)
    target_exp = target_exp.expand(-1, -1, K, -1)

    bce = F.binary_cross_entropy_with_logits(
        logits_exp,
        target_exp,
        reduction="none",
    )  # [B, T, K_pred, K_gt]

    vm_4d = valid_mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
    masked_bce = bce * vm_4d

    cost_per_sample = masked_bce.sum(dim=1)  # [B, K, K]
    valid_frames = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # [B,1]
    cost = cost_per_sample / valid_frames.unsqueeze(-1)  # [B, K, K]

    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost[b].detach().cpu().numpy())
        for r, c in zip(row_ind, col_ind):
            if has_silence:
                pred_bin_full[b, :, 1 + c] = (logits_sel[b, :, r] >= 0.0).float()
            else:
                pred_bin_full[b, :, c] = (logits_sel[b, :, r] >= 0.0).float()

    return pred_bin_full