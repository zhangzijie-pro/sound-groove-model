import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


class DERTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.fa = 0.0
        self.miss = 0.0
        self.conf = 0.0
        self.gt_active = 0.0
        self.pred_active = 0.0

    def update(self, info: dict):
        self.fa += float(info.get("fa", 0.0))
        self.miss += float(info.get("miss", 0.0))
        self.conf += float(info.get("conf", 0.0))
        self.gt_active += float(info.get("gt_active", 0.0))
        self.pred_active += float(info.get("pred_active", 0.0))

    def value(self):
        denom = max(self.gt_active, 1e-8)
        der = (self.fa + self.miss + self.conf) / denom
        return der

    def detail(self):
        return {
            "fa": self.fa,
            "miss": self.miss,
            "conf": self.conf,
            "gt_active": self.gt_active,
            "pred_active": self.pred_active,
            "der": self.value(),
        }


@torch.no_grad()
def hungarian_match_logits(diar_logits, target_matrix, valid_mask=None):
    """
    Align predicted speaker dimensions to target speaker dimensions by Hungarian matching.

    Args:
        diar_logits:  [B, T, N_pred]
        target_matrix:[B, T, N_tgt]
        valid_mask:   [B, T] or None

    Returns:
        pred_bin_full: thresholded binary predictions after best alignment
                       [B, T, N_pred]
    """
    device = diar_logits.device
    target_matrix = target_matrix.float().to(device)

    B, T, N_pred = diar_logits.shape
    _, _, N_tgt = target_matrix.shape
    N = min(N_pred, N_tgt)

    if valid_mask is None:
        valid_mask = torch.ones((B, T), dtype=torch.bool, device=device)
    else:
        valid_mask = valid_mask.bool().to(device)

    pred_bin_full = torch.zeros((B, T, N_pred), dtype=torch.float32, device=device)

    if N == 0:
        return pred_bin_full

    logits_sel = diar_logits[..., :N]      # [B,T,N]
    target_sel = target_matrix[..., :N]    # [B,T,N]

    logits_exp = logits_sel.unsqueeze(-1)   # [B,T,N,1]
    target_exp = target_sel.unsqueeze(-2)   # [B,T,1,N]

    logits_exp = logits_exp.expand(-1, -1, -1, N)
    target_exp = target_exp.expand(-1, -1, N, -1)

    bce = F.binary_cross_entropy_with_logits(
        logits_exp,
        target_exp,
        reduction="none",
    )  # [B,T,N_pred,N_tgt]

    vm_4d = valid_mask.unsqueeze(-1).unsqueeze(-1)  # [B,T,1,1]
    masked_bce = bce * vm_4d

    cost_per_sample = masked_bce.sum(dim=1)  # [B,N,N]
    valid_frames = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # [B,1]
    cost = cost_per_sample / valid_frames.unsqueeze(-1)  # [B,N,N]

    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost[b].detach().cpu().numpy())
        for r, c in zip(row_ind, col_ind):
            pred_bin_full[b, :, c] = (logits_sel[b, :, r] >= 0.0).float()

    return pred_bin_full


@torch.no_grad()
def diarization_error_rate_pit(
    diar_logits,
    target_matrix,
    target_activity=None,
    valid_mask=None,
    return_detail=False,
):
    """
    Training-time proxy DER using PIT-aligned binary outputs.
    """
    device = diar_logits.device
    target_matrix = target_matrix.float().to(device)

    B, T, N_pred = diar_logits.shape
    _, _, N_tgt = target_matrix.shape

    pred_bin_full = hungarian_match_logits(
        diar_logits,
        target_matrix,
        valid_mask=valid_mask,
    )  # [B,T,N_pred]

    N = min(pred_bin_full.size(-1), N_tgt)
    pred_bin = pred_bin_full[..., :N]
    target_sel = target_matrix[..., :N]

    gt_activity = (target_matrix.sum(dim=-1) > 0)
    pred_activity = (pred_bin.sum(dim=-1) > 0)

    if valid_mask is None:
        valid_mask = torch.ones_like(gt_activity, dtype=torch.bool, device=device)
    else:
        valid_mask = valid_mask.bool().to(device)

    fa = ((pred_activity == 1) & (gt_activity == 0) & valid_mask).sum().float()
    miss = ((pred_activity == 0) & (gt_activity == 1) & valid_mask).sum().float()

    frame_slot_mismatch = ((pred_bin != target_sel).float().sum(dim=-1) > 0)
    conf = (frame_slot_mismatch & gt_activity & valid_mask).sum().float()

    gt_active = (gt_activity & valid_mask).sum().float()
    pred_active = (pred_activity & valid_mask).sum().float()

    der = (fa + miss + conf) / gt_active.clamp_min(1.0)

    if return_detail:
        return der, {
            "fa": float(fa.item()),
            "miss": float(miss.item()),
            "conf": float(conf.item()),
            "gt_active": float(gt_active.item()),
            "pred_active": float(pred_active.item()),
        }
    return der


@torch.no_grad()
def compute_activity_metrics(pred_act_logits, target_act, valid_mask, threshold=0.5):
    pred_act = (torch.sigmoid(pred_act_logits) >= threshold)
    pred_act = pred_act & valid_mask.bool()
    target_act = target_act.bool() & valid_mask.bool()

    tp = ((pred_act == 1) & (target_act == 1)).sum().item()
    fp = ((pred_act == 1) & (target_act == 0)).sum().item()
    fn = ((pred_act == 0) & (target_act == 1)).sum().item()

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return prec, rec, f1


@torch.no_grad()
def compute_count_acc_from_existence(exist_logits, target_count, threshold=0.5):
    """
    For query/attractor-based diarization:
    count = number of active speaker queries
    """
    pred_count = (torch.sigmoid(exist_logits) >= threshold).sum(dim=1)  # [B]
    gt = target_count.long().to(pred_count.device)
    acc = (pred_count == gt).float().mean().item()
    return acc


@torch.no_grad()
def compute_existence_acc(exist_logits, exist_targets, threshold=0.5):
    """
    Query-level existence accuracy against permutation-aligned existence targets.

    Args:
        exist_logits:  [B, N]
        exist_targets: [B, N]
    """
    pred_exist = (torch.sigmoid(exist_logits) >= threshold).float()
    gt_exist = exist_targets.float().to(pred_exist.device)
    acc = (pred_exist == gt_exist).float().mean().item()
    return acc


@torch.no_grad()
def compute_count_mae_from_existence(exist_logits, target_count, threshold=0.5):
    pred_count = (torch.sigmoid(exist_logits) >= threshold).sum(dim=1).float()
    gt = target_count.float().to(pred_count.device)
    mae = (pred_count - gt).abs().mean().item()
    return mae
