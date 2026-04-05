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
    device = diar_logits.device
    target_matrix = target_matrix.float().to(device)

    B, T, K_logit = diar_logits.shape
    _, _, K_tgt = target_matrix.shape

    has_silence = (K_logit == K_tgt + 1)

    if has_silence:
        silence_logits = diar_logits[..., :1]   # [B, T, 1]
        speaker_logits = diar_logits[..., 1:]   # [B, T, K_logit-1]
    else:
        silence_logits = None
        speaker_logits = diar_logits

    K_spk_logit = speaker_logits.size(-1)
    K = min(K_spk_logit, K_tgt)

    if valid_mask is None:
        valid_mask = torch.ones((B, T), dtype=torch.bool, device=device)
    else:
        valid_mask = valid_mask.bool().to(device)

    pred_bin_full = torch.zeros((B, T, K_logit), dtype=torch.float32, device=device)

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

@torch.no_grad()
def diarization_error_rate_pit(
    diar_logits,
    target_matrix,
    target_activity=None,
    valid_mask=None,
    return_detail=False,
):
    device = diar_logits.device
    target_matrix = target_matrix.float().to(device)

    B, T, K_logit = diar_logits.shape
    _, _, K_tgt = target_matrix.shape

    has_silence = (K_logit == K_tgt + 1)

    pred_bin_full = hungarian_match_logits(
        diar_logits,
        target_matrix,
        valid_mask=valid_mask,
    )  # [B, T, K_logit]

    if has_silence:
        pred_bin = pred_bin_full[..., 1:]
    else:
        pred_bin = pred_bin_full

    K = min(pred_bin.size(-1), K_tgt)
    pred_bin = pred_bin[..., :K]              # [B, T, K]
    target_sel = target_matrix[..., :K]       # [B, T, K]

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
def compute_count_acc(pred_count_logits, target_count):
    pred = pred_count_logits.argmax(dim=1)  # [B], 0..K
    gt = target_count.long()
    gt = gt.clamp(0, pred_count_logits.size(1) - 1)
    acc = (pred == gt).float().mean().item()
    return acc
