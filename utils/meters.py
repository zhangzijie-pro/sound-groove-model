import torch


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
    """
    全局累计 DER 统计器：
    最终用全局 sum(fa/miss/conf) / sum(gt_active)
    """
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
def diarization_error_rate_pit(
    slot_logits,
    target_matrix,
    target_activity=None,
    valid_mask=None,
    return_detail=False,
):
    """
    内部训练用帧级 PIT-DER，支持 slot_logits K > target_matrix K 的情况
    （例如加了 silence prototype 后 K=5/6，但 target 仍是 4）
    """
    from utils.matching import hungarian_match_logits

    target_matrix = target_matrix.float()

    K_logit = slot_logits.shape[-1]
    K_tgt   = target_matrix.shape[-1]
    K = min(K_logit, K_tgt)

    pred_bin_full = hungarian_match_logits(slot_logits, target_matrix, valid_mask=valid_mask)  # [B,T,K_logit]

    pred_bin = pred_bin_full[:, :, :K]          # [B, T, K]
    target_sel = target_matrix[:, :, :K]        # [B, T, K]

    gt_activity = (target_matrix.sum(dim=-1) > 0)
    pred_activity = (pred_bin.sum(dim=-1) > 0)

    if valid_mask is None:
        valid_mask = torch.ones_like(gt_activity, dtype=torch.bool)
    else:
        valid_mask = valid_mask.bool()

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
    pred = pred_count_logits.argmax(dim=1)  # [B], 0..K-1
    gt = target_count.long()
    if gt.numel() > 0 and gt.min().item() >= 1:
        gt = gt - 1
    gt = gt.clamp(0, pred_count_logits.size(1) - 1)
    acc = (pred == gt).float().mean().item()
    return acc