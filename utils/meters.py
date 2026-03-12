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

@torch.no_grad()
def diarization_error_rate_pit(
    slot_logits,
    target_matrix,
    target_activity,
    valid_mask=None,
    return_detail=False,
):
    """
    slot_logits:   [B,T,K]
    target_matrix: [B,T,K]
    target_activity: [B,T]
    """
    from utils.matching import hungarian_match_logits

    pred_bin = hungarian_match_logits(slot_logits, target_matrix, valid_mask=valid_mask)  # [B,T,K]

    pred_activity = (pred_bin.sum(dim=-1) > 0).float()
    gt_activity = (target_matrix.sum(dim=-1) > 0).float()

    if valid_mask is None:
        valid_mask = torch.ones_like(gt_activity, dtype=torch.bool)
    else:
        valid_mask = valid_mask.bool()

    fa = ((pred_activity == 1) & (gt_activity == 0) & valid_mask).sum().float()
    miss = ((pred_activity == 0) & (gt_activity == 1) & valid_mask).sum().float()

    conf = (((pred_bin != target_matrix).float().sum(dim=-1) > 0) & (gt_activity == 1) & valid_mask).sum().float()

    denom = (gt_activity == 1).float()[valid_mask].sum().clamp_min(1.0)
    der = (fa + miss + conf) / denom

    if return_detail:
        return der, {
            "fa": float(fa.item()),
            "miss": float(miss.item()),
            "conf": float(conf.item()),
            "gt_active": float(denom.item()),
            "pred_active": float(pred_activity[valid_mask].sum().item()),
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
