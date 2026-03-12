import numpy as np
import torch
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


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits: [B, C]
    targets: [B]
    """
    if logits.numel() == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    correct = (pred == targets).sum().item()
    return correct / max(1, int(targets.size(0)))


def compute_eer(labels, scores):
    """
    labels: list/np[int], 1=same, 0=diff
    scores: list/np[float], 越大越像同一个人
    rule: score >= th -> same
    return: eer, th
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.size == 0 or scores.size == 0:
        return 1.0, 0.0
    if labels.size != scores.size:
        raise ValueError(f"labels.size ({labels.size}) != scores.size ({scores.size})")

    P = int(labels.sum())
    N = int(labels.size - P)
    if P == 0 or N == 0:
        return 1.0, float(scores.max())

    order = np.argsort(scores)[::-1]
    scores_s = scores[order]
    labels_s = labels[order]

    tp = 0
    fp = 0

    best_eer = 0.5
    best_th = float(scores_s[0] + 1e-6)
    best_diff = abs(0.0 - 1.0)

    i = 0
    while i < labels_s.size:
        th = scores_s[i]
        j = i
        while j < labels_s.size and scores_s[j] == th:
            if labels_s[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1

        far = fp / N               # false accept rate
        frr = 1.0 - (tp / P)       # false reject rate
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer = (far + frr) / 2.0
            best_th = float(th)

        i = j

    return float(best_eer), float(best_th)


def roc_points(labels, scores, num_th=200):
    labels = list(labels)
    scores = list(scores)
    if len(scores) == 0:
        return [], []
    mn, mx = float(min(scores)), float(max(scores))
    if mx == mn:
        mx = mn + 1e-6

    ths = [mn + (mx - mn) * i / (num_th - 1) for i in range(num_th)]
    P = sum(labels)
    N = len(labels) - P

    tpr, fpr = [], []
    for th in ths:
        tp = sum(1 for l, s in zip(labels, scores) if l == 1 and s >= th)
        fp = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= th)
        tpr.append(tp / max(1, P))
        fpr.append(fp / max(1, N))
    return fpr, tpr


def det_points(labels, scores, num_th=400):
    labels = list(labels)
    scores = list(scores)
    if len(scores) == 0:
        return [], []
    mn, mx = float(min(scores)), float(max(scores))
    if mx == mn:
        mx = mn + 1e-6

    ths = [mn + (mx - mn) * i / (num_th - 1) for i in range(num_th)]
    P = sum(labels)
    N = len(labels) - P

    fars, frrs = [], []
    for th in ths:
        fa = sum(1 for l, s in zip(labels, scores) if l == 0 and s >= th)
        fr = sum(1 for l, s in zip(labels, scores) if l == 1 and s < th)
        fars.append(fa / max(1, N))
        frrs.append(fr / max(1, P))
    return fars, frrs


def recall_at_k(embeddings: torch.Tensor, labels: torch.Tensor, ks=(1, 5, 10)):
    """
    embeddings: [M, D] (建议已归一化)
    labels: [M] speaker id int
    """
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be [M,D], got {tuple(embeddings.shape)}")
    if labels.dim() != 1 or labels.size(0) != embeddings.size(0):
        raise ValueError("labels shape mismatch")

    # [M,M]
    sims = embeddings @ embeddings.t()
    sims.fill_diagonal_(-1e9)
    idx = torch.argsort(sims, dim=1, descending=True)

    M = sims.size(0)
    res = {}
    for k in ks:
        k = int(k)
        hit = 0
        topk = idx[:, :k]  # [M,k]
        for i in range(M):
            if (labels[topk[i]] == labels[i]).any().item():
                hit += 1
        res[k] = hit / max(1, M)
    return res


def l2norm(x: torch.Tensor, eps=1e-12):
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

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

    # 多标签逐槽位不一致也算 confusion
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