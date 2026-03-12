import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.loss.aamsoftmax import AAMSoftmax
from speaker_verification.loss.pit import PITLoss


def _to_scalar_loss(x, device):
    if torch.is_tensor(x):
        return x

    if isinstance(x, (float, int)):
        return torch.tensor(float(x), device=device)

    if isinstance(x, (tuple, list)):
        if len(x) == 0:
            return torch.tensor(0.0, device=device)
        return _to_scalar_loss(x[0], device)

    if isinstance(x, dict):
        for k in ("loss", "total", "diar_loss", "ver_loss"):
            if k in x:
                return _to_scalar_loss(x[k], device)

        s = 0.0
        found = False
        for v in x.values():
            if torch.is_tensor(v):
                s = s + v
                found = True
            elif isinstance(v, (float, int)):
                s = s + float(v)
                found = True

        if found:
            return _to_scalar_loss(s, device)

        return torch.tensor(0.0, device=device)

    raise TypeError(f"Unsupported loss type: {type(x)}")


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        embedding_dim=192,
        num_classes=1000,
        max_spk=5,
        lambda_ver=1.0,
        lambda_pit=1.0,
        lambda_act=1.0,
        lambda_cnt=0.2,
        pos_weight=2.0,
        pit_pos_weight=1.5,
    ):
        super().__init__()
        self.ver_loss = AAMSoftmax(embedding_dim, num_classes)
        self.pit_loss = PITLoss(pos_weight=pit_pos_weight)

        self.max_spk = int(max_spk)
        self.lambda_ver = float(lambda_ver)
        self.lambda_pit = float(lambda_pit)
        self.lambda_act = float(lambda_act)
        self.lambda_cnt = float(lambda_cnt)

        self.register_buffer(
            "act_pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32)
        )

    def forward(
        self,
        emb,
        slot_logits,         # [B,T,K]
        pred_activity,       # [B,T]
        pred_count,          # [B,K]
        label,               # [B]
        target_matrix,       # [B,T,K]
        target_activity,     # [B,T]
        target_count,        # [B]
        valid_mask=None,
    ):
        device = emb.device
        label = label.long().to(device)

        # 1) SV loss
        ver_loss = self.ver_loss(emb, label)
        ver_loss = _to_scalar_loss(ver_loss, device)

        # 2) PIT loss
        pit_loss = self.pit_loss(slot_logits, target_matrix, valid_mask=valid_mask)
        pit_loss = _to_scalar_loss(pit_loss, device)

        # 3) activity loss
        if valid_mask is None:
            valid_mask = torch.ones_like(target_activity, dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.bool().to(device)

        act_loss_raw = F.binary_cross_entropy_with_logits(
            pred_activity,
            target_activity.float(),
            reduction="none",
            pos_weight=self.act_pos_weight.to(device),
        )
        act_loss = act_loss_raw[valid_mask].mean() if valid_mask.any() else pred_activity.new_tensor(0.0)

        # 4) count loss
        tc = target_count.long().to(device)
        if tc.min().item() >= 1:
            tc = tc - 1
        tc = tc.clamp(0, pred_count.size(1) - 1)
        cnt_loss = F.cross_entropy(pred_count, tc)

        total = (
            self.lambda_ver * ver_loss
            + self.lambda_pit * pit_loss
            + self.lambda_act * act_loss
            + self.lambda_cnt * cnt_loss
        )
        return total