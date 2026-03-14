import torch
import torch.nn as nn
import torch.nn.functional as F

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
        max_spk=10,
        lambda_pit=1.0,
        lambda_act=1.0,
        lambda_cnt=0.2,
        lambda_frm=0.5,
        pos_weight=2.0,
        pit_pos_weight=1.5,
        proto_eps=1e-8,
    ):
        super().__init__()
        self.pit_loss = PITLoss(pos_weight=pit_pos_weight)

        self.max_spk = int(max_spk)
        self.lambda_pit = float(lambda_pit)
        self.lambda_act = float(lambda_act)
        self.lambda_cnt = float(lambda_cnt)
        self.lambda_frm = float(lambda_frm)
        self.proto_eps = float(proto_eps)

        self.register_buffer(
            "act_pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32)
        )

    def _frame_proto_loss(
        self,
        frame_embeds,
        target_matrix,
        valid_mask=None,
    ):
        """
        bug unsupport(类内拉近 + 类间推远)
        同一 speaker 的帧接近自己的 prototype

        远离其他 speaker 的 prototype
        """
        device = frame_embeds.device
        B, T, D = frame_embeds.shape
        K = target_matrix.size(-1)

        target_matrix = target_matrix.float().to(device)

        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.bool().to(device)

        total_loss = frame_embeds.new_tensor(0.0)
        total_count = frame_embeds.new_tensor(0.0)

        frame_embeds = F.normalize(frame_embeds, p=2, dim=-1)

        for b in range(B):
            vb = valid_mask[b]                       # [T]
            fb = frame_embeds[b]                     # [T,D]
            tb = target_matrix[b]                    # [T,K]

            if not vb.any():
                continue
            fb_valid = fb[vb]                        # [Tv,D]
            tb_valid = tb[vb]                        # [Tv,K]

            # speaker prototype: [K,D]
            # numer = sum_t mask(t,k) * f_t
            weights = tb_valid                       # [Tv,K]
            denom = weights.sum(dim=0).clamp_min(self.proto_eps)  # [K]

            # [K,D]
            protos = torch.einsum("tk,td->kd", weights, fb_valid) / denom.unsqueeze(-1)
            protos = F.normalize(protos, p=2, dim=-1)

            # [Tv,K]
            sim = torch.einsum("td,kd->tk", fb_valid, protos)

            # loss = 1 - cos
            # pos_loss = (1.0 - sim) * weights
            pos_loss = ((1.0 - sim).clamp_min(0.0)) * weights

            pos_count = weights.sum()
            if pos_count > 0:
                total_loss = total_loss + pos_loss.sum()
                total_count = total_count + pos_count

        if total_count.item() == 0:
            return frame_embeds.new_tensor(0.0)

        return total_loss / total_count

    def forward(
        self,
        frame_embeds,       # [B,T,D]
        slot_logits,        # [B,T,K]
        pred_activity,      # [B,T]
        pred_count,         # [B,K]
        target_matrix,      # [B,T,K]
        target_activity,    # [B,T]
        target_count,       # [B]
        valid_mask=None,    # [B,T]
        return_detail=False,
    ):
        device = frame_embeds.device
        target_matrix = target_matrix.to(device)
        target_activity = target_activity.to(device)
        target_count = target_count.to(device)

        # 1) PIT
        pit_loss = self.pit_loss(slot_logits, target_matrix, valid_mask=valid_mask)
        pit_loss = _to_scalar_loss(pit_loss, device)

        # 2) activity
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

        # 3) count
        tc = target_count.long().to(device)
        if tc.numel() > 0 and tc.min().item() >= 1:
            tc = tc - 1
        tc = tc.clamp(0, pred_count.size(1) - 1)
        cnt_loss = F.cross_entropy(pred_count, tc)

        # 4) frame prototype consistency
        frm_loss = self._frame_proto_loss(
            frame_embeds=frame_embeds,
            target_matrix=target_matrix,
            valid_mask=valid_mask,
        )

        total = (
            self.lambda_pit * pit_loss
            + self.lambda_act * act_loss
            + self.lambda_cnt * cnt_loss
            + self.lambda_frm * frm_loss
        )

        if return_detail:
            return {
                "total": total,
                "pit_loss": pit_loss.detach(),
                "act_loss": act_loss.detach(),
                "cnt_loss": cnt_loss.detach(),
                "frm_loss": frm_loss.detach(),
            }

        return total