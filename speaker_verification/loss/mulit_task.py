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
        proto_temperature=0.05,   # ← 理论最优锐利值
        margin=0.3,               # ← 新增：margin push negatives
    ):
        super().__init__()
        self.pit_loss = PITLoss(pos_weight=pit_pos_weight)

        self.max_spk = int(max_spk)
        self.lambda_pit = float(lambda_pit)
        self.lambda_act = float(lambda_act)
        self.lambda_cnt = float(lambda_cnt)
        self.lambda_frm = float(lambda_frm)
        self.proto_eps = float(proto_eps)
        self.proto_temperature = float(proto_temperature)
        self.margin = float(margin)                     # 新增

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
        device = frame_embeds.device
        B, T, D = frame_embeds.shape

        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.bool().to(device)

        frame_embeds = F.normalize(frame_embeds, p=2, dim=-1)

        total_proto_loss = frame_embeds.new_tensor(0.0)
        total_proto_count = frame_embeds.new_tensor(0.0)

        total_temp_loss = frame_embeds.new_tensor(0.0)
        total_temp_count = frame_embeds.new_tensor(0.0)

        for b in range(B):
            vb = valid_mask[b]
            if not vb.any():
                continue

            f = frame_embeds[b][vb]                # [Tv, D]
            target = target_matrix[b][vb].float()  # [Tv, K]

            # ---------- prototype contrastive ----------
            spk_mask = target.sum(dim=0) > self.proto_eps
            n_active = int(spk_mask.sum().item())
            if n_active >= 2:
                weights = target[:, spk_mask]   # [Tv, n_active]
                denom = weights.sum(dim=0).clamp_min(self.proto_eps)

                protos = (weights.unsqueeze(-1) * f.unsqueeze(1)).sum(dim=0) / denom.unsqueeze(-1)
                protos = F.normalize(protos, p=2, dim=-1)    # [n_active, D]

                sim = torch.matmul(f, protos.T).float() / self.proto_temperature

                pos_mask = weights > 0.5
                has_pos = pos_mask.any(dim=1)
                if has_pos.any():
                    sim_valid = sim[has_pos]                    # [N_valid, n_active]
                    pos_mask_valid = pos_mask[has_pos]

                    # === 关键理论增强：margin push negatives ===
                    sim_denom = sim_valid.clone()
                    sim_denom[~pos_mask_valid] -= self.margin   # neg sim 下推，严格 contrastive

                    log_denom = torch.logsumexp(sim_denom, dim=1, keepdim=True)

                    neg_inf = torch.full_like(sim_valid, torch.finfo(sim_valid.dtype).min)
                    pos_logits = torch.where(pos_mask_valid, sim_valid, neg_inf)
                    log_pos = torch.logsumexp(pos_logits, dim=1, keepdim=True)

                    proto_loss = -(log_pos - log_denom).squeeze(-1)

                    total_proto_loss += proto_loss.sum()
                    total_proto_count += proto_loss.numel()

            # ---------- temporal consistency（你已实现的优秀部分，保留） ----------
            if f.size(0) >= 2:
                target_bin = (target > 0.5).float()
                same_set = (target_bin[1:] == target_bin[:-1]).all(dim=1)   # [Tv-1]

                if same_set.any():
                    diff = f[1:] - f[:-1]
                    temp_loss = (diff.pow(2).sum(dim=1))[same_set]

                    total_temp_loss += temp_loss.sum()
                    total_temp_count += temp_loss.numel()

        proto_loss = (
            total_proto_loss / total_proto_count
            if total_proto_count.item() > 0
            else frame_embeds.new_tensor(0.0)
        )

        temp_loss = (
            total_temp_loss / total_temp_count
            if total_temp_count.item() > 0
            else frame_embeds.new_tensor(0.0)
        )

        return proto_loss + 0.3 * temp_loss   # temporal 权重保持稳定

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
        target_matrix = target_matrix.to(device).float()
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

        # 4) frame prototype + temporal（已理论最优）
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