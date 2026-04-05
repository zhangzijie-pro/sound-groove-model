import torch
from tqdm import tqdm

from speaker_verification.utils.meters import (
    AverageMeter,
    DERTracker,
    diarization_error_rate_pit,
    compute_activity_metrics,
    compute_count_acc_from_existence,
    compute_existence_acc,
    compute_count_mae_from_existence,
)


def _remove_short_runs(binary_seq: torch.Tensor, min_active_frames: int = 3) -> torch.Tensor:
    """
    binary_seq: [T] float/bool tensor with values in {0,1}
    Remove active runs shorter than min_active_frames.
    """
    out = binary_seq.clone()
    T = out.numel()
    t = 0
    while t < T:
        if float(out[t].item()) < 0.5:
            t += 1
            continue

        end = t + 1
        while end < T and float(out[end].item()) >= 0.5:
            end += 1

        if end - t < min_active_frames:
            out[t:end] = 0.0

        t = end
    return out


@torch.no_grad()
def _decode_predictions(
    diar_logits: torch.Tensor,
    exist_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    activity_threshold: float = 0.5,
    exist_threshold: float = 0.5,
    min_active_frames: int = 3,
):
    """
    Decode one sample.

    Args:
        diar_logits: [T, N]
        exist_logits: [N]
        valid_mask: [T]

    Returns:
        pred_bin_full: [T, N] binary speaker activity after existence gating
        pred_count: int
        pred_activity_logits: [T]
        pred_exist: [N]
        keep_mask: [N]
    """
    diar_prob = torch.sigmoid(diar_logits)          # [T,N]
    exist_prob = torch.sigmoid(exist_logits)        # [N]

    keep_mask = exist_prob >= exist_threshold       # [N]
    pred_exist = keep_mask.float()
    pred_count = int(keep_mask.sum().item())

    pred_bin_full = torch.zeros_like(diar_prob, dtype=torch.float32)

    if pred_count > 0:
        pred_bin_full[:, keep_mask] = (diar_prob[:, keep_mask] >= activity_threshold).float()

        # light smoothing on each active query stream
        for n in range(pred_bin_full.size(1)):
            if bool(keep_mask[n].item()):
                pred_bin_full[:, n] = _remove_short_runs(
                    pred_bin_full[:, n],
                    min_active_frames=min_active_frames,
                )

    pred_bin_full = pred_bin_full * valid_mask.unsqueeze(-1).float()

    # activity logits for VAD-like metric
    pred_activity_logits = diar_logits.max(dim=-1).values  # [T]

    return pred_bin_full, pred_count, pred_activity_logits, pred_exist, keep_mask


@torch.no_grad()
def validate(
    model,
    loss_fn,
    loader,
    device,
    max_batches: int = -1,
    activity_threshold: float = 0.5,
    exist_threshold: float = 0.5,
    min_active_frames: int = 3,
):
    model.eval()

    val_loss_meter = AverageMeter()
    pit_meter = AverageMeter()
    exist_meter = AverageMeter()
    pull_meter = AverageMeter()
    sep_meter = AverageMeter()
    smooth_meter = AverageMeter()

    der_tracker = DERTracker()

    count_acc_sum = 0.0
    count_acc_n = 0

    count_mae_sum = 0.0
    count_mae_n = 0

    act_prec_sum = 0.0
    act_rec_sum = 0.0
    act_f1_sum = 0.0
    act_n = 0

    exist_acc_sum = 0.0
    exist_acc_n = 0

    total_steps = len(loader) if max_batches is None or max_batches < 0 else min(len(loader), max_batches)
    pbar = tqdm(loader, desc="VALID", total=total_steps, ncols=170)

    for bi, batch in enumerate(pbar):
        if max_batches is not None and max_batches >= 0 and bi >= max_batches:
            break

        fbank = batch["fbank"].to(device, non_blocking=True)                  # [B,T,F]
        target_matrix = batch["target_matrix"].to(device, non_blocking=True)  # [B,T,K]
        target_count = batch["target_count"].to(device, non_blocking=True)    # [B]
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)        # [B,T]

        frame_embeds, attractors, exist_logits, diar_logits = model(fbank)

        loss_dict = loss_fn(
            frame_embeds=frame_embeds,
            attractors=attractors,
            exist_logits=exist_logits,
            diar_logits=diar_logits,
            target_matrix=target_matrix,
            target_count=target_count,
            valid_mask=valid_mask,
            return_detail=True,
        )

        bs = fbank.size(0)

        val_loss_meter.update(float(loss_dict["total"].item()), bs)
        pit_meter.update(float(loss_dict["pit_loss"].item()), bs)
        exist_meter.update(float(loss_dict["exist_loss"].item()), bs)
        pull_meter.update(float(loss_dict["pull_loss"].item()), bs)
        sep_meter.update(float(loss_dict["sep_loss"].item()), bs)
        smooth_meter.update(float(loss_dict["smooth_loss"].item()), bs)

        # ------------------------------------------------------------
        # 1) training-time proxy DER (still PIT-based)
        # ------------------------------------------------------------
        _, der_info = diarization_error_rate_pit(
            diar_logits,
            target_matrix,
            valid_mask=valid_mask,
            return_detail=True,
        )
        der_tracker.update(der_info)

        # ------------------------------------------------------------
        # 2) count / existence metrics
        # ------------------------------------------------------------
        batch_count_acc = compute_count_acc_from_existence(
            exist_logits=exist_logits,
            target_count=target_count,
            threshold=exist_threshold,
        )
        count_acc_sum += batch_count_acc
        count_acc_n += 1

        batch_count_mae = compute_count_mae_from_existence(
            exist_logits=exist_logits,
            target_count=target_count,
            threshold=exist_threshold,
        )
        count_mae_sum += batch_count_mae
        count_mae_n += 1

        batch_exist_acc = compute_existence_acc(
            exist_logits=exist_logits,
            exist_targets=loss_dict["exist_targets"],
            threshold=exist_threshold,
        )
        exist_acc_sum += batch_exist_acc
        exist_acc_n += 1

        # ------------------------------------------------------------
        # 3) activity metrics
        # use diar max as frame activity surrogate
        # ------------------------------------------------------------
        target_activity = (target_matrix.sum(dim=-1) > 0).float()  # [B,T]

        ap, ar, af1 = compute_activity_metrics(
            pred_act_logits=diar_logits.max(dim=-1).values,
            target_act=target_activity,
            valid_mask=valid_mask,
            threshold=activity_threshold,
        )
        act_prec_sum += ap
        act_rec_sum += ar
        act_f1_sum += af1
        act_n += 1

        # ------------------------------------------------------------
        # 4) optional per-sample decode (debug / sanity path)
        # not used for main metrics aggregation currently, but useful
        # to ensure inference path is consistent with validate path
        # ------------------------------------------------------------
        # You can later expose this for richer validation outputs.
        for b in range(bs):
            vm = valid_mask[b].bool()
            _ = _decode_predictions(
                diar_logits=diar_logits[b][vm],
                exist_logits=exist_logits[b],
                valid_mask=torch.ones_like(valid_mask[b][vm], dtype=torch.bool),
                activity_threshold=activity_threshold,
                exist_threshold=exist_threshold,
                min_active_frames=min_active_frames,
            )

        pbar.set_postfix(
            total=f"{val_loss_meter.avg:.4f}",
            pit=f"{pit_meter.avg:.4f}",
            der=f"{der_tracker.value() * 100.0:.2f}%",
            cacc=f"{(count_acc_sum / max(count_acc_n, 1)) * 100.0:.2f}%",
            cmae=f"{(count_mae_sum / max(count_mae_n, 1)):.3f}",
            af1=f"{(act_f1_sum / max(act_n, 1)) * 100.0:.2f}%",
            eacc=f"{(exist_acc_sum / max(exist_acc_n, 1)) * 100.0:.2f}%",
        )

    der_detail = der_tracker.detail()

    out = {
        "val_loss": val_loss_meter.avg,
        "pit_loss": pit_meter.avg,
        "exist_loss": exist_meter.avg,
        "pull_loss": pull_meter.avg,
        "sep_loss": sep_meter.avg,
        "smooth_loss": smooth_meter.avg,
        "der": der_detail["der"] * 100.0,
        "der_detail": {
            "fa": der_detail["fa"],
            "miss": der_detail["miss"],
            "conf": der_detail["conf"],
            "gt_active": der_detail["gt_active"],
            "pred_active": der_detail["pred_active"],
        },
        "count_acc": count_acc_sum / max(1, count_acc_n),
        "count_mae": count_mae_sum / max(1, count_mae_n),
        "act_prec": act_prec_sum / max(1, act_n),
        "act_rec": act_rec_sum / max(1, act_n),
        "act_f1": act_f1_sum / max(1, act_n),
        "exist_acc": exist_acc_sum / max(1, exist_acc_n),
    }
    return out
