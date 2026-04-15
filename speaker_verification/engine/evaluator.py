import torch
from tqdm import tqdm

from speaker_verification.utils.meters import (
    AverageMeter,
    DERTracker,
    diarization_error_rate_decoded,
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
    speaker_activity_threshold: float = 0.6,
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
        pred_activity: [T]
        pred_exist: [N]
        keep_mask: [N]
    """
    diar_prob = torch.sigmoid(diar_logits)          # [T,N]
    exist_prob = torch.sigmoid(exist_logits)        # [N]

    keep_mask = exist_prob >= exist_threshold       # [N]
    if not bool(keep_mask.any().item()):
        # Avoid all-zero collapse when existence head briefly becomes over-conservative.
        keep_mask = diar_prob.max(dim=0).values >= speaker_activity_threshold

    pred_exist = keep_mask.float()
    pred_count = int(keep_mask.sum().item())

    pred_bin_full = torch.zeros_like(diar_prob, dtype=torch.float32)

    if pred_count > 0:
        pred_bin_full[:, keep_mask] = (diar_prob[:, keep_mask] >= speaker_activity_threshold).float()
        for n in range(pred_bin_full.size(1)):
            if bool(keep_mask[n].item()):
                pred_bin_full[:, n] = _remove_short_runs(
                    pred_bin_full[:, n],
                    min_active_frames=min_active_frames,
                )

    pred_bin_full = pred_bin_full * valid_mask.unsqueeze(-1).float()
    pred_activity = (pred_bin_full.sum(dim=-1) > 0).float()
    pred_activity = _remove_short_runs(pred_activity, min_active_frames=min_active_frames)
    pred_activity = pred_activity * valid_mask.float()

    return pred_bin_full, pred_count, pred_activity, pred_exist, keep_mask


@torch.no_grad()
def validate(
    model,
    loss_fn,
    loader,
    device,
    max_batches: int = -1,
    speaker_activity_threshold: float = 0.6,
    speaker_activity_sweep_thresholds=None,
    exist_threshold: float = 0.5,
    min_active_frames: int = 3,
):
    model.eval()

    val_loss_meter = AverageMeter()
    pit_meter = AverageMeter()
    exist_meter = AverageMeter()

    der_tracker = DERTracker()
    sweep_thresholds = []
    if speaker_activity_sweep_thresholds is not None:
        sweep_thresholds = sorted({float(thr) for thr in speaker_activity_sweep_thresholds})
    sweep_trackers = {thr: DERTracker() for thr in sweep_thresholds}

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

        _, _, exist_logits, diar_logits, _ = model(
            fbank,
            valid_mask=valid_mask,
        )

        loss_dict = loss_fn(
            exist_logits=exist_logits,
            diar_logits=diar_logits,
            target_matrix=target_matrix,
            valid_mask=valid_mask,
            return_detail=True,
        )

        bs = fbank.size(0)

        val_loss_meter.update(float(loss_dict["total"].item()), bs)
        pit_meter.update(float(loss_dict["pit_loss"].item()), bs)
        exist_meter.update(float(loss_dict["exist_loss"].item()), bs)

        # ------------------------------------------------------------
        # 1) decoded DER aligned with inference-time gating
        # ------------------------------------------------------------
        decoded_preds = []
        decoded_activity = []
        for b in range(bs):
            pred_bin_full, _, pred_activity, _, _ = _decode_predictions(
                diar_logits=diar_logits[b],
                exist_logits=exist_logits[b],
                valid_mask=valid_mask[b],
                speaker_activity_threshold=speaker_activity_threshold,
                exist_threshold=exist_threshold,
                min_active_frames=min_active_frames,
            )
            decoded_preds.append(pred_bin_full)
            decoded_activity.append(pred_activity)

        decoded_pred_batch = torch.stack(decoded_preds, dim=0)
        decoded_activity_batch = torch.stack(decoded_activity, dim=0)

        _, der_info = diarization_error_rate_decoded(
            decoded_pred_batch,
            target_matrix,
            valid_mask=valid_mask,
            return_detail=True,
        )
        der_tracker.update(der_info)

        for sweep_thr, sweep_tracker in sweep_trackers.items():
            if abs(sweep_thr - float(speaker_activity_threshold)) < 1e-9:
                sweep_tracker.update(der_info)
                continue

            sweep_preds = []
            for b in range(bs):
                pred_bin_full, _, _, _, _ = _decode_predictions(
                    diar_logits=diar_logits[b],
                    exist_logits=exist_logits[b],
                    valid_mask=valid_mask[b],
                    speaker_activity_threshold=sweep_thr,
                    exist_threshold=exist_threshold,
                    min_active_frames=min_active_frames,
                )
                sweep_preds.append(pred_bin_full)

            sweep_pred_batch = torch.stack(sweep_preds, dim=0)
            _, sweep_info = diarization_error_rate_decoded(
                sweep_pred_batch,
                target_matrix,
                valid_mask=valid_mask,
                return_detail=True,
            )
            sweep_tracker.update(sweep_info)

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
        # ------------------------------------------------------------
        target_activity = (target_matrix.sum(dim=-1) > 0) & valid_mask.bool()   # [B,T]
        pred_activity = decoded_activity_batch.bool() & valid_mask.bool()

        tp = ((pred_activity == 1) & (target_activity == 1)).sum().item()
        fp = ((pred_activity == 1) & (target_activity == 0)).sum().item()
        fn = ((pred_activity == 0) & (target_activity == 1)).sum().item()

        ap = tp / max(tp + fp, 1)
        ar = tp / max(tp + fn, 1)
        af1 = 2 * ap * ar / max(ap + ar, 1e-8)
        act_prec_sum += ap
        act_rec_sum += ar
        act_f1_sum += af1
        act_n += 1

        pbar.set_postfix(
            total=f"{val_loss_meter.avg:.4f}",
            pit=f"{pit_meter.avg:.4f}",
            exist=f"{exist_meter.avg:.4f}",
            der=f"{der_tracker.value() * 100.0:.2f}%",
            cacc=f"{(count_acc_sum / max(count_acc_n, 1)) * 100.0:.2f}%",
            cmae=f"{(count_mae_sum / max(count_mae_n, 1)):.3f}",
            af1=f"{(act_f1_sum / max(act_n, 1)) * 100.0:.2f}%",
            eacc=f"{(exist_acc_sum / max(exist_acc_n, 1)) * 100.0:.2f}%",
        )

    der_detail = der_tracker.detail()
    sweep_detail = {
        str(thr): tracker.detail()["der"] * 100.0
        for thr, tracker in sweep_trackers.items()
    }
    if sweep_detail:
        best_sweep_threshold, best_sweep_der = min(
            ((float(thr), der) for thr, der in sweep_detail.items()),
            key=lambda item: item[1],
        )
    else:
        best_sweep_threshold = None
        best_sweep_der = None

    out = {
        "val_loss": val_loss_meter.avg,
        "pit_loss": pit_meter.avg,
        "exist_loss": exist_meter.avg,
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
    if best_sweep_der is not None:
        out["sweep_der"] = best_sweep_der
        out["sweep_threshold"] = best_sweep_threshold
        out["sweep_detail"] = sweep_detail
    return out
