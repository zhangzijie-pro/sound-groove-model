import torch
from tqdm import tqdm

from speaker_verification.utils.meters import (
    AverageMeter,
    DERTracker,
    diarization_error_rate_pit,
    compute_count_acc,
    compute_activity_metrics,
)


@torch.no_grad()
def validate(model, loss_fn, loader, device, max_batches=-1, activity_threshold=0.5):
    model.eval()

    val_loss_meter = AverageMeter()
    pit_meter = AverageMeter()
    act_meter = AverageMeter()
    cnt_meter = AverageMeter()
    frm_meter = AverageMeter()
    der_tracker = DERTracker()

    count_acc_sum = 0.0
    count_acc_n = 0
    act_prec_sum = 0.0
    act_rec_sum = 0.0
    act_f1_sum = 0.0
    act_n = 0

    total_steps = len(loader) if max_batches is None or max_batches < 0 else min(len(loader), max_batches)
    pbar = tqdm(loader, desc="VALID", total=total_steps, ncols=120)

    for bi, batch in enumerate(pbar):
        if max_batches is not None and max_batches >= 0 and bi >= max_batches:
            break

        fbank = batch["fbank"].to(device, non_blocking=True)
        target_matrix = batch["target_matrix"].to(device, non_blocking=True)
        target_activity = batch["target_activity"].to(device, non_blocking=True)
        target_count = batch["target_count"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        _, frame_embeddings, slot_logits, pred_activity, pred_count = model(
            fbank, return_diarization=True
        )

        loss_dict = loss_fn(
            frame_embeds=frame_embeddings,
            slot_logits=slot_logits,
            pred_activity=pred_activity,
            pred_count=pred_count,
            target_matrix=target_matrix,
            target_activity=target_activity,
            target_count=target_count,
            valid_mask=valid_mask,
            return_detail=True,
        )

        loss = loss_dict["total"]
        bs = fbank.size(0)

        val_loss_meter.update(float(loss.item()), bs)
        pit_meter.update(float(loss_dict["pit_loss"].item()), bs)
        act_meter.update(float(loss_dict["act_loss"].item()), bs)
        cnt_meter.update(float(loss_dict["cnt_loss"].item()), bs)
        frm_meter.update(float(loss_dict["frm_loss"].item()), bs)

        _, info = diarization_error_rate_pit(
            slot_logits,
            target_matrix,
            target_activity,
            valid_mask=valid_mask,
            return_detail=True,
        )
        der_tracker.update(info)

        count_acc_sum += compute_count_acc(pred_count, target_count)
        count_acc_n += 1

        ap, ar, af1 = compute_activity_metrics(
            pred_activity, target_activity, valid_mask, threshold=activity_threshold
        )
        act_prec_sum += ap
        act_rec_sum += ar
        act_f1_sum += af1
        act_n += 1

        pbar.set_postfix(
            loss=f"{val_loss_meter.avg:.4f}",
            DER=f"{der_tracker.value() * 100.0:.2f}%",
            CAcc=f"{(count_acc_sum / max(count_acc_n, 1)) * 100:.1f}%",
            ActF1=f"{(act_f1_sum / max(act_n, 1)) * 100:.1f}%",
        )

    der_detail = der_tracker.detail()

    return {
        "val_loss": val_loss_meter.avg,
        "pit_loss": pit_meter.avg,
        "act_loss": act_meter.avg,
        "cnt_loss": cnt_meter.avg,
        "frm_loss": frm_meter.avg,
        "der": der_detail["der"] * 100.0,
        "der_detail": {
            "fa": der_detail["fa"],
            "miss": der_detail["miss"],
            "conf": der_detail["conf"],
            "gt_active": der_detail["gt_active"],
            "pred_active": der_detail["pred_active"],
        },
        "count_acc": count_acc_sum / max(1, count_acc_n),
        "act_prec": act_prec_sum / max(1, act_n),
        "act_rec": act_rec_sum / max(1, act_n),
        "act_f1": act_f1_sum / max(1, act_n),
    }