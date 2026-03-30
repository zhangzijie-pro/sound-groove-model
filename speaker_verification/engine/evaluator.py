import torch
import torch.nn.functional as F
from tqdm import tqdm

from speaker_verification.utils.meters import (
    AverageMeter,
    DERTracker,
    diarization_error_rate_pit,
    compute_count_acc,
    compute_activity_metrics,
)
from speaker_verification.interfaces.diar_interface import ChunkInferenceResult, SlotResult
from speaker_verification.interfaces.global_tracker import GlobalSpeakerTracker


def _extract_pred_local_ids_and_count(slot_logits, pred_activity, pred_count, threshold=0.5):
    """
    slot_logits: [T, K] or [T, K+1]
    pred_activity: [T]
    pred_count: [K+1] logits
    """
    activity_prob = torch.sigmoid(pred_activity)
    active_mask = activity_prob >= threshold

    n_spk = int(pred_count.argmax(dim=0).item())

    if slot_logits.size(-1) >= 2:
        # 若含 silence slot，则去掉第0类
        speaker_logits = slot_logits[:, 1:] if slot_logits.size(-1) == pred_count.numel() else slot_logits
    else:
        speaker_logits = slot_logits

    local_ids = speaker_logits.argmax(dim=-1) + 1  # 从1开始
    local_ids = local_ids * active_mask.long()

    return local_ids, max(n_spk, 0), active_mask


def _build_slot_results_from_frame_embeds(frame_embeds, local_frame_ids, max_slots=None):
    """
    frame_embeds: [T, D]
    local_frame_ids: [T], 0=inactive, 1..N=local slot
    """
    slots = []
    slot_ids = sorted(set(local_frame_ids.tolist()) - {0})
    if max_slots is not None:
        slot_ids = slot_ids[:max_slots]

    for sid in slot_ids:
        mask = local_frame_ids == int(sid)
        n = int(mask.sum().item())
        if n <= 0:
            continue
        proto = F.normalize(frame_embeds[mask].mean(dim=0), p=2, dim=-1)
        slots.append(
            SlotResult(
                slot=int(sid),
                name=f"slot_{sid}",
                score=None,
                is_known=False,
                num_frames=n,
                duration_sec=float(n) * 0.01,
                prototype=proto.detach().cpu(),
            )
        )
    return slots


def _mean_proto_cos(slots_a, slots_b):
    """
    按 slot id 对齐后计算 prototype cosine
    """
    map_a = {int(s.slot): s for s in slots_a}
    map_b = {int(s.slot): s for s in slots_b}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(common) == 0:
        return None, 0

    vals = []
    for sid in common:
        a = F.normalize(map_a[sid].prototype.float(), dim=0)
        b = F.normalize(map_b[sid].prototype.float(), dim=0)
        vals.append(float(torch.dot(a, b).item()))
    return sum(vals) / len(vals), len(vals)


@torch.no_grad()
def validate(
    model,
    loss_fn,
    loader,
    device,
    max_batches=-1,
    activity_threshold=0.5,
    frame_quantizer=None,
    tracker_quantizer=None,
    run_quant_tracker=False,
):
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

    proto_cos_sum = 0.0
    proto_cos_n = 0
    tracker_updates = 0

    tracker = GlobalSpeakerTracker(
        match_threshold=0.72,
        momentum=0.9,
        max_misses=30,
        device=str(device),
        quantizer=tracker_quantizer,
    ) if run_quant_tracker else None

    total_steps = len(loader) if max_batches is None or max_batches < 0 else min(len(loader), max_batches)
    pbar = tqdm(loader, desc="VALID", total=total_steps, ncols=140)

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

        if frame_quantizer is not None:
            frame_embeddings_q = frame_quantizer.quant_dequant(frame_embeddings)

            for b in range(bs):
                local_ids, n_spk, _ = _extract_pred_local_ids_and_count(
                    slot_logits[b],
                    pred_activity[b],
                    pred_count[b],
                    threshold=activity_threshold,
                )

                slots_fp = _build_slot_results_from_frame_embeds(
                    frame_embeddings[b].detach().cpu(),
                    local_ids.detach().cpu(),
                    max_slots=max(n_spk, 1),
                )
                slots_q = _build_slot_results_from_frame_embeds(
                    frame_embeddings_q[b].detach().cpu(),
                    local_ids.detach().cpu(),
                    max_slots=max(n_spk, 1),
                )

                cos_val, n_pair = _mean_proto_cos(slots_fp, slots_q)
                if cos_val is not None:
                    proto_cos_sum += cos_val
                    proto_cos_n += 1

                if tracker is not None and len(slots_q) > 0:
                    chunk_result = ChunkInferenceResult(
                        num_speakers=len(slots_q),
                        dominant_speaker=f"slot_{slots_q[0].slot}" if len(slots_q) > 0 else None,
                        dominant_speaker_slot=slots_q[0].slot if len(slots_q) > 0 else None,
                        activity_ratio=float((local_ids > 0).float().mean().item()),
                        slots=slots_q,
                        segments=[],
                        frame_activity_prob=torch.sigmoid(pred_activity[b]).detach().cpu(),
                        local_frame_ids=local_ids.detach().cpu(),
                        global_frame_ids=None,
                    )
                    _ = tracker.update(chunk_result)
                    tracker_updates += 1

        pbar.set_postfix(
            loss=f"{val_loss_meter.avg:.4f}",
            DER=f"{der_tracker.value() * 100.0:.2f}%",
            CAcc=f"{(count_acc_sum / max(count_acc_n, 1)) * 100:.1f}%",
            ActF1=f"{(act_f1_sum / max(act_n, 1)) * 100:.1f}%",
            ProtoQ=f"{(proto_cos_sum / max(proto_cos_n, 1)):.4f}" if proto_cos_n > 0 else "NA",
        )

    der_detail = der_tracker.detail()

    out = {
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
        "proto_cos_q": (proto_cos_sum / proto_cos_n) if proto_cos_n > 0 else None,
        "tracker_updates_q": tracker_updates,
    }
    return out