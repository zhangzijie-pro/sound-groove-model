import torch
import torch.nn.functional as F
from tqdm import tqdm

from speaker_verification.utils.meters import (
    AverageMeter,
    DERTracker,
    diarization_error_rate_pit,
    compute_activity_metrics,
)
from speaker_verification.interfaces.diar_interface import (
    ChunkInferenceResult,
    SlotResult,
    build_slot_activity_masks,
    dominant_local_ids_from_masks,
    fill_short_inactive_gaps,
    remove_short_active_runs,
)
from speaker_verification.interfaces.global_tracker import GlobalSpeakerTracker


def _extract_pred_local_ids_and_count(
    diar_logits,
    activity_threshold=0.5,
    slot_threshold=0.5,
    min_active_frames=3,
    min_slot_run=3,
    fill_gap_frames=3,
    slot_presence_frames=8,
):
    diar_prob = torch.sigmoid(diar_logits)
    activity_prob = diar_prob.amax(dim=-1)
    active_mask = activity_prob >= activity_threshold
    active_mask = fill_short_inactive_gaps(active_mask, max_gap_frames=fill_gap_frames)
    active_mask = remove_short_active_runs(active_mask, min_active_frames=min_active_frames)

    slot_masks = build_slot_activity_masks(
        diar_prob=diar_prob,
        threshold=slot_threshold,
        min_active_frames=min_active_frames,
        fill_gap_frames=fill_gap_frames,
        global_activity_mask=active_mask,
    )
    slot_present = slot_masks.sum(dim=0) >= int(slot_presence_frames)
    if not bool(slot_present.any().item()):
        zeros = torch.zeros(diar_logits.size(0), dtype=torch.long, device=diar_logits.device)
        return zeros, 0, active_mask

    slot_masks = slot_masks[:, slot_present]
    slot_scores = diar_prob[:, slot_present]
    local_ids = dominant_local_ids_from_masks(
        slot_scores=slot_scores,
        slot_masks=slot_masks,
        min_run=min_slot_run,
    )
    n_spk = int(slot_present.sum().item())
    return local_ids, n_spk, active_mask


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
    slot_threshold=0.5,
    min_active_frames=3,
    min_slot_run=3,
    fill_gap_frames=3,
    slot_presence_frames=8,
    frame_quantizer=None,
    tracker_quantizer=None,
    run_quant_tracker=False,
    tracker_match_threshold=0.72,
    tracker_momentum=0.9,
    tracker_max_misses=30,
):
    model.eval()

    val_loss_meter = AverageMeter()
    diar_meter = AverageMeter()
    smooth_meter = AverageMeter()
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
        match_threshold=tracker_match_threshold,
        momentum=tracker_momentum,
        max_misses=tracker_max_misses,
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
        target_count = batch["target_count"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        frame_embeddings, diar_logits = model(
            fbank
        )

        loss_dict = loss_fn(
            diar_logits=diar_logits,
            target_matrix=target_matrix,
            valid_mask=valid_mask,
            return_detail=True,
        )

        loss = loss_dict["total"]
        bs = fbank.size(0)

        val_loss_meter.update(float(loss.item()), bs)
        diar_meter.update(float(loss_dict["diar_loss"].item()), bs)
        smooth_meter.update(float(loss_dict["smooth_loss"].item()), bs)

        _, info = diarization_error_rate_pit(
            diar_logits,
            target_matrix,
            valid_mask=valid_mask,
            return_detail=True,
        )
        der_tracker.update(info)

        pred_count = []
        diar_prob_batch = torch.sigmoid(diar_logits)
        for b in range(bs):
            valid_frames = valid_mask[b].bool()
            local_ids, n_spk, _ = _extract_pred_local_ids_and_count(
                diar_logits[b][valid_frames],
                activity_threshold=activity_threshold,
                slot_threshold=slot_threshold,
                min_active_frames=min_active_frames,
                min_slot_run=min_slot_run,
                fill_gap_frames=fill_gap_frames,
                slot_presence_frames=slot_presence_frames,
            )
            del local_ids
            pred_count.append(n_spk)
        pred_count = torch.tensor(pred_count, device=device)
        count_acc_sum += (pred_count == target_count.long()).float().mean().item()
        count_acc_n += 1

        target_activity = (target_matrix.sum(dim=-1) > 0).float()
        pred_activity_logits = diar_logits.amax(dim=-1)
        ap, ar, af1 = compute_activity_metrics(
            pred_activity_logits, target_activity, valid_mask, threshold=activity_threshold
        )
        act_prec_sum += ap
        act_rec_sum += ar
        act_f1_sum += af1
        act_n += 1

        if frame_quantizer is not None:
            frame_embeddings_q = frame_quantizer.quant_dequant(frame_embeddings)

            for b in range(bs):
                local_ids, n_spk, _ = _extract_pred_local_ids_and_count(
                    diar_logits[b][valid_mask[b].bool()],
                    activity_threshold=activity_threshold,
                    slot_threshold=slot_threshold,
                    min_active_frames=min_active_frames,
                    min_slot_run=min_slot_run,
                    fill_gap_frames=fill_gap_frames,
                    slot_presence_frames=slot_presence_frames,
                )
                padded_local_ids = torch.zeros_like(valid_mask[b], dtype=torch.long)
                padded_local_ids[valid_mask[b].bool()] = local_ids

                slots_fp = _build_slot_results_from_frame_embeds(
                    frame_embeddings[b].detach().cpu(),
                    padded_local_ids.detach().cpu(),
                    max_slots=max(n_spk, 1),
                )
                slots_q = _build_slot_results_from_frame_embeds(
                    frame_embeddings_q[b].detach().cpu(),
                    padded_local_ids.detach().cpu(),
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
                        activity_ratio=float((padded_local_ids > 0).float().mean().item()),
                        slots=slots_q,
                        segments=[],
                        frame_activity_prob=torch.sigmoid(pred_activity_logits[b]).detach().cpu(),
                        local_frame_ids=padded_local_ids.detach().cpu(),
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
        "diar_loss": diar_meter.avg,
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
        "act_prec": act_prec_sum / max(1, act_n),
        "act_rec": act_rec_sum / max(1, act_n),
        "act_f1": act_f1_sum / max(1, act_n),
        "proto_cos_q": (proto_cos_sum / proto_cos_n) if proto_cos_n > 0 else None,
        "tracker_updates_q": tracker_updates,
    }
    return out
