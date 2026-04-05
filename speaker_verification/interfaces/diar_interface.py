from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple

import torch
import torch.nn.functional as F
import torchaudio

from speaker_verification.audio.features import TARGET_SR, wav_to_fbank_infer
from speaker_verification.models.resowave import ResoWave


class SpeakerBankProtocol(Protocol):
    def identify(self, embedding: torch.Tensor) -> Dict[str, Any]:
        ...


class EmptySpeakerBank:
    def identify(self, embedding: torch.Tensor) -> Dict[str, Any]:
        return {"name": "unknown", "score": None, "is_known": False}


@dataclass
class SlotResult:
    slot: int
    name: str
    score: Optional[float]
    is_known: bool
    num_frames: int
    duration_sec: float
    prototype: torch.Tensor


@dataclass
class SegmentResult:
    slot: int
    name: str
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass
class ChunkInferenceResult:
    num_speakers: int
    dominant_speaker: Optional[str]
    dominant_speaker_slot: Optional[int]
    activity_ratio: float
    slots: List[SlotResult]
    segments: List[SegmentResult]
    frame_activity_prob: torch.Tensor
    local_frame_ids: torch.Tensor
    global_frame_ids: Optional[torch.Tensor] = None


def smooth_sequence(seq: torch.Tensor, min_run: int = 3) -> torch.Tensor:
    if seq.numel() == 0:
        return seq
    seq = seq.clone()
    t, total = 0, seq.numel()
    while t < total:
        end = t + 1
        while end < total and seq[end] == seq[t]:
            end += 1
        if end - t < min_run:
            left = seq[t - 1] if t > 0 else None
            right = seq[end] if end < total else None
            if left is not None:
                seq[t:end] = left
            elif right is not None:
                seq[t:end] = right
        t = end
    return seq


def remove_short_active_runs(active_mask: torch.Tensor, min_active_frames: int = 3) -> torch.Tensor:
    if active_mask.numel() == 0:
        return active_mask
    active_mask = active_mask.clone()
    t, total = 0, active_mask.numel()
    while t < total:
        value = bool(active_mask[t].item())
        end = t + 1
        while end < total and bool(active_mask[end].item()) == value:
            end += 1
        if value and end - t < min_active_frames:
            active_mask[t:end] = False
        t = end
    return active_mask


def fill_short_inactive_gaps(active_mask: torch.Tensor, max_gap_frames: int = 3) -> torch.Tensor:
    if active_mask.numel() == 0 or max_gap_frames <= 0:
        return active_mask
    active_mask = active_mask.clone()
    t, total = 0, active_mask.numel()
    while t < total:
        value = bool(active_mask[t].item())
        end = t + 1
        while end < total and bool(active_mask[end].item()) == value:
            end += 1
        if (not value) and (end - t) <= max_gap_frames:
            left_active = t > 0 and bool(active_mask[t - 1].item())
            right_active = end < total and bool(active_mask[end].item())
            if left_active and right_active:
                active_mask[t:end] = True
        t = end
    return active_mask


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return float(torch.dot(a, b).item())


def merge_similar_slots(
    slot_results: List[SlotResult],
    sim_threshold: float = 0.85,
) -> Tuple[List[SlotResult], Dict[int, int]]:
    if len(slot_results) <= 1:
        return slot_results, {int(slot.slot): int(slot.slot) for slot in slot_results}

    used = [False] * len(slot_results)
    merged: List[SlotResult] = []
    merge_map: Dict[int, int] = {}

    for i in range(len(slot_results)):
        if used[i]:
            continue
        base = slot_results[i]
        used[i] = True
        rep_slot = int(base.slot)

        total_frames = base.num_frames
        total_duration = base.duration_sec
        prototype = base.prototype.clone()
        merge_map[rep_slot] = rep_slot

        for j in range(i + 1, len(slot_results)):
            if used[j]:
                continue
            cur = slot_results[j]
            if cosine_sim(base.prototype, cur.prototype) >= sim_threshold:
                used[j] = True
                merge_map[int(cur.slot)] = rep_slot
                total_frames += cur.num_frames
                total_duration += cur.duration_sec
                prototype = F.normalize(prototype + cur.prototype, dim=0)

        base.num_frames = total_frames
        base.duration_sec = round(total_duration, 4)
        base.prototype = prototype
        merged.append(base)

    merged = sorted(merged, key=lambda item: item.num_frames, reverse=True)
    return merged, merge_map


def smooth_active_labels(local_frame_ids: torch.Tensor, min_run: int = 3) -> torch.Tensor:
    if local_frame_ids.numel() == 0:
        return local_frame_ids
    out = local_frame_ids.clone()
    t, total = 0, out.numel()
    while t < total:
        if int(out[t].item()) <= 0:
            t += 1
            continue
        end = t + 1
        while end < total and int(out[end].item()) > 0:
            end += 1
        out[t:end] = smooth_sequence(out[t:end], min_run=min_run)
        t = end
    return out


def build_slot_activity_masks(
    diar_prob: torch.Tensor,
    threshold: float,
    min_active_frames: int,
    fill_gap_frames: int,
    global_activity_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    slot_masks: List[torch.Tensor] = []
    for slot_idx in range(diar_prob.size(-1)):
        slot_mask = diar_prob[:, slot_idx] >= threshold
        if global_activity_mask is not None:
            slot_mask = slot_mask & global_activity_mask
        slot_mask = fill_short_inactive_gaps(slot_mask, max_gap_frames=fill_gap_frames)
        slot_mask = remove_short_active_runs(slot_mask, min_active_frames=min_active_frames)
        slot_masks.append(slot_mask)

    if not slot_masks:
        return torch.zeros(
            diar_prob.size(0),
            0,
            dtype=torch.bool,
            device=diar_prob.device,
        )
    return torch.stack(slot_masks, dim=-1)


def build_slot_results_from_masks(
    frame_embeds: torch.Tensor,
    slot_masks: torch.Tensor,
    frame_shift_sec: float,
    speaker_bank: SpeakerBankProtocol,
    slot_ids: Optional[List[int]] = None,
) -> List[SlotResult]:
    if slot_masks.dim() != 2:
        raise ValueError(f"slot_masks must be [T, K], got {tuple(slot_masks.shape)}")

    if slot_ids is None:
        slot_ids = list(range(1, slot_masks.size(1) + 1))

    slots: List[SlotResult] = []
    for column, slot_id in enumerate(slot_ids):
        mask = slot_masks[:, column].bool()
        num_frames = int(mask.sum().item())
        if num_frames <= 0:
            continue
        prototype = F.normalize(frame_embeds[mask].mean(dim=0), p=2, dim=-1)
        ident = speaker_bank.identify(prototype)
        slots.append(
            SlotResult(
                slot=int(slot_id),
                name=ident.get("name", f"speaker_{slot_id}"),
                score=ident.get("score", None),
                is_known=bool(ident.get("is_known", False)),
                num_frames=num_frames,
                duration_sec=round(num_frames * frame_shift_sec, 4),
                prototype=prototype.detach().cpu(),
            )
        )
    return slots


def dominant_local_ids_from_masks(
    slot_scores: torch.Tensor,
    slot_masks: torch.Tensor,
    min_run: int = 3,
) -> torch.Tensor:
    num_frames = slot_scores.size(0)
    if slot_scores.numel() == 0 or slot_masks.numel() == 0:
        return torch.zeros(num_frames, dtype=torch.long, device=slot_scores.device)

    masked_scores = slot_scores.masked_fill(~slot_masks, float("-inf"))
    local_frame_ids = masked_scores.argmax(dim=-1) + 1
    active_mask = slot_masks.any(dim=-1)
    local_frame_ids = local_frame_ids * active_mask.long()
    return smooth_active_labels(local_frame_ids, min_run=min_run)


def frames_to_segments(
    local_frame_ids: torch.Tensor,
    frame_shift_sec: float,
    slot_to_name: Dict[int, str],
    offset_sec: float = 0.0,
) -> List[SegmentResult]:
    segments: List[SegmentResult] = []
    total = local_frame_ids.numel()
    t = 0
    while t < total:
        slot = int(local_frame_ids[t].item())
        if slot <= 0:
            t += 1
            continue
        start = t
        t += 1
        while t < total and int(local_frame_ids[t].item()) == slot:
            t += 1
        end = t
        segments.append(
            SegmentResult(
                slot=slot,
                name=slot_to_name.get(slot, f"slot_{slot}"),
                start_sec=round(offset_sec + start * frame_shift_sec, 4),
                end_sec=round(offset_sec + end * frame_shift_sec, 4),
                duration_sec=round((end - start) * frame_shift_sec, 4),
            )
        )
    return segments


def slot_masks_to_segments(
    slot_masks: torch.Tensor,
    frame_shift_sec: float,
    slot_to_name: Dict[int, str],
    offset_sec: float = 0.0,
) -> List[SegmentResult]:
    if slot_masks.dim() != 2:
        raise ValueError(f"slot_masks must be [T, K], got {tuple(slot_masks.shape)}")

    segments: List[SegmentResult] = []
    num_frames, num_slots = slot_masks.shape
    for slot_idx in range(num_slots):
        mask = slot_masks[:, slot_idx].bool()
        t = 0
        slot_id = slot_idx + 1
        while t < num_frames:
            if not bool(mask[t].item()):
                t += 1
                continue
            start = t
            t += 1
            while t < num_frames and bool(mask[t].item()):
                t += 1
            end = t
            segments.append(
                SegmentResult(
                    slot=slot_id,
                    name=slot_to_name.get(slot_id, f"slot_{slot_id}"),
                    start_sec=round(offset_sec + start * frame_shift_sec, 4),
                    end_sec=round(offset_sec + end * frame_shift_sec, 4),
                    duration_sec=round((end - start) * frame_shift_sec, 4),
                )
            )
    segments.sort(key=lambda item: (item.start_sec, item.slot))
    return segments


class SpeakerAwareDiarizationInterface:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        feat_dim: int = 80,
        channels: int = 512,
        emb_dim: int = 192,
        max_mix_speakers: int = 4,
        activity_threshold: float = 0.5,
        slot_threshold: Optional[float] = None,
        frame_shift_sec: float = 0.01,
        min_active_frames: int = 3,
        min_slot_run: int = 3,
        slot_presence_frames: int = 8,
        speaker_bank: Optional[SpeakerBankProtocol] = None,
        fbank_energy_threshold: float = 0.020,
        min_activity_ratio: float = 0.10,
        min_mean_activity_prob: float = 0.35,
        fill_gap_frames: int = 3,
        cluster_min_frames: int = 12,
        cluster_silhouette_threshold: float = 0.02,
        cluster_count_prior_bias: float = 0.04,
        cluster_merge_threshold: float = 0.94,
        post_ffn_hidden_dim: Optional[int] = None,
        post_ffn_dropout: float = 0.1,
        head_dropout: float = 0.1,
    ):
        del cluster_min_frames
        del cluster_silhouette_threshold
        del cluster_count_prior_bias

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.frame_shift_sec = float(frame_shift_sec)
        self.speaker_bank = speaker_bank or EmptySpeakerBank()
        self.fbank_energy_threshold = float(fbank_energy_threshold)
        self.min_activity_ratio = float(min_activity_ratio)
        self.min_mean_activity_prob = float(min_mean_activity_prob)
        self.slot_merge_threshold = float(cluster_merge_threshold)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        model_cfg = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg, dict) else {}
        validate_cfg = ckpt_cfg.get("validate", {}) if isinstance(ckpt_cfg, dict) else {}

        feat_dim = int(model_cfg.get("in_channels", feat_dim))
        channels = int(model_cfg.get("channels", channels))
        emb_dim = int(model_cfg.get("embedding_dim", emb_dim))
        max_mix_speakers = int(model_cfg.get("max_mix_speakers", max_mix_speakers))
        post_ffn_hidden_dim = model_cfg.get("post_ffn_hidden_dim", post_ffn_hidden_dim)
        post_ffn_dropout = float(model_cfg.get("post_ffn_dropout", post_ffn_dropout))
        head_dropout = float(model_cfg.get("head_dropout", head_dropout))
        activity_threshold = float(validate_cfg.get("activity_threshold", activity_threshold))
        if slot_threshold is None:
            slot_threshold = validate_cfg.get("slot_threshold", None)
        min_active_frames = int(validate_cfg.get("min_active_frames", min_active_frames))
        min_slot_run = int(validate_cfg.get("min_slot_run", min_slot_run))
        fill_gap_frames = int(validate_cfg.get("fill_gap_frames", fill_gap_frames))
        slot_presence_frames = int(validate_cfg.get("slot_presence_frames", slot_presence_frames))
        self.activity_threshold = float(activity_threshold)
        self.slot_threshold = float(activity_threshold if slot_threshold is None else slot_threshold)
        self.min_active_frames = int(min_active_frames)
        self.min_slot_run = int(min_slot_run)
        self.slot_presence_frames = int(slot_presence_frames)
        self.fill_gap_frames = int(fill_gap_frames)
        self.feat_dim = feat_dim
        self.max_mix_speakers = max_mix_speakers

        self.model = ResoWave(
            in_channels=feat_dim,
            channels=channels,
            embedding_dim=emb_dim,
            max_mix_speakers=max_mix_speakers,
            post_ffn_hidden_dim=post_ffn_hidden_dim,
            post_ffn_dropout=post_ffn_dropout,
            head_dropout=head_dropout,
        ).to(self.device)

        state_dict = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def _empty_result(self, num_frames: int = 0) -> ChunkInferenceResult:
        zeros_prob = torch.zeros(num_frames, dtype=torch.float32)
        zeros_ids = torch.zeros(num_frames, dtype=torch.long)
        return ChunkInferenceResult(
            num_speakers=0,
            dominant_speaker=None,
            dominant_speaker_slot=None,
            activity_ratio=0.0,
            slots=[],
            segments=[],
            frame_activity_prob=zeros_prob,
            local_frame_ids=zeros_ids,
            global_frame_ids=zeros_ids.clone(),
        )

    @torch.no_grad()
    def infer_wav(
        self,
        wav: torch.Tensor,
        sample_rate: int = 16000,
        crop_sec: Optional[float] = None,
        crop_mode: Literal["tail", "center", "none"] = "tail",
        normalize: bool = True,
        offset_sec: float = 0.0,
    ) -> ChunkInferenceResult:
        if wav.dim() == 2 and wav.shape[0] == 1:
            wav = wav.squeeze(0)
        if wav.dim() != 1:
            raise ValueError(f"Expected mono wav [T] or [1,T], got {tuple(wav.shape)}")

        if sample_rate != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sample_rate, TARGET_SR)
        if normalize and wav.abs().max() > 0:
            wav = wav / wav.abs().max()

        fbank = wav_to_fbank_infer(
            wav_16k=wav,
            n_mels=self.feat_dim,
            crop_sec=crop_sec,
            crop_mode=crop_mode,
        )
        return self.infer_fbank(fbank, offset_sec=offset_sec)

    @torch.no_grad()
    def infer_fbank(self, fbank: torch.Tensor, offset_sec: float = 0.0) -> ChunkInferenceResult:
        if fbank.dim() == 2:
            fbank = fbank.unsqueeze(0)
        if fbank.dim() != 3:
            raise ValueError(f"Expected [T,F] or [1,T,F], got {tuple(fbank.shape)}")

        num_frames = int(fbank.shape[1])
        fbank = fbank.to(self.device).float()

        if torch.norm(fbank, dim=-1).mean().item() < self.fbank_energy_threshold:
            return self._empty_result(num_frames)

        frame_embeds, diar_logits = self.model(fbank)

        frame_embeds = frame_embeds[0]
        diar_logits = diar_logits[0]
        diar_prob = torch.sigmoid(diar_logits)
        activity_prob = diar_prob.amax(dim=-1)
        global_activity_mask = activity_prob >= self.activity_threshold
        global_activity_mask = fill_short_inactive_gaps(
            global_activity_mask,
            max_gap_frames=self.fill_gap_frames,
        )
        global_activity_mask = remove_short_active_runs(
            global_activity_mask,
            min_active_frames=self.min_active_frames,
        )

        activity_ratio = float(global_activity_mask.float().mean().item())
        mean_activity_prob = float(activity_prob.mean().item())
        if activity_ratio < self.min_activity_ratio or mean_activity_prob < self.min_mean_activity_prob:
            out = self._empty_result(num_frames)
            out.activity_ratio = activity_ratio
            out.frame_activity_prob = activity_prob.detach().cpu()
            return out

        slot_masks = build_slot_activity_masks(
            diar_prob=diar_prob,
            threshold=self.slot_threshold,
            min_active_frames=self.min_active_frames,
            fill_gap_frames=self.fill_gap_frames,
            global_activity_mask=global_activity_mask,
        )

        active_raw_slot_ids = [
            slot_idx + 1
            for slot_idx in range(slot_masks.size(1))
            if int(slot_masks[:, slot_idx].sum().item()) >= self.slot_presence_frames
        ]
        if not active_raw_slot_ids:
            out = self._empty_result(num_frames)
            out.activity_ratio = activity_ratio
            out.frame_activity_prob = activity_prob.detach().cpu()
            return out

        raw_slot_masks = torch.stack(
            [slot_masks[:, slot_id - 1] for slot_id in active_raw_slot_ids],
            dim=-1,
        )
        raw_slots = build_slot_results_from_masks(
            frame_embeds=frame_embeds,
            slot_masks=raw_slot_masks,
            frame_shift_sec=self.frame_shift_sec,
            speaker_bank=self.speaker_bank,
            slot_ids=active_raw_slot_ids,
        )

        merged_slots, merge_map = merge_similar_slots(
            raw_slots,
            sim_threshold=self.slot_merge_threshold,
        )

        final_masks: List[torch.Tensor] = []
        rep_to_final: Dict[int, int] = {}
        for merged in merged_slots:
            rep_slot = int(merged.slot)
            merged_mask = torch.zeros(num_frames, dtype=torch.bool, device=raw_slot_masks.device)
            for raw_slot_id in active_raw_slot_ids:
                if merge_map.get(raw_slot_id, raw_slot_id) == rep_slot:
                    merged_mask |= slot_masks[:, raw_slot_id - 1]
            if not bool(merged_mask.any().item()):
                continue
            rep_to_final[rep_slot] = len(final_masks) + 1
            final_masks.append(merged_mask)

        if not final_masks:
            out = self._empty_result(num_frames)
            out.activity_ratio = activity_ratio
            out.frame_activity_prob = activity_prob.detach().cpu()
            return out

        final_slot_masks = torch.stack(final_masks, dim=-1)
        final_slot_scores = torch.zeros(
            num_frames,
            final_slot_masks.size(1),
            dtype=diar_prob.dtype,
            device=diar_prob.device,
        )
        for raw_slot_id in active_raw_slot_ids:
            rep_slot = merge_map.get(raw_slot_id, raw_slot_id)
            final_slot_id = rep_to_final[rep_slot] - 1
            final_slot_scores[:, final_slot_id] = torch.maximum(
                final_slot_scores[:, final_slot_id],
                diar_prob[:, raw_slot_id - 1],
            )

        final_slots = build_slot_results_from_masks(
            frame_embeds=frame_embeds,
            slot_masks=final_slot_masks,
            frame_shift_sec=self.frame_shift_sec,
            speaker_bank=self.speaker_bank,
            slot_ids=list(range(1, final_slot_masks.size(1) + 1)),
        )
        slot_to_name = {slot.slot: slot.name for slot in final_slots}
        local_frame_ids = dominant_local_ids_from_masks(
            slot_scores=final_slot_scores,
            slot_masks=final_slot_masks,
            min_run=self.min_slot_run,
        )
        segments = slot_masks_to_segments(
            slot_masks=final_slot_masks.detach().cpu(),
            frame_shift_sec=self.frame_shift_sec,
            slot_to_name=slot_to_name,
            offset_sec=offset_sec,
        )

        dominant_slot = max(final_slots, key=lambda item: item.num_frames).slot
        dominant_name = slot_to_name[dominant_slot]

        return ChunkInferenceResult(
            num_speakers=len(final_slots),
            dominant_speaker=dominant_name,
            dominant_speaker_slot=dominant_slot,
            activity_ratio=float(final_slot_masks.any(dim=-1).float().mean().item()),
            slots=final_slots,
            segments=segments,
            frame_activity_prob=activity_prob.detach().cpu(),
            local_frame_ids=local_frame_ids.detach().cpu(),
            global_frame_ids=None,
        )

    def to_jsonable(self, result: ChunkInferenceResult) -> Dict[str, Any]:
        out = {
            "num_speakers": result.num_speakers,
            "dominant_speaker": result.dominant_speaker,
            "dominant_speaker_slot": result.dominant_speaker_slot,
            "activity_ratio": round(result.activity_ratio, 6),
            "slots": [
                {
                    "slot": slot.slot,
                    "speaker_id": slot.slot,
                    "name": slot.name,
                    "score": slot.score,
                    "is_known": slot.is_known,
                    "num_frames": slot.num_frames,
                    "duration_sec": slot.duration_sec,
                }
                for slot in result.slots
            ],
            "segments": [
                {
                    "slot": seg.slot,
                    "speaker_id": seg.slot,
                    "name": seg.name,
                    "start_sec": seg.start_sec,
                    "end_sec": seg.end_sec,
                    "duration_sec": seg.duration_sec,
                }
                for seg in result.segments
            ],
        }
        if result.global_frame_ids is not None:
            out["active_global_ids"] = sorted(set(result.global_frame_ids.tolist()) - {0})
        return out
