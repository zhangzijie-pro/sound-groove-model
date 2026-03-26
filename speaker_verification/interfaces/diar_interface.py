from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Literal, Tuple

import torch
import torch.nn.functional as F
import torchaudio

from speaker_verification.models.resowave import ResoWave
from speaker_verification.audio.features import TARGET_SR, wav_to_fbank_infer


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
    t, T = 0, seq.numel()
    while t < T:
        end = t + 1
        while end < T and seq[end] == seq[t]:
            end += 1
        if end - t < min_run:
            left = seq[t - 1] if t > 0 else None
            right = seq[end] if end < T else None
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
    t, T = 0, active_mask.numel()
    while t < T:
        v = bool(active_mask[t].item())
        end = t + 1
        while end < T and bool(active_mask[end].item()) == v:
            end += 1
        if v and end - t < min_active_frames:
            active_mask[t:end] = False
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
        return slot_results, {int(s.slot): int(s.slot) for s in slot_results}

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
        total_dur = base.duration_sec
        proto = base.prototype.clone()

        merge_map[rep_slot] = rep_slot

        for j in range(i + 1, len(slot_results)):
            if used[j]:
                continue
            cur = slot_results[j]
            if cosine_sim(base.prototype, cur.prototype) >= sim_threshold:
                used[j] = True
                merge_map[int(cur.slot)] = rep_slot
                total_frames += cur.num_frames
                total_dur += cur.duration_sec
                proto = F.normalize(proto + cur.prototype, dim=0)

        base.num_frames = total_frames
        base.duration_sec = round(total_dur, 4)
        base.prototype = proto
        merged.append(base)

    merged = sorted(merged, key=lambda x: x.num_frames, reverse=True)
    return merged, merge_map


def frames_to_segments(
    local_frame_ids: torch.Tensor,
    frame_shift_sec: float,
    slot_to_name: Dict[int, str],
) -> List[SegmentResult]:
    segments: List[SegmentResult] = []
    T = local_frame_ids.numel()
    t = 0
    while t < T:
        slot = int(local_frame_ids[t].item())
        if slot <= 0:
            t += 1
            continue
        start = t
        t += 1
        while t < T and int(local_frame_ids[t].item()) == slot:
            t += 1
        end = t
        segments.append(
            SegmentResult(
                slot=slot,
                name=slot_to_name.get(slot, f"slot_{slot}"),
                start_sec=round(start * frame_shift_sec, 4),
                end_sec=round(end * frame_shift_sec, 4),
                duration_sec=round((end - start) * frame_shift_sec, 4),
            )
        )
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
        frame_shift_sec: float = 0.01,
        min_active_frames: int = 3,
        min_slot_run: int = 3,
        speaker_bank: Optional[SpeakerBankProtocol] = None,
        fbank_energy_threshold: float = 0.020,
        min_activity_ratio: float = 0.10,
        min_mean_activity_prob: float = 0.35,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.activity_threshold = float(activity_threshold)
        self.frame_shift_sec = float(frame_shift_sec)
        self.min_active_frames = int(min_active_frames)
        self.min_slot_run = int(min_slot_run)
        self.feat_dim = int(feat_dim)
        self.speaker_bank = speaker_bank or EmptySpeakerBank()
        self.fbank_energy_threshold = float(fbank_energy_threshold)
        self.min_activity_ratio = float(min_activity_ratio)
        self.min_mean_activity_prob = float(min_mean_activity_prob)
        self.max_mix_speakers = int(max_mix_speakers)

        self.model = ResoWave(
            in_channels=feat_dim,
            channels=channels,
            embd_dim=emb_dim,
            max_mix_speakers=max_mix_speakers,
        ).to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
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
    ) -> ChunkInferenceResult:
        if wav.dim() == 2 and wav.shape[0] == 1:
            wav = wav.squeeze(0)
        if wav.dim() != 1:
            raise ValueError(f"Expected mono wav [T] or [1,T], got {wav.shape}")

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
        return self.infer_fbank(fbank)

    @torch.no_grad()
    def infer_fbank(self, fbank: torch.Tensor) -> ChunkInferenceResult:
        if fbank.dim() == 2:
            fbank = fbank.unsqueeze(0)
        if fbank.dim() != 3:
            raise ValueError(f"Expected [T,F] or [1,T,F], got {tuple(fbank.shape)}")

        num_frames = int(fbank.shape[1])
        fbank = fbank.to(self.device).float()

        if torch.norm(fbank, dim=-1).mean().item() < self.fbank_energy_threshold:
            return self._empty_result(num_frames)

        _, frame_embeds, slot_logits, activity_logits, count_logits = self.model(
            fbank, return_diarization=True
        )

        frame_embeds = frame_embeds[0]
        slot_logits = slot_logits[0]
        activity_logits = activity_logits[0]
        count_logits = count_logits[0]

        pred_count = int(count_logits.argmax(dim=0).item())
        activity_prob = torch.sigmoid(activity_logits)
        active_mask = activity_prob >= self.activity_threshold
        active_mask = remove_short_active_runs(active_mask, self.min_active_frames)

        activity_ratio = float(active_mask.float().mean().item())
        mean_activity_prob = float(activity_prob.mean().item())

        if pred_count <= 0 or activity_ratio < self.min_activity_ratio or mean_activity_prob < self.min_mean_activity_prob:
            out = self._empty_result(num_frames)
            out.activity_ratio = activity_ratio
            out.frame_activity_prob = activity_prob.detach().cpu()
            return out

        speaker_logits = slot_logits[:, 1:] if slot_logits.size(-1) == self.max_mix_speakers + 1 else slot_logits
        raw_slot_ids = speaker_logits.argmax(dim=-1)
        raw_slot_ids = smooth_sequence(raw_slot_ids, self.min_slot_run)

        slot_counts: Dict[int, int] = {}
        for s in raw_slot_ids[active_mask].tolist():
            s = int(s)
            slot_counts[s] = slot_counts.get(s, 0) + 1

        raw_kept_slots = [k for k, _ in sorted(slot_counts.items(), key=lambda x: x[1], reverse=True)][:max(pred_count, 1)]

        raw_slots: List[SlotResult] = []
        for raw_slot in raw_kept_slots:
            mask = (raw_slot_ids == raw_slot) & active_mask
            n = int(mask.sum().item())
            if n <= 0:
                continue
            proto = F.normalize(frame_embeds[mask].mean(dim=0), p=2, dim=-1)
            ident = self.speaker_bank.identify(proto)
            raw_slots.append(
                SlotResult(
                    slot=int(raw_slot),
                    name=ident.get("name", f"slot_{raw_slot}"),
                    score=ident.get("score", None),
                    is_known=bool(ident.get("is_known", False)),
                    num_frames=n,
                    duration_sec=round(n * self.frame_shift_sec, 4),
                    prototype=proto.detach().cpu(),
                )
            )

        if len(raw_slots) == 0:
            out = self._empty_result(num_frames)
            out.activity_ratio = activity_ratio
            out.frame_activity_prob = activity_prob.detach().cpu()
            return out

        merged_slots, merge_map = merge_similar_slots(raw_slots, sim_threshold=0.85)

        if len(merged_slots) >= 2:
            if cosine_sim(merged_slots[0].prototype, merged_slots[1].prototype) >= 0.90:
                merged_slots, merge_map = merge_similar_slots(merged_slots, sim_threshold=0.90)

        merged_slots = merged_slots[: min(pred_count, len(merged_slots))]
        if len(merged_slots) == 0:
            out = self._empty_result(num_frames)
            out.activity_ratio = activity_ratio
            out.frame_activity_prob = activity_prob.detach().cpu()
            return out

        rep_slots = [int(s.slot) for s in merged_slots]
        rep_set = set(rep_slots)

        raw_to_final: Dict[int, int] = {}
        next_local = 1
        final_slots: List[SlotResult] = []
        slot_to_name: Dict[int, str] = {}

        for s in merged_slots:
            rep = int(s.slot)
            if rep not in rep_set:
                continue
            local_id = next_local
            next_local += 1
            for raw_slot, mapped_rep in merge_map.items():
                if int(mapped_rep) == rep:
                    raw_to_final[int(raw_slot)] = local_id

            final_slot = SlotResult(
                slot=local_id,
                name=s.name,
                score=s.score,
                is_known=s.is_known,
                num_frames=s.num_frames,
                duration_sec=s.duration_sec,
                prototype=s.prototype,
            )
            final_slots.append(final_slot)
            slot_to_name[local_id] = s.name

        local_frame_ids = torch.zeros_like(raw_slot_ids, dtype=torch.long)
        for raw_slot, local_id in raw_to_final.items():
            local_frame_ids[(raw_slot_ids == int(raw_slot)) & active_mask] = int(local_id)

        kept_activity_ratio = float((local_frame_ids > 0).float().mean().item())
        if kept_activity_ratio < self.min_activity_ratio:
            out = self._empty_result(num_frames)
            out.activity_ratio = kept_activity_ratio
            out.frame_activity_prob = activity_prob.detach().cpu()
            return out

        local_frame_ids_cpu = local_frame_ids.detach().cpu()
        segments = frames_to_segments(
            local_frame_ids=local_frame_ids_cpu,
            frame_shift_sec=self.frame_shift_sec,
            slot_to_name=slot_to_name,
        )

        dominant_slot = max(final_slots, key=lambda x: x.num_frames).slot
        dominant_name = slot_to_name[dominant_slot]

        return ChunkInferenceResult(
            num_speakers=len(final_slots),
            dominant_speaker=dominant_name,
            dominant_speaker_slot=dominant_slot,
            activity_ratio=kept_activity_ratio,
            slots=final_slots,
            segments=segments,
            frame_activity_prob=activity_prob.detach().cpu(),
            local_frame_ids=local_frame_ids_cpu,
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
                    "slot": s.slot,
                    "name": s.name,
                    "score": s.score,
                    "is_known": s.is_known,
                    "num_frames": s.num_frames,
                    "duration_sec": s.duration_sec,
                }
                for s in result.slots
            ],
            "segments": [
                {
                    "slot": seg.slot,
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