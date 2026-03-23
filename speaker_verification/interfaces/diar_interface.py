from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Literal

import torch
import torch.nn.functional as F
import torchaudio

from speaker_verification.models.resowave import ResoWave
from speaker_verification.audio.features import TARGET_SR, wav_to_fbank_infer


class SpeakerBankProtocol(Protocol):
    def identify(self, embedding: torch.Tensor) -> Dict[str, Any]:
        """
        Input:
            embedding: [D]
        Return example:
            {
                "name": "zhangsan",
                "score": 0.87,
                "is_known": True,
            }
        """
        ...


class EmptySpeakerBank:
    def identify(self, embedding: torch.Tensor) -> Dict[str, Any]:
        return {
            "name": "unknown",
            "score": None,
            "is_known": False,
        }


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
    activity_ratio: float
    slots: List[SlotResult]
    segments: List[SegmentResult]
    frame_activity_prob: torch.Tensor
    frame_slot_ids: torch.Tensor


def smooth_sequence(seq: torch.Tensor, min_run: int = 3) -> torch.Tensor:
    if seq.numel() == 0:
        return seq
    seq = seq.clone()
    t = 0
    T = seq.numel()
    while t < T:
        end = t + 1
        while end < T and seq[end] == seq[t]:
            end += 1
        run_len = end - t
        if run_len < min_run:
            left_val = seq[t - 1] if t > 0 else None
            right_val = seq[end] if end < T else None
            if left_val is not None:
                seq[t:end] = left_val
            elif right_val is not None:
                seq[t:end] = right_val
        t = end
    return seq


def remove_short_active_runs(active_mask: torch.Tensor, min_active_frames: int = 3) -> torch.Tensor:
    if active_mask.numel() == 0:
        return active_mask
    active_mask = active_mask.clone()
    t = 0
    T = active_mask.numel()
    while t < T:
        val = bool(active_mask[t].item())
        end = t + 1
        while end < T and bool(active_mask[end].item()) == val:
            end += 1
        run_len = end - t
        if val and run_len < min_active_frames:
            active_mask[t:end] = False
        t = end
    return active_mask

def filter_active_mask_by_kept_slots(
    slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
    kept_slots: List[int],
) -> torch.Tensor:
    if active_mask.numel() == 0:
        return active_mask

    if len(kept_slots) == 0:
        return torch.zeros_like(active_mask, dtype=torch.bool)

    keep_mask = torch.zeros_like(active_mask, dtype=torch.bool)
    for s in kept_slots:
        keep_mask |= (slot_ids == int(s))

    return active_mask & keep_mask

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return float(torch.dot(a, b).item())

def merge_similar_slots(
    slot_results,
    sim_threshold: float = 0.85,
):
    if len(slot_results) <= 1:
        return slot_results

    kept = []
    used = [False] * len(slot_results)

    for i in range(len(slot_results)):
        if used[i]:
            continue

        base = slot_results[i]
        used[i] = True

        merged_num_frames = base.num_frames
        merged_duration = base.duration_sec
        merged_proto = base.prototype.clone()

        for j in range(i + 1, len(slot_results)):
            if used[j]:
                continue

            sim = cosine_sim(base.prototype, slot_results[j].prototype)
            if sim >= sim_threshold:
                used[j] = True
                merged_num_frames += slot_results[j].num_frames
                merged_duration += slot_results[j].duration_sec
                merged_proto = F.normalize(
                    merged_proto + slot_results[j].prototype, dim=0
                )

        base.num_frames = merged_num_frames
        base.duration_sec = round(merged_duration, 4)
        base.prototype = merged_proto
        kept.append(base)

    kept = sorted(kept, key=lambda x: x.num_frames, reverse=True)
    return kept

def frames_to_segments(
    slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
    frame_shift_sec: float,
    slot_to_name: Dict[int, str],
) -> List[SegmentResult]:
    segments: List[SegmentResult] = []
    T = slot_ids.numel()
    t = 0

    while t < T:
        if not bool(active_mask[t].item()):
            t += 1
            continue

        slot = int(slot_ids[t].item())
        start = t
        t += 1

        while t < T and bool(active_mask[t].item()) and int(slot_ids[t].item()) == slot:
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
        wav_rms_threshold: float = 0.008,
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

        self.wav_rms_threshold = float(wav_rms_threshold)
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
        return ChunkInferenceResult(
            num_speakers=0,
            dominant_speaker=None,
            activity_ratio=0.0,
            slots=[],
            segments=[],
            frame_activity_prob=torch.zeros(num_frames, dtype=torch.float32),
            frame_slot_ids=torch.zeros(num_frames, dtype=torch.long),
        )

    @staticmethod
    def _compute_rms(wav: torch.Tensor) -> float:
        if wav.numel() == 0:
            return 0.0
        return float(torch.sqrt(torch.mean(wav.float() ** 2) + 1e-12).item())

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
        fbank = wav_to_fbank_infer( wav_16k=wav, n_mels=self.feat_dim, crop_sec=crop_sec, crop_mode=crop_mode, ) # → [T_frames, 80] 
        result = self.infer_fbank(fbank) 
        return result

    @torch.no_grad()
    def infer_fbank(self, fbank: torch.Tensor) -> ChunkInferenceResult:
        """
        fbank: [T,F] or [1,T,F]
        """
        if fbank.dim() == 2:
            fbank = fbank.unsqueeze(0)
        if fbank.dim() != 3:
            raise ValueError(f"Expected [T,F] or [1,T,F], got {tuple(fbank.shape)}")

        num_frames = int(fbank.shape[1])
        fbank = fbank.to(self.device).float()

        # 前置静音门
        fbank_energy = torch.norm(fbank, dim=-1).mean().item()
        if fbank_energy < self.fbank_energy_threshold:
            return self._empty_result(num_frames=num_frames)

        _, frame_embeds, slot_logits, activity_logits, count_logits = self.model(
            fbank, return_diarization=True
        )

        frame_embeds = frame_embeds[0]        # [T,D]
        slot_logits = slot_logits[0]          # [T,1+K] or [T,K]
        activity_logits = activity_logits[0]  # [T]
        count_logits = count_logits[0]        # [K+1]

        # count: 0..K
        pred_count = int(count_logits.argmax(dim=0).item())

        activity_prob = torch.sigmoid(activity_logits)
        mean_activity_prob = float(activity_prob.mean().item())

        active_mask = activity_prob >= self.activity_threshold
        active_mask = remove_short_active_runs(active_mask, self.min_active_frames)
        activity_ratio = float(active_mask.float().mean().item())

        # 第一层：语音活动不成立，直接 0 人
        if (
            pred_count <= 0
            or activity_ratio < self.min_activity_ratio
            or mean_activity_prob < self.min_mean_activity_prob
        ):
            return ChunkInferenceResult(
                num_speakers=0,
                dominant_speaker=None,
                activity_ratio=activity_ratio,
                slots=[],
                segments=[],
                frame_activity_prob=activity_prob.detach().cpu(),
                frame_slot_ids=torch.zeros_like(activity_prob, dtype=torch.long).cpu(),
            )

        # 只取 speaker logits，不取 silence
        if slot_logits.size(-1) == self.max_mix_speakers + 1:
            speaker_logits = slot_logits[:, 1:]   # [T,K]
        else:
            speaker_logits = slot_logits

        slot_ids = speaker_logits.argmax(dim=-1)  # 统一用 0..K-1
        slot_ids = smooth_sequence(slot_ids, self.min_slot_run)

        # 统计 active 帧内每个 slot 的占用
        slot_counts: Dict[int, int] = {}
        if active_mask.any():
            active_slot_ids = slot_ids[active_mask]
            for s in active_slot_ids.tolist():
                s = int(s)
                slot_counts[s] = slot_counts.get(s, 0) + 1

        # 先按 count 取 top-N 原始候选
        raw_kept_slots = [
            k for k, _ in sorted(slot_counts.items(), key=lambda x: x[1], reverse=True)
        ][:max(pred_count, 1)]

        raw_slot_results: List[SlotResult] = []

        for slot in raw_kept_slots:
            mask = (slot_ids == slot) & active_mask
            num_frames_slot = int(mask.sum().item())
            if num_frames_slot <= 0:
                continue

            proto = frame_embeds[mask].mean(dim=0)
            proto = F.normalize(proto, p=2, dim=-1)

            ident = self.speaker_bank.identify(proto)
            name = ident.get("name", f"slot_{slot}")

            raw_slot_results.append(
                SlotResult(
                    slot=slot,
                    name=name,
                    score=ident.get("score", None),
                    is_known=bool(ident.get("is_known", False)),
                    num_frames=num_frames_slot,
                    duration_sec=round(num_frames_slot * self.frame_shift_sec, 4),
                    prototype=proto.detach().cpu(),
                )
            )

        if len(raw_slot_results) == 0:
            return ChunkInferenceResult(
                num_speakers=0,
                dominant_speaker=None,
                activity_ratio=activity_ratio,
                slots=[],
                segments=[],
                frame_activity_prob=activity_prob.detach().cpu(),
                frame_slot_ids=torch.zeros_like(slot_ids, dtype=torch.long).cpu(),
            )

        # merge 必须放在循环外
        merged_slot_results = merge_similar_slots(
            raw_slot_results,
            sim_threshold=0.85,
        )

        # 单说话人优先约束：
        # 如果 top2 prototype 很像，强制向 1 人收缩
        if len(merged_slot_results) >= 2:
            sim01 = cosine_sim(
                merged_slot_results[0].prototype,
                merged_slot_results[1].prototype
            )
            if sim01 >= 0.90:
                merged_slot_results = merge_similar_slots(
                    merged_slot_results,
                    sim_threshold=0.90,
                )

        # 最终人数不要超过 pred_count，也不要超过 merge 后有效 slot 数
        final_num_speakers = min(pred_count, len(merged_slot_results))
        merged_slot_results = merged_slot_results[:final_num_speakers]

        if len(merged_slot_results) == 0:
            return ChunkInferenceResult(
                num_speakers=0,
                dominant_speaker=None,
                activity_ratio=activity_ratio,
                slots=[],
                segments=[],
                frame_activity_prob=activity_prob.detach().cpu(),
                frame_slot_ids=torch.zeros_like(slot_ids, dtype=torch.long).cpu(),
            )

        kept_slots = [s.slot for s in merged_slot_results]
        slot_to_name = {s.slot: s.name for s in merged_slot_results}

        # 关键：segments 只能来自最终保留的 slots
        active_mask_kept = filter_active_mask_by_kept_slots(
            slot_ids=slot_ids,
            active_mask=active_mask,
            kept_slots=kept_slots,
        )

        kept_activity_ratio = float(active_mask_kept.float().mean().item())
        if kept_activity_ratio < self.min_activity_ratio:
            return ChunkInferenceResult(
                num_speakers=0,
                dominant_speaker=None,
                activity_ratio=kept_activity_ratio,
                slots=[],
                segments=[],
                frame_activity_prob=activity_prob.detach().cpu(),
                frame_slot_ids=torch.zeros_like(slot_ids, dtype=torch.long).cpu(),
            )

        segments = frames_to_segments(
            slot_ids=slot_ids.detach().cpu(),
            active_mask=active_mask_kept.detach().cpu(),
            frame_shift_sec=self.frame_shift_sec,
            slot_to_name=slot_to_name,
        )

        dominant_speaker = max(merged_slot_results, key=lambda x: x.num_frames).name

        return ChunkInferenceResult(
            num_speakers=len(merged_slot_results),
            dominant_speaker=dominant_speaker,
            activity_ratio=kept_activity_ratio,
            slots=merged_slot_results,
            segments=segments,
            frame_activity_prob=activity_prob.detach().cpu(),
            frame_slot_ids=slot_ids.detach().cpu(),
        )

    def to_jsonable(self, result: ChunkInferenceResult) -> Dict[str, Any]:
        return {
            "num_speakers": result.num_speakers,
            "dominant_speaker": result.dominant_speaker,
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