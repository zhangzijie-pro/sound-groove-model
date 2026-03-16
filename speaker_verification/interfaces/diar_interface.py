from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import torch
import torch.nn.functional as F

from speaker_verification.models.resowave import ResoWave


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
    dominant_speaker: str
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
        if not active_mask[t]:
            t += 1
            continue
        slot = int(slot_ids[t].item())
        start = t
        t += 1
        while t < T and active_mask[t] and int(slot_ids[t].item()) == slot:
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
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.activity_threshold = float(activity_threshold)
        self.frame_shift_sec = float(frame_shift_sec)
        self.min_active_frames = int(min_active_frames)
        self.min_slot_run = int(min_slot_run)
        self.speaker_bank = speaker_bank or EmptySpeakerBank()

        self.model = ResoWave(
            in_channels=feat_dim,
            channels=channels,
            embd_dim=emb_dim,
            max_mix_speakers=max_mix_speakers,
        ).to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def infer_fbank(self, fbank: torch.Tensor) -> ChunkInferenceResult:
        """
        fbank: [T,80] or [1,T,80]
        """
        if fbank.dim() == 2:
            fbank = fbank.unsqueeze(0)
        if fbank.dim() != 3:
            raise ValueError(f"Expected [T,F] or [1,T,F], got {tuple(fbank.shape)}")

        fbank = fbank.to(self.device)

        _, frame_embeds, slot_logits, activity_logits, count_logits = self.model(
            fbank, return_diarization=True
        )

        frame_embeds = frame_embeds[0]      # [T,D]
        slot_logits = slot_logits[0]        # [T,K]
        activity_logits = activity_logits[0]
        count_logits = count_logits[0]

        pred_count = int(count_logits.argmax().item() + 1)

        activity_prob = torch.sigmoid(activity_logits)
        active_mask = activity_prob >= self.activity_threshold
        active_mask = remove_short_active_runs(active_mask, self.min_active_frames)

        slot_ids = slot_logits.argmax(dim=-1)
        slot_ids = smooth_sequence(slot_ids, self.min_slot_run)

        slot_results: List[SlotResult] = []
        slot_to_name: Dict[int, str] = {}

        for slot in range(pred_count):
            mask = (slot_ids == slot) & active_mask
            num_frames = int(mask.sum().item())
            if num_frames <= 0:
                continue

            proto = frame_embeds[mask].mean(dim=0)
            proto = F.normalize(proto, p=2, dim=-1)

            ident = self.speaker_bank.identify(proto)
            name = ident.get("name", f"slot_{slot}")
            slot_to_name[slot] = name

            slot_results.append(
                SlotResult(
                    slot=slot,
                    name=name,
                    score=ident.get("score", None),
                    is_known=bool(ident.get("is_known", False)),
                    num_frames=num_frames,
                    duration_sec=round(num_frames * self.frame_shift_sec, 4),
                    prototype=proto.detach().cpu(),
                )
            )

        segments = frames_to_segments(
            slot_ids=slot_ids.detach().cpu(),
            active_mask=active_mask.detach().cpu(),
            frame_shift_sec=self.frame_shift_sec,
            slot_to_name=slot_to_name,
        )

        dominant_speaker = "unknown"
        if slot_results:
            dominant_speaker = max(slot_results, key=lambda x: x.num_frames).name

        return ChunkInferenceResult(
            num_speakers=pred_count,
            dominant_speaker=dominant_speaker,
            activity_ratio=float(active_mask.float().mean().item()),
            slots=slot_results,
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