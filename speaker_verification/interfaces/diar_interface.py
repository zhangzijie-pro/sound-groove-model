from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from speaker_verification.models.resowave import ResoWave
from speaker_verification.audio.features import TARGET_SR, wav_to_fbank_infer

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError:
    KMeans = None
    silhouette_score = None


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


def fill_short_inactive_gaps(active_mask: torch.Tensor, max_gap_frames: int = 3) -> torch.Tensor:
    if active_mask.numel() == 0 or max_gap_frames <= 0:
        return active_mask
    active_mask = active_mask.clone()
    t, T = 0, active_mask.numel()
    while t < T:
        v = bool(active_mask[t].item())
        end = t + 1
        while end < T and bool(active_mask[end].item()) == v:
            end += 1
        if (not v) and (end - t) <= max_gap_frames:
            left_active = t > 0 and bool(active_mask[t - 1].item())
            right_active = end < T and bool(active_mask[end].item())
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


def smooth_active_labels(local_frame_ids: torch.Tensor, min_run: int = 3) -> torch.Tensor:
    if local_frame_ids.numel() == 0:
        return local_frame_ids
    out = local_frame_ids.clone()
    t, T = 0, out.numel()
    while t < T:
        if int(out[t].item()) <= 0:
            t += 1
            continue
        end = t + 1
        while end < T and int(out[end].item()) > 0:
            end += 1
        out[t:end] = smooth_sequence(out[t:end], min_run=min_run)
        t = end
    return out


def build_slot_results(
    frame_embeds: torch.Tensor,
    local_frame_ids: torch.Tensor,
    frame_shift_sec: float,
    speaker_bank: SpeakerBankProtocol,
) -> List[SlotResult]:
    slots: List[SlotResult] = []
    slot_ids = sorted(set(local_frame_ids.tolist()) - {0})
    for slot_id in slot_ids:
        mask = local_frame_ids == int(slot_id)
        n = int(mask.sum().item())
        if n <= 0:
            continue
        proto = F.normalize(frame_embeds[mask].mean(dim=0), p=2, dim=-1)
        ident = speaker_bank.identify(proto)
        slots.append(
            SlotResult(
                slot=int(slot_id),
                name=ident.get("name", f"speaker_{slot_id}"),
                score=ident.get("score", None),
                is_known=bool(ident.get("is_known", False)),
                num_frames=n,
                duration_sec=round(n * frame_shift_sec, 4),
                prototype=proto.detach().cpu(),
            )
        )
    return slots


def adaptive_cluster_frame_embeddings(
    frame_embeds: torch.Tensor,
    active_mask: torch.Tensor,
    count_logits: torch.Tensor,
    max_speakers: int,
    min_cluster_frames: int,
    min_silhouette: float,
    count_prior_bias: float,
    merge_sim_threshold: float,
) -> Tuple[torch.Tensor, int]:
    device = frame_embeds.device
    local_frame_ids = torch.zeros(frame_embeds.size(0), dtype=torch.long, device=device)

    active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
    if active_idx.numel() == 0:
        return local_frame_ids, 0

    active_embeds = F.normalize(frame_embeds[active_idx], p=2, dim=-1)
    n_active = int(active_idx.numel())

    count_probs = torch.softmax(count_logits.float(), dim=-1)
    count_prior = int(count_probs.argmax(dim=0).item())
    count_prior_conf = float(count_probs[count_prior].item())

    if n_active < max(min_cluster_frames * 2, 12) or max_speakers <= 1 or KMeans is None:
        local_frame_ids[active_idx] = 1
        return local_frame_ids, 1

    candidate_max = min(int(max_speakers), max(1, n_active // max(1, min_cluster_frames)))
    if candidate_max <= 1:
        local_frame_ids[active_idx] = 1
        return local_frame_ids, 1

    x_np = active_embeds.detach().cpu().numpy()
    active_embeds_cpu = active_embeds.detach().cpu()

    best_k = 1
    best_score = float("-inf")
    best_labels: Optional[np.ndarray] = None

    for k in range(2, candidate_max + 1):
        try:
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x_np)
        except Exception:
            continue

        counts = np.bincount(labels, minlength=k)
        if counts.min() < min_cluster_frames:
            continue

        try:
            sil = float(silhouette_score(x_np, labels, metric="cosine"))
        except Exception:
            continue

        centers: List[torch.Tensor] = []
        for cid in range(k):
            mask = torch.from_numpy(labels == cid)
            center = F.normalize(active_embeds_cpu[mask].mean(dim=0), p=2, dim=-1)
            centers.append(center)

        max_center_sim = -1.0
        if len(centers) >= 2:
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    max_center_sim = max(max_center_sim, float(torch.dot(centers[i], centers[j]).item()))

        balance_bonus = 0.05 * float(counts.min() / max(counts.max(), 1))
        score = sil + balance_bonus

        if k == count_prior and count_prior_conf >= 0.35:
            score += count_prior_bias * count_prior_conf
        elif count_prior_conf >= 0.65:
            score -= 0.01 * abs(k - max(count_prior, 1))

        if max_center_sim > merge_sim_threshold:
            score -= 0.10 * (max_center_sim - merge_sim_threshold)

        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels

    use_multi_cluster = best_labels is not None and (
        best_score >= min_silhouette or (count_prior >= 2 and count_prior_conf >= 0.75)
    )

    if not use_multi_cluster:
        local_frame_ids[active_idx] = 1
        return local_frame_ids, 1

    local_frame_ids[active_idx] = torch.from_numpy(best_labels).to(device=device, dtype=torch.long) + 1
    return local_frame_ids, int(best_k)


def frames_to_segments(
    local_frame_ids: torch.Tensor,
    frame_shift_sec: float,
    slot_to_name: Dict[int, str],
    offset_sec: float = 0.0,
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
                start_sec=round(offset_sec + start * frame_shift_sec, 4),
                end_sec=round(offset_sec + end * frame_shift_sec, 4),
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
        fill_gap_frames: int = 3,
        cluster_min_frames: int = 12,
        cluster_silhouette_threshold: float = 0.02,
        cluster_count_prior_bias: float = 0.04,
        cluster_merge_threshold: float = 0.94,
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
        self.fill_gap_frames = int(fill_gap_frames)
        self.cluster_min_frames = int(cluster_min_frames)
        self.cluster_silhouette_threshold = float(cluster_silhouette_threshold)
        self.cluster_count_prior_bias = float(cluster_count_prior_bias)
        self.cluster_merge_threshold = float(cluster_merge_threshold)

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
        offset_sec: float = 0.0,
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

        _, frame_embeds, slot_logits, activity_logits, count_logits = self.model(
            fbank, return_diarization=True
        )

        frame_embeds = frame_embeds[0]
        slot_logits = slot_logits[0]
        activity_logits = activity_logits[0]
        count_logits = count_logits[0]

        activity_prob = torch.sigmoid(activity_logits)
        active_mask = activity_prob >= self.activity_threshold
        active_mask = fill_short_inactive_gaps(active_mask, self.fill_gap_frames)
        active_mask = remove_short_active_runs(active_mask, self.min_active_frames)

        activity_ratio = float(active_mask.float().mean().item())
        mean_activity_prob = float(activity_prob.mean().item())

        if activity_ratio < self.min_activity_ratio or mean_activity_prob < self.min_mean_activity_prob:
            out = self._empty_result(num_frames)
            out.activity_ratio = activity_ratio
            out.frame_activity_prob = activity_prob.detach().cpu()
            return out

        local_frame_ids, _ = adaptive_cluster_frame_embeddings(
            frame_embeds=frame_embeds,
            active_mask=active_mask,
            count_logits=count_logits,
            max_speakers=self.max_mix_speakers,
            min_cluster_frames=self.cluster_min_frames,
            min_silhouette=self.cluster_silhouette_threshold,
            count_prior_bias=self.cluster_count_prior_bias,
            merge_sim_threshold=self.cluster_merge_threshold,
        )
        local_frame_ids = smooth_active_labels(local_frame_ids, min_run=self.min_slot_run)

        raw_slots = build_slot_results(
            frame_embeds=frame_embeds,
            local_frame_ids=local_frame_ids,
            frame_shift_sec=self.frame_shift_sec,
            speaker_bank=self.speaker_bank,
        )

        if len(raw_slots) == 0:
            out = self._empty_result(num_frames)
            out.activity_ratio = activity_ratio
            out.frame_activity_prob = activity_prob.detach().cpu()
            return out

        merged_slots, merge_map = merge_similar_slots(
            raw_slots,
            sim_threshold=self.cluster_merge_threshold,
        )
        raw_to_final: Dict[int, int] = {}
        next_local = 1
        final_slots: List[SlotResult] = []
        slot_to_name: Dict[int, str] = {}

        for s in merged_slots:
            local_id = next_local
            next_local += 1
            rep = int(s.slot)
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

        merged_frame_ids = torch.zeros_like(local_frame_ids, dtype=torch.long)
        for raw_slot, local_id in raw_to_final.items():
            merged_frame_ids[local_frame_ids == int(raw_slot)] = int(local_id)

        kept_activity_ratio = float((merged_frame_ids > 0).float().mean().item())
        if kept_activity_ratio < self.min_activity_ratio:
            out = self._empty_result(num_frames)
            out.activity_ratio = kept_activity_ratio
            out.frame_activity_prob = activity_prob.detach().cpu()
            return out

        local_frame_ids_cpu = merged_frame_ids.detach().cpu()
        segments = frames_to_segments(
            local_frame_ids=local_frame_ids_cpu,
            frame_shift_sec=self.frame_shift_sec,
            slot_to_name=slot_to_name,
            offset_sec=offset_sec,
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
                    "speaker_id": s.slot,
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
