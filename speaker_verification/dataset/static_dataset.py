import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from speaker_verification.audio.features import wav_to_fbank_infer


class StaticMixDataset(Dataset):
    """
    Online diarization dataset.

    Supported manifest entries:
    - legacy pt-pack entries
    - Train_Ali_near close-talk sources for online synthetic mixing
    - VoxConverse diarized recordings for direct chunk supervision
    """

    def __init__(
        self,
        out_dir: str = "processed/real_diar_dataset",
        manifest: str = "train_manifest.jsonl",
        crop_sec: float = 4.0,
        shuffle: bool = True,
        crop_mode: str = "random",
        max_speakers: int | None = None,
    ):
        super().__init__()
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.manifest_path = self.out_dir / manifest
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")

        self.crop_mode = str(crop_mode)
        if self.crop_mode not in {"random", "center", "start"}:
            raise ValueError(f"Unsupported crop_mode: {self.crop_mode}")

        spk2id_path = self.out_dir / "spk2id.json"
        if spk2id_path.is_file():
            with spk2id_path.open("r", encoding="utf-8") as f:
                self.spk2id = json.load(f)
        else:
            self.spk2id = {}

        self.num_classes = len(self.spk2id)
        self.crop_frames = int(float(crop_sec) * 100)
        self.base_seed = 20260410
        self.is_train_manifest = manifest.lower().startswith("train")
        self.mode = "legacy"
        self.items: List[str] = []
        self.entries: List[Dict[str, Any]] = []

        self.target_sr = 16000
        self.n_mels = 80
        self.chunk_sec = float(crop_sec)
        self.chunk_samples = int(round(self.chunk_sec * self.target_sr))
        self.max_mix = int(max_speakers or 3)

        self.virtual_samples = 1
        self.min_segment_sec = 0.08
        self.min_source_sec = 1.2
        self.max_source_sec = 3.6
        self.max_offset_sec = 2.5
        self.same_session_train_prob = 0.75
        self.same_session_val_prob = 1.0
        self.gain_db_low = -2.0
        self.gain_db_high = 2.0
        self.source_rms = 0.05
        self.source_rms_jitter_db = 2.0
        self.ali_leak_scale = 0.12
        self.ali_mask_smooth_sec = 0.04
        self.mix_rms = 0.06
        self.mix_rms_jitter_db = 1.5
        self.vox_max_crop_retry = 8

        self.speaker_count_choices: List[int] = [1]
        self.speaker_count_weights: List[float] = [1.0]
        self.source_kind_choices: List[str] = ["ali_synth"]
        self.source_kind_weights: List[float] = [1.0]

        self.ali_entries: List[Dict[str, Any]] = []
        self.ali_entries_by_speaker: Dict[str, List[Dict[str, Any]]] = {}
        self.ali_entries_by_session: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.ali_speaker_ids: List[str] = []
        self.vox_entries: List[Dict[str, Any]] = []

        raw_items: List[Dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_items.append(json.loads(line))

        if raw_items and "pt" in raw_items[0]:
            self.mode = "legacy"
            self.items = [str(item["pt"]) for item in raw_items]
            if shuffle:
                random.shuffle(self.items)
            return

        self.mode = "online"
        self.entries = raw_items
        self._init_online_mode(max_speakers=max_speakers)

    @staticmethod
    def _session_id(entry: Dict[str, Any]) -> str:
        if entry.get("session_id"):
            return str(entry["session_id"])
        recording_id = str(entry.get("recording_id", ""))
        parts = recording_id.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
        return recording_id

    def _init_online_mode(self, max_speakers: int | None) -> None:
        meta_path = self.out_dir / "dataset_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing dataset meta: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        self.target_sr = int(meta.get("target_sr", 16000))
        self.n_mels = int(meta.get("n_mels", 80))
        self.chunk_sec = float(meta.get("chunk_sec", self.chunk_sec))
        self.chunk_samples = int(round(self.chunk_sec * self.target_sr))
        self.max_mix = int(max_speakers or meta.get("max_mix", 3))

        split_key = "train" if self.is_train_manifest else "val"
        self.base_seed = int(meta.get("split_seeds", {}).get(split_key, self.base_seed))
        self.virtual_samples = max(1, int(meta.get("virtual_samples", {}).get(split_key, len(self.entries) or 1)))

        ali_cfg = meta.get("ali_near", {})
        self.min_segment_sec = float(ali_cfg.get("min_segment_sec", self.min_segment_sec))
        self.min_source_sec = max(0.2, min(float(ali_cfg.get("min_source_sec", self.min_source_sec)), self.chunk_sec))
        self.max_source_sec = max(self.min_source_sec, min(float(ali_cfg.get("max_source_sec", self.max_source_sec)), self.chunk_sec))
        self.max_offset_sec = float(ali_cfg.get("max_offset_sec", self.max_offset_sec))
        self.same_session_train_prob = max(0.0, min(float(ali_cfg.get("same_session_train_prob", self.same_session_train_prob)), 1.0))
        self.same_session_val_prob = max(0.0, min(float(ali_cfg.get("same_session_val_prob", self.same_session_val_prob)), 1.0))
        self.gain_db_low = float(ali_cfg.get("gain_db_low", self.gain_db_low))
        self.gain_db_high = float(ali_cfg.get("gain_db_high", self.gain_db_high))
        self.source_rms = float(ali_cfg.get("source_rms", self.source_rms))
        self.source_rms_jitter_db = float(ali_cfg.get("source_rms_jitter_db", self.source_rms_jitter_db))
        self.ali_leak_scale = float(ali_cfg.get("leak_scale", self.ali_leak_scale))
        self.ali_mask_smooth_sec = float(ali_cfg.get("mask_smooth_sec", self.ali_mask_smooth_sec))
        self.mix_rms = float(ali_cfg.get("mix_rms", self.mix_rms))
        self.mix_rms_jitter_db = float(ali_cfg.get("mix_rms_jitter_db", self.mix_rms_jitter_db))
        self.vox_max_crop_retry = int(meta.get("voxconverse", {}).get("max_crop_retry", self.vox_max_crop_retry))

        for entry in self.entries:
            corpus = str(entry.get("corpus", "")).lower()
            if corpus == "ali_near":
                speaker_id = str(entry["speaker_id"])
                session_id = self._session_id(entry)
                entry["session_id"] = session_id
                self.ali_entries.append(entry)
                self.ali_entries_by_speaker.setdefault(speaker_id, []).append(entry)
                self.ali_entries_by_session.setdefault(session_id, {}).setdefault(speaker_id, []).append(entry)
            elif corpus == "voxconverse":
                self.vox_entries.append(entry)

        self.ali_speaker_ids = sorted(self.ali_entries_by_speaker.keys())

        if not self.ali_entries and not self.vox_entries:
            raise ValueError(f"No supported entries found in {self.manifest_path}")

        count_probs = meta.get("speaker_count_probs", {"1": 1.0})
        count_choices: List[int] = []
        count_weights: List[float] = []
        for raw_count, raw_weight in count_probs.items():
            count = int(raw_count)
            weight = float(raw_weight)
            if weight <= 0.0:
                continue
            if count < 1 or count > self.max_mix:
                continue
            if self.ali_entries and count <= len(self.ali_speaker_ids):
                count_choices.append(count)
                count_weights.append(weight)

        if count_choices:
            weight_sum = sum(count_weights)
            self.speaker_count_choices = count_choices
            self.speaker_count_weights = [weight / weight_sum for weight in count_weights]

        sampling_cfg = meta.get("source_sampling", {}).get(split_key, {})
        source_choices: List[str] = []
        source_weights: List[float] = []
        for kind, weight in sampling_cfg.items():
            weight = float(weight)
            if weight <= 0.0:
                continue
            if kind == "ali_synth" and self.ali_entries:
                source_choices.append(kind)
                source_weights.append(weight)
            elif kind == "vox_chunk" and self.vox_entries:
                source_choices.append(kind)
                source_weights.append(weight)

        if not source_choices:
            if self.ali_entries:
                source_choices.append("ali_synth")
                source_weights.append(1.0)
            if self.vox_entries:
                source_choices.append("vox_chunk")
                source_weights.append(1.0)

        weight_sum = sum(source_weights)
        self.source_kind_choices = source_choices
        self.source_kind_weights = [weight / weight_sum for weight in source_weights]

    def __len__(self) -> int:
        if self.mode == "legacy":
            return len(self.items)
        return self.virtual_samples

    def _load_legacy_item(self, idx: int) -> Dict[str, Any]:
        rel_pt = self.items[idx]
        abs_pt = self.out_dir / rel_pt
        pack = torch.load(abs_pt, map_location="cpu", weights_only=False)

        fbank = pack["fbank"].float()
        target_matrix = pack["target_matrix"].float()
        spk_label = int(pack.get("spk_label", -1))

        time_steps = fbank.size(0)
        if time_steps > self.crop_frames:
            if self.crop_mode == "random":
                start = random.randint(0, time_steps - self.crop_frames)
            elif self.crop_mode == "center":
                start = (time_steps - self.crop_frames) // 2
            else:
                start = 0
            fbank = fbank[start:start + self.crop_frames]
            target_matrix = target_matrix[start:start + self.crop_frames]
            valid_mask = torch.ones(self.crop_frames, dtype=torch.bool)
        elif time_steps < self.crop_frames:
            pad = self.crop_frames - time_steps
            fbank = F.pad(fbank, (0, 0, 0, pad))
            target_matrix = F.pad(target_matrix, (0, 0, 0, pad))
            valid_mask = torch.zeros(self.crop_frames, dtype=torch.bool)
            valid_mask[:time_steps] = True
        else:
            valid_mask = torch.ones(self.crop_frames, dtype=torch.bool)

        target_activity = (target_matrix.sum(dim=-1) > 0).float()
        speaker_presence = (target_matrix.sum(dim=0) > 0).float()
        target_count = int(speaker_presence.sum().item())
        return {
            "fbank": fbank,
            "spk_label": torch.tensor(spk_label, dtype=torch.long),
            "target_matrix": target_matrix,
            "target_activity": target_activity,
            "target_count": torch.tensor(target_count, dtype=torch.long),
            "speaker_presence": speaker_presence,
            "valid_mask": valid_mask,
        }

    def _rng_for_index(self, idx: int):
        return random if self.is_train_manifest else random.Random(self.base_seed + int(idx))

    def _sample_weighted(self, choices: Sequence[Any], weights: Sequence[float], rng) -> Any:
        if len(choices) == 1:
            return choices[0]
        value = rng.random()
        accum = 0.0
        for choice, weight in zip(choices, weights):
            accum += weight
            if value <= accum:
                return choice
        return choices[-1]

    def _sample_speaker_count(self, rng) -> int:
        return int(self._sample_weighted(self.speaker_count_choices, self.speaker_count_weights, rng))

    def _sample_source_kind(self, rng) -> str:
        return str(self._sample_weighted(self.source_kind_choices, self.source_kind_weights, rng))

    def _num_feature_frames(self, sample_count: int | None = None) -> int:
        sample_count = self.chunk_samples if sample_count is None else int(sample_count)
        frame_length = int(round(0.025 * self.target_sr))
        frame_shift = int(round(0.010 * self.target_sr))
        if sample_count < frame_length:
            return 1
        return 1 + (sample_count - frame_length) // frame_shift

    def _segments_to_activity(self, segments: Sequence[Tuple[float, float]], num_frames: int) -> torch.Tensor:
        frame_length = int(round(0.025 * self.target_sr))
        frame_shift = int(round(0.010 * self.target_sr))
        centers = (
            torch.arange(num_frames, dtype=torch.float32) * float(frame_shift)
            + float(frame_length) / 2.0
        ) / float(self.target_sr)
        activity = torch.zeros(num_frames, dtype=torch.float32)
        for seg_start, seg_end in segments:
            if seg_end - seg_start <= 0.0:
                continue
            active = (centers >= float(seg_start)) & (centers <= float(seg_end))
            activity[active] = 1.0
        return activity

    def _crop_segments(
        self,
        segments: Sequence[Sequence[float]],
        crop_start_sec: float,
        crop_sec: float,
        offset_sec: float,
        clip_sec: float | None = None,
    ) -> List[Tuple[float, float]]:
        cropped: List[Tuple[float, float]] = []
        crop_end_sec = crop_start_sec + crop_sec
        limit_sec = self.chunk_sec if clip_sec is None else float(clip_sec)
        for raw_start, raw_end in segments:
            raw_start = float(raw_start)
            raw_end = float(raw_end)
            if raw_end <= crop_start_sec or raw_start >= crop_end_sec:
                continue
            seg_start = max(0.0, raw_start - crop_start_sec) + offset_sec
            seg_end = min(crop_sec, raw_end - crop_start_sec) + offset_sec
            seg_start = max(0.0, min(limit_sec, seg_start))
            seg_end = max(0.0, min(limit_sec, seg_end))
            if seg_end - seg_start <= 0.0:
                continue
            cropped.append((seg_start, seg_end))
        return cropped

    def _segments_to_sample_mask(
        self,
        segments: Sequence[Tuple[float, float]],
        num_samples: int,
        clip_sec: float,
    ) -> torch.Tensor:
        mask = torch.zeros(num_samples, dtype=torch.float32)
        for seg_start, seg_end in segments:
            start = max(0, int(round(float(seg_start) / clip_sec * num_samples)))
            end = min(num_samples, int(round(float(seg_end) / clip_sec * num_samples)))
            if end > start:
                mask[start:end] = 1.0
        return mask

    def _smooth_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.numel() <= 1:
            return mask
        kernel = max(1, int(round(self.ali_mask_smooth_sec * self.target_sr)))
        if kernel <= 1:
            return mask
        if kernel % 2 == 0:
            kernel += 1
        return F.avg_pool1d(mask.view(1, 1, -1), kernel_size=kernel, stride=1, padding=kernel // 2).view(-1)

    def _apply_focus_mask(self, wav: torch.Tensor, activity_mask: torch.Tensor) -> torch.Tensor:
        if activity_mask.numel() != wav.numel():
            raise ValueError("activity mask size does not match waveform size")
        if float(activity_mask.max().item()) <= 0.0:
            return wav
        smoothed = self._smooth_mask(activity_mask)
        focus = self.ali_leak_scale + (1.0 - self.ali_leak_scale) * smoothed.clamp(0.0, 1.0)
        return wav * focus

    def _normalize_source_wave(self, wav: torch.Tensor, activity_mask: torch.Tensor, rng) -> torch.Tensor:
        active = wav[activity_mask > 0.5]
        ref = active if active.numel() > 0 else wav
        rms = ref.pow(2).mean().sqrt()
        if float(rms.item()) > 1e-6:
            target_rms = self.source_rms * (10.0 ** (rng.uniform(-self.source_rms_jitter_db, self.source_rms_jitter_db) / 20.0))
            wav = wav * (target_rms / rms)
        wav = wav * (10.0 ** (rng.uniform(self.gain_db_low, self.gain_db_high) / 20.0))
        return wav

    def _normalize_mix_wave(self, wav: torch.Tensor, rng) -> torch.Tensor:
        rms = wav.pow(2).mean().sqrt()
        if float(rms.item()) > 1e-6:
            target_rms = self.mix_rms * (10.0 ** (rng.uniform(-self.mix_rms_jitter_db, self.mix_rms_jitter_db) / 20.0))
            wav = wav * (target_rms / rms)
        peak = wav.abs().max()
        if float(peak.item()) > 0.99:
            wav = wav / peak * 0.99
        return wav

    def _load_crop_wave(self, entry: Dict[str, Any], crop_start_sec: float, crop_sec: float) -> torch.Tensor:
        source_sr = int(entry.get("sample_rate", self.target_sr))
        frame_offset = max(0, int(round(crop_start_sec * source_sr)))
        num_frames = max(1, int(round(crop_sec * source_sr)))
        target_samples = max(1, int(round(crop_sec * self.target_sr)))

        wav, sr = torchaudio.load(
            entry["wav_path"],
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
        wav = wav.mean(dim=0).to(torch.float32)

        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        if wav.numel() < target_samples:
            wav = F.pad(wav, (0, target_samples - wav.numel()))
        elif wav.numel() > target_samples:
            wav = wav[:target_samples]

        return wav.contiguous()

    def _pick_crop_start(self, entry: Dict[str, Any], crop_sec: float, rng) -> float:
        duration_sec = float(entry.get("duration_sec", 0.0))
        if duration_sec <= crop_sec:
            return 0.0

        segments = [
            (float(start), float(end))
            for start, end in entry.get("segments", [])
            if float(end) - float(start) >= self.min_segment_sec
        ]
        if not segments:
            return rng.uniform(0.0, max(duration_sec - crop_sec, 0.0))

        seg_start, seg_end = rng.choice(segments)
        anchor = rng.uniform(seg_start, max(seg_start + 1e-3, seg_end))
        low = max(0.0, anchor - crop_sec)
        high = min(duration_sec - crop_sec, anchor)
        if high < low:
            return max(0.0, min(duration_sec - crop_sec, seg_start))
        return rng.uniform(low, high)

    def _pick_group_crop_start(self, entries: Sequence[Dict[str, Any]], crop_sec: float, rng) -> float:
        duration_candidates = [
            float(entry.get("duration_sec", 0.0))
            for entry in entries
            if float(entry.get("duration_sec", 0.0)) > 0.0
        ]
        duration_sec = min(duration_candidates) if duration_candidates else crop_sec
        if duration_sec <= crop_sec:
            return 0.0

        segments: List[Tuple[float, float]] = []
        for entry in entries:
            for start, end in entry.get("segments", []):
                start = float(start)
                end = float(end)
                if end - start >= self.min_segment_sec:
                    segments.append((start, end))

        if not segments:
            return rng.uniform(0.0, max(duration_sec - crop_sec, 0.0))

        seg_start, seg_end = rng.choice(segments)
        anchor = rng.uniform(seg_start, max(seg_start + 1e-3, seg_end))
        low = max(0.0, anchor - crop_sec)
        high = min(duration_sec - crop_sec, anchor)
        if high < low:
            return max(0.0, min(duration_sec - crop_sec, seg_start))
        return rng.uniform(low, high)

    def _sample_source_duration(self, speaker_count: int, rng) -> float:
        if speaker_count <= 1:
            return self.chunk_sec
        return rng.uniform(self.min_source_sec, self.max_source_sec)

    def _sample_offset(self, source_duration_sec: float, speaker_slot: int, rng) -> float:
        max_offset = max(0.0, min(self.chunk_sec - source_duration_sec, self.max_offset_sec))
        if max_offset <= 0.0:
            return 0.0
        if speaker_slot == 0 and rng.random() < 0.35:
            return 0.0
        return rng.uniform(0.0, max_offset)

    def _select_ali_entries(self, rng, count: int) -> Tuple[List[Dict[str, Any]], bool]:
        same_session_prob = self.same_session_train_prob if self.is_train_manifest else self.same_session_val_prob
        if count > 1 and same_session_prob > 0.0 and rng.random() < same_session_prob:
            candidates = [
                (session_id, speaker_map)
                for session_id, speaker_map in self.ali_entries_by_session.items()
                if len(speaker_map) >= count
            ]
            if candidates:
                _, speaker_map = rng.choice(candidates)
                picked_speakers = rng.sample(sorted(speaker_map.keys()), count)
                return [rng.choice(speaker_map[speaker_id]) for speaker_id in picked_speakers], True

        picked_speakers = rng.sample(self.ali_speaker_ids, count)
        return [rng.choice(self.ali_entries_by_speaker[speaker_id]) for speaker_id in picked_speakers], False

    def _build_item_from_targets(self, wav: torch.Tensor, target_matrix: torch.Tensor) -> Dict[str, Any]:
        fbank = wav_to_fbank_infer(
            wav,
            n_mels=self.n_mels,
            crop_sec=None,
            crop_mode="none",
        ).float()

        expected_frames = int(target_matrix.size(0))
        if fbank.size(0) != expected_frames:
            if fbank.size(0) < expected_frames:
                target_matrix = target_matrix[:fbank.size(0)]
            else:
                target_matrix = F.pad(target_matrix, (0, 0, 0, fbank.size(0) - expected_frames))

        valid_mask = torch.ones(fbank.size(0), dtype=torch.bool)
        target_activity = (target_matrix.sum(dim=-1) > 0).float()
        speaker_presence = (target_matrix.sum(dim=0) > 0).float()
        target_count = int(speaker_presence.sum().item())

        return {
            "fbank": fbank,
            "spk_label": torch.tensor(-1, dtype=torch.long),
            "target_matrix": target_matrix,
            "target_activity": target_activity,
            "target_count": torch.tensor(target_count, dtype=torch.long),
            "speaker_presence": speaker_presence,
            "valid_mask": valid_mask,
        }

    def _build_ali_item(self, idx: int, rng) -> Dict[str, Any]:
        count = self._sample_speaker_count(rng)
        selected_entries, is_same_session = self._select_ali_entries(rng, count)

        num_frames = self._num_feature_frames()
        mix_wav = torch.zeros(self.chunk_samples, dtype=torch.float32)
        target_matrix = torch.zeros(num_frames, self.max_mix, dtype=torch.float32)

        shared_crop_start_sec = 0.0
        if is_same_session:
            shared_crop_start_sec = self._pick_group_crop_start(selected_entries, self.chunk_sec, rng)

        for speaker_slot, entry in enumerate(selected_entries):
            if is_same_session:
                source_duration_sec = self.chunk_sec
                crop_start_sec = shared_crop_start_sec
                offset_sec = 0.0
            else:
                source_duration_sec = self._sample_source_duration(count, rng)
                crop_start_sec = self._pick_crop_start(entry, source_duration_sec, rng)
                offset_sec = self._sample_offset(source_duration_sec, speaker_slot, rng)

            wav = self._load_crop_wave(entry, crop_start_sec, source_duration_sec)
            source_segments = self._crop_segments(
                entry.get("segments", []),
                crop_start_sec=crop_start_sec,
                crop_sec=source_duration_sec,
                offset_sec=0.0,
                clip_sec=source_duration_sec,
            )
            source_mask = self._segments_to_sample_mask(source_segments, wav.numel(), max(source_duration_sec, 1e-6))
            wav = self._apply_focus_mask(wav, source_mask)
            wav = self._normalize_source_wave(wav, source_mask, rng)

            start_sample = int(round(offset_sec * self.target_sr))
            end_sample = min(self.chunk_samples, start_sample + wav.numel())
            if end_sample > start_sample:
                mix_wav[start_sample:end_sample] += wav[:end_sample - start_sample]

            mixed_segments = self._crop_segments(
                entry.get("segments", []),
                crop_start_sec=crop_start_sec,
                crop_sec=source_duration_sec,
                offset_sec=offset_sec,
            )
            target_matrix[:, speaker_slot] = self._segments_to_activity(mixed_segments, num_frames)

        mix_wav = self._normalize_mix_wave(mix_wav, rng)
        return self._build_item_from_targets(mix_wav, target_matrix)

    def _pick_vox_entry(self, idx: int, rng) -> Dict[str, Any]:
        if self.is_train_manifest:
            return rng.choice(self.vox_entries)
        return self.vox_entries[idx % len(self.vox_entries)]

    def _pick_vox_crop_start(self, entry: Dict[str, Any], rng) -> float:
        duration_sec = float(entry.get("duration_sec", 0.0))
        if duration_sec <= self.chunk_sec:
            return 0.0

        segments: List[Tuple[float, float]] = []
        for speaker in entry.get("speakers", []):
            for start, end in speaker.get("segments", []):
                start = float(start)
                end = float(end)
                if end - start >= self.min_segment_sec:
                    segments.append((start, end))

        if not segments:
            return rng.uniform(0.0, max(duration_sec - self.chunk_sec, 0.0))

        seg_start, seg_end = rng.choice(segments)
        anchor = rng.uniform(seg_start, max(seg_start + 1e-3, seg_end))
        low = max(0.0, anchor - self.chunk_sec)
        high = min(duration_sec - self.chunk_sec, anchor)
        if high < low:
            return max(0.0, min(duration_sec - self.chunk_sec, seg_start))
        return rng.uniform(low, high)

    def _vox_target_for_crop(
        self,
        entry: Dict[str, Any],
        crop_start_sec: float,
        allow_clip: bool,
    ) -> Tuple[torch.Tensor, int]:
        num_frames = self._num_feature_frames()
        speaker_items: List[Tuple[int, str, torch.Tensor]] = []

        for speaker in entry.get("speakers", []):
            local_segments = self._crop_segments(
                speaker.get("segments", []),
                crop_start_sec=crop_start_sec,
                crop_sec=self.chunk_sec,
                offset_sec=0.0,
            )
            if not local_segments:
                continue
            activity = self._segments_to_activity(local_segments, num_frames)
            active_frames = int(activity.sum().item())
            if active_frames <= 0:
                continue
            speaker_items.append((active_frames, str(speaker.get("speaker_id", "")), activity))

        active_speaker_count = len(speaker_items)
        if active_speaker_count == 0:
            return torch.zeros(num_frames, self.max_mix, dtype=torch.float32), 0
        if active_speaker_count > self.max_mix and not allow_clip:
            return torch.zeros(num_frames, self.max_mix, dtype=torch.float32), active_speaker_count

        speaker_items.sort(key=lambda item: (-item[0], item[1]))
        speaker_items = speaker_items[:self.max_mix]
        target_matrix = torch.zeros(num_frames, self.max_mix, dtype=torch.float32)
        for speaker_slot, (_, _, activity) in enumerate(speaker_items):
            target_matrix[:, speaker_slot] = activity
        return target_matrix, active_speaker_count

    def _build_vox_item(self, idx: int, rng) -> Dict[str, Any]:
        entry = self._pick_vox_entry(idx, rng)
        last_target = torch.zeros(self._num_feature_frames(), self.max_mix, dtype=torch.float32)
        last_crop_start = 0.0

        for attempt in range(max(1, self.vox_max_crop_retry)):
            crop_start_sec = self._pick_vox_crop_start(entry, rng)
            allow_clip = attempt + 1 >= self.vox_max_crop_retry
            target_matrix, active_speaker_count = self._vox_target_for_crop(
                entry,
                crop_start_sec=crop_start_sec,
                allow_clip=allow_clip,
            )
            last_target = target_matrix
            last_crop_start = crop_start_sec
            if active_speaker_count > 0 and (active_speaker_count <= self.max_mix or allow_clip):
                break

        wav = self._load_crop_wave(entry, last_crop_start, self.chunk_sec)
        wav = self._normalize_mix_wave(wav, rng)
        return self._build_item_from_targets(wav, last_target)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == "legacy":
            return self._load_legacy_item(idx)

        rng = self._rng_for_index(idx)
        source_kind = self._sample_source_kind(rng)
        if source_kind == "vox_chunk":
            return self._build_vox_item(idx, rng)
        return self._build_ali_item(idx, rng)
