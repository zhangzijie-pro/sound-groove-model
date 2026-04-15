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
        assert self.manifest_path.is_file(), f"Missing manifest: {self.manifest_path}"
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
        self.entries_by_speaker: Dict[str, List[Dict[str, Any]]] = {}
        self.entries_by_session: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.speaker_ids: List[str] = []
        self.virtual_samples = 0
        self.target_sr = 16000
        self.n_mels = 80
        self.chunk_sec = float(crop_sec)
        self.chunk_samples = int(round(self.chunk_sec * self.target_sr))
        self.max_mix = int(max_speakers or 3)
        self.min_segment_sec = 0.08
        self.max_offset_sec = 2.5
        self.min_source_sec = 1.4
        self.max_source_sec = 3.4
        self.same_session_train_prob = 0.80
        self.same_session_val_prob = 1.00
        self.gain_db_low = -3.0
        self.gain_db_high = 3.0
        self.crop_rms = 0.05
        self.speaker_count_choices: List[int] = [1]
        self.speaker_count_weights: List[float] = [1.0]

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

        self.mode = "synth"
        self.entries = raw_items
        self._init_synth_mode(max_speakers=max_speakers)

    @staticmethod
    def _session_id(entry: Dict[str, Any]) -> str:
        if entry.get("session_id"):
            return str(entry["session_id"])
        recording_id = str(entry.get("recording_id", ""))
        parts = recording_id.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
        return recording_id

    def _init_synth_mode(self, max_speakers: int | None):
        meta_path = self.out_dir / "dataset_meta.json"
        assert meta_path.is_file(), f"Missing dataset meta: {meta_path}"
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        self.target_sr = int(meta.get("target_sr", 16000))
        self.n_mels = int(meta.get("n_mels", 80))
        self.chunk_sec = float(meta.get("chunk_sec", self.chunk_sec))
        self.chunk_samples = int(round(self.chunk_sec * self.target_sr))
        self.max_mix = int(max_speakers or meta.get("max_mix", 3))

        synth_cfg = meta.get("synth", {})
        self.min_segment_sec = float(synth_cfg.get("min_segment_sec", 0.08))
        self.max_offset_sec = float(synth_cfg.get("max_offset_sec", 2.5))
        self.min_source_sec = float(synth_cfg.get("min_source_sec", 1.4))
        self.max_source_sec = float(synth_cfg.get("max_source_sec", 3.4))
        self.min_source_sec = max(0.2, min(self.min_source_sec, self.chunk_sec))
        self.max_source_sec = max(self.min_source_sec, min(self.max_source_sec, self.chunk_sec))
        self.same_session_train_prob = float(synth_cfg.get("same_session_train_prob", 0.80))
        self.same_session_val_prob = float(synth_cfg.get("same_session_val_prob", 1.00))
        self.same_session_train_prob = max(0.0, min(self.same_session_train_prob, 1.0))
        self.same_session_val_prob = max(0.0, min(self.same_session_val_prob, 1.0))
        self.gain_db_low = float(synth_cfg.get("gain_db_low", -3.0))
        self.gain_db_high = float(synth_cfg.get("gain_db_high", 3.0))

        split_key = "train" if self.is_train_manifest else "val"
        split_seeds = meta.get("split_seeds", {})
        self.base_seed = int(split_seeds.get(split_key, self.base_seed))
        virtual_samples = meta.get("virtual_samples", {})
        self.virtual_samples = int(virtual_samples.get(split_key, len(self.entries)))
        self.virtual_samples = max(self.virtual_samples, 1)

        for entry in self.entries:
            speaker_id = str(entry["speaker_id"])
            session_id = self._session_id(entry)
            entry["session_id"] = session_id
            self.entries_by_speaker.setdefault(speaker_id, []).append(entry)
            self.entries_by_session.setdefault(session_id, {}).setdefault(speaker_id, []).append(entry)
        self.speaker_ids = sorted(self.entries_by_speaker.keys())
        if not self.speaker_ids:
            raise ValueError(f"No speaker entries found in {self.manifest_path}")

        count_probs = meta.get("speaker_count_probs", {"1": 1.0})
        choices: List[int] = []
        weights: List[float] = []
        for raw_count, raw_weight in count_probs.items():
            count = int(raw_count)
            weight = float(raw_weight)
            if weight <= 0.0:
                continue
            if count < 1 or count > self.max_mix or count > len(self.speaker_ids):
                continue
            choices.append(count)
            weights.append(weight)

        if not choices:
            choices = [1]
            weights = [1.0]

        weight_sum = sum(weights)
        self.speaker_count_choices = choices
        self.speaker_count_weights = [w / weight_sum for w in weights]

    def __len__(self):
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

    def _sample_speaker_count(self, rng) -> int:
        if len(self.speaker_count_choices) == 1:
            return self.speaker_count_choices[0]
        value = rng.random()
        accum = 0.0
        for count, weight in zip(self.speaker_count_choices, self.speaker_count_weights):
            accum += weight
            if value <= accum:
                return count
        return self.speaker_count_choices[-1]

    def _select_entries(self, rng, count: int) -> Tuple[List[Dict[str, Any]], bool]:
        same_session_prob = self.same_session_train_prob if self.is_train_manifest else self.same_session_val_prob
        if count > 1 and same_session_prob > 0.0 and rng.random() < same_session_prob:
            candidates = [
                (session_id, speaker_map)
                for session_id, speaker_map in self.entries_by_session.items()
                if len(speaker_map) >= count
            ]
            if candidates:
                _, speaker_map = rng.choice(candidates)
                picked_speakers = rng.sample(sorted(speaker_map.keys()), count)
                return [rng.choice(speaker_map[speaker_id]) for speaker_id in picked_speakers], True

        picked_speakers = rng.sample(self.speaker_ids, count)
        return [rng.choice(self.entries_by_speaker[speaker_id]) for speaker_id in picked_speakers], False

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
        anchor_high = max(seg_start + 1e-3, seg_end)
        anchor = rng.uniform(seg_start, anchor_high)
        low = max(0.0, anchor - crop_sec)
        high = min(duration_sec - crop_sec, anchor)
        if high < low:
            return max(0.0, min(duration_sec - crop_sec, seg_start))
        return rng.uniform(low, high)

    def _pick_group_crop_start(self, entries: Sequence[Dict[str, Any]], crop_sec: float, rng) -> float:
        duration_candidates = [float(entry.get("duration_sec", 0.0)) for entry in entries if float(entry.get("duration_sec", 0.0)) > 0.0]
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
        anchor_high = max(seg_start + 1e-3, seg_end)
        anchor = rng.uniform(seg_start, anchor_high)
        low = max(0.0, anchor - crop_sec)
        high = min(duration_sec - crop_sec, anchor)
        if high < low:
            return max(0.0, min(duration_sec - crop_sec, seg_start))
        return rng.uniform(low, high)

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

    def _crop_segments(
        self,
        segments: Sequence[Sequence[float]],
        crop_start_sec: float,
        crop_sec: float,
        offset_sec: float,
    ) -> List[Tuple[float, float]]:
        cropped: List[Tuple[float, float]] = []
        crop_end_sec = crop_start_sec + crop_sec
        for raw_start, raw_end in segments:
            if float(raw_end) <= crop_start_sec or float(raw_start) >= crop_end_sec:
                continue
            seg_start = max(0.0, float(raw_start) - crop_start_sec) + offset_sec
            seg_end = min(crop_sec, float(raw_end) - crop_start_sec) + offset_sec
            seg_start = max(0.0, min(self.chunk_sec, seg_start))
            seg_end = max(0.0, min(self.chunk_sec, seg_end))
            if seg_end - seg_start <= 0.0:
                continue
            cropped.append((seg_start, seg_end))
        return cropped

    def _normalize_wave(self, wav: torch.Tensor, rng) -> torch.Tensor:
        rms = wav.pow(2).mean().sqrt()
        if float(rms.item()) > 1e-5:
            wav = wav * (self.crop_rms / rms)
        gain_db = rng.uniform(self.gain_db_low, self.gain_db_high)
        wav = wav * (10.0 ** (gain_db / 20.0))
        return wav

    def _num_feature_frames(self) -> int:
        frame_length = int(round(0.025 * self.target_sr))
        frame_shift = int(round(0.010 * self.target_sr))
        if self.chunk_samples < frame_length:
            return 1
        return 1 + (self.chunk_samples - frame_length) // frame_shift

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

    def _build_synth_item(self, idx: int) -> Dict[str, Any]:
        rng = random if self.is_train_manifest else random.Random(self.base_seed + int(idx))
        count = self._sample_speaker_count(rng)
        selected_entries, is_same_session = self._select_entries(rng, count)

        mix_wav = torch.zeros(self.chunk_samples, dtype=torch.float32)
        num_frames = self._num_feature_frames()
        target_matrix = torch.zeros(num_frames, self.max_mix, dtype=torch.float32)

        shared_crop_start_sec = 0.0
        if is_same_session:
            shared_crop_start_sec = self._pick_group_crop_start(selected_entries, self.chunk_sec, rng)

        for speaker_slot, entry in enumerate(selected_entries):
            if is_same_session:
                source_duration_sec = self.chunk_sec
                offset_sec = 0.0
                crop_start_sec = shared_crop_start_sec
            else:
                source_duration_sec = self._sample_source_duration(count, rng)
                offset_sec = self._sample_offset(source_duration_sec, speaker_slot, rng)
                crop_start_sec = self._pick_crop_start(entry, source_duration_sec, rng)
            wav = self._load_crop_wave(entry, crop_start_sec, source_duration_sec)
            wav = self._normalize_wave(wav, rng)
            start_sample = int(round(offset_sec * self.target_sr))
            end_sample = min(self.chunk_samples, start_sample + wav.numel())
            if end_sample > start_sample:
                mix_wav[start_sample:end_sample] += wav[:end_sample - start_sample]

            local_segments = self._crop_segments(
                entry.get("segments", []),
                crop_start_sec=crop_start_sec,
                crop_sec=source_duration_sec,
                offset_sec=offset_sec,
            )
            target_matrix[:, speaker_slot] = self._segments_to_activity(local_segments, num_frames)

        peak = mix_wav.abs().max()
        if float(peak.item()) > 0.99:
            mix_wav = mix_wav / peak * 0.99

        fbank = wav_to_fbank_infer(
            mix_wav,
            n_mels=self.n_mels,
            crop_sec=None,
            crop_mode="none",
        ).float()

        if fbank.size(0) != num_frames:
            if fbank.size(0) < num_frames:
                target_matrix = target_matrix[:fbank.size(0)]
            else:
                pad_frames = fbank.size(0) - num_frames
                target_matrix = F.pad(target_matrix, (0, 0, 0, pad_frames))
            num_frames = fbank.size(0)

        valid_mask = torch.ones(num_frames, dtype=torch.bool)
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

    def __getitem__(self, idx) -> Dict[str, Any]:
        if self.mode == "legacy":
            return self._load_legacy_item(idx)
        return self._build_synth_item(idx)
