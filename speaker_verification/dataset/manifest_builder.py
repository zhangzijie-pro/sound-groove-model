import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torchaudio


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/")


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, items: Iterable[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def session_id_from_recording_id(recording_id: str) -> str:
    parts = str(recording_id).split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return str(recording_id)


def build_audio_stem_map(root: Path) -> Dict[str, Path]:
    audio_map: Dict[str, Path] = {}
    if not root.is_dir():
        return audio_map
    for suffix in ("*.wav", "*.flac", "*.mp3", "*.m4a"):
        for path in root.rglob(suffix):
            audio_map.setdefault(path.stem, path)
    return audio_map


def parse_textgrid(path: str | Path) -> Tuple[List[Tuple[float, float]], float]:
    path = Path(path)
    segments: List[Tuple[float, float]] = []
    intervals: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    in_interval = False
    duration_sec = 0.0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if duration_sec <= 0.0 and line.startswith("xmax ="):
                try:
                    duration_sec = float(line.split("=", 1)[1].strip())
                except ValueError:
                    pass

            if line.startswith("intervals ["):
                if in_interval:
                    intervals.append(current)
                current = {}
                in_interval = True
                continue

            if not in_interval:
                continue

            if line.startswith("xmin ="):
                try:
                    current["start"] = float(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("xmax ="):
                try:
                    current["end"] = float(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("text ="):
                text = line.split("=", 1)[1].strip()
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                current["text"] = text

        if in_interval:
            intervals.append(current)

    for item in intervals:
        start = float(item.get("start", 0.0))
        end = float(item.get("end", 0.0))
        text = str(item.get("text", "")).strip()
        if end <= start or not text:
            continue
        segments.append((start, end))

    if duration_sec <= 0.0 and segments:
        duration_sec = max(end for _, end in segments)

    return segments, duration_sec


def parse_rttm(path: str | Path) -> Dict[str, List[Tuple[float, float]]]:
    speaker_segments: Dict[str, List[Tuple[float, float]]] = {}
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue

            speaker_id = parts[7]
            try:
                start = float(parts[3])
                duration = float(parts[4])
            except ValueError:
                continue

            if duration <= 0.0:
                continue
            speaker_segments.setdefault(speaker_id, []).append((start, start + duration))

    for speaker_id, segments in speaker_segments.items():
        segments.sort()
        speaker_segments[speaker_id] = segments
    return speaker_segments


def split_records_by_group(
    records: Sequence[Dict[str, Any]],
    group_key: str,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not records:
        return [], []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record[group_key]), []).append(record)

    group_ids = sorted(grouped.keys())
    if len(group_ids) == 1:
        return list(records), []

    rng = random.Random(seed)
    rng.shuffle(group_ids)
    val_count = max(1, int(round(len(group_ids) * val_ratio)))
    val_count = min(val_count, len(group_ids) - 1)
    val_groups = set(group_ids[:val_count])

    train_records = [
        record
        for group_id in sorted(set(group_ids) - val_groups)
        for record in sorted(grouped[group_id], key=lambda item: str(item.get("recording_id", "")))
    ]
    val_records = [
        record
        for group_id in sorted(val_groups)
        for record in sorted(grouped[group_id], key=lambda item: str(item.get("recording_id", "")))
    ]
    return train_records, val_records


def build_ali_near_records(root: Path, split_names: Sequence[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for split_name in split_names:
        split_dir = root / split_name
        layout_root = split_dir if split_dir.is_dir() else root
        audio_map = build_audio_stem_map(layout_root / "audio_dir")
        textgrid_map = {
            path.stem: path
            for path in (layout_root / "textgrid_dir").glob("*.TextGrid")
        }

        if not audio_map or not textgrid_map:
            continue

        for stem in sorted(set(audio_map.keys()) & set(textgrid_map.keys())):
            segments, tg_duration = parse_textgrid(textgrid_map[stem])
            if not segments:
                continue
            info = torchaudio.info(str(audio_map[stem]))
            duration_sec = (
                float(info.num_frames) / float(info.sample_rate)
                if info.sample_rate > 0
                else float(tg_duration)
            )
            speech_sec = sum(end - start for start, end in segments)
            speaker_id = stem.split("_")[-1]
            records.append(
                {
                    "corpus": "ali_near",
                    "entry_type": "close_talk_source",
                    "dataset": "Train_Ali_near",
                    "split": split_name,
                    "recording_id": stem,
                    "session_id": session_id_from_recording_id(stem),
                    "speaker_id": speaker_id,
                    "global_speaker_id": f"ali:{speaker_id}",
                    "wav_path": normalize_path(audio_map[stem]),
                    "annotation_path": normalize_path(textgrid_map[stem]),
                    "annotation_type": "textgrid",
                    "sample_rate": int(info.sample_rate),
                    "num_samples": int(info.num_frames),
                    "duration_sec": float(round(duration_sec, 6)),
                    "speech_sec": float(round(speech_sec, 6)),
                    "segments": [[float(round(start, 4)), float(round(end, 4))] for start, end in segments],
                }
            )
        break
    return records


def build_voxconverse_records(root: Path) -> List[Dict[str, Any]]:
    label_root = root / "labels" / "dev"
    audio_root = root / "voxconverse_dev_wav"
    audio_map = build_audio_stem_map(audio_root)

    if not label_root.is_dir() or not audio_map:
        return []

    records: List[Dict[str, Any]] = []
    for rttm_path in sorted(label_root.glob("*.rttm")):
        recording_id = rttm_path.stem
        wav_path = audio_map.get(recording_id)
        if wav_path is None:
            continue

        speaker_segments = parse_rttm(rttm_path)
        if not speaker_segments:
            continue

        info = torchaudio.info(str(wav_path))
        duration_sec = float(info.num_frames) / float(info.sample_rate) if info.sample_rate > 0 else 0.0
        speakers = []
        total_speech_sec = 0.0
        for speaker_id, segments in sorted(speaker_segments.items()):
            speech_sec = sum(end - start for start, end in segments)
            total_speech_sec += speech_sec
            speakers.append(
                {
                    "speaker_id": speaker_id,
                    "global_speaker_id": f"vox:{recording_id}:{speaker_id}",
                    "speech_sec": float(round(speech_sec, 6)),
                    "segments": [[float(round(start, 4)), float(round(end, 4))] for start, end in segments],
                }
            )

        records.append(
            {
                "corpus": "voxconverse",
                "entry_type": "diar_recording",
                "dataset": "voxconverse_dev",
                "split": "dev",
                "recording_id": recording_id,
                "session_id": recording_id,
                "wav_path": normalize_path(wav_path),
                "annotation_path": normalize_path(rttm_path),
                "annotation_type": "rttm",
                "sample_rate": int(info.sample_rate),
                "num_samples": int(info.num_frames),
                "duration_sec": float(round(duration_sec, 6)),
                "total_speech_sec": float(round(total_speech_sec, 6)),
                "num_speakers": int(len(speakers)),
                "speakers": speakers,
            }
        )
    return records


@dataclass
class BuildCfg:
    seed: int = 1234
    target_sr: int = 16000
    n_mels: int = 80
    datasets_root: str = ""
    output_root: str = ""
    ali_splits: Tuple[str, ...] = ("Train_Ali_near",)
    val_ratio: float = 0.10
    max_mix_speakers: int = 3
    chunk_sec: float = 4.0
    train_virtual_samples: int = 24000
    val_virtual_samples: int = 2400
    prob_1spk: float = 0.10
    prob_2spk: float = 0.55
    prob_3spk: float = 0.35
    train_ali_prob: float = 0.65
    train_vox_prob: float = 0.35
    val_ali_prob: float = 0.50
    val_vox_prob: float = 0.50
    same_session_train_prob: float = 0.75
    same_session_val_prob: float = 1.0
    min_segment_sec: float = 0.08
    min_source_sec: float = 1.2
    max_source_sec: float = 3.6
    max_offset_sec: float = 2.5
    gain_db_low: float = -2.0
    gain_db_high: float = 2.0
    source_rms: float = 0.05
    source_rms_jitter_db: float = 2.0
    ali_leak_scale: float = 0.12
    ali_mask_smooth_sec: float = 0.04
    mix_rms: float = 0.06
    mix_rms_jitter_db: float = 1.5
    vox_max_crop_retry: int = 8

    def resolved_datasets_root(self, project_root: Path) -> Path:
        if self.datasets_root:
            return Path(self.datasets_root).expanduser().resolve()
        return (project_root / ".." / "datasets").resolve()

    def resolved_output_root(self, project_root: Path) -> Path:
        if self.output_root:
            return Path(self.output_root).expanduser().resolve()
        return (project_root / "processed" / "real_diar_dataset").resolve()


def build_online_dataset(cfg: BuildCfg, project_root: Path) -> Path:
    datasets_root = cfg.resolved_datasets_root(project_root)
    output_root = cfg.resolved_output_root(project_root)
    ensure_dir(output_root)

    ali_records = build_ali_near_records(datasets_root, cfg.ali_splits)
    if not ali_records:
        raise FileNotFoundError(f"No Train_Ali_near records found under {datasets_root}")

    vox_records = build_voxconverse_records(datasets_root / "voxconverse")

    ali_train, ali_val = split_records_by_group(
        ali_records,
        group_key="session_id",
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )
    vox_train, vox_val = split_records_by_group(
        vox_records,
        group_key="recording_id",
        val_ratio=cfg.val_ratio,
        seed=cfg.seed + 101,
    )

    train_manifest = ali_train + vox_train
    val_manifest = ali_val + vox_val
    write_jsonl(output_root / "train_manifest.jsonl", train_manifest)
    write_jsonl(output_root / "val_manifest.jsonl", val_manifest)

    speaker_ids = sorted(
        {
            record["global_speaker_id"]
            for record in ali_records
        }
        | {
            speaker["global_speaker_id"]
            for record in vox_records
            for speaker in record.get("speakers", [])
        }
    )
    spk2id = {speaker_id: idx for idx, speaker_id in enumerate(speaker_ids)}
    save_json(output_root / "spk2id.json", spk2id)

    meta = {
        "schema_version": "online_diar_manifest_v2",
        "datasets_root": normalize_path(datasets_root),
        "target_sr": int(cfg.target_sr),
        "n_mels": int(cfg.n_mels),
        "chunk_sec": float(cfg.chunk_sec),
        "max_mix": int(cfg.max_mix_speakers),
        "val_ratio": float(cfg.val_ratio),
        "virtual_samples": {
            "train": int(cfg.train_virtual_samples),
            "val": int(cfg.val_virtual_samples),
        },
        "speaker_count_probs": {
            "1": float(cfg.prob_1spk),
            "2": float(cfg.prob_2spk),
            "3": float(cfg.prob_3spk),
        },
        "split_seeds": {
            "train": int(cfg.seed + 17),
            "val": int(cfg.seed + 29),
        },
        "source_sampling": {
            "train": {
                "ali_synth": float(cfg.train_ali_prob),
                "vox_chunk": float(cfg.train_vox_prob),
            },
            "val": {
                "ali_synth": float(cfg.val_ali_prob),
                "vox_chunk": float(cfg.val_vox_prob),
            },
        },
        "ali_near": {
            "same_session_train_prob": float(cfg.same_session_train_prob),
            "same_session_val_prob": float(cfg.same_session_val_prob),
            "min_segment_sec": float(cfg.min_segment_sec),
            "min_source_sec": float(cfg.min_source_sec),
            "max_source_sec": float(cfg.max_source_sec),
            "max_offset_sec": float(cfg.max_offset_sec),
            "gain_db_low": float(cfg.gain_db_low),
            "gain_db_high": float(cfg.gain_db_high),
            "source_rms": float(cfg.source_rms),
            "source_rms_jitter_db": float(cfg.source_rms_jitter_db),
            "leak_scale": float(cfg.ali_leak_scale),
            "mask_smooth_sec": float(cfg.ali_mask_smooth_sec),
            "mix_rms": float(cfg.mix_rms),
            "mix_rms_jitter_db": float(cfg.mix_rms_jitter_db),
        },
        "voxconverse": {
            "max_crop_retry": int(cfg.vox_max_crop_retry),
        },
        "source_summary": {
            "ali_near": {
                "train": int(len(ali_train)),
                "val": int(len(ali_val)),
            },
            "voxconverse": {
                "train": int(len(vox_train)),
                "val": int(len(vox_val)),
            },
        },
        "build_time": datetime.now().isoformat(timespec="seconds"),
        "build_cfg": asdict(cfg),
    }
    save_json(output_root / "dataset_meta.json", meta)
    return output_root
