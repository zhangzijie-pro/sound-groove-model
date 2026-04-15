import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torchaudio
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

PROCESSED_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PROCESSED_ROOT.parent
DATASETS_ROOT = (PROCESSED_ROOT / ".." / ".." / "datasets").resolve()
AISHELL4_ROOT = DATASETS_ROOT
OUTPUT_ROOT = PROJECT_ROOT / "processed" / "real_diar_dataset"
AISHELL4_TRAIN_SOURCE_SPLITS = ["Train_Ali_near"]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def session_id_from_recording_id(recording_id: str) -> str:
    parts = str(recording_id).split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return str(recording_id)


def save_json(path: str | Path, obj: Any):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, items: Iterable[dict]):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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


def collect_aishell4_records(root: Path, split_names: List[str]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []

    for split in split_names:
        split_dir = root / split
        layout_root = split_dir if split_dir.is_dir() else root
        audio_map = build_audio_stem_map(layout_root / "audio_dir")
        textgrid_map = {path.stem: path for path in (layout_root / "textgrid_dir").glob("*.TextGrid")}

        if not audio_map or not textgrid_map:
            raise FileNotFoundError(f"Missing AISHELL4 audio_dir/textgrid_dir under {layout_root}")

        for stem in sorted(set(audio_map.keys()) & set(textgrid_map.keys())):
            records.append(
                {
                    "dataset": "aishell4",
                    "split": split,
                    "recording_id": stem,
                    "session_id": session_id_from_recording_id(stem),
                    "speaker_id": stem.split("_")[-1],
                    "wav_path": normalize_path(audio_map[stem]),
                    "textgrid_path": normalize_path(textgrid_map[stem]),
                    "annotation_type": "textgrid",
                }
            )
        break

    return records


def split_train_val_records(
    records: List[Dict[str, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not records:
        return [], []

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for record in records:
        session_id = record.get("session_id") or session_id_from_recording_id(record["recording_id"])
        grouped.setdefault(session_id, []).append(record)

    session_ids = sorted(grouped)
    random.Random(seed).shuffle(session_ids)
    val_count = max(1, int(round(len(session_ids) * val_ratio)))
    val_count = min(val_count, len(session_ids) - 1)
    val_sessions = set(session_ids[:val_count])

    val_records = [
        record
        for session_id in sorted(val_sessions)
        for record in sorted(grouped[session_id], key=lambda x: x["recording_id"])
    ]
    train_records = [
        record
        for session_id in sorted(set(session_ids) - val_sessions)
        for record in sorted(grouped[session_id], key=lambda x: x["recording_id"])
    ]
    return train_records, val_records


def build_source_entry(record: Dict[str, str]) -> Dict[str, Any]:
    segments, tg_duration = parse_textgrid(record["textgrid_path"])
    if not segments:
        raise ValueError(f"No active segments parsed from {record['textgrid_path']}")

    info = torchaudio.info(record["wav_path"])
    duration_sec = float(info.num_frames) / float(info.sample_rate) if info.sample_rate > 0 else float(tg_duration)
    speech_sec = sum(end - start for start, end in segments)

    return {
        "dataset": record["dataset"],
        "split": record["split"],
        "recording_id": record["recording_id"],
        "session_id": record.get("session_id") or session_id_from_recording_id(record["recording_id"]),
        "speaker_id": record["speaker_id"],
        "wav_path": record["wav_path"],
        "textgrid_path": record["textgrid_path"],
        "annotation_type": record["annotation_type"],
        "sample_rate": int(info.sample_rate),
        "num_samples": int(info.num_frames),
        "duration_sec": float(round(duration_sec, 6)),
        "speech_sec": float(round(speech_sec, 6)),
        "segments": [[float(round(start, 4)), float(round(end, 4))] for start, end in segments],
    }


@dataclass
class BuildCfg:
    seed: int = 1234
    target_sr: int = 16000
    n_mels: int = 80
    aishell4_root: str = str(AISHELL4_ROOT)
    real_out_dir: str = str(OUTPUT_ROOT)
    real_chunk_sec: float = 4.0
    real_max_mix: int = 3
    real_val_ratio: float = 0.10
    synth_train_samples: int = 20000
    synth_val_samples: int = 2000
    synth_prob_1spk: float = 0.10
    synth_prob_2spk: float = 0.60
    synth_prob_3spk: float = 0.30
    synth_max_offset_sec: float = 2.5
    synth_min_segment_sec: float = 0.08
    synth_min_source_sec: float = 1.4
    synth_max_source_sec: float = 3.4
    synth_same_session_train_prob: float = 0.80
    synth_same_session_val_prob: float = 1.00
    synth_gain_db_low: float = -3.0
    synth_gain_db_high: float = 3.0


def save_meta(cfg: BuildCfg, output_root: Path, train_count: int, val_count: int):
    meta = {
        "schema_version": "aishell4_synth_mix_v1",
        "sources": ["aishell4_near"],
        "target_sr": cfg.target_sr,
        "n_mels": cfg.n_mels,
        "chunk_sec": cfg.real_chunk_sec,
        "max_mix": cfg.real_max_mix,
        "val_ratio": cfg.real_val_ratio,
        "source_count": {
            "train": train_count,
            "val": val_count,
        },
        "virtual_samples": {
            "train": int(cfg.synth_train_samples),
            "val": int(cfg.synth_val_samples),
        },
        "speaker_count_probs": {
            "1": float(cfg.synth_prob_1spk),
            "2": float(cfg.synth_prob_2spk),
            "3": float(cfg.synth_prob_3spk),
        },
        "synth": {
            "max_offset_sec": float(cfg.synth_max_offset_sec),
            "min_segment_sec": float(cfg.synth_min_segment_sec),
            "min_source_sec": float(cfg.synth_min_source_sec),
            "max_source_sec": float(cfg.synth_max_source_sec),
            "same_session_train_prob": float(cfg.synth_same_session_train_prob),
            "same_session_val_prob": float(cfg.synth_same_session_val_prob),
            "gain_db_low": float(cfg.synth_gain_db_low),
            "gain_db_high": float(cfg.synth_gain_db_high),
        },
        "split_seeds": {
            "train": int(cfg.seed + 17),
            "val": int(cfg.seed + 29),
        },
        "build_time": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(output_root / "dataset_meta.json", meta)


def save_speaker_index(entries: List[Dict[str, Any]], output_root: Path):
    speakers = sorted({str(item["speaker_id"]) for item in entries})
    spk2id = {speaker_id: idx for idx, speaker_id in enumerate(speakers)}
    save_json(output_root / "spk2id.json", spk2id)


def build_real_diar_dataset(cfg: BuildCfg):
    set_seed(cfg.seed)
    output_root = Path(cfg.real_out_dir)
    ensure_dir(output_root)

    records = collect_aishell4_records(Path(cfg.aishell4_root), AISHELL4_TRAIN_SOURCE_SPLITS)
    train_records, val_records = split_train_val_records(records, val_ratio=cfg.real_val_ratio, seed=cfg.seed)
    if not train_records:
        raise FileNotFoundError("No AISHELL4 training records found under Train_Ali_near")

    train_entries = [build_source_entry(record) for record in tqdm(train_records, desc="Parse train sources")]
    val_entries = [build_source_entry(record) for record in tqdm(val_records, desc="Parse val sources")]

    write_jsonl(output_root / "train_manifest.jsonl", train_entries)
    write_jsonl(output_root / "val_manifest.jsonl", val_entries)
    save_speaker_index(train_entries + val_entries, output_root)
    save_meta(cfg, output_root, train_count=len(train_entries), val_count=len(val_entries))

    print(
        "[SYNTH DIAR] build complete | "
        f"train_sources={len(train_entries)} val_sources={len(val_entries)} "
        f"train_virtual_samples={cfg.synth_train_samples} val_virtual_samples={cfg.synth_val_samples}"
    )


def main():
    cfg = BuildCfg()
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    build_real_diar_dataset(cfg)
    print("[ALL DONE]")


if __name__ == "__main__":
    main()
