import io
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from speaker_verification.audio.features import load_wav_mono, wav_to_fbank_infer

PROCESSED_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PROCESSED_ROOT.parent
DATASETS_ROOT = (PROCESSED_ROOT / ".." / ".." / "datasets").resolve()
AISHELL4_ROOT = DATASETS_ROOT
VOXCONVERSE_ROOT = DATASETS_ROOT / "voxconverse"
OUTPUT_ROOT = PROJECT_ROOT / "processed" / "real_diar_dataset"

AISHELL4_TRAIN_SOURCE_SPLITS = ["Train_Ali_near"]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    items: List[dict] = []
    if not path.is_file():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, items: List[dict]):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def dedupe_preserve_order(items: Iterable[dict]) -> List[dict]:
    seen = set()
    out: List[dict] = []
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def backup_files(paths: Iterable[str | Path], backup_root: str | Path, relative_root: str | Path):
    backup_root = Path(backup_root)
    relative_root = Path(relative_root)
    for src in paths:
        src = Path(src)
        if not src.is_file():
            continue
        rel = src.relative_to(relative_root)
        dst = backup_root / rel
        ensure_dir(dst.parent)
        if not dst.exists():
            shutil.copy2(src, dst)


def parse_rttm(path: str | Path) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            if dur <= 0:
                continue
            segments.append(
                {
                    "recording_id": parts[1],
                    "start": start,
                    "end": start + dur,
                    "speaker": parts[7],
                }
            )
    return segments


def parse_textgrid(path: str | Path) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    path = Path(path)
    recording_id = path.stem
    speaker_name = recording_id.split("_")[-1]

    intervals: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    in_interval = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

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
        segments.append(
            {
                "recording_id": recording_id,
                "start": start,
                "end": end,
                "speaker": speaker_name,
            }
        )

    return segments


def build_audio_stem_map(root: Path) -> Dict[str, Path]:
    audio_map: Dict[str, Path] = {}
    if not root.is_dir():
        return audio_map
    for suffix in ("*.wav", "*.flac", "*.mp3", "*.m4a"):
        for path in root.rglob(suffix):
            audio_map.setdefault(path.stem, path)
    return audio_map


def collect_split_records(dataset: str, root: Path, split_names: List[str]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    global_audio_map = build_audio_stem_map(root)
    seen = set()

    for split in split_names:
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"[WARN] Missing split dir: {split_dir}")
            continue

        local_audio_map = build_audio_stem_map(split_dir)
        for rttm_path in sorted(split_dir.rglob("*.rttm")):
            stem = rttm_path.stem
            wav_path = local_audio_map.get(stem) or global_audio_map.get(stem)
            if wav_path is None:
                print(f"[WARN] Missing audio for RTTM: {rttm_path}")
                continue

            key = (dataset, split, stem)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "recording_id": stem,
                    "wav_path": normalize_path(wav_path),
                    "rttm_path": normalize_path(rttm_path),
                }
            )

    return records


def collect_aishell4_records(root: Path, split_names: List[str]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []

    for split in split_names:
        split_dir = root / split
        layout_root = split_dir if split_dir.is_dir() else root
        audio_map = build_audio_stem_map(layout_root / "audio_dir")
        textgrid_map = {path.stem: path for path in (layout_root / "textgrid_dir").glob("*.TextGrid")}

        if not audio_map or not textgrid_map:
            if split_dir.is_dir():
                records.extend(collect_split_records("aishell4", root, [split]))
                continue
            print(f"[WARN] Missing AISHELL4 split dir: {split_dir}. Fallback to flat layout under {root}")

        for stem in sorted(set(audio_map.keys()) & set(textgrid_map.keys())):
            records.append(
                {
                    "dataset": "aishell4",
                    "split": split,
                    "recording_id": stem,
                    "wav_path": normalize_path(audio_map[stem]),
                    "rttm_path": normalize_path(textgrid_map[stem]),
                    "annotation_type": "textgrid",
                }
            )
        break

    return records


def sliding_window_starts(total_frames: int, chunk_frames: int, hop_frames: int) -> List[int]:
    if total_frames <= chunk_frames:
        return [0]

    starts = list(range(0, total_frames - chunk_frames + 1, hop_frames))
    tail_start = total_frames - chunk_frames
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def build_chunk_target_from_rttm(
    segments: List[Dict[str, Any]],
    chunk_start_sec: float,
    frame_count: int,
    max_speakers: int,
    skip_exceeding: bool,
) -> Tuple[Optional[torch.Tensor], List[str]]:
    chunk_end_sec = chunk_start_sec + frame_count / 100.0
    speaker_stats: Dict[str, Dict[str, float]] = {}

    for seg in segments:
        ov_start = max(chunk_start_sec, float(seg["start"]))
        ov_end = min(chunk_end_sec, float(seg["end"]))
        if ov_end <= ov_start:
            continue
        spk = str(seg["speaker"])
        info = speaker_stats.setdefault(spk, {"dur": 0.0, "first": ov_start})
        info["dur"] += ov_end - ov_start
        info["first"] = min(info["first"], ov_start)

    if not speaker_stats:
        return torch.zeros(frame_count, max_speakers, dtype=torch.float32), []

    ordered_speakers = [
        spk
        for spk, _ in sorted(
            speaker_stats.items(),
            key=lambda kv: (kv[1]["first"], -kv[1]["dur"], kv[0]),
        )
    ]

    if len(ordered_speakers) > max_speakers and skip_exceeding:
        return None, ordered_speakers

    speaker_names = ordered_speakers[:max_speakers]
    target_matrix = torch.zeros(frame_count, max_speakers, dtype=torch.float32)

    for slot, speaker_name in enumerate(speaker_names):
        for seg in segments:
            if str(seg["speaker"]) != speaker_name:
                continue
            ov_start = max(chunk_start_sec, float(seg["start"]))
            ov_end = min(chunk_end_sec, float(seg["end"]))
            if ov_end <= ov_start:
                continue
            f0 = max(0, int(math.floor((ov_start - chunk_start_sec) * 100.0)))
            f1 = min(frame_count, int(math.ceil((ov_end - chunk_start_sec) * 100.0)))
            if f1 > f0:
                target_matrix[f0:f1, slot] = 1.0

    return target_matrix, speaker_names


def finalize_pack(fbank: torch.Tensor, target_matrix: torch.Tensor, **extra: Any) -> Dict[str, Any]:
    # Clone to break any view/storage sharing before serialization.
    fbank = fbank.detach().clone().contiguous().float().cpu()
    target_matrix = target_matrix.detach().clone().contiguous().float().cpu()
    target_activity = (target_matrix.sum(dim=-1) > 0).float()
    target_count = int((target_matrix.sum(dim=0) > 0).sum().item())
    pack = {
        "fbank": fbank,
        "spk_label": -1,
        "target_matrix": target_matrix,
        "target_activity": target_activity,
        "target_count": target_count,
    }
    pack.update(extra)
    return pack


def atomic_torch_save(obj: Any, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    with path.open("wb") as f:
        f.write(buffer.getbuffer())
        f.flush()


@dataclass
class BuildCfg:
    seed: int = 1234
    incremental: bool = True
    backup_manifest: bool = True
    backup_dir_name: str = "_backups"
    backup_tag: str = ""

    target_sr: int = 16000
    n_mels: int = 80

    aishell4_root: str = str(AISHELL4_ROOT)
    voxconverse_root: str = str(VOXCONVERSE_ROOT)
    real_out_dir: str = str(OUTPUT_ROOT)

    real_chunk_sec: float = 4.0
    real_hop_sec: float = 2.0
    real_max_mix: int = 8
    real_val_ratio: float = 0.10
    real_min_active_sec: float = 0.05
    real_keep_empty_prob_train: float = 0.30
    real_keep_empty_prob_val: float = 1.0
    skip_chunks_exceeding_max_mix: bool = True


def save_meta(cfg: BuildCfg, output_root: Path):
    meta = {
        "schema_version": "real_diar_v1",
        "sources": ["aishell4_near"],
        "train_source_splits": {
            "aishell4": AISHELL4_TRAIN_SOURCE_SPLITS,
        },
        "target_sr": cfg.target_sr,
        "n_mels": cfg.n_mels,
        "chunk_sec": cfg.real_chunk_sec,
        "hop_sec": cfg.real_hop_sec,
        "max_mix": cfg.real_max_mix,
        "val_ratio": cfg.real_val_ratio,
        "skip_chunks_exceeding_max_mix": cfg.skip_chunks_exceeding_max_mix,
    }
    save_json(output_root / "dataset_meta.json", meta)


def split_train_val_records(
    records: List[Dict[str, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not records:
        return [], []

    if val_ratio <= 0.0:
        return list(records), []

    by_dataset: Dict[str, List[Dict[str, str]]] = {}
    for record in records:
        by_dataset.setdefault(str(record["dataset"]), []).append(record)

    train_records: List[Dict[str, str]] = []
    val_records: List[Dict[str, str]] = []

    for dataset_name, dataset_records in by_dataset.items():
        dataset_records = sorted(dataset_records, key=lambda x: (x["split"], x["recording_id"]))
        if len(dataset_records) == 1:
            train_records.extend(dataset_records)
            continue

        rng = random.Random(seed + sum(ord(ch) for ch in dataset_name))
        shuffled = list(dataset_records)
        rng.shuffle(shuffled)

        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        val_count = min(val_count, len(shuffled) - 1)

        val_records.extend(shuffled[:val_count])
        train_records.extend(shuffled[val_count:])

    train_records.sort(key=lambda x: (x["dataset"], x["split"], x["recording_id"]))
    val_records.sort(key=lambda x: (x["dataset"], x["split"], x["recording_id"]))
    return train_records, val_records


def build_real_diar_split(
    split_name: str,
    records: List[Dict[str, str]],
    cfg: BuildCfg,
    output_root: Path,
    keep_empty_prob: float,
):
    ensure_dir(output_root)
    manifest_path = output_root / f"{split_name}_manifest.jsonl"
    spk2id_path = output_root / "spk2id.json"

    existing_items = load_jsonl(manifest_path) if cfg.incremental and manifest_path.is_file() else []
    existing_relpts = {normalize_path(item["pt"]) for item in existing_items}
    manifest_items = list(existing_items) if cfg.incremental else []

    spk2id = load_json(spk2id_path) if spk2id_path.is_file() else {}
    next_spk_id = max(spk2id.values(), default=-1) + 1

    chunk_frames = int(round(cfg.real_chunk_sec * 100.0))
    hop_frames = max(1, int(round(cfg.real_hop_sec * 100.0)))

    new_items: List[Dict[str, Any]] = []
    skipped_many_speakers = 0

    for record in tqdm(records, desc=f"Build {split_name}"):
        wav = load_wav_mono(record["wav_path"], target_sr=cfg.target_sr, trim_silence=False)
        feat = wav_to_fbank_infer(
            wav,
            n_mels=cfg.n_mels,
            crop_sec=None,
            crop_mode="none",
        ).float().cpu()
        annotation_type = str(record.get("annotation_type", "rttm")).lower()
        if annotation_type == "textgrid":
            segments = parse_textgrid(record["rttm_path"])
        else:
            segments = parse_rttm(record["rttm_path"])
        starts = sliding_window_starts(feat.size(0), chunk_frames=chunk_frames, hop_frames=hop_frames)

        for chunk_idx, start_frame in enumerate(starts):
            end_frame = min(feat.size(0), start_frame + chunk_frames)
            chunk_feat = feat[start_frame:end_frame]
            frame_count = int(chunk_feat.size(0))
            if frame_count <= 0:
                continue

            chunk_start_sec = start_frame / 100.0
            chunk_end_sec = chunk_start_sec + frame_count / 100.0
            rel_pt = normalize_path(
                Path("packs") / split_name / f"{record['dataset']}__{record['recording_id']}__{chunk_idx:06d}.pt"
            )
            abs_pt = output_root / rel_pt

            if cfg.incremental and rel_pt in existing_relpts and abs_pt.is_file():
                continue

            target_matrix, speaker_names = build_chunk_target_from_rttm(
                segments=segments,
                chunk_start_sec=chunk_start_sec,
                frame_count=frame_count,
                max_speakers=cfg.real_max_mix,
                skip_exceeding=cfg.skip_chunks_exceeding_max_mix,
            )
            if target_matrix is None:
                skipped_many_speakers += 1
                continue

            active_sec = float((target_matrix.sum(dim=-1) > 0).float().sum().item()) / 100.0
            if active_sec < cfg.real_min_active_sec and random.random() > keep_empty_prob:
                continue

            speaker_global_ids: List[int] = []
            for speaker_name in speaker_names:
                global_name = f"{record['dataset']}::{record['recording_id']}::{speaker_name}"
                if global_name not in spk2id:
                    spk2id[global_name] = next_spk_id
                    next_spk_id += 1
                speaker_global_ids.append(int(spk2id[global_name]))

            pack = finalize_pack(
                fbank=chunk_feat,
                target_matrix=target_matrix,
                dataset=record["dataset"],
                split=record["split"],
                recording_id=record["recording_id"],
                wav_path=record["wav_path"],
                rttm_path=record["rttm_path"],
                annotation_type=annotation_type,
                chunk_index=int(chunk_idx),
                chunk_start_sec=float(chunk_start_sec),
                chunk_end_sec=float(chunk_end_sec),
                speaker_names=speaker_names,
                speaker_global_ids=speaker_global_ids,
            )

            ensure_dir(abs_pt.parent)
            atomic_torch_save(pack, abs_pt)

            item = {
                "pt": rel_pt,
                "dataset": record["dataset"],
                "split": record["split"],
                "recording_id": record["recording_id"],
                "wav_path": record["wav_path"],
                "rttm_path": record["rttm_path"],
                "annotation_type": annotation_type,
                "chunk_start_sec": float(chunk_start_sec),
                "chunk_end_sec": float(chunk_end_sec),
                "target_count": int(pack["target_count"]),
            }
            manifest_items.append(item)
            new_items.append(item)

    if cfg.backup_manifest:
        backup_root = output_root / cfg.backup_dir_name / (cfg.backup_tag or datetime.now().strftime("%Y%m%d_%H%M%S"))
        backup_files([manifest_path, spk2id_path], backup_root=backup_root, relative_root=output_root)

    write_jsonl(manifest_path, dedupe_preserve_order(manifest_items))
    save_json(spk2id_path, spk2id)
    print(
        f"[REAL DIAR] {split_name}: added={len(new_items)} total={len(manifest_items)} "
        f"skipped_many_speakers={skipped_many_speakers}"
    )


def build_real_diar_dataset(cfg: BuildCfg):
    set_seed(cfg.seed)

    output_root = Path(cfg.real_out_dir)
    ensure_dir(output_root)
    save_meta(cfg, output_root)

    aishell_all = collect_aishell4_records(Path(cfg.aishell4_root), AISHELL4_TRAIN_SOURCE_SPLITS)

    aishell_train, aishell_val = split_train_val_records(
        aishell_all,
        val_ratio=cfg.real_val_ratio,
        seed=cfg.seed,
    )

    train_records = aishell_train
    val_records = aishell_val

    if not train_records:
        raise FileNotFoundError(
            "No AISHELL4 training records found. Check aishell4_root and Train_Ali_near layout."
        )

    build_real_diar_split(
        split_name="train",
        records=train_records,
        cfg=cfg,
        output_root=output_root,
        keep_empty_prob=cfg.real_keep_empty_prob_train,
    )

    if val_records:
        build_real_diar_split(
            split_name="val",
            records=val_records,
            cfg=cfg,
            output_root=output_root,
            keep_empty_prob=cfg.real_keep_empty_prob_val,
        )
    else:
        print("[REAL DIAR] No validation records found. Only train_manifest was generated.")


def main():
    cfg = BuildCfg()
    if not cfg.backup_tag:
        cfg.backup_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    build_real_diar_dataset(cfg)
    print("[ALL DONE]")


if __name__ == "__main__":
    main()
