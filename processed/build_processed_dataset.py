import os
import json
import math
import random
import shutil
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from speaker_verification.audio.features import wav_to_fbank, load_wav_mono

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path: str) -> List[dict]:
    items = []
    if not os.path.isfile(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(json.loads(s))
    return items


def write_jsonl(path: str, items: List[dict]):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def append_jsonl(path: str, items: List[dict]):
    if not items:
        return
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def dedupe_preserve_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    out = []
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def load_txt_pairs(path: str) -> List[Tuple[int, str]]:
    items: List[Tuple[int, str]] = []
    if not os.path.isfile(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            sp = s.split(maxsplit=1)
            if len(sp) != 2:
                continue
            items.append((int(sp[0]), sp[1]))
    return items


def save_txt_pairs(path: str, items: List[Tuple[int, str]]):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for spk_id, p in items:
            f.write(f"{spk_id} {p}\n")


def normalize_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def copy_to_backup(path: str | Path, backup_root: str | Path, relative_to: str | Path):
    src = Path(path)
    if not src.is_file():
        return
    rel = src.relative_to(relative_to)
    dst = Path(backup_root) / rel
    ensure_dir(dst.parent)
    if dst.exists():
        return
    shutil.copy2(src, dst)
    print(f"[Backup] {src} -> {dst}")


def backup_files(paths: Iterable[str | Path], output_root: str | Path, backup_dir_name: str, backup_tag: str):
    backup_root = Path(output_root) / backup_dir_name / backup_tag
    for path in paths:
        copy_to_backup(path, backup_root, output_root)


def next_available_index(out_dir: str | Path, prefix: str, suffix: str = ".pt") -> int:
    root = Path(out_dir)
    if not root.is_dir():
        return 0
    max_idx = -1
    for path in root.glob(f"{prefix}*{suffix}"):
        stem = path.stem
        parts = stem.split("_")
        for token in parts:
            if token.isdigit():
                max_idx = max(max_idx, int(token))
                break
    return max_idx + 1


def backup_metadata_files(cfg: "BuildCfg", output_root: str, paths: Iterable[str | Path]):
    if not cfg.backup_manifest:
        return
    if not cfg.backup_tag:
        cfg.backup_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_files(paths, output_root=output_root, backup_dir_name=cfg.backup_dir_name, backup_tag=cfg.backup_tag)


def ensure_static_mix_compatibility(cfg: "BuildCfg"):
    meta_path = Path(cfg.static_out_dir) / "mix_meta.json"
    if not cfg.incremental or not meta_path.is_file():
        return
    meta = load_json(str(meta_path))
    expected = {
        "label_mode": "framewise_speech_mask",
        "mix_domain": "linear_mel_energy",
    }
    mismatched = {
        key: {"existing": meta.get(key), "expected": value}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatched:
        raise RuntimeError(
            "Existing static mix directory was built with an older preprocessing schema. "
            f"Use a new static_out_dir or clean {cfg.static_out_dir} before incremental rebuild. "
            f"Mismatched meta: {mismatched}"
        )


def safe_key(path: str, root: str) -> str:
    rel = os.path.relpath(path, root).replace("\\", "/")
    return os.path.splitext(rel)[0].replace("/", "__")


def db_to_gain(db: float) -> float:
    return 10 ** (db / 20.0)


def db_to_power_gain(db: float) -> float:
    return 10 ** (db / 10.0)


def list_pt_files(root: str) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    out = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".pt"):
                out.append(os.path.join(r, f))
    return out


def smooth_curve_1d(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size <= 1 or x.numel() <= 2:
        return x
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    y = x.view(1, 1, -1)
    y = F.pad(y, (pad, pad), mode="replicate")
    weight = torch.ones(1, 1, kernel_size, dtype=x.dtype, device=x.device) / kernel_size
    return F.conv1d(y, weight).view(-1)


def remove_short_true_runs(mask: torch.Tensor, min_len: int) -> torch.Tensor:
    if mask.numel() == 0 or min_len <= 1:
        return mask
    out = mask.clone()
    t = 0
    total = out.numel()
    while t < total:
        if not bool(out[t].item()):
            t += 1
            continue
        end = t + 1
        while end < total and bool(out[end].item()):
            end += 1
        if end - t < min_len:
            out[t:end] = False
        t = end
    return out


def fill_short_false_runs(mask: torch.Tensor, max_gap: int) -> torch.Tensor:
    if mask.numel() == 0 or max_gap <= 0:
        return mask
    out = mask.clone()
    t = 0
    total = out.numel()
    while t < total:
        if bool(out[t].item()):
            t += 1
            continue
        end = t + 1
        while end < total and not bool(out[end].item()):
            end += 1
        left_active = t > 0 and bool(out[t - 1].item())
        right_active = end < total and bool(out[end].item())
        if left_active and right_active and (end - t) <= max_gap:
            out[t:end] = True
        t = end
    return out


def estimate_speech_mask_from_fbank(
    feat: torch.Tensor,
    threshold_ratio: float,
    smooth_frames: int,
    min_speech_frames: int,
    fill_gap_frames: int,
) -> torch.Tensor:
    if feat.dim() != 2:
        raise ValueError(f"Expected [T,F] fbank, got {tuple(feat.shape)}")
    if feat.size(0) == 0:
        return torch.zeros(0, dtype=torch.bool)

    energy = feat.float().mean(dim=-1)
    energy = smooth_curve_1d(energy, kernel_size=smooth_frames)
    lo = torch.quantile(energy, 0.10)
    hi = torch.quantile(energy, 0.95)
    span = float((hi - lo).item())
    if span < 1e-6:
        return torch.ones(feat.size(0), dtype=torch.bool)

    threshold = lo + float(threshold_ratio) * (hi - lo)
    mask = energy >= threshold
    mask = fill_short_false_runs(mask, max_gap=int(fill_gap_frames))
    mask = remove_short_true_runs(mask, min_len=int(min_speech_frames))
    if not bool(mask.any().item()):
        peak_index = int(torch.argmax(energy).item())
        mask[max(0, peak_index - 1): min(mask.numel(), peak_index + 2)] = True
    return mask


def align_mask_length(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    mask = mask.bool().view(-1)
    if mask.numel() == target_len:
        return mask
    if mask.numel() > target_len:
        return mask[:target_len]
    out = torch.zeros(target_len, dtype=torch.bool)
    out[: mask.numel()] = mask
    return out


def log_fbank_to_linear(feat: torch.Tensor) -> torch.Tensor:
    return torch.exp(feat.float())


def linear_fbank_to_log(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(feat.float(), min=eps))


def finalize_pack(fbank: torch.Tensor, target_matrix: torch.Tensor, spk_label: int, **extra: Any) -> Dict[str, Any]:
    fbank = fbank.float().cpu()
    target_matrix = target_matrix.float().cpu()
    target_activity = (target_matrix.sum(dim=-1) > 0).float()
    target_count = int((target_matrix.sum(dim=0) > 0).sum().item())
    pack = {
        "fbank": fbank,
        "spk_label": int(spk_label),
        "target_matrix": target_matrix,
        "target_activity": target_activity,
        "target_count": target_count,
    }
    pack.update(extra)
    return pack


def load_single_speaker_feat_pack(
    path: str,
    cfg: "BuildCfg",
    feat_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    norm_path = os.path.normpath(path)
    if feat_cache is not None and norm_path in feat_cache:
        return feat_cache[norm_path]

    obj = torch.load(norm_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        feat = obj["fbank"] if "fbank" in obj else next(v for v in obj.values() if torch.is_tensor(v))
        wav_path = obj.get("wav_path")
        speaker = obj.get("speaker")
    else:
        feat = obj
        wav_path = None
        speaker = None

    if not torch.is_tensor(feat):
        feat = torch.tensor(feat)
    feat = feat.float().cpu()
    if feat.dim() == 3 and feat.size(0) == 1:
        feat = feat[0]
    if feat.dim() != 2:
        raise ValueError(f"Unexpected feat shape {tuple(feat.shape)}: {path}")
    if feat.size(1) != cfg.n_mels and feat.size(0) == cfg.n_mels:
        feat = feat.transpose(0, 1)
    if feat.size(1) != cfg.n_mels:
        raise ValueError(f"Expected n_mels={cfg.n_mels}, got {tuple(feat.shape)}: {path}")

    speech_mask = obj.get("speech_mask") if isinstance(obj, dict) else None
    if speech_mask is None:
        speech_mask = estimate_speech_mask_from_fbank(
            feat,
            threshold_ratio=cfg.vad_threshold_ratio,
            smooth_frames=cfg.vad_smooth_frames,
            min_speech_frames=cfg.vad_min_speech_frames,
            fill_gap_frames=cfg.vad_fill_gap_frames,
        )
    else:
        if not torch.is_tensor(speech_mask):
            speech_mask = torch.tensor(speech_mask)
        speech_mask = align_mask_length(speech_mask.bool().cpu(), feat.size(0))

    record = {
        "fbank": feat,
        "speech_mask": speech_mask,
        "wav_path": wav_path,
        "speaker": speaker,
    }
    if feat_cache is not None and feat.size(0) <= 1600:
        feat_cache[norm_path] = record
    return record


def choose_segment_frames(
    total_frames: int,
    crop_frames: int,
    min_ratio: float,
    max_ratio: float,
) -> int:
    max_frames = min(total_frames, crop_frames)
    min_frames = max(8, int(round(crop_frames * min_ratio)))
    max_target = max(min_frames, int(round(crop_frames * max_ratio)))
    max_frames = min(max_frames, max_target)
    min_frames = min(min_frames, max_frames)
    return random.randint(min_frames, max_frames)


def sample_feature_segment(
    feat: torch.Tensor,
    speech_mask: torch.Tensor,
    crop_frames: int,
    min_ratio: float,
    max_ratio: float,
    min_speech_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_frames = feat.size(0)
    seg_frames = choose_segment_frames(total_frames, crop_frames, min_ratio=min_ratio, max_ratio=max_ratio)
    if total_frames <= seg_frames:
        return feat, align_mask_length(speech_mask, feat.size(0))

    best_start = 0
    best_active = -1
    for _ in range(10):
        start = random.randint(0, total_frames - seg_frames)
        candidate_mask = speech_mask[start:start + seg_frames]
        active = int(candidate_mask.sum().item())
        if active > best_active:
            best_start = start
            best_active = active
        if active >= min_speech_frames:
            best_start = start
            break
    end = best_start + seg_frames
    return feat[best_start:end], speech_mask[best_start:end]


def sample_destination_start(
    canvas_frames: int,
    seg_frames: int,
    allow_overlap: bool,
    max_offset_ratio: float,
    occupied_mask: Optional[torch.Tensor] = None,
) -> int:
    if seg_frames >= canvas_frames:
        return 0

    max_start = canvas_frames - seg_frames
    if occupied_mask is None or allow_overlap or not bool(occupied_mask.any().item()):
        center = max_start // 2
        radius = min(max_start, int(round(canvas_frames * max_offset_ratio)))
        lo = max(0, center - radius)
        hi = min(max_start, center + radius)
        if lo > hi:
            lo, hi = 0, max_start
        return random.randint(lo, hi)

    best_start = 0
    best_overlap = None
    for _ in range(16):
        start = random.randint(0, max_start)
        overlap = int(occupied_mask[start:start + seg_frames].sum().item())
        if best_overlap is None or overlap < best_overlap:
            best_start = start
            best_overlap = overlap
            if overlap == 0:
                break
    return best_start


# =========================================================
# Config
# =========================================================
@dataclass
class BuildCfg:
    # stage
    stage: str = "all"  # cn | augment_cn | mix | add_single | add_negative | all

    # common
    seed: int = 1234
    incremental: bool = True
    backup_manifest: bool = True
    backup_dir_name: str = "_backups"
    backup_tag: str = ""

    # CN-Celeb2 preprocess
    cn_root: str = "../CN-Celeb_flac"
    cn_out_dir: str = "../processed/cn_celeb2"
    target_sr: int = 16000
    n_mels: int = 80
    min_sec: float = 1.0
    min_speech_ratio_per_utt: float = 0.08
    val_utt_ratio: float = 0.1
    min_utts_per_spk: int = 2
    vad_threshold_ratio: float = 0.22
    vad_smooth_frames: int = 5
    vad_min_speech_frames: int = 4
    vad_fill_gap_frames: int = 6

    # static mix
    static_out_dir: str = "../processed/static_mix_cnceleb2"
    num_train_mixes: int = 10000
    num_val_mixes: int = 1000
    min_mix: int = 2
    max_mix: int = 4
    crop_sec: float = 4.0
    spk_snr_min: float = -5.0
    spk_snr_max: float = 5.0
    noise_fbank_pt_dir: str = ""
    noise_prob: float = 0.3
    noise_snr_min: float = -10.0
    noise_snr_max: float = 0.0
    allow_overlap: bool = True
    max_offset_ratio: float = 0.35
    mix_source_dur_min_ratio: float = 0.35
    mix_source_dur_max_ratio: float = 0.95
    mix_use_augmented_sources: bool = False

    # add single
    add_single_train: int = 3000
    add_single_val: int = 500
    add_single_subdir: str = "single_from_cnceleb2"
    skip_existing: bool = True
    single_source_dur_min_ratio: float = 0.35
    single_source_dur_max_ratio: float = 0.95
    single_max_pad_ratio: float = 0.40

    # add negative
    add_neg_train: int = 2000
    add_neg_val: int = 500
    add_neg_subdir: str = "neg_augmented"

    # augmentation
    num_aug_per_utt_train: int = 4
    num_aug_per_utt_val: int = 1
    aug_subdir: str = "fbank_pt_aug"
    augmentation_version: str = "v2"


# =========================================================
# Stage 1: preprocess CN-Celeb2 -> single-speaker fbank
# =========================================================
def preprocess_cn_celeb2(cfg: BuildCfg):
    set_seed(cfg.seed)

    cn_out_dir = Path(cfg.cn_out_dir)
    ensure_dir(cn_out_dir)
    feat_dir = cn_out_dir / "fbank_pt"
    ensure_dir(feat_dir)

    data_dir = Path(cfg.cn_root) / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    spk2files = defaultdict(list)
    for spk in sorted(os.listdir(data_dir)):
        spk_path = data_dir / spk
        if not spk_path.is_dir():
            continue
        for fn in sorted(os.listdir(spk_path)):
            if fn.lower().endswith((".flac", ".wav")):
                spk2files[spk].append(str(spk_path / fn))

    kept_spks = sorted(spk for spk, fs in spk2files.items() if len(fs) >= cfg.min_utts_per_spk)

    train_list_path = cn_out_dir / "train_fbank_list.txt"
    val_list_path = cn_out_dir / "val_fbank_list.txt"
    spk_train_path = cn_out_dir / "spk_to_utterances_train.json"
    spk_val_path = cn_out_dir / "spk_to_utterances_val.json"
    spk2id_path = cn_out_dir / "spk2id.json"
    val_meta_path = cn_out_dir / "val_meta.jsonl"

    if spk2id_path.is_file():
        spk2id = load_json(str(spk2id_path))
    else:
        spk2id = {}
    next_spk_id = max(spk2id.values(), default=-1) + 1
    for spk in kept_spks:
        if spk not in spk2id:
            spk2id[spk] = next_spk_id
            next_spk_id += 1

    existing_train_items = load_txt_pairs(str(train_list_path))
    existing_val_items = load_txt_pairs(str(val_list_path))
    spk_to_utters_train = defaultdict(list, load_json(str(spk_train_path)) if spk_train_path.is_file() else {})
    spk_to_utters_val = defaultdict(list, load_json(str(spk_val_path)) if spk_val_path.is_file() else {})
    val_meta = load_jsonl(str(val_meta_path))

    train_relpaths = {normalize_path(path) for _, path in existing_train_items}
    val_relpaths = {normalize_path(path) for _, path in existing_val_items}
    existing_feat_paths = train_relpaths | val_relpaths
    existing_val_meta_keys = {(item.get("speaker"), normalize_path(item.get("feat_path", ""))) for item in val_meta}

    train_items = list(existing_train_items)
    val_items = list(existing_val_items)

    new_train = 0
    new_val = 0
    skipped_existing = 0

    for spk in tqdm(kept_spks, desc="Preprocess CN-Celeb2"):
        files = sorted(spk2files[spk])
        train_count = len(spk_to_utters_train.get(spk, []))
        val_count = len(spk_to_utters_val.get(spk, []))

        for wav_path in files:
            key = safe_key(wav_path, cfg.cn_root)
            feat_path = normalize_path(feat_dir / f"{spk}__{key}.pt")
            if cfg.incremental and feat_path in existing_feat_paths and Path(feat_path).is_file():
                skipped_existing += 1
                continue

            try:
                wav = load_wav_mono(wav_path, target_sr=cfg.target_sr)
                if len(wav) < int(cfg.target_sr * cfg.min_sec):
                    continue

                feat = wav_to_fbank(
                    wav,
                    n_mels=cfg.n_mels,
                    num_crops=1,
                    crop_sec=max(cfg.min_sec, float(len(wav)) / cfg.target_sr),
                )[0]
                speech_mask = estimate_speech_mask_from_fbank(
                    feat,
                    threshold_ratio=cfg.vad_threshold_ratio,
                    smooth_frames=cfg.vad_smooth_frames,
                    min_speech_frames=cfg.vad_min_speech_frames,
                    fill_gap_frames=cfg.vad_fill_gap_frames,
                )
                speech_ratio = float(speech_mask.float().mean().item())
                if speech_ratio < cfg.min_speech_ratio_per_utt:
                    continue

                torch.save(
                    {
                        "fbank": feat,
                        "speech_mask": speech_mask,
                        "speech_ratio": speech_ratio,
                        "speaker": spk,
                        "spk_id": int(spk2id[spk]),
                        "wav_path": normalize_path(wav_path),
                    },
                    feat_path,
                )

                label = int(spk2id[spk])
                assign_to_val = train_count > 0 and val_count < max(1, int((train_count + val_count + 1) * cfg.val_utt_ratio))
                if assign_to_val:
                    spk_to_utters_val[spk].append(feat_path)
                    val_items.append((label, feat_path))
                    meta_item = {"speaker": spk, "feat_path": feat_path, "spk_id": label}
                    if (spk, feat_path) not in existing_val_meta_keys:
                        val_meta.append(meta_item)
                        existing_val_meta_keys.add((spk, feat_path))
                    val_count += 1
                    new_val += 1
                else:
                    spk_to_utters_train[spk].append(feat_path)
                    train_items.append((label, feat_path))
                    train_count += 1
                    new_train += 1

                existing_feat_paths.add(feat_path)
            except Exception as e:
                print(f"[WARN] skip {wav_path}: {e}")

    train_items = dedupe_preserve_order(train_items)
    val_items = dedupe_preserve_order(val_items)
    spk_to_utters_train = {spk: dedupe_preserve_order(paths) for spk, paths in spk_to_utters_train.items()}
    spk_to_utters_val = {spk: dedupe_preserve_order(paths) for spk, paths in spk_to_utters_val.items()}
    val_meta = dedupe_preserve_order(val_meta)

    backup_metadata_files(
        cfg,
        output_root=str(cn_out_dir),
        paths=[train_list_path, val_list_path, spk_train_path, spk_val_path, spk2id_path, val_meta_path],
    )
    save_txt_pairs(str(train_list_path), train_items)
    save_txt_pairs(str(val_list_path), val_items)
    save_json(str(spk_train_path), spk_to_utters_train)
    save_json(str(spk_val_path), spk_to_utters_val)
    save_json(str(spk2id_path), spk2id)
    write_jsonl(str(val_meta_path), val_meta)

    print(
        f"[CN DONE] speakers={len(spk2id)} new_train={new_train} "
        f"new_val={new_val} skipped_existing={skipped_existing} out_dir={cfg.cn_out_dir}"
    )


# =========================================================
# Stage 1.5: augment single-speaker features
# =========================================================
def time_mask(feat: torch.Tensor, max_width: int = 20) -> torch.Tensor:
    x = feat.clone()
    T = x.size(0)
    if T <= 4:
        return x
    w = random.randint(0, min(max_width, max(1, T // 8)))
    if w <= 0:
        return x
    s = random.randint(0, max(0, T - w))
    x[s:s + w] = 0.0
    return x


def freq_mask(feat: torch.Tensor, max_width: int = 8) -> torch.Tensor:
    x = feat.clone()
    fdim = x.size(1)
    if fdim <= 4:
        return x
    w = random.randint(0, min(max_width, max(1, fdim // 6)))
    if w <= 0:
        return x
    s = random.randint(0, max(0, fdim - w))
    x[:, s:s + w] = 0.0
    return x


def global_gain(feat: torch.Tensor, scale_min: float = 0.85, scale_max: float = 1.18) -> torch.Tensor:
    return feat * random.uniform(scale_min, scale_max)


def add_gaussian_noise(feat: torch.Tensor, sigma_min: float = 0.002, sigma_max: float = 0.012) -> torch.Tensor:
    return feat + torch.randn_like(feat) * random.uniform(sigma_min, sigma_max)


def temporal_warp(feat: torch.Tensor, rate_min: float = 0.90, rate_max: float = 1.10) -> torch.Tensor:
    x = feat.transpose(0, 1).unsqueeze(0)
    t = x.size(-1)
    rate = random.uniform(rate_min, rate_max)
    warped_t = max(8, int(t * rate))
    y = F.interpolate(x, size=warped_t, mode="linear", align_corners=False)
    z = F.interpolate(y, size=t, mode="linear", align_corners=False)
    return z.squeeze(0).transpose(0, 1).contiguous()


def smooth_perturb(feat: torch.Tensor) -> torch.Tensor:
    x = feat.transpose(0, 1).unsqueeze(0)
    kernel = random.choice([3, 5])
    pad = kernel // 2
    weight = torch.ones(x.size(1), 1, kernel, dtype=x.dtype, device=x.device) / kernel
    y = F.conv1d(F.pad(x, (pad, pad), mode="replicate"), weight, groups=x.size(1))
    y = 0.72 * x + 0.28 * y
    return y.squeeze(0).transpose(0, 1).contiguous()


def spectral_tilt(feat: torch.Tensor, tilt_db_min: float = -5.0, tilt_db_max: float = 5.0) -> torch.Tensor:
    x = feat.clone()
    fdim = x.size(1)
    db_span = random.uniform(tilt_db_min, tilt_db_max)
    ramp = torch.linspace(-0.5, 0.5, fdim, dtype=x.dtype, device=x.device)
    gains = torch.pow(10.0, (ramp * db_span) / 20.0)
    return x * gains.unsqueeze(0)


def formant_shift(feat: torch.Tensor, shift_min: float = 0.94, shift_max: float = 1.08) -> torch.Tensor:
    x = feat.transpose(0, 1).unsqueeze(0).unsqueeze(0)
    fdim, tdim = x.size(2), x.size(3)
    shifted_bins = max(8, int(fdim * random.uniform(shift_min, shift_max)))
    y = F.interpolate(x, size=(shifted_bins, tdim), mode="bilinear", align_corners=False)
    z = F.interpolate(y, size=(fdim, tdim), mode="bilinear", align_corners=False)
    return z.squeeze(0).squeeze(0).transpose(0, 1).contiguous()


def energy_contour_perturb(feat: torch.Tensor, min_points: int = 4, max_points: int = 8) -> torch.Tensor:
    x = feat.clone()
    tdim = x.size(0)
    num_points = random.randint(min_points, max_points)
    control = torch.empty(num_points, dtype=x.dtype, device=x.device).uniform_(0.82, 1.20)
    envelope = F.interpolate(control.view(1, 1, -1), size=tdim, mode="linear", align_corners=False).view(-1)
    return x * envelope.unsqueeze(1)


def spectral_contrast_perturb(feat: torch.Tensor, factor_min: float = 0.82, factor_max: float = 1.18) -> torch.Tensor:
    x = feat.clone()
    mean = x.mean(dim=1, keepdim=True)
    factor = random.uniform(factor_min, factor_max)
    return mean + (x - mean) * factor


def normalize_feat(feat: torch.Tensor) -> torch.Tensor:
    return (feat - feat.mean()) / (feat.std() + 1e-5)


def apply_profile(feat: torch.Tensor, profile: str) -> torch.Tensor:
    x = feat
    if profile == "emotion_excited":
        x = temporal_warp(x, 1.02, 1.10)
        x = energy_contour_perturb(x)
        x = spectral_contrast_perturb(x, 1.02, 1.20)
    elif profile == "emotion_calm":
        x = temporal_warp(x, 0.92, 1.00)
        x = smooth_perturb(x)
        x = spectral_contrast_perturb(x, 0.82, 0.98)
    elif profile == "timbre_bright":
        x = spectral_tilt(x, 1.5, 5.5)
        x = formant_shift(x, 1.00, 1.08)
    elif profile == "timbre_dark":
        x = spectral_tilt(x, -5.5, -1.5)
        x = formant_shift(x, 0.94, 1.00)
    elif profile == "tone_soft":
        x = smooth_perturb(x)
        x = global_gain(x, 0.88, 1.02)
    elif profile == "tone_tense":
        x = spectral_contrast_perturb(x, 1.05, 1.18)
        x = add_gaussian_noise(x, 0.0015, 0.006)
    return x


def augment_fbank(feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    x = feat.clone().float()
    profile = random.choice(
        [
            "emotion_excited",
            "emotion_calm",
            "timbre_bright",
            "timbre_dark",
            "tone_soft",
            "tone_tense",
        ]
    )
    x = apply_profile(x, profile)

    ops = []
    if random.random() < 0.90:
        ops.append(global_gain)
    if random.random() < 0.75:
        ops.append(add_gaussian_noise)
    if random.random() < 0.60:
        ops.append(time_mask)
    if random.random() < 0.60:
        ops.append(freq_mask)
    if random.random() < 0.45:
        ops.append(energy_contour_perturb)
    if random.random() < 0.35:
        ops.append(spectral_tilt)

    random.shuffle(ops)
    applied_ops = [profile]
    for op in ops:
        x = op(x)
        applied_ops.append(op.__name__)

    x = normalize_feat(x)
    return x, {"profile": profile, "ops": applied_ops}


def build_augmented_cn_lists(cfg: BuildCfg, split: str, num_aug_per_utt: int):
    list_path = Path(cfg.cn_out_dir) / f"{split}_fbank_list.txt"
    items = load_txt_pairs(str(list_path))
    if not items:
        raise FileNotFoundError(list_path)

    aug_root = Path(cfg.cn_out_dir) / cfg.aug_subdir / split
    ensure_dir(aug_root)

    aug_list_path = Path(cfg.cn_out_dir) / f"{split}_fbank_list_aug.txt"
    aug_map_path = Path(cfg.cn_out_dir) / f"spk_to_utterances_{split}_aug.json"
    existing_items = load_txt_pairs(str(aug_list_path))
    out_items = list(existing_items) if cfg.incremental else []
    existing_paths = {normalize_path(path) for _, path in out_items}

    new_augments = 0
    for spk_id, pt_path in tqdm(items, desc=f"Augment {split} singles"):
        base_path = normalize_path(pt_path)
        if base_path not in existing_paths:
            out_items.append((spk_id, base_path))
            existing_paths.add(base_path)

        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
        feat = obj["fbank"].float()
        stem = Path(pt_path).stem

        for k in range(num_aug_per_utt):
            out_path = normalize_path(aug_root / f"{stem}__{cfg.augmentation_version}_aug{k}.pt")
            if cfg.incremental and out_path in existing_paths and Path(out_path).is_file():
                continue

            aug_feat, aug_meta = augment_fbank(feat)
            aug_speech_mask = estimate_speech_mask_from_fbank(
                aug_feat,
                threshold_ratio=cfg.vad_threshold_ratio,
                smooth_frames=cfg.vad_smooth_frames,
                min_speech_frames=cfg.vad_min_speech_frames,
                fill_gap_frames=cfg.vad_fill_gap_frames,
            )
            aug_obj = dict(obj)
            aug_obj["fbank"] = aug_feat
            aug_obj["speech_mask"] = aug_speech_mask
            aug_obj["speech_ratio"] = float(aug_speech_mask.float().mean().item())
            aug_obj["source"] = "augmented_cn_single"
            aug_obj["aug_index"] = k
            aug_obj["aug_version"] = cfg.augmentation_version
            aug_obj["aug_profile"] = aug_meta["profile"]
            aug_obj["aug_ops"] = aug_meta["ops"]

            torch.save(aug_obj, out_path)
            out_items.append((spk_id, out_path))
            existing_paths.add(out_path)
            new_augments += 1

    out_items = dedupe_preserve_order(out_items)
    spk2id = load_json(str(Path(cfg.cn_out_dir) / "spk2id.json"))
    id2spk = {int(v): k for k, v in spk2id.items()}
    spk_to_utts: Dict[str, List[str]] = defaultdict(list)

    for spk_id, path in out_items:
        spk = id2spk[int(spk_id)]
        spk_to_utts[spk].append(normalize_path(path))

    spk_to_utts = {spk: dedupe_preserve_order(paths) for spk, paths in spk_to_utts.items()}

    backup_metadata_files(cfg, output_root=cfg.cn_out_dir, paths=[aug_list_path, aug_map_path])
    save_txt_pairs(str(aug_list_path), out_items)
    save_json(str(aug_map_path), spk_to_utts)
    print(f"[AUG DONE] split={split} added={new_augments} total_items={len(out_items)}")


def build_augmented_cn(cfg: BuildCfg):
    set_seed(cfg.seed)
    build_augmented_cn_lists(cfg, "train", cfg.num_aug_per_utt_train)
    build_augmented_cn_lists(cfg, "val", cfg.num_aug_per_utt_val)


# =========================================================
# Stage 2: build static mixed-speaker dataset
# =========================================================
@torch.no_grad()
def load_feat_any(path: str, target_sr: int = 16000, n_mels: int = 80) -> torch.Tensor:
    lp = path.lower()

    if lp.endswith(".pt"):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            feat = obj["fbank"] if "fbank" in obj else next(v for v in obj.values() if torch.is_tensor(v))
        else:
            feat = obj
        if not torch.is_tensor(feat):
            feat = torch.tensor(feat)
        feat = feat.float().cpu()
        if feat.dim() == 3 and feat.size(0) == 1:
            feat = feat[0]
        if feat.dim() != 2:
            raise ValueError(f"Unexpected feat shape {tuple(feat.shape)}: {path}")
        if feat.size(1) != n_mels and feat.size(0) == n_mels:
            feat = feat.transpose(0, 1)
        if feat.size(1) != n_mels:
            raise ValueError(f"Expected n_mels={n_mels}, got {tuple(feat.shape)}: {path}")
        return feat

    wav = load_wav_mono(path, target_sr=target_sr)
    feat = wav_to_fbank(wav, n_mels=n_mels)[0]
    if not torch.is_tensor(feat):
        feat = torch.tensor(feat)
    return feat.float().cpu()


def generate_one_mix(
    speakers: List[str],
    spk_to_utters: Dict[str, List[str]],
    spk2id: Dict[str, int],
    cfg: BuildCfg,
    feat_cache: Dict[str, Dict[str, Any]],
    noise_pts: List[str],
):
    crop_frames = int(cfg.crop_sec * 100)
    k = random.randint(cfg.min_mix, cfg.max_mix)
    spks = random.sample(speakers, k)
    sv_spk = random.choice(spks)

    mixed_linear = torch.zeros(crop_frames, cfg.n_mels, dtype=torch.float32)
    target_matrix = torch.zeros(crop_frames, cfg.max_mix, dtype=torch.float32)
    occupied_mask = torch.zeros(crop_frames, dtype=torch.bool)

    def get_source(p: str) -> Dict[str, Any]:
        return load_single_speaker_feat_pack(p, cfg=cfg, feat_cache=feat_cache)

    for local_slot, spk in enumerate(spks):
        utt = random.choice(spk_to_utters[spk])
        source = get_source(utt)
        feat_seg, speech_mask_seg = sample_feature_segment(
            feat=source["fbank"],
            speech_mask=source["speech_mask"],
            crop_frames=crop_frames,
            min_ratio=cfg.mix_source_dur_min_ratio,
            max_ratio=cfg.mix_source_dur_max_ratio,
            min_speech_frames=cfg.vad_min_speech_frames,
        )
        seg_frames = int(feat_seg.size(0))
        if seg_frames <= 0:
            continue

        dst_s0 = sample_destination_start(
            canvas_frames=crop_frames,
            seg_frames=seg_frames,
            allow_overlap=cfg.allow_overlap,
            max_offset_ratio=cfg.max_offset_ratio,
            occupied_mask=occupied_mask,
        )
        dst_e0 = min(crop_frames, dst_s0 + seg_frames)
        seg_frames = dst_e0 - dst_s0
        if seg_frames <= 0:
            continue

        feat_seg = feat_seg[:seg_frames]
        speech_mask_seg = align_mask_length(speech_mask_seg, seg_frames)
        power_gain = db_to_power_gain(random.uniform(cfg.spk_snr_min, cfg.spk_snr_max))
        mixed_linear[dst_s0:dst_e0] += log_fbank_to_linear(feat_seg) * power_gain
        target_matrix[dst_s0:dst_e0, local_slot] = torch.maximum(
            target_matrix[dst_s0:dst_e0, local_slot],
            speech_mask_seg.float(),
        )
        occupied_mask[dst_s0:dst_e0] |= speech_mask_seg

    if noise_pts and random.random() < cfg.noise_prob:
        npt = random.choice(noise_pts)
        nfeat = load_feat_any(npt, target_sr=cfg.target_sr, n_mels=cfg.n_mels)
        if nfeat.size(0) > crop_frames:
            start = random.randint(0, nfeat.size(0) - crop_frames)
            nfeat = nfeat[start:start + crop_frames]
        elif nfeat.size(0) < crop_frames:
            nfeat = F.pad(nfeat, (0, 0, 0, crop_frames - nfeat.size(0)))
        noise_gain = db_to_power_gain(random.uniform(cfg.noise_snr_min, cfg.noise_snr_max))
        mixed_linear += log_fbank_to_linear(nfeat) * noise_gain

    mixed = linear_fbank_to_log(mixed_linear)
    return finalize_pack(
        fbank=mixed,
        target_matrix=target_matrix,
        spk_label=int(spk2id[sv_spk]),
        speaker_names=spks,
    )


def generate_mix_split(
    split_name: str,
    num_mixes: int,
    spk_to_utters: Dict[str, List[str]],
    spk2id: Dict[str, int],
    cfg: BuildCfg,
    noise_pts: List[str],
):
    speakers = sorted(spk_to_utters.keys())
    if len(speakers) < cfg.max_mix:
        raise ValueError(f"[{split_name}] speakers={len(speakers)} < max_mix={cfg.max_mix}")

    static_out_dir = Path(cfg.static_out_dir)
    ensure_dir(static_out_dir)
    mix_dir = static_out_dir / "mix_pt" / split_name
    ensure_dir(mix_dir)

    manifest_path = static_out_dir / f"{split_name}_manifest.jsonl"
    feat_cache: Dict[str, Dict[str, Any]] = {}
    existing_items = load_jsonl(str(manifest_path))
    existing_relpts = {normalize_path(item["pt"]) for item in existing_items if "pt" in item}
    start_idx = len(existing_items) if cfg.incremental else 0
    new_items = []

    for idx in tqdm(range(start_idx, start_idx + num_mixes), desc=f"Generate {split_name} mixes"):
        pack = generate_one_mix(
            speakers=speakers,
            spk_to_utters=spk_to_utters,
            spk2id=spk2id,
            cfg=cfg,
            feat_cache=feat_cache,
            noise_pts=noise_pts,
        )
        rel_pt = normalize_path(Path("mix_pt") / split_name / f"{idx:08d}.pt")
        if cfg.incremental and rel_pt in existing_relpts:
            continue
        abs_pt = static_out_dir / rel_pt
        ensure_dir(abs_pt.parent)
        torch.save(pack, abs_pt)
        new_items.append({"pt": rel_pt})

    if new_items:
        backup_metadata_files(cfg, output_root=cfg.static_out_dir, paths=[manifest_path])
        append_jsonl(str(manifest_path), new_items)

    print(f"[MIX DONE] {split_name}: added={len(new_items)} manifest={manifest_path}")


def build_static_mix(cfg: BuildCfg):
    set_seed(cfg.seed)
    ensure_static_mix_compatibility(cfg)

    train_map_path_aug = os.path.join(cfg.cn_out_dir, "spk_to_utterances_train_aug.json")
    val_map_path_aug = os.path.join(cfg.cn_out_dir, "spk_to_utterances_val_aug.json")

    if cfg.mix_use_augmented_sources and os.path.isfile(train_map_path_aug) and os.path.isfile(val_map_path_aug):
        train_map_path = train_map_path_aug
        val_map_path = val_map_path_aug
    else:
        train_map_path = os.path.join(cfg.cn_out_dir, "spk_to_utterances_train.json")
        val_map_path = os.path.join(cfg.cn_out_dir, "spk_to_utterances_val.json")
        if not os.path.isfile(train_map_path) and os.path.isfile(train_map_path_aug):
            train_map_path = train_map_path_aug
        if not os.path.isfile(val_map_path) and os.path.isfile(val_map_path_aug):
            val_map_path = val_map_path_aug

    spk2id_path = os.path.join(cfg.cn_out_dir, "spk2id.json")

    if not os.path.isfile(train_map_path):
        raise FileNotFoundError(train_map_path)
    if not os.path.isfile(val_map_path):
        raise FileNotFoundError(val_map_path)
    if not os.path.isfile(spk2id_path):
        raise FileNotFoundError(spk2id_path)

    spk_to_utters_train = load_json(train_map_path)
    spk_to_utters_val = load_json(val_map_path)
    spk2id = load_json(spk2id_path)

    ensure_dir(cfg.static_out_dir)
    spk2id_out = os.path.join(cfg.static_out_dir, "spk2id.json")
    mix_meta_path = os.path.join(cfg.static_out_dir, "mix_meta.json")
    backup_metadata_files(cfg, output_root=cfg.static_out_dir, paths=[spk2id_out, mix_meta_path])
    save_json(spk2id_out, spk2id)
    save_json(
        mix_meta_path,
        {
            "cn_out_dir": cfg.cn_out_dir,
            "static_out_dir": cfg.static_out_dir,
            "num_train_mixes": cfg.num_train_mixes,
            "num_val_mixes": cfg.num_val_mixes,
            "min_mix": cfg.min_mix,
            "max_mix": cfg.max_mix,
            "crop_sec": cfg.crop_sec,
            "crop_frames": int(cfg.crop_sec * 100),
            "spk_snr_min": cfg.spk_snr_min,
            "spk_snr_max": cfg.spk_snr_max,
            "noise_fbank_pt_dir": cfg.noise_fbank_pt_dir,
            "noise_prob": cfg.noise_prob,
            "noise_snr_min": cfg.noise_snr_min,
            "noise_snr_max": cfg.noise_snr_max,
            "allow_overlap": cfg.allow_overlap,
            "max_offset_ratio": cfg.max_offset_ratio,
            "mix_source_dur_min_ratio": cfg.mix_source_dur_min_ratio,
            "mix_source_dur_max_ratio": cfg.mix_source_dur_max_ratio,
            "mix_use_augmented_sources": cfg.mix_use_augmented_sources,
            "label_mode": "framewise_speech_mask",
            "mix_domain": "linear_mel_energy",
            "seed": cfg.seed,
            "incremental": cfg.incremental,
            "augmentation_version": cfg.augmentation_version,
        },
    )

    noise_pts = list_pt_files(cfg.noise_fbank_pt_dir)
    print(f"[StaticMix] train_spks={len(spk_to_utters_train)} val_spks={len(spk_to_utters_val)} noise_pts={len(noise_pts)}")

    generate_mix_split("train", cfg.num_train_mixes, spk_to_utters_train, spk2id, cfg, noise_pts)
    generate_mix_split("val", cfg.num_val_mixes, spk_to_utters_val, spk2id, cfg, noise_pts)


# =========================================================
# Stage 3: append single-speaker packs into static mix
# =========================================================
def infer_max_mix_from_manifest(static_mix_dir: str, manifest_name: str) -> int:
    manifest_path = os.path.join(static_mix_dir, manifest_name)
    items = load_jsonl(manifest_path)
    if not items:
        raise RuntimeError(f"Empty manifest: {manifest_path}")
    pack = torch.load(os.path.join(static_mix_dir, items[0]["pt"]), map_location="cpu", weights_only=False)
    target_matrix = pack["target_matrix"]
    if not torch.is_tensor(target_matrix):
        target_matrix = torch.tensor(target_matrix)
    return int(target_matrix.shape[1])


def collect_cn_single_items(cn_out_dir: str, split: str) -> List[Tuple[int, str]]:
    aug_list_path = os.path.join(cn_out_dir, f"{split}_fbank_list_aug.txt")
    list_path = aug_list_path if os.path.isfile(aug_list_path) else os.path.join(cn_out_dir, f"{split}_fbank_list.txt")
    if not os.path.isfile(list_path):
        raise FileNotFoundError(list_path)
    return load_txt_pairs(list_path)


def build_single_pack_from_cn_pt(cn_pt_path: str, spk_id: int, max_mix: int, cfg: BuildCfg):
    source = load_single_speaker_feat_pack(cn_pt_path, cfg=cfg)
    crop_frames = int(cfg.crop_sec * 100)
    feat_seg, speech_mask_seg = sample_feature_segment(
        feat=source["fbank"],
        speech_mask=source["speech_mask"],
        crop_frames=crop_frames,
        min_ratio=cfg.single_source_dur_min_ratio,
        max_ratio=cfg.single_source_dur_max_ratio,
        min_speech_frames=cfg.vad_min_speech_frames,
    )

    canvas = torch.zeros(crop_frames, cfg.n_mels, dtype=torch.float32)
    target_matrix = torch.zeros(crop_frames, max_mix, dtype=torch.float32)
    start = sample_destination_start(
        canvas_frames=crop_frames,
        seg_frames=int(feat_seg.size(0)),
        allow_overlap=True,
        max_offset_ratio=cfg.single_max_pad_ratio,
        occupied_mask=None,
    )
    end = min(crop_frames, start + int(feat_seg.size(0)))
    seg_len = end - start
    if seg_len > 0:
        canvas[start:end] = feat_seg[:seg_len]
        target_matrix[start:end, 0] = align_mask_length(speech_mask_seg, seg_len).float()

    return finalize_pack(
        fbank=canvas,
        target_matrix=target_matrix,
        spk_label=int(spk_id),
        source="cn_celeb2_single",
        speaker_name=source.get("speaker"),
        wav_path=source.get("wav_path"),
        source_pt=normalize_path(cn_pt_path),
    )


def get_existing_relpts(static_mix_dir: str, manifest_name: str) -> set:
    return {x["pt"].replace("\\", "/") for x in load_jsonl(os.path.join(static_mix_dir, manifest_name)) if "pt" in x}


def append_single_samples(cfg: BuildCfg):
    set_seed(cfg.seed)
    ensure_static_mix_compatibility(cfg)

    train_manifest = Path(cfg.static_out_dir) / "train_manifest.jsonl"
    val_manifest = Path(cfg.static_out_dir) / "val_manifest.jsonl"
    if not train_manifest.is_file():
        raise FileNotFoundError(train_manifest)
    if not val_manifest.is_file():
        raise FileNotFoundError(val_manifest)

    max_mix = infer_max_mix_from_manifest(cfg.static_out_dir, "train_manifest.jsonl")
    backup_metadata_files(cfg, output_root=cfg.static_out_dir, paths=[train_manifest, val_manifest])

    for split, n_add in [("train", cfg.add_single_train), ("val", cfg.add_single_val)]:
        items = collect_cn_single_items(cfg.cn_out_dir, split)
        random.shuffle(items)
        items = items[: min(n_add, len(items))]

        existing = get_existing_relpts(cfg.static_out_dir, f"{split}_manifest.jsonl")
        appended = []
        out_dir = Path(cfg.static_out_dir) / cfg.add_single_subdir / split
        ensure_dir(out_dir)
        start_idx = next_available_index(out_dir, prefix="single_")

        for idx, (spk_id, cn_pt_path) in enumerate(items):
            try:
                base = Path(cn_pt_path).stem
                rel_pt = normalize_path(Path(cfg.add_single_subdir) / split / f"single_{start_idx + idx:06d}_{base}.pt")
                if cfg.skip_existing and rel_pt in existing:
                    continue

                pack = build_single_pack_from_cn_pt(cn_pt_path, spk_id, max_mix, cfg)
                abs_pt = Path(cfg.static_out_dir) / rel_pt
                ensure_dir(abs_pt.parent)
                torch.save(pack, abs_pt)
                appended.append({"pt": rel_pt, "source": "cn_celeb2_single"})
            except Exception as e:
                print(f"[WARN] single skip {cn_pt_path}: {e}")

        append_jsonl(str(Path(cfg.static_out_dir) / f"{split}_manifest.jsonl"), appended)
        print(f"[ADD SINGLE] {split}: added={len(appended)}")


# =========================================================
# Stage 4: append synthetic negative packs
# =========================================================
NEG_TYPES = [
    "pure_silence",
    "low_white_noise",
    "pink_noise",
    "brown_noise",
    "hum_50hz",
    "hum_100hz",
    "fan_like_noise",
    "impulse_clicks",
    "mixed_noise",
]
NEG_WEIGHTS = [0.18, 0.14, 0.10, 0.08, 0.10, 0.08, 0.14, 0.06, 0.12]
NEG_DURATIONS = [2.0, 3.0, 4.0, 5.0, 6.0]


def apply_fade(wav: torch.Tensor, sr: int, fade_ms: float = 20.0):
    fade_len = max(1, int(sr * fade_ms / 1000))
    fade = torch.linspace(0.0, 1.0, fade_len, dtype=wav.dtype)
    if wav.numel() >= 2 * fade_len:
        wav[:fade_len] *= fade
        wav[-fade_len:] *= fade.flip(0)
    return wav


def normalize_peak(wav: torch.Tensor, peak: float = 0.8):
    m = wav.abs().max().item()
    return wav if m < 1e-8 else wav / m * peak


def gen_white_noise(sec, sr, scale=0.01):
    return torch.randn(int(sec * sr), dtype=torch.float32) * scale


def gen_pure_silence(sec, sr):
    return torch.zeros(int(sec * sr), dtype=torch.float32)


def gen_pink_noise(sec, sr, scale=0.01):
    n = int(sec * sr)
    x = torch.randn(n, dtype=torch.float32)
    X = torch.fft.rfft(x)
    freqs = torch.linspace(1.0, X.shape[0], X.shape[0], dtype=torch.float32)
    X = X / torch.sqrt(freqs)
    y = torch.fft.irfft(X, n=n).float()
    return y / (y.std() + 1e-6) * scale


def gen_brown_noise(sec, sr, scale=0.01):
    n = int(sec * sr)
    x = torch.randn(n, dtype=torch.float32)
    y = torch.cumsum(x, dim=0)
    y = y - y.mean()
    return y / (y.std() + 1e-6) * scale


def gen_hum(sec, sr, freq=50.0, amp=0.02):
    n = int(sec * sr)
    t = torch.arange(n, dtype=torch.float32) / sr
    y = amp * torch.sin(2 * math.pi * freq * t)
    y += 0.4 * amp * torch.sin(2 * math.pi * freq * 2 * t)
    y += 0.2 * amp * torch.sin(2 * math.pi * freq * 3 * t)
    return y


def gen_fan_like_noise(sec, sr):
    n = int(sec * sr)
    t = torch.arange(n, dtype=torch.float32) / sr
    base = gen_hum(sec, sr, freq=random.choice([45.0, 50.0, 60.0, 90.0]), amp=0.01)
    white = gen_white_noise(sec, sr, scale=0.005)
    lfo_freq = random.uniform(0.1, 0.4)
    env = 0.7 + 0.3 * torch.sin(2 * math.pi * lfo_freq * t + random.uniform(0, 2 * math.pi))
    return (base + white) * env


def gen_impulse_clicks(sec, sr):
    n = int(sec * sr)
    y = gen_white_noise(sec, sr, scale=0.002)
    for _ in range(random.randint(3, 15)):
        pos = random.randint(0, n - 1)
        width = random.randint(8, 80)
        amp = random.uniform(0.02, 0.08)
        end = min(n, pos + width)
        y[pos:end] += torch.hann_window(end - pos, periodic=False) * amp
    return y


def gen_mixed_noise(sec, sr):
    parts = []
    if random.random() < 0.7:
        parts.append(gen_white_noise(sec, sr, scale=random.uniform(0.003, 0.01)))
    if random.random() < 0.6:
        parts.append(gen_pink_noise(sec, sr, scale=random.uniform(0.003, 0.01)))
    if random.random() < 0.6:
        parts.append(gen_hum(sec, sr, freq=random.choice([50.0, 60.0, 100.0, 120.0]), amp=random.uniform(0.005, 0.02)))
    if random.random() < 0.5:
        parts.append(gen_impulse_clicks(sec, sr) * random.uniform(0.2, 0.8))
    if not parts:
        parts = [gen_white_noise(sec, sr, scale=0.006)]
    return sum(parts)


def synth_negative_waveform(kind: str, sec: float, sr: int):
    if kind == "pure_silence":
        y = gen_pure_silence(sec, sr)
    elif kind == "low_white_noise":
        y = gen_white_noise(sec, sr, scale=random.uniform(0.002, 0.01))
    elif kind == "pink_noise":
        y = gen_pink_noise(sec, sr, scale=random.uniform(0.003, 0.012))
    elif kind == "brown_noise":
        y = gen_brown_noise(sec, sr, scale=random.uniform(0.003, 0.012))
    elif kind == "hum_50hz":
        y = gen_hum(sec, sr, freq=50.0, amp=random.uniform(0.005, 0.02))
    elif kind == "hum_100hz":
        y = gen_hum(sec, sr, freq=100.0, amp=random.uniform(0.005, 0.02))
    elif kind == "fan_like_noise":
        y = gen_fan_like_noise(sec, sr)
    elif kind == "impulse_clicks":
        y = gen_impulse_clicks(sec, sr)
    elif kind == "mixed_noise":
        y = gen_mixed_noise(sec, sr)
    else:
        raise ValueError(kind)

    y = apply_fade(y, sr, 20.0)
    return normalize_peak(y, peak=random.uniform(0.2, 0.8))


def build_negative_pack(fbank: torch.Tensor, max_mix: int):
    T = int(fbank.shape[0])
    return finalize_pack(
        fbank=fbank.float(),
        target_matrix=torch.zeros(T, max_mix, dtype=torch.float32),
        spk_label=-1,
    )


def add_negative_samples(cfg: BuildCfg):
    set_seed(cfg.seed)
    ensure_static_mix_compatibility(cfg)

    max_mix = infer_max_mix_from_manifest(cfg.static_out_dir, "train_manifest.jsonl")

    for split, n_add in [("train", cfg.add_neg_train), ("val", cfg.add_neg_val)]:
        out_dir = Path(cfg.static_out_dir) / cfg.add_neg_subdir / split
        ensure_dir(out_dir)

        manifest_path = Path(cfg.static_out_dir) / f"{split}_manifest.jsonl"
        backup_metadata_files(cfg, output_root=cfg.static_out_dir, paths=[manifest_path])
        new_items = []
        start_idx = next_available_index(out_dir, prefix="neg_")
        existing = get_existing_relpts(cfg.static_out_dir, f"{split}_manifest.jsonl")

        for i in range(start_idx, start_idx + n_add):
            sec = random.choice(NEG_DURATIONS)
            kind = random.choices(NEG_TYPES, weights=NEG_WEIGHTS, k=1)[0]

            wav = synth_negative_waveform(kind, sec, cfg.target_sr)
            fbank = wav_to_fbank(
                wav,
                n_mels=cfg.n_mels,
                num_crops=1,
                crop_sec=max(cfg.crop_sec, sec),
            )[0]
            pack = build_negative_pack(fbank, max_mix=max_mix)

            fname = f"neg_{kind}_{i:06d}.pt"
            abs_pt = out_dir / fname
            rel_pt = normalize_path(abs_pt.relative_to(cfg.static_out_dir))
            if cfg.skip_existing and rel_pt in existing:
                continue

            torch.save(pack, abs_pt)
            new_items.append(
                {
                    "pt": rel_pt,
                    "source": "synthetic_negative",
                    "noise_type": kind,
                    "duration_sec": sec,
                }
            )

            generated = i - start_idx + 1
            if generated % 200 == 0:
                print(f"[{split}] negative generated {generated}/{n_add}")

        append_jsonl(str(manifest_path), new_items)
        print(f"[ADD NEG] {split}: added={len(new_items)}")


# =========================================================
# CLI
# =========================================================
def parse_args():
    defaults = BuildCfg()
    parser = argparse.ArgumentParser(description="Unified preprocessing script for Speaker-Verification")

    parser.add_argument("--stage", type=str, default="all",
                        choices=["cn", "augment_cn", "mix", "add_single", "add_negative", "all"])

    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--incremental", dest="incremental", action="store_true")
    parser.add_argument("--no-incremental", dest="incremental", action="store_false")
    parser.set_defaults(incremental=defaults.incremental)
    parser.add_argument("--backup_manifest", dest="backup_manifest", action="store_true")
    parser.add_argument("--no-backup_manifest", dest="backup_manifest", action="store_false")
    parser.set_defaults(backup_manifest=defaults.backup_manifest)
    parser.add_argument("--backup_dir_name", type=str, default=defaults.backup_dir_name)

    parser.add_argument("--cn_root", type=str, default=defaults.cn_root)
    parser.add_argument("--cn_out_dir", type=str, default=defaults.cn_out_dir)
    parser.add_argument("--static_out_dir", type=str, default=defaults.static_out_dir)

    parser.add_argument("--target_sr", type=int, default=defaults.target_sr)
    parser.add_argument("--n_mels", type=int, default=defaults.n_mels)
    parser.add_argument("--min_sec", type=float, default=defaults.min_sec)
    parser.add_argument("--min_speech_ratio_per_utt", type=float, default=defaults.min_speech_ratio_per_utt)
    parser.add_argument("--val_utt_ratio", type=float, default=defaults.val_utt_ratio)
    parser.add_argument("--min_utts_per_spk", type=int, default=defaults.min_utts_per_spk)
    parser.add_argument("--vad_threshold_ratio", type=float, default=defaults.vad_threshold_ratio)
    parser.add_argument("--vad_smooth_frames", type=int, default=defaults.vad_smooth_frames)
    parser.add_argument("--vad_min_speech_frames", type=int, default=defaults.vad_min_speech_frames)
    parser.add_argument("--vad_fill_gap_frames", type=int, default=defaults.vad_fill_gap_frames)

    parser.add_argument("--num_train_mixes", type=int, default=defaults.num_train_mixes)
    parser.add_argument("--num_val_mixes", type=int, default=defaults.num_val_mixes)
    parser.add_argument("--min_mix", type=int, default=defaults.min_mix)
    parser.add_argument("--max_mix", type=int, default=defaults.max_mix)
    parser.add_argument("--crop_sec", type=float, default=defaults.crop_sec)
    parser.add_argument("--spk_snr_min", type=float, default=defaults.spk_snr_min)
    parser.add_argument("--spk_snr_max", type=float, default=defaults.spk_snr_max)
    parser.add_argument("--noise_fbank_pt_dir", type=str, default=defaults.noise_fbank_pt_dir)
    parser.add_argument("--noise_prob", type=float, default=defaults.noise_prob)
    parser.add_argument("--noise_snr_min", type=float, default=defaults.noise_snr_min)
    parser.add_argument("--noise_snr_max", type=float, default=defaults.noise_snr_max)
    parser.add_argument("--allow_overlap", dest="allow_overlap", action="store_true")
    parser.add_argument("--no-allow_overlap", dest="allow_overlap", action="store_false")
    parser.set_defaults(allow_overlap=defaults.allow_overlap)
    parser.add_argument("--max_offset_ratio", type=float, default=defaults.max_offset_ratio)
    parser.add_argument("--mix_source_dur_min_ratio", type=float, default=defaults.mix_source_dur_min_ratio)
    parser.add_argument("--mix_source_dur_max_ratio", type=float, default=defaults.mix_source_dur_max_ratio)
    parser.add_argument("--mix_use_augmented_sources", dest="mix_use_augmented_sources", action="store_true")
    parser.add_argument("--no-mix_use_augmented_sources", dest="mix_use_augmented_sources", action="store_false")
    parser.set_defaults(mix_use_augmented_sources=defaults.mix_use_augmented_sources)

    parser.add_argument("--add_single_train", type=int, default=defaults.add_single_train)
    parser.add_argument("--add_single_val", type=int, default=defaults.add_single_val)
    parser.add_argument("--add_neg_train", type=int, default=defaults.add_neg_train)
    parser.add_argument("--add_neg_val", type=int, default=defaults.add_neg_val)
    parser.add_argument("--single_source_dur_min_ratio", type=float, default=defaults.single_source_dur_min_ratio)
    parser.add_argument("--single_source_dur_max_ratio", type=float, default=defaults.single_source_dur_max_ratio)
    parser.add_argument("--single_max_pad_ratio", type=float, default=defaults.single_max_pad_ratio)
    parser.add_argument("--num_aug_per_utt_train", type=int, default=defaults.num_aug_per_utt_train)
    parser.add_argument("--num_aug_per_utt_val", type=int, default=defaults.num_aug_per_utt_val)
    parser.add_argument("--aug_subdir", type=str, default=defaults.aug_subdir)
    parser.add_argument("--augmentation_version", type=str, default=defaults.augmentation_version)

    args = parser.parse_args()
    return BuildCfg(**vars(args))


def main():
    cfg = parse_args()
    if not cfg.backup_tag:
        cfg.backup_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    if cfg.stage == "cn":
        preprocess_cn_celeb2(cfg)
    elif cfg.stage == "augment_cn":
        build_augmented_cn(cfg)
    elif cfg.stage == "mix":
        build_static_mix(cfg)
    elif cfg.stage == "add_single":
        append_single_samples(cfg)
    elif cfg.stage == "add_negative":
        add_negative_samples(cfg)
    elif cfg.stage == "all":
        preprocess_cn_celeb2(cfg)
        build_augmented_cn(cfg)
        build_static_mix(cfg)
        append_single_samples(cfg)
        add_negative_samples(cfg)
    else:
        raise ValueError(cfg.stage)

    print("[ALL DONE]")


if __name__ == "__main__":
    main()
