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

try:
    import torchaudio
except ImportError:
    torchaudio = None

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


def safe_key(path: str, root: str) -> str:
    rel = os.path.relpath(path, root).replace("\\", "/")
    return os.path.splitext(rel)[0].replace("/", "__")


def db_to_gain(db: float) -> float:
    return 10 ** (db / 20.0)


def list_pt_files(root: str) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    out = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".pt"):
                out.append(os.path.join(r, f))
    return out


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
    val_utt_ratio: float = 0.1
    min_utts_per_spk: int = 2

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

    # add single
    add_single_train: int = 3000
    add_single_val: int = 500
    add_single_subdir: str = "single_from_cnceleb2"
    skip_existing: bool = True

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

                torch.save(
                    {
                        "fbank": feat,
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
            aug_obj = dict(obj)
            aug_obj["fbank"] = aug_feat
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
    feat = wav_to_fbank(wav, n_mels=n_mels)
    if not torch.is_tensor(feat):
        feat = torch.tensor(feat)
    return feat.float().cpu()


def crop_or_pad_feat(x: torch.Tensor, crop_frames: int) -> torch.Tensor:
    T = x.size(0)
    if T >= crop_frames:
        s = random.randint(0, T - crop_frames)
        return x[s:s + crop_frames]
    return F.pad(x, (0, 0, 0, crop_frames - T))


def build_segment_with_offset(crop_frames: int, allow_overlap: bool, max_offset_ratio: float):
    max_shift = int(crop_frames * max_offset_ratio) if allow_overlap else 0
    offset = random.randint(-max_shift, max_shift) if max_shift > 0 else 0

    src_s0 = 0
    dst_s0 = offset
    if dst_s0 < 0:
        src_s0 = -dst_s0
        dst_s0 = 0

    length = crop_frames - dst_s0
    length = min(length, crop_frames - src_s0)
    return src_s0, dst_s0, length


def generate_one_mix(
    speakers: List[str],
    spk_to_utters: Dict[str, List[str]],
    spk2id: Dict[str, int],
    cfg: BuildCfg,
    feat_cache: Dict[str, torch.Tensor],
    noise_pts: List[str],
):
    crop_frames = int(cfg.crop_sec * 100)
    k = random.randint(cfg.min_mix, cfg.max_mix)
    spks = random.sample(speakers, k)
    sv_spk = random.choice(spks)

    mixed = torch.zeros(crop_frames, cfg.n_mels, dtype=torch.float32)
    target_matrix = torch.zeros(crop_frames, cfg.max_mix, dtype=torch.float32)
    target_activity = torch.zeros(crop_frames, dtype=torch.float32)

    def get_feat(p: str):
        p = os.path.normpath(p)
        if p in feat_cache:
            return feat_cache[p]
        feat = load_feat_any(p, target_sr=cfg.target_sr, n_mels=cfg.n_mels)
        if feat.size(0) <= 800:
            feat_cache[p] = feat
        return feat

    for local_slot, spk in enumerate(spks):
        utt = random.choice(spk_to_utters[spk])
        feat = crop_or_pad_feat(get_feat(utt), crop_frames)
        gain = db_to_gain(random.uniform(cfg.spk_snr_min, cfg.spk_snr_max))

        src_s0, dst_s0, length = build_segment_with_offset(
            crop_frames=crop_frames,
            allow_overlap=cfg.allow_overlap,
            max_offset_ratio=cfg.max_offset_ratio,
        )
        if length <= 0:
            continue

        seg = feat[src_s0:src_s0 + length] * gain
        mixed[dst_s0:dst_s0 + length] += seg
        target_matrix[dst_s0:dst_s0 + length, local_slot] = 1.0
        target_activity[dst_s0:dst_s0 + length] = 1.0

    if noise_pts and random.random() < cfg.noise_prob:
        npt = random.choice(noise_pts)
        nfeat = crop_or_pad_feat(load_feat_any(npt, target_sr=cfg.target_sr, n_mels=cfg.n_mels), crop_frames)
        mixed += nfeat * db_to_gain(random.uniform(cfg.noise_snr_min, cfg.noise_snr_max))

    return {
        "fbank": mixed,
        "spk_label": int(spk2id[sv_spk]),
        "target_matrix": target_matrix,
        "target_activity": target_activity,
        "target_count": int(k),
        "speaker_names": spks,
    }


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
    feat_cache: Dict[str, torch.Tensor] = {}
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

    train_map_path_aug = os.path.join(cfg.cn_out_dir, "spk_to_utterances_train_aug.json")
    val_map_path_aug = os.path.join(cfg.cn_out_dir, "spk_to_utterances_val_aug.json")

    train_map_path = train_map_path_aug if os.path.isfile(train_map_path_aug) else os.path.join(cfg.cn_out_dir, "spk_to_utterances_train.json")
    val_map_path = val_map_path_aug if os.path.isfile(val_map_path_aug) else os.path.join(cfg.cn_out_dir, "spk_to_utterances_val.json")

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


def build_single_pack_from_cn_pt(cn_pt_path: str, spk_id: int, max_mix: int):
    obj = torch.load(cn_pt_path, map_location="cpu", weights_only=False)
    fbank = obj["fbank"]
    if not torch.is_tensor(fbank):
        fbank = torch.tensor(fbank)
    fbank = fbank.float().cpu()
    T = int(fbank.shape[0])

    target_matrix = torch.zeros(T, max_mix, dtype=torch.float32)
    target_matrix[:, 0] = 1.0

    pack = {
        "fbank": fbank,
        "spk_label": int(spk_id),
        "target_matrix": target_matrix,
        "target_activity": torch.ones(T, dtype=torch.float32),
        "target_count": 1,
        "source": "cn_celeb2_single",
    }
    if "speaker" in obj:
        pack["speaker_name"] = obj["speaker"]
    if "wav_path" in obj:
        pack["wav_path"] = obj["wav_path"]
    return pack


def get_existing_relpts(static_mix_dir: str, manifest_name: str) -> set:
    return {x["pt"].replace("\\", "/") for x in load_jsonl(os.path.join(static_mix_dir, manifest_name)) if "pt" in x}


def append_single_samples(cfg: BuildCfg):
    set_seed(cfg.seed)

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

                pack = build_single_pack_from_cn_pt(cn_pt_path, spk_id, max_mix)
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
FRAME_SHIFT_MS = 10
FRAME_LEN_MS = 25


def waveform_to_fbank(wav: torch.Tensor, sr: int, n_mels: int) -> torch.Tensor:
    if torchaudio is None:
        raise ImportError("torchaudio is required for synthetic negative generation")

    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    wav = wav.unsqueeze(0)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=int(sr * FRAME_LEN_MS / 1000),
        hop_length=int(sr * FRAME_SHIFT_MS / 1000),
        win_length=int(sr * FRAME_LEN_MS / 1000),
        n_mels=n_mels,
        power=2.0,
        center=True,
        normalized=False,
    )(wav)

    mel = torch.clamp(mel, min=1e-10).log()
    mel = mel.squeeze(0).transpose(0, 1).contiguous()
    return (mel - mel.mean()) / (mel.std() + 1e-5)


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
    return {
        "fbank": fbank.float(),
        "target_matrix": torch.zeros(T, max_mix, dtype=torch.float32),
        "target_activity": torch.zeros(T, dtype=torch.float32),
        "spk_label": -1,
        "target_count": 0,
    }


def add_negative_samples(cfg: BuildCfg):
    set_seed(cfg.seed)

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
            fbank = waveform_to_fbank(wav, sr=cfg.target_sr, n_mels=cfg.n_mels)
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
    parser = argparse.ArgumentParser(description="Unified preprocessing script for Speaker-Verification")

    parser.add_argument("--stage", type=str, default="all",
                        choices=["cn", "augment_cn", "mix", "add_single", "add_negative", "all"])

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--incremental", dest="incremental", action="store_true")
    parser.add_argument("--no-incremental", dest="incremental", action="store_false")
    parser.set_defaults(incremental=True)
    parser.add_argument("--backup_manifest", dest="backup_manifest", action="store_true")
    parser.add_argument("--no-backup_manifest", dest="backup_manifest", action="store_false")
    parser.set_defaults(backup_manifest=True)
    parser.add_argument("--backup_dir_name", type=str, default="_backups")

    parser.add_argument("--cn_root", type=str, default="../CN-Celeb_flac")
    parser.add_argument("--cn_out_dir", type=str, default="../processed/cn_celeb2")
    parser.add_argument("--static_out_dir", type=str, default="../processed/static_mix_cnceleb2")

    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--min_sec", type=float, default=1.0)
    parser.add_argument("--val_utt_ratio", type=float, default=0.1)
    parser.add_argument("--min_utts_per_spk", type=int, default=2)

    parser.add_argument("--num_train_mixes", type=int, default=100000)
    parser.add_argument("--num_val_mixes", type=int, default=20000)
    parser.add_argument("--min_mix", type=int, default=2)
    parser.add_argument("--max_mix", type=int, default=4)
    parser.add_argument("--crop_sec", type=float, default=4.0)
    parser.add_argument("--spk_snr_min", type=float, default=-5.0)
    parser.add_argument("--spk_snr_max", type=float, default=5.0)
    parser.add_argument("--noise_fbank_pt_dir", type=str, default="")
    parser.add_argument("--noise_prob", type=float, default=0.3)
    parser.add_argument("--noise_snr_min", type=float, default=-10.0)
    parser.add_argument("--noise_snr_max", type=float, default=0.0)
    parser.add_argument("--allow_overlap", action="store_true")
    parser.add_argument("--max_offset_ratio", type=float, default=0.35)

    parser.add_argument("--add_single_train", type=int, default=3000)
    parser.add_argument("--add_single_val", type=int, default=500)
    parser.add_argument("--add_neg_train", type=int, default=4000)
    parser.add_argument("--add_neg_val", type=int, default=500)
    parser.add_argument("--num_aug_per_utt_train", type=int, default=4)
    parser.add_argument("--num_aug_per_utt_val", type=int, default=1)
    parser.add_argument("--aug_subdir", type=str, default="fbank_pt_aug")
    parser.add_argument("--augmentation_version", type=str, default="v2")

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
