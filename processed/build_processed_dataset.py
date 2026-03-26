import os
import json
import math
import random
import shutil
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import torchaudio
except ImportError:
    torchaudio = None

from speaker_verification.audio.features import wav_to_fbank, load_wav_mono

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def backup_file(path: str):
    if not os.path.isfile(path):
        return
    bak = path + ".bak"
    if not os.path.isfile(bak):
        shutil.copy2(path, bak)
        print(f"[Backup] {path} -> {bak}")


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
    stage: str = "all"  # cn | mix | add_single | add_negative | all

    # common
    seed: int = 1234

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
    num_train_mixes: int = 100000
    num_val_mixes: int = 10000
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
    add_single_train: int = 4000
    add_single_val: int = 500
    add_single_subdir: str = "single_from_cnceleb2"
    backup_manifest: bool = True
    skip_existing: bool = True

    # add negative
    add_neg_train: int = 4000
    add_neg_val: int = 500
    add_neg_subdir: str = "neg_augmented"


# =========================================================
# Stage 1: preprocess CN-Celeb2 -> single-speaker fbank
# =========================================================
def preprocess_cn_celeb2(cfg: BuildCfg):
    set_seed(cfg.seed)

    ensure_dir(cfg.cn_out_dir)
    feat_dir = os.path.join(cfg.cn_out_dir, "fbank_pt")
    ensure_dir(feat_dir)

    data_dir = os.path.join(cfg.cn_root, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    spk2files = defaultdict(list)
    for spk in os.listdir(data_dir):
        spk_path = os.path.join(data_dir, spk)
        if not os.path.isdir(spk_path):
            continue
        for fn in os.listdir(spk_path):
            if fn.lower().endswith((".flac", ".wav")):
                spk2files[spk].append(os.path.join(spk_path, fn))

    kept_spks = sorted([spk for spk, fs in spk2files.items() if len(fs) >= cfg.min_utts_per_spk])
    spk2id = {spk: i for i, spk in enumerate(kept_spks)}

    train_list_path = os.path.join(cfg.cn_out_dir, "train_fbank_list.txt")
    val_list_path = os.path.join(cfg.cn_out_dir, "val_fbank_list.txt")
    spk_train_path = os.path.join(cfg.cn_out_dir, "spk_to_utterances_train.json")
    spk_val_path = os.path.join(cfg.cn_out_dir, "spk_to_utterances_val.json")
    spk2id_path = os.path.join(cfg.cn_out_dir, "spk2id.json")
    val_meta_path = os.path.join(cfg.cn_out_dir, "val_meta.jsonl")

    spk_to_utters_train = defaultdict(list)
    spk_to_utters_val = defaultdict(list)
    val_meta = []

    with open(train_list_path, "w", encoding="utf-8") as ftrain, open(val_list_path, "w", encoding="utf-8") as fval:
        for spk in tqdm(kept_spks, desc="Preprocess CN-Celeb2"):
            files = sorted(spk2files[spk])
            random.shuffle(files)

            n_val = max(1, int(len(files) * cfg.val_utt_ratio))
            if n_val >= len(files):
                n_val = len(files) - 1

            val_files = set(files[:n_val])

            for wav_path in files:
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

                    key = safe_key(wav_path, cfg.cn_root)
                    feat_path = os.path.join(feat_dir, f"{spk}__{key}.pt").replace("\\", "/")

                    torch.save(
                        {
                            "fbank": feat,
                            "speaker": spk,
                            "spk_id": int(spk2id[spk]),
                            "wav_path": wav_path.replace("\\", "/"),
                        },
                        feat_path,
                    )

                    label = int(spk2id[spk])
                    if wav_path in val_files:
                        spk_to_utters_val[spk].append(feat_path)
                        fval.write(f"{label} {feat_path}\n")
                        val_meta.append(
                            {
                                "speaker": spk,
                                "feat_path": feat_path,
                                "spk_id": label,
                            }
                        )
                    else:
                        spk_to_utters_train[spk].append(feat_path)
                        ftrain.write(f"{label} {feat_path}\n")

                except Exception as e:
                    print(f"[WARN] skip {wav_path}: {e}")

    save_json(spk_train_path, spk_to_utters_train)
    save_json(spk_val_path, spk_to_utters_val)
    save_json(spk2id_path, spk2id)
    write_jsonl(val_meta_path, val_meta)

    print(f"[CN DONE] speakers={len(spk2id)} out_dir={cfg.cn_out_dir}")


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

    ensure_dir(cfg.static_out_dir)
    mix_dir = os.path.join(cfg.static_out_dir, "mix_pt", split_name)
    ensure_dir(mix_dir)

    manifest_path = os.path.join(cfg.static_out_dir, f"{split_name}_manifest.jsonl")
    feat_cache: Dict[str, torch.Tensor] = {}

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for idx in tqdm(range(num_mixes), desc=f"Generate {split_name} mixes"):
            pack = generate_one_mix(
                speakers=speakers,
                spk_to_utters=spk_to_utters,
                spk2id=spk2id,
                cfg=cfg,
                feat_cache=feat_cache,
                noise_pts=noise_pts,
            )
            rel_pt = os.path.join("mix_pt", split_name, f"{idx:08d}.pt").replace("\\", "/")
            abs_pt = os.path.join(cfg.static_out_dir, rel_pt)
            torch.save(pack, abs_pt)
            mf.write(json.dumps({"pt": rel_pt}, ensure_ascii=False) + "\n")

    print(f"[MIX DONE] {split_name}: {manifest_path}")


def build_static_mix(cfg: BuildCfg):
    set_seed(cfg.seed)

    train_map_path = os.path.join(cfg.cn_out_dir, "spk_to_utterances_train.json")
    val_map_path = os.path.join(cfg.cn_out_dir, "spk_to_utterances_val.json")
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
    save_json(os.path.join(cfg.static_out_dir, "spk2id.json"), spk2id)
    save_json(
        os.path.join(cfg.static_out_dir, "mix_meta.json"),
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
    list_path = os.path.join(cn_out_dir, f"{split}_fbank_list.txt")
    if not os.path.isfile(list_path):
        raise FileNotFoundError(list_path)

    out = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            sp = s.split(maxsplit=1)
            if len(sp) != 2:
                continue
            out.append((int(sp[0]), sp[1]))
    return out


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

    train_manifest = os.path.join(cfg.static_out_dir, "train_manifest.jsonl")
    val_manifest = os.path.join(cfg.static_out_dir, "val_manifest.jsonl")
    if not os.path.isfile(train_manifest):
        raise FileNotFoundError(train_manifest)
    if not os.path.isfile(val_manifest):
        raise FileNotFoundError(val_manifest)

    if cfg.backup_manifest:
        backup_file(train_manifest)
        backup_file(val_manifest)

    max_mix = infer_max_mix_from_manifest(cfg.static_out_dir, "train_manifest.jsonl")

    for split, n_add in [("train", cfg.add_single_train), ("val", cfg.add_single_val)]:
        items = collect_cn_single_items(cfg.cn_out_dir, split)
        random.shuffle(items)
        items = items[: min(n_add, len(items))]

        existing = get_existing_relpts(cfg.static_out_dir, f"{split}_manifest.jsonl")
        appended = []
        out_dir = os.path.join(cfg.static_out_dir, cfg.add_single_subdir, split)
        ensure_dir(out_dir)

        for idx, (spk_id, cn_pt_path) in enumerate(items):
            try:
                base = Path(cn_pt_path).stem
                rel_pt = os.path.join(cfg.add_single_subdir, split, f"single_{idx:06d}_{base}.pt").replace("\\", "/")
                if cfg.skip_existing and rel_pt in existing:
                    continue

                pack = build_single_pack_from_cn_pt(cn_pt_path, spk_id, max_mix)
                abs_pt = os.path.join(cfg.static_out_dir, rel_pt)
                ensure_dir(os.path.dirname(abs_pt))
                torch.save(pack, abs_pt)
                appended.append({"pt": rel_pt, "source": "cn_celeb2_single"})
            except Exception as e:
                print(f"[WARN] single skip {cn_pt_path}: {e}")

        append_jsonl(os.path.join(cfg.static_out_dir, f"{split}_manifest.jsonl"), appended)
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
        out_dir = os.path.join(cfg.static_out_dir, cfg.add_neg_subdir, split)
        ensure_dir(out_dir)

        manifest_path = os.path.join(cfg.static_out_dir, f"{split}_manifest.jsonl")
        new_items = []

        for i in range(n_add):
            sec = random.choice(NEG_DURATIONS)
            kind = random.choices(NEG_TYPES, weights=NEG_WEIGHTS, k=1)[0]

            wav = synth_negative_waveform(kind, sec, cfg.target_sr)
            fbank = waveform_to_fbank(wav, sr=cfg.target_sr, n_mels=cfg.n_mels)
            pack = build_negative_pack(fbank, max_mix=max_mix)

            fname = f"neg_{kind}_{i:06d}.pt"
            abs_pt = os.path.join(out_dir, fname)
            rel_pt = os.path.relpath(abs_pt, cfg.static_out_dir).replace("\\", "/")

            torch.save(pack, abs_pt)
            new_items.append(
                {
                    "pt": rel_pt,
                    "source": "synthetic_negative",
                    "noise_type": kind,
                    "duration_sec": sec,
                }
            )

            if (i + 1) % 200 == 0:
                print(f"[{split}] negative generated {i+1}/{n_add}")

        append_jsonl(manifest_path, new_items)
        print(f"[ADD NEG] {split}: added={len(new_items)}")


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Unified preprocessing script for Speaker-Verification")

    parser.add_argument("--stage", type=str, default="all",
                        choices=["cn", "mix", "add_single", "add_negative", "all"])

    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--cn_root", type=str, default="../CN-Celeb_flac")
    parser.add_argument("--cn_out_dir", type=str, default="../processed/cn_celeb2")
    parser.add_argument("--static_out_dir", type=str, default="../processed/static_mix_cnceleb2")

    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--min_sec", type=float, default=1.0)
    parser.add_argument("--val_utt_ratio", type=float, default=0.1)
    parser.add_argument("--min_utts_per_spk", type=int, default=2)

    parser.add_argument("--num_train_mixes", type=int, default=100000)
    parser.add_argument("--num_val_mixes", type=int, default=10000)
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

    parser.add_argument("--add_single_train", type=int, default=4000)
    parser.add_argument("--add_single_val", type=int, default=500)
    parser.add_argument("--add_neg_train", type=int, default=4000)
    parser.add_argument("--add_neg_val", type=int, default=500)

    args = parser.parse_args()
    return BuildCfg(**vars(args))


def main():
    cfg = parse_args()
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    if cfg.stage == "cn":
        preprocess_cn_celeb2(cfg)
    elif cfg.stage == "mix":
        build_static_mix(cfg)
    elif cfg.stage == "add_single":
        append_single_samples(cfg)
    elif cfg.stage == "add_negative":
        add_negative_samples(cfg)
    elif cfg.stage == "all":
        preprocess_cn_celeb2(cfg)
        build_static_mix(cfg)
        append_single_samples(cfg)
        add_negative_samples(cfg)
    else:
        raise ValueError(cfg.stage)

    print("[ALL DONE]")


if __name__ == "__main__":
    main()