import os
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

try:
    import torchaudio
except ImportError:
    torchaudio = None


PROCESSED_DIR = "./static_mix_cnceleb2"
TRAIN_MANIFEST = "train_manifest.jsonl"
VAL_MANIFEST = "val_manifest.jsonl"

OUT_SUBDIR = "neg_augmented"
SEED = 42

# 生成数量
NUM_TRAIN_NEG = 4000
NUM_VAL_NEG = 500

# 时长（秒）
DURATION_CHOICES = [2.0, 3.0, 4.0, 5.0, 6.0]

# 音频参数
SR = 16000
N_MELS = 80
FRAME_SHIFT_MS = 10
FRAME_LEN_MS = 25

NEGATIVE_TYPES = [
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

TYPE_WEIGHTS = [0.18, 0.14, 0.10, 0.08, 0.10, 0.08, 0.14, 0.06, 0.12]


# =========================
# 工具函数
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def ensure_manifest_jsonl(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing manifest: {path}")


def load_manifest_lines(path: str) -> List[dict]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            lines.append(json.loads(s))
    return lines


def append_manifest_lines(path: str, items: List[dict]):
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_any_sample_shape(processed_dir: str, manifest_name: str) -> Tuple[int, int]:
    manifest_path = os.path.join(processed_dir, manifest_name)
    samples = load_manifest_lines(manifest_path)
    if not samples:
        raise RuntimeError(f"Empty manifest: {manifest_path}")

    first_rel = samples[0]["pt"]
    first_abs = os.path.join(processed_dir, first_rel)
    pack = torch.load(first_abs, map_location="cpu", weights_only=False)

    fbank = pack["fbank"].float()
    target_matrix = pack["target_matrix"].float()

    feat_dim = int(fbank.shape[1])
    max_spk = int(target_matrix.shape[1])
    return feat_dim, max_spk


def waveform_to_fbank(wav: torch.Tensor, sr: int = 16000, n_mels: int = 80) -> torch.Tensor:
    """
    返回 [T, n_mels]
    """
    if torchaudio is None:
        raise ImportError("torchaudio is required for waveform-based augmentation.")

    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    wav = wav.unsqueeze(0)  # [1, N]

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=int(sr * FRAME_LEN_MS / 1000),
        hop_length=int(sr * FRAME_SHIFT_MS / 1000),
        win_length=int(sr * FRAME_LEN_MS / 1000),
        n_mels=n_mels,
        power=2.0,
        center=True,
        normalized=False,
    )(wav)  # [1, n_mels, T]

    mel = torch.clamp(mel, min=1e-10).log()
    mel = mel.squeeze(0).transpose(0, 1).contiguous()  # [T, n_mels]

    # 做一个温和标准化，避免负样本整体数值分布太奇怪
    mel = (mel - mel.mean()) / (mel.std() + 1e-5)
    return mel


def rand_duration_sec() -> float:
    return random.choice(DURATION_CHOICES)


def apply_fade(wav: torch.Tensor, sr: int, fade_ms: float = 20.0) -> torch.Tensor:
    fade_len = max(1, int(sr * fade_ms / 1000))
    fade = torch.linspace(0.0, 1.0, fade_len, dtype=wav.dtype)
    if wav.numel() >= 2 * fade_len:
        wav[:fade_len] *= fade
        wav[-fade_len:] *= fade.flip(0)
    return wav


def normalize_peak(wav: torch.Tensor, peak: float = 0.8) -> torch.Tensor:
    m = wav.abs().max().item()
    if m < 1e-8:
        return wav
    return wav / m * peak


# =========================
# 噪声生成器
# =========================
def gen_pure_silence(sec: float, sr: int) -> torch.Tensor:
    n = int(sec * sr)
    return torch.zeros(n, dtype=torch.float32)


def gen_white_noise(sec: float, sr: int, scale: float = 0.01) -> torch.Tensor:
    n = int(sec * sr)
    return torch.randn(n, dtype=torch.float32) * scale


def gen_pink_noise(sec: float, sr: int, scale: float = 0.01) -> torch.Tensor:
    """
    简单频域法生成 pink noise
    """
    n = int(sec * sr)
    x = torch.randn(n, dtype=torch.float32)

    X = torch.fft.rfft(x)
    freqs = torch.linspace(1.0, X.shape[0], X.shape[0], dtype=torch.float32)
    X = X / torch.sqrt(freqs)
    y = torch.fft.irfft(X, n=n).float()
    y = y / (y.std() + 1e-6) * scale
    return y


def gen_brown_noise(sec: float, sr: int, scale: float = 0.01) -> torch.Tensor:
    n = int(sec * sr)
    x = torch.randn(n, dtype=torch.float32)
    y = torch.cumsum(x, dim=0)
    y = y - y.mean()
    y = y / (y.std() + 1e-6) * scale
    return y


def gen_hum(sec: float, sr: int, freq: float = 50.0, amp: float = 0.02) -> torch.Tensor:
    n = int(sec * sr)
    t = torch.arange(n, dtype=torch.float32) / sr
    y = amp * torch.sin(2 * math.pi * freq * t)
    y += 0.4 * amp * torch.sin(2 * math.pi * freq * 2 * t)
    y += 0.2 * amp * torch.sin(2 * math.pi * freq * 3 * t)
    return y


def gen_fan_like_noise(sec: float, sr: int) -> torch.Tensor:
    """
    风扇/空调感：低频嗡声 + 轻微宽带噪声 + 缓慢包络变化
    """
    n = int(sec * sr)
    t = torch.arange(n, dtype=torch.float32) / sr

    base = gen_hum(sec, sr, freq=random.choice([45.0, 50.0, 60.0, 90.0]), amp=0.01)
    white = gen_white_noise(sec, sr, scale=0.005)

    lfo_freq = random.uniform(0.1, 0.4)
    env = 0.7 + 0.3 * torch.sin(2 * math.pi * lfo_freq * t + random.uniform(0, 2 * math.pi))
    y = (base + white) * env
    return y


def gen_impulse_clicks(sec: float, sr: int) -> torch.Tensor:
    """
    偶发脉冲/键盘/碰撞感
    """
    n = int(sec * sr)
    y = gen_white_noise(sec, sr, scale=0.002)

    num_clicks = random.randint(3, 15)
    for _ in range(num_clicks):
        pos = random.randint(0, n - 1)
        width = random.randint(8, 80)
        amp = random.uniform(0.02, 0.08)
        end = min(n, pos + width)
        pulse = torch.hann_window(end - pos, periodic=False) * amp
        y[pos:end] += pulse
    return y


def gen_mixed_noise(sec: float, sr: int) -> torch.Tensor:
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

    y = sum(parts)
    return y


def synth_negative_waveform(kind: str, sec: float, sr: int) -> torch.Tensor:
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
        raise ValueError(f"Unknown kind: {kind}")

    y = apply_fade(y, sr, fade_ms=20.0)
    y = normalize_peak(y, peak=random.uniform(0.2, 0.8))
    return y


def build_negative_pack(
    fbank: torch.Tensor,
    max_spk: int,
) -> dict:
    """
    与你当前 processed 包格式兼容：
    - fbank: [T, 80]
    - target_matrix: [T, K] 全 0
    - target_activity: [T] 全 0
    - target_count: 0
    - spk_label: -1
    """
    T = fbank.shape[0]
    target_matrix = torch.zeros(T, max_spk, dtype=torch.float32)
    target_activity = torch.zeros(T, dtype=torch.float32)

    pack = {
        "fbank": fbank.float(),
        "target_matrix": target_matrix,
        "target_activity": target_activity,
        "spk_label": -1,
        "target_count": 0,
    }
    return pack


def save_negative_samples(
    processed_dir: str,
    out_subdir: str,
    manifest_name: str,
    num_samples: int,
    feat_dim: int,
    max_spk: int,
):
    out_dir = os.path.join(processed_dir, out_subdir, Path(manifest_name).stem)
    os.makedirs(out_dir, exist_ok=True)

    new_items = []

    for i in range(num_samples):
        sec = rand_duration_sec()
        kind = random.choices(NEGATIVE_TYPES, weights=TYPE_WEIGHTS, k=1)[0]

        wav = synth_negative_waveform(kind, sec, SR)
        fbank = waveform_to_fbank(wav, sr=SR, n_mels=feat_dim)

        pack = build_negative_pack(fbank=fbank, max_spk=max_spk)

        fname = f"neg_{kind}_{i:06d}.pt"
        abs_pt = os.path.join(out_dir, fname)
        rel_pt = os.path.relpath(abs_pt, processed_dir).replace("\\", "/")

        torch.save(pack, abs_pt)

        new_items.append({
            "pt": rel_pt,
            "source": "synthetic_negative",
            "noise_type": kind,
            "duration_sec": sec,
        })

        if (i + 1) % 200 == 0:
            print(f"[{manifest_name}] generated {i+1}/{num_samples}")

    manifest_path = os.path.join(processed_dir, manifest_name)
    append_manifest_lines(manifest_path, new_items)
    print(f"[DONE] appended {len(new_items)} items -> {manifest_path}")


def main():
    set_seed(SEED)

    processed_dir = os.path.abspath(PROCESSED_DIR)
    train_manifest_path = os.path.join(processed_dir, TRAIN_MANIFEST)
    val_manifest_path = os.path.join(processed_dir, VAL_MANIFEST)

    ensure_manifest_jsonl(train_manifest_path)
    ensure_manifest_jsonl(val_manifest_path)

    feat_dim, max_spk = read_any_sample_shape(processed_dir, TRAIN_MANIFEST)
    print(f"[INFO] inferred feat_dim={feat_dim}, max_spk={max_spk}")

    if feat_dim != N_MELS:
        print(f"[WARN] inferred feat_dim={feat_dim}, script config N_MELS={N_MELS}, using inferred feat_dim.")

    save_negative_samples(
        processed_dir=processed_dir,
        out_subdir=OUT_SUBDIR,
        manifest_name=TRAIN_MANIFEST,
        num_samples=NUM_TRAIN_NEG,
        feat_dim=feat_dim,
        max_spk=max_spk,
    )

    save_negative_samples(
        processed_dir=processed_dir,
        out_subdir=OUT_SUBDIR,
        manifest_name=VAL_MANIFEST,
        num_samples=NUM_VAL_NEG,
        feat_dim=feat_dim,
        max_spk=max_spk,
    )

    print("[ALL DONE]")


if __name__ == "__main__":
    main()