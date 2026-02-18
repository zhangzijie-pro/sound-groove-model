import random
import numpy as np
import torch
import torchaudio

TARGET_SR = 16000

def load_wav_mono(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    返回: wav [T] float32, target_sr, mono
    + 轻量去静音（防止实时录音大量静音导致 embedding 飘）
    """
    wav, sr = torchaudio.load(path)   # [C,T]
    wav = wav.mean(dim=0)             # [T]
    wav = wav.to(torch.float32)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    abs_w = wav.abs()
    if abs_w.numel() > 0:
        thr = max(1e-4, float(abs_w.max()) * 0.02)  # 动态阈值：max 的 2%
        idx = torch.nonzero(abs_w > thr, as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            wav = wav[idx[0].item() : idx[-1].item() + 1]

    return wav.contiguous()


def random_crop_or_repeat(wav: torch.Tensor, length: int) -> torch.Tensor:
    """
    wav: [T]
    length: samples
    """
    T = wav.numel()
    if T == length:
        return wav
    if T < length:
        reps = int(np.ceil(length / T))
        wav = wav.repeat(reps)[:length]
        return wav
    start = random.randint(0, T - length)
    return wav[start : start + length]


def wav_to_fbank(
    wav_16k: torch.Tensor,
    n_mels: int = 80,
    *,
    num_crops: int = 1,
    crop_sec: float = 3.0,
) -> torch.Tensor:
    """
    输入 wav: [T] (16kHz)
    输出 feat: [N, T_frames, n_mels]
      - N = num_crops
      - 每个 crop 长度 = crop_sec 秒（不足则 repeat，过长则 random crop）
    """
    assert wav_16k.dim() == 1, f"wav must be 1D [T], got {wav_16k.shape}"
    wav_16k = wav_16k.to(torch.float32)

    crop_len = int(TARGET_SR * float(crop_sec))
    crops = []
    for _ in range(int(num_crops)):
        w = random_crop_or_repeat(wav_16k, crop_len)  # [crop_len]
        w = w.unsqueeze(0)  # [1, T]
        feat = torchaudio.compliance.kaldi.fbank(
            w,
            sample_frequency=TARGET_SR,
            num_mel_bins=int(n_mels),
            frame_length=25,
            frame_shift=10,
            use_energy=False,
            window_type="povey",
            dither=0.0,
        )  # [frames, n_mels]
        crops.append(feat)

    feat = torch.stack(crops, dim=0)  # [N, frames, n_mels]
    return feat