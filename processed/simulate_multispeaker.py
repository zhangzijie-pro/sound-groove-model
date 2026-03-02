import torchaudio
import torch
import random
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

NOISE_DIR = "/path/to/musan/noise"  # 修改成你的噪声音库路径

def load_random_wav(spk_dir):
    files = list(Path(spk_dir).rglob("*.wav"))
    if not files:
        return None, None
    file = random.choice(files)
    wav, sr = torchaudio.load(file)
    return wav.mean(0), sr  # mono


def add_reverb(wav, sr, rt60=0.3):
    decay = torch.exp(-torch.arange(len(wav)) / (sr * rt60))
    return wav * decay[:len(wav)]


def mix_random_speakers(speakers_list, max_spk=10, min_len_sec=5, max_len_sec=20, sr=16000):
    n_spk = random.choices(range(0, max_spk + 1), weights=[1, 3, 5, 4, 3, 2, 1, 1, 0.5, 0.3, 0.2])[0]
    
    if n_spk == 0:
        # 纯噪声或静音
        length = random.randint(min_len_sec * sr, max_len_sec * sr)
        mixed = torch.zeros(length)
        labels = torch.zeros((length // 100, 0))  # 无说话人
        return mixed, labels, 0

    selected_spks = random.sample(speakers_list, min(n_spk, len(speakers_list)))
    wavs = []
    start_times = []
    spk_ids = []

    max_length = 0
    for i, spk_dir in enumerate(selected_spks):
        wav, _ = load_random_wav(spk_dir)
        if wav is None:
            continue
        wav = wav / (wav.abs().max() + 1e-8)  # 归一
        # 随机音量
        gain_db = random.uniform(-10, 10)
        wav *= 10 ** (gain_db / 20)
        # 随机起始位置（允许 overlap）
        start_sec = random.uniform(0, max_len_sec - min_len_sec)
        start = int(start_sec * sr)
        wavs.append(wav)
        start_times.append(start)
        spk_ids.append(i)
        max_length = max(max_length, start + len(wav))

    # 统一长度
    mixed = torch.zeros(max_length)
    # 准备 label: (T_frame, n_spk)，每 10ms 一帧
    frame_hop = sr // 100  # 10ms frame
    n_frames = (max_length + frame_hop - 1) // frame_hop
    labels = torch.zeros((n_frames, n_spk))

    for i, (wav, start, spk_idx) in enumerate(zip(wavs, start_times, spk_ids)):
        end = start + len(wav)
        mixed[start:end] += wav[:end-start]

        start_frame = start // frame_hop
        end_frame = (end + frame_hop - 1) // frame_hop
        labels[start_frame:end_frame, spk_idx] = 1.0

    # 加背景噪声
    if random.random() < 0.8:
        noise_files = list(Path(NOISE_DIR).rglob("*.wav"))
        if noise_files:
            noise, _ = torchaudio.load(random.choice(noise_files))
            noise = noise.mean(0)[:max_length]
            noise_gain_db = random.uniform(-5, 15)
            mixed += noise[:len(mixed)] * (10 ** (noise_gain_db / 20))

    mixed = mixed.clamp(-1, 1)

    return mixed, labels, n_spk


def generate_dataset(cnceleb_root, output_dir, num_samples=5000, max_spk=10):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    speakers = [str(d) for d in Path(cnceleb_root).iterdir() if d.is_dir()]

    for i in tqdm(range(num_samples)):
        mixed, labels, n_spk = mix_random_speakers(speakers, max_spk=max_spk)

        wav_path = Path(output_dir) / f"mix_{i:06d}_{n_spk}spk.wav"
        label_path = Path(output_dir) / f"mix_{i:06d}_{n_spk}spk.pt"

        torchaudio.save(str(wav_path), mixed.unsqueeze(0), 16000)
        torch.save(labels, label_path)

        if i % 500 == 0:
            print(f"Generated {i} samples, avg speakers: {n_spk}")


if __name__ == "__main__":
    cnceleb_root = "/path/to/CN-Celeb"          # 修改路径
    output_dir   = "./cnceleb_simulated_random"
    generate_dataset(cnceleb_root, output_dir, num_samples=10000)