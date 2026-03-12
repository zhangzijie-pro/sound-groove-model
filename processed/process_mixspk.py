import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from speaker_verification.audio.features import wav_to_fbank, load_wav_mono
    _HAS_AUDIO = True
except Exception:
    _HAS_AUDIO = False


@dataclass
class StaticMixPrepCfg:
    # preprocess_cnceleb2_train.py 的输出目录
    processed_dir: str = "../processed/cn_celeb2"

    # 静态混合输出目录
    out_dir: str = "../processed/static_mix_cnceleb2"

    # 训练/验证各生成多少条
    num_train_mixes: int = 100_000
    num_val_mixes: int = 10_000

    # 每条样本混合说话人数范围
    min_mix: int = 2
    max_mix: int = 4

    # 每条样本时长（秒）
    crop_sec: float = 4.0

    # 说话人增益范围（dB）
    spk_snr_min: float = -5.0
    spk_snr_max: float = 5.0

    # 噪声设置（可选）
    noise_fbank_pt_dir: str = ""
    noise_prob: float = 0.3
    noise_snr_min: float = -10.0
    noise_snr_max: float = 0.0

    # 是否允许重叠
    allow_overlap: bool = True

    # 控制偏移范围，越大越分散，越小越重叠
    max_offset_ratio: float = 0.35

    # 随机种子
    seed: int = 1234


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def db_to_gain(db: float) -> float:
    return 10 ** (db / 20.0)


def list_pt_files(root: str) -> List[str]:
    if not root or (not os.path.isdir(root)):
        return []
    out = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".pt"):
                out.append(os.path.join(r, f))
    return out


@torch.no_grad()
def load_feat_any(path: str) -> torch.Tensor:
    """
    统一读取特征:
      - .pt: 支持 Tensor 或 dict{'fbank': Tensor}
      - 音频: 自动转 fbank
    返回:
      [T,80] float32 CPU tensor
    """
    lp = path.lower()

    if lp.endswith(".pt"):
        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict):
            if "fbank" in obj:
                feat = obj["fbank"]
            else:
                tens = [v for v in obj.values() if torch.is_tensor(v)]
                if not tens:
                    raise ValueError(f"PT dict has no tensor: {path}")
                feat = tens[0]
        else:
            feat = obj

        if not torch.is_tensor(feat):
            feat = torch.tensor(feat)

        feat = feat.float().cpu()

        if feat.dim() == 3 and feat.size(0) == 1:
            feat = feat[0]

        if feat.dim() != 2:
            raise ValueError(f"Unexpected feat shape {tuple(feat.shape)} in {path}")

        if feat.size(1) != 80 and feat.size(0) == 80:
            feat = feat.transpose(0, 1)

        if feat.size(1) != 80:
            raise ValueError(f"Expected mel=80, got shape {tuple(feat.shape)} in {path}")

        return feat

    if not _HAS_AUDIO:
        raise RuntimeError(f"Audio backend not available, but got audio file: {path}")

    wav = load_wav_mono(path, target_sr=16000)
    feat = wav_to_fbank(wav, n_mels=80)

    if not torch.is_tensor(feat):
        feat = torch.tensor(feat)

    feat = feat.float().cpu()
    if feat.dim() != 2:
        raise ValueError(f"Unexpected fbank shape {tuple(feat.shape)} for audio {path}")

    return feat


def crop_or_pad_feat(x: torch.Tensor, crop_frames: int) -> torch.Tensor:
    """
    x: [T,80] -> [crop_frames,80]
    """
    T = x.size(0)
    if T >= crop_frames:
        s = random.randint(0, T - crop_frames)
        return x[s:s + crop_frames]

    pad = crop_frames - T
    return F.pad(x, (0, 0, 0, pad))


def resolve_path(p: str) -> str:
    return os.path.normpath(p)


def build_segment_with_offset(
    feat: torch.Tensor,
    crop_frames: int,
    allow_overlap: bool,
    max_offset_ratio: float,
):
    """
    返回:
      src_s0, dst_s0, length
    """
    if allow_overlap:
        max_shift = int(crop_frames * max_offset_ratio)
        offset = random.randint(-max_shift, max_shift)
    else:
        offset = 0

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
    crop_frames: int,
    cfg: StaticMixPrepCfg,
    feat_cache: Dict[str, torch.Tensor],
    noise_pts: List[str],
):
    """
    生成单条混合样本

    输出 pack:
      fbank: [T,80]
      spk_label: int                    # 用于SV分支的主标签
      target_matrix: [T,K]              # K=max_mix，逐帧多槽位0/1监督
      target_activity: [T]
      target_count: int
      speaker_names: List[str]
    """
    def get_feat(p: str) -> torch.Tensor:
        p2 = resolve_path(p)
        if p2 in feat_cache:
            return feat_cache[p2]

        feat = load_feat_any(p2)

        # 小缓存，避免内存爆炸
        if feat.size(0) <= 800:
            feat_cache[p2] = feat
        return feat

    k = random.randint(cfg.min_mix, cfg.max_mix)
    spks = random.sample(speakers, k)

    mixed = torch.zeros(crop_frames, 80, dtype=torch.float32)
    target_matrix = torch.zeros(crop_frames, cfg.max_mix, dtype=torch.float32)
    target_activity = torch.zeros(crop_frames, dtype=torch.float32)

    # 用于 SV 分支，随机从当前混合说话人中选一个主标签
    sv_spk = random.choice(spks)

    for local_slot, spk in enumerate(spks):
        utt = random.choice(spk_to_utters[spk])
        feat = crop_or_pad_feat(get_feat(utt), crop_frames)

        snr_db = random.uniform(cfg.spk_snr_min, cfg.spk_snr_max)
        gain = db_to_gain(snr_db)

        src_s0, dst_s0, length = build_segment_with_offset(
            feat=feat,
            crop_frames=crop_frames,
            allow_overlap=cfg.allow_overlap,
            max_offset_ratio=cfg.max_offset_ratio,
        )

        if length <= 0:
            continue

        seg = feat[src_s0:src_s0 + length] * gain
        mixed[dst_s0:dst_s0 + length] += seg

        # 多标签，多槽位
        target_matrix[dst_s0:dst_s0 + length, local_slot] = 1.0
        target_activity[dst_s0:dst_s0 + length] = 1.0

    # 可选加噪
    if noise_pts and (random.random() < cfg.noise_prob):
        npt = random.choice(noise_pts)
        nfeat = crop_or_pad_feat(load_feat_any(npt), crop_frames)
        ndb = random.uniform(cfg.noise_snr_min, cfg.noise_snr_max)
        mixed += nfeat * db_to_gain(ndb)

    return {
        "fbank": mixed,                              # [T,80]
        "spk_label_name": sv_spk,                    # 字符串speaker名，后面统一映射
        "target_matrix": target_matrix,              # [T,K]
        "target_activity": target_activity,          # [T]
        "target_count": int(k),                      # 1..K
        "speaker_names": spks,                       # 当前样本实际有哪些speaker
    }


def generate_split(
    split_name: str,
    num_mixes: int,
    spk_to_utters: Dict[str, List[str]],
    spk2id: Dict[str, int],
    cfg: StaticMixPrepCfg,
    noise_pts: List[str],
):
    speakers = sorted(list(spk_to_utters.keys()))
    assert len(speakers) >= cfg.max_mix, f"[{split_name}] speakers={len(speakers)} < max_mix={cfg.max_mix}"

    mix_dir = os.path.join(cfg.out_dir, "mix_pt", split_name)
    ensure_dir(mix_dir)

    manifest_path = os.path.join(cfg.out_dir, f"{split_name}_manifest.jsonl")
    crop_frames = int(cfg.crop_sec * 100)  # 10ms hop -> 100 fps

    feat_cache: Dict[str, torch.Tensor] = {}

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for idx in tqdm(range(num_mixes), desc=f"Generating {split_name} mixes"):
            pack = generate_one_mix(
                speakers=speakers,
                spk_to_utters=spk_to_utters,
                crop_frames=crop_frames,
                cfg=cfg,
                feat_cache=feat_cache,
                noise_pts=noise_pts,
            )

            spk_label_name = pack.pop("spk_label_name")
            spk_label = int(spk2id[spk_label_name])

            pack["spk_label"] = spk_label

            rel_pt = os.path.join("mix_pt", split_name, f"{idx:08d}.pt").replace("\\", "/")
            abs_pt = os.path.join(cfg.out_dir, rel_pt)

            torch.save(pack, abs_pt)
            mf.write(json.dumps({"pt": rel_pt}, ensure_ascii=False) + "\n")

    print(f"✅ [{split_name}] Manifest: {manifest_path}")


def main():
    cfg = StaticMixPrepCfg()

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_map_path = os.path.join(cfg.processed_dir, "spk_to_utterances_train.json")
    val_map_path = os.path.join(cfg.processed_dir, "spk_to_utterances_val.json")
    spk2id_path = os.path.join(cfg.processed_dir, "spk2id.json")

    assert os.path.isfile(train_map_path), f"Missing {train_map_path}"
    assert os.path.isfile(val_map_path), f"Missing {val_map_path}"
    assert os.path.isfile(spk2id_path), f"Missing {spk2id_path}"

    with open(train_map_path, "r", encoding="utf-8") as f:
        spk_to_utters_train: Dict[str, List[str]] = json.load(f)

    with open(val_map_path, "r", encoding="utf-8") as f:
        spk_to_utters_val: Dict[str, List[str]] = json.load(f)

    with open(spk2id_path, "r", encoding="utf-8") as f:
        spk2id: Dict[str, int] = json.load(f)

    ensure_dir(cfg.out_dir)

    # 复制一份统一 spk2id 到混合目录
    with open(os.path.join(cfg.out_dir, "spk2id.json"), "w", encoding="utf-8") as f:
        json.dump(spk2id, f, ensure_ascii=False, indent=2)

    meta = {
        "processed_dir": cfg.processed_dir,
        "out_dir": cfg.out_dir,
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
    }
    with open(os.path.join(cfg.out_dir, "mix_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    noise_pts = list_pt_files(cfg.noise_fbank_pt_dir)
    print(
        f"[StaticMixPrep] "
        f"train_spks={len(spk_to_utters_train)} | "
        f"val_spks={len(spk_to_utters_val)} | "
        f"noise_pts={len(noise_pts)}"
    )

    generate_split(
        split_name="train",
        num_mixes=cfg.num_train_mixes,
        spk_to_utters=spk_to_utters_train,
        spk2id=spk2id,
        cfg=cfg,
        noise_pts=noise_pts,
    )

    generate_split(
        split_name="val",
        num_mixes=cfg.num_val_mixes,
        spk_to_utters=spk_to_utters_val,
        spk2id=spk2id,
        cfg=cfg,
        noise_pts=noise_pts,
    )

    print(f"✅ Done. out_dir = {cfg.out_dir}")
    print(f"✅ train_manifest = {os.path.join(cfg.out_dir, 'train_manifest.jsonl')}")
    print(f"✅ val_manifest   = {os.path.join(cfg.out_dir, 'val_manifest.jsonl')}")
    print(f"✅ spk2id         = {os.path.join(cfg.out_dir, 'spk2id.json')}")
    print(f"✅ meta           = {os.path.join(cfg.out_dir, 'mix_meta.json')}")


if __name__ == "__main__":
    main()