import os
import json
import random
import argparse
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_txt_pairs(path: str) -> List[Tuple[int, str]]:
    items = []
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
    with open(path, "w", encoding="utf-8") as f:
        for spk_id, p in items:
            f.write(f"{spk_id} {p}\n")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def time_mask(feat: torch.Tensor, max_width: int = 20) -> torch.Tensor:
    x = feat.clone()
    T = x.size(0)
    if T <= 4:
        return x
    w = random.randint(0, min(max_width, max(1, T // 8)))
    if w <= 0:
        return x
    s = random.randint(0, max(0, T - w))
    x[s:s+w] = 0.0
    return x


def freq_mask(feat: torch.Tensor, max_width: int = 8) -> torch.Tensor:
    x = feat.clone()
    Fdim = x.size(1)
    if Fdim <= 4:
        return x
    w = random.randint(0, min(max_width, max(1, Fdim // 6)))
    if w <= 0:
        return x
    s = random.randint(0, max(0, Fdim - w))
    x[:, s:s+w] = 0.0
    return x


def global_gain(feat: torch.Tensor, scale_min: float = 0.85, scale_max: float = 1.15) -> torch.Tensor:
    g = random.uniform(scale_min, scale_max)
    return feat * g


def add_gaussian_noise(feat: torch.Tensor, sigma_min: float = 0.002, sigma_max: float = 0.01) -> torch.Tensor:
    sigma = random.uniform(sigma_min, sigma_max)
    return feat + torch.randn_like(feat) * sigma


def temporal_warp(feat: torch.Tensor, rate_min: float = 0.92, rate_max: float = 1.08) -> torch.Tensor:
    """
    在 fbank 域做近似时间拉伸：
    先沿时间轴插值到新长度，再插回原长度。
    """
    x = feat.transpose(0, 1).unsqueeze(0)  # [1, F, T]
    T = x.size(-1)
    rate = random.uniform(rate_min, rate_max)
    T2 = max(8, int(T * rate))
    y = F.interpolate(x, size=T2, mode="linear", align_corners=False)
    z = F.interpolate(y, size=T, mode="linear", align_corners=False)
    return z.squeeze(0).transpose(0, 1).contiguous()


def smooth_perturb(feat: torch.Tensor) -> torch.Tensor:
    """
    轻微平滑，模拟不同说话力度 / 情绪状态下的局部谱形变化
    """
    x = feat.transpose(0, 1).unsqueeze(0)  # [1, F, T]
    k = random.choice([3, 5])
    pad = k // 2
    weight = torch.ones(x.size(1), 1, k, dtype=x.dtype, device=x.device) / k
    y = F.conv1d(F.pad(x, (pad, pad), mode="replicate"), weight, groups=x.size(1))
    y = 0.7 * x + 0.3 * y
    return y.squeeze(0).transpose(0, 1).contiguous()


def normalize_feat(feat: torch.Tensor) -> torch.Tensor:
    return (feat - feat.mean()) / (feat.std() + 1e-5)


def augment_fbank(feat: torch.Tensor) -> torch.Tensor:
    x = feat.clone().float()

    ops = []

    if random.random() < 0.8:
        ops.append(global_gain)
    if random.random() < 0.7:
        ops.append(add_gaussian_noise)
    if random.random() < 0.6:
        ops.append(temporal_warp)
    if random.random() < 0.5:
        ops.append(smooth_perturb)
    if random.random() < 0.5:
        ops.append(time_mask)
    if random.random() < 0.5:
        ops.append(freq_mask)

    random.shuffle(ops)
    for op in ops:
        x = op(x)

    x = normalize_feat(x)
    return x


def build_augmented_cn_lists(
    cn_out_dir: str,
    split: str,
    num_aug_per_utt: int = 2,
):
    list_path = os.path.join(cn_out_dir, f"{split}_fbank_list.txt")
    items = load_txt_pairs(list_path)

    aug_root = os.path.join(cn_out_dir, "fbank_pt_aug", split)
    ensure_dir(aug_root)

    out_items = []
    for spk_id, pt_path in tqdm(items, desc=f"Augment {split} singles"):
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
        feat = obj["fbank"].float()

        out_items.append((spk_id, pt_path))

        stem = os.path.splitext(os.path.basename(pt_path))[0]
        speaker = obj.get("speaker", "unknown")

        for k in range(num_aug_per_utt):
            aug_feat = augment_fbank(feat)
            aug_obj = dict(obj)
            aug_obj["fbank"] = aug_feat
            aug_obj["source"] = "augmented_cn_single"
            aug_obj["aug_index"] = k

            out_path = os.path.join(aug_root, f"{stem}__aug{k}.pt")
            torch.save(aug_obj, out_path)
            out_items.append((spk_id, out_path))

    save_txt_pairs(os.path.join(cn_out_dir, f"{split}_fbank_list_aug.txt"), out_items)

    # 同步生成 speaker -> utterances map
    spk2id = load_json(os.path.join(cn_out_dir, "spk2id.json"))
    id2spk = {int(v): k for k, v in spk2id.items()}
    spk_to_utts: Dict[str, List[str]] = {}

    for spk_id, path in out_items:
        spk = id2spk[int(spk_id)]
        spk_to_utts.setdefault(spk, []).append(path)

    save_json(
        os.path.join(cn_out_dir, f"spk_to_utterances_{split}_aug.json"),
        spk_to_utts
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cn_out_dir", type=str, default="../processed/cn_celeb2")
    parser.add_argument("--num_aug_per_utt", type=int, default=2)
    args = parser.parse_args()

    random.seed(1234)
    torch.manual_seed(1234)

    build_augmented_cn_lists(args.cn_out_dir, "train", args.num_aug_per_utt)
    build_augmented_cn_lists(args.cn_out_dir, "val", max(1, args.num_aug_per_utt // 2))
    print("[AUG CN DONE]")


if __name__ == "__main__":
    main()