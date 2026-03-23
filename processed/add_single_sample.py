import os
import json
from pathlib import Path
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class AddSingleSpeakerCfg:
    # 单说话人 fbank 目录（由 preprocess_cnceleb2_train.py 生成）
    cn_processed_dir: str = "../processed/cn_celeb2"

    # 混合说话人目录（由 static_mix 生成）
    static_mix_dir: str = "../processed/static_mix_cnceleb2"

    train_manifest_name: str = "train_manifest.jsonl"
    val_manifest_name: str = "val_manifest.jsonl"

    # 追加多少单说话人样本
    num_train_add: int = 4000
    num_val_add: int = 500

    # 若为 True，追加前备份原 manifest
    backup_manifest: bool = True

    # 若为 True，跳过已经追加过的 rel_pt，避免重复
    skip_existing: bool = True

    # 统一写入到 static_mix_cnceleb2/single_from_cnceleb2/{train,val}/
    out_subdir: str = "single_from_cnceleb2"

    seed: int = 1234


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    return items


def append_jsonl(path: str, items: List[dict]):
    with open(path, "a", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def backup_file(path: str):
    if not os.path.isfile(path):
        return
    dst = path + ".bak"
    if not os.path.isfile(dst):
        shutil.copy2(path, dst)
        print(f"[Backup] {path} -> {dst}")
    else:
        print(f"[Backup] already exists: {dst}")


def infer_max_mix_from_static_mix(static_mix_dir: str, manifest_name: str) -> int:
    manifest_path = os.path.join(static_mix_dir, manifest_name)
    items = load_jsonl(manifest_path)
    if len(items) == 0:
        raise RuntimeError(f"Empty manifest: {manifest_path}")

    rel_pt = items[0]["pt"]
    abs_pt = os.path.join(static_mix_dir, rel_pt)
    pack = torch.load(abs_pt, map_location="cpu", weights_only=False)

    if "target_matrix" not in pack:
        raise RuntimeError(f"target_matrix not found in sample: {abs_pt}")

    target_matrix = pack["target_matrix"]
    if not torch.is_tensor(target_matrix):
        target_matrix = torch.tensor(target_matrix)

    if target_matrix.dim() != 2:
        raise RuntimeError(f"Unexpected target_matrix shape {tuple(target_matrix.shape)} in {abs_pt}")

    max_mix = int(target_matrix.shape[1])
    return max_mix


def collect_cn_single_pt(
    cn_processed_dir: str,
    split: str,
) -> List[Tuple[int, str]]:
    """
    从 cn_celeb2 的 train_fbank_list.txt / val_fbank_list.txt 读取：
      每行格式: "<label> <feat_path>"
    返回:
      [(spk_id, feat_path), ...]
    """
    list_name = f"{split}_fbank_list.txt"
    list_path = os.path.join(cn_processed_dir, list_name)
    if not os.path.isfile(list_path):
        raise FileNotFoundError(f"Missing {list_path}")

    out = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            sp = s.split(maxsplit=1)
            if len(sp) != 2:
                continue
            spk_id = int(sp[0])
            feat_path = sp[1]
            out.append((spk_id, feat_path))
    return out


def build_single_pack_from_cn_pt(
    cn_pt_path: str,
    spk_id: int,
    max_mix: int,
):
    """
    将 cn_celeb2 的单说话人 fbank_pt 转为 static_mix 兼容 pack
    输出字段对齐当前 StaticMixDataset:
      - fbank: [T,80]
      - spk_label: int
      - target_matrix: [T,max_mix]
      - target_activity: [T]
      - target_count: 1
    """
    obj = torch.load(cn_pt_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "fbank" not in obj:
        raise RuntimeError(f"Unexpected CN pack format: {cn_pt_path}")

    fbank = obj["fbank"]
    if not torch.is_tensor(fbank):
        fbank = torch.tensor(fbank)
    fbank = fbank.float().cpu()

    if fbank.dim() != 2:
        raise RuntimeError(f"Unexpected fbank shape {tuple(fbank.shape)} in {cn_pt_path}")

    T = int(fbank.shape[0])
    target_matrix = torch.zeros(T, max_mix, dtype=torch.float32)
    target_matrix[:, 0] = 1.0  # 单说话人统一放在 slot 0

    target_activity = torch.ones(T, dtype=torch.float32)

    pack = {
        "fbank": fbank,
        "spk_label": int(spk_id),
        "target_matrix": target_matrix,
        "target_activity": target_activity,
        "target_count": 1,
    }

    # 把原字段带上，方便后查
    if "speaker" in obj:
        pack["speaker_name"] = obj["speaker"]
    if "wav_path" in obj:
        pack["wav_path"] = obj["wav_path"]
    pack["source"] = "cn_celeb2_single"

    return pack


def get_existing_relpts(static_mix_dir: str, manifest_name: str) -> set:
    manifest_path = os.path.join(static_mix_dir, manifest_name)
    if not os.path.isfile(manifest_path):
        return set()
    items = load_jsonl(manifest_path)
    rels = set()
    for x in items:
        pt = x.get("pt", "")
        if pt:
            rels.add(pt.replace("\\", "/"))
    return rels


def add_split(
    split: str,
    cn_items: List[Tuple[int, str]],
    cfg: AddSingleSpeakerCfg,
    max_mix: int,
):
    static_manifest_path = os.path.join(cfg.static_mix_dir, f"{split}_manifest.jsonl")
    if not os.path.isfile(static_manifest_path):
        raise FileNotFoundError(f"Missing {static_manifest_path}")

    n_add = cfg.num_train_add if split == "train" else cfg.num_val_add

    existing_relpts = get_existing_relpts(cfg.static_mix_dir, f"{split}_manifest_name_dummy").copy()
    # 上面这一行为了统一变量初始化，下一行会真正覆盖
    existing_relpts = get_existing_relpts(cfg.static_mix_dir, f"{split}_manifest.jsonl")

    random.shuffle(cn_items)
    picked = cn_items[: min(n_add, len(cn_items))]

    out_dir = os.path.join(cfg.static_mix_dir, cfg.out_subdir, split)
    ensure_dir(out_dir)

    manifest_to_append = []
    built = 0
    skipped = 0

    for idx, (spk_id, cn_pt_path) in enumerate(picked):
        try:
            pack = build_single_pack_from_cn_pt(
                cn_pt_path=cn_pt_path,
                spk_id=spk_id,
                max_mix=max_mix,
            )

            base = Path(cn_pt_path).stem
            rel_pt = os.path.join(cfg.out_subdir, split, f"single_{idx:06d}_{base}.pt").replace("\\", "/")

            if cfg.skip_existing and rel_pt in existing_relpts:
                skipped += 1
                continue

            abs_pt = os.path.join(cfg.static_mix_dir, rel_pt)
            ensure_dir(os.path.dirname(abs_pt))
            torch.save(pack, abs_pt)

            manifest_to_append.append({
                "pt": rel_pt,
                "source": "cn_celeb2_single",
            })
            built += 1

        except Exception as e:
            print(f"[WARN] skip {cn_pt_path}: {e}")
            continue

    append_jsonl(static_manifest_path, manifest_to_append)

    print(f"[{split}] added={built}, skipped={skipped}, manifest={static_manifest_path}")


def main():
    cfg = AddSingleSpeakerCfg()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_manifest_path = os.path.join(cfg.static_mix_dir, cfg.train_manifest_name)
    val_manifest_path = os.path.join(cfg.static_mix_dir, cfg.val_manifest_name)

    if not os.path.isfile(train_manifest_path):
        raise FileNotFoundError(train_manifest_path)
    if not os.path.isfile(val_manifest_path):
        raise FileNotFoundError(val_manifest_path)

    if cfg.backup_manifest:
        backup_file(train_manifest_path)
        backup_file(val_manifest_path)

    max_mix = infer_max_mix_from_static_mix(cfg.static_mix_dir, cfg.train_manifest_name)
    print(f"[INFO] inferred max_mix={max_mix}")

    train_cn_items = collect_cn_single_pt(cfg.cn_processed_dir, "train")
    val_cn_items = collect_cn_single_pt(cfg.cn_processed_dir, "val")

    print(f"[INFO] cn train single items = {len(train_cn_items)}")
    print(f"[INFO] cn val single items   = {len(val_cn_items)}")

    add_split(
        split="train",
        cn_items=train_cn_items,
        cfg=cfg,
        max_mix=max_mix,
    )

    add_split(
        split="val",
        cn_items=val_cn_items,
        cfg=cfg,
        max_mix=max_mix,
    )

    print("[ALL DONE]")
    print(f"train manifest: {train_manifest_path}")
    print(f"val manifest:   {val_manifest_path}")


if __name__ == "__main__":
    main()
