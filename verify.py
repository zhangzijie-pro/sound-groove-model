import os
import json
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from speaker_verification.models.resowave import ResoWave


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class StaticMixEvalDataset(Dataset):
    def __init__(self, root_dir, manifest_path):
        self.root_dir = root_dir
        self.items = load_jsonl(manifest_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        meta = self.items[idx]

        # 常见字段兜底
        pt_path = meta.get("pt_path") or meta.get("feat_path") or meta.get("path")
        if pt_path is None:
            raise KeyError(f"manifest item missing pt_path/feat_path/path: {meta}")

        if not os.path.isabs(pt_path):
            pt_path = os.path.join(self.root_dir, pt_path)

        sample = torch.load(pt_path, map_location="cpu")

        feat = sample["feat"].float()                       # [T,80]
        target_matrix = sample["target_matrix"].float()    # [T,K]
        target_activity = sample["target_activity"].float()# [T]
        target_count = torch.tensor(sample["target_count"]).long()

        speakers = (
            sample.get("speakers")
            or sample.get("speaker_ids")
            or meta.get("speakers")
            or meta.get("speaker_ids")
            or None
        )

        return {
            "feat": feat,
            "target_matrix": target_matrix,
            "target_activity": target_activity,
            "target_count": target_count,
            "speakers": speakers,
            "meta": meta,
        }


def collate_fn(batch):
    B = len(batch)
    T_max = max(x["feat"].size(0) for x in batch)
    F_dim = batch[0]["feat"].size(1)
    K = batch[0]["target_matrix"].size(1)

    feats = torch.zeros(B, T_max, F_dim, dtype=torch.float32)
    target_matrix = torch.zeros(B, T_max, K, dtype=torch.float32)
    target_activity = torch.zeros(B, T_max, dtype=torch.float32)
    target_count = torch.zeros(B, dtype=torch.long)
    valid_mask = torch.zeros(B, T_max, dtype=torch.bool)

    speakers = []
    metas = []

    for i, x in enumerate(batch):
        t = x["feat"].size(0)
        feats[i, :t] = x["feat"]
        target_matrix[i, :t] = x["target_matrix"]
        target_activity[i, :t] = x["target_activity"]
        target_count[i] = x["target_count"]
        valid_mask[i, :t] = True
        speakers.append(x["speakers"])
        metas.append(x["meta"])

    return {
        "feat": feats,
        "target_matrix": target_matrix,
        "target_activity": target_activity,
        "target_count": target_count,
        "valid_mask": valid_mask,
        "speakers": speakers,
        "metas": metas,
    }


def cosine_sim(a, b):
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return torch.matmul(a, b.transpose(-1, -2))


def load_model(ckpt_path, device, channels=512, embd_dim=192, max_mix_speakers=5):
    model = ResoWave(
        in_channels=80,
        channels=channels,
        embd_dim=embd_dim,
        max_mix_speakers=max_mix_speakers,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_speaker_bank(bank_path, device):
    """
    bank 格式建议：
    {
        "names": ["zhangsan", "lisi", ...],
        "embeddings": tensor/list, shape [N,D]
    }
    """
    if bank_path is None:
        return None

    bank = torch.load(bank_path, map_location=device)
    names = bank["names"]
    embs = torch.as_tensor(bank["embeddings"], dtype=torch.float32, device=device)
    embs = F.normalize(embs, p=2, dim=-1)
    return {"names": names, "embeddings": embs}


def decode_predictions(slot_logits, activity_logits, count_logits, valid_mask, activity_th=0.5):
    """
    返回：
      pred_slot_id: [B,T] argmax 槽位
      pred_active:  [B,T] bool
      pred_count:   [B] 1..K
    """
    slot_id = slot_logits.argmax(dim=-1)  # [B,T]
    act_prob = torch.sigmoid(activity_logits)
    pred_active = act_prob >= activity_th

    pred_count = count_logits.argmax(dim=-1) + 1  # [B]
    pred_active = pred_active & valid_mask

    return slot_id, pred_active, pred_count


def build_slot_prototypes(frame_embeds, slot_id, pred_active, pred_count):
    """
    对每个样本构造预测槽位 prototype
    返回 list[dict]:
      [
        {
          "slot_ids": [...],
          "prototypes": [S,D],
          "durations": [S]
        },
        ...
      ]
    """
    B, T, D = frame_embeds.shape
    out = []

    for b in range(B):
        fb = frame_embeds[b]          # [T,D]
        sb = slot_id[b]               # [T]
        ab = pred_active[b]           # [T]
        num_spk = int(pred_count[b].item())

        slot_stats = []
        for k in range(num_spk):
            mask = (sb == k) & ab
            dur = int(mask.sum().item())
            if dur <= 0:
                continue

            proto = fb[mask].mean(dim=0)
            proto = F.normalize(proto, p=2, dim=-1)
            slot_stats.append((k, proto, dur))

        if len(slot_stats) == 0:
            out.append({"slot_ids": [], "prototypes": None, "durations": []})
        else:
            slot_ids = [x[0] for x in slot_stats]
            protos = torch.stack([x[1] for x in slot_stats], dim=0)
            durs = [x[2] for x in slot_stats]
            out.append({"slot_ids": slot_ids, "prototypes": protos, "durations": durs})

    return out


def identify_slots(slot_pack, bank, sim_th=0.45):
    if bank is None:
        return None

    results = []
    names = bank["names"]
    bank_embs = bank["embeddings"]

    for item in slot_pack:
        protos = item["prototypes"]
        if protos is None or protos.numel() == 0:
            results.append([])
            continue

        sims = cosine_sim(protos, bank_embs)   # [S,N]
        vals, idxs = sims.max(dim=-1)

        sample_ret = []
        for i in range(protos.size(0)):
            name = names[idxs[i].item()] if vals[i].item() >= sim_th else "unknown"
            sample_ret.append({
                "slot": item["slot_ids"][i],
                "name": name,
                "score": float(vals[i].item()),
                "duration_frames": int(item["durations"][i]),
            })
        results.append(sample_ret)

    return results


def compute_activity_f1(pred_active, target_activity, valid_mask):
    pa = pred_active[valid_mask]
    ta = target_activity.bool()[valid_mask]

    tp = ((pa == 1) & (ta == 1)).sum().item()
    fp = ((pa == 1) & (ta == 0)).sum().item()
    fn = ((pa == 0) & (ta == 1)).sum().item()

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return prec, rec, f1


def compute_simple_slot_accuracy(slot_logits, target_matrix, valid_mask):
    pred_slot = slot_logits.argmax(dim=-1)
    gt_slot = target_matrix.argmax(dim=-1)

    mask = valid_mask & (target_matrix.sum(dim=-1) > 0)
    if mask.sum().item() == 0:
        return 0.0

    acc = (pred_slot[mask] == gt_slot[mask]).float().mean().item()
    return acc


def compute_dominant_speaker(slot_identity_result):
    """
    根据 duration_frames 最大者作为主说话人
    """
    if slot_identity_result is None or len(slot_identity_result) == 0:
        return None

    best = max(slot_identity_result, key=lambda x: x["duration_frames"])
    return best["name"]


@torch.no_grad()
def verify(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(
        args.ckpt,
        device=device,
        channels=args.channels,
        embd_dim=args.embd_dim,
        max_mix_speakers=args.max_mix_speakers,
    )
    bank = load_speaker_bank(args.speaker_bank, device=device)

    dataset = StaticMixEvalDataset(args.data_root, args.manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    total_count_correct = 0
    total_count_num = 0

    slot_acc_sum = 0.0
    slot_acc_n = 0

    act_p_sum = 0.0
    act_r_sum = 0.0
    act_f1_sum = 0.0
    act_n = 0

    examples = []

    for batch in loader:
        feat = batch["feat"].to(device)
        target_matrix = batch["target_matrix"].to(device)
        target_activity = batch["target_activity"].to(device)
        target_count = batch["target_count"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        emb, frame_embeds, slot_logits, activity_logits, count_logits = model(feat, return_diarization=True)

        pred_slot_id, pred_active, pred_count = decode_predictions(
            slot_logits, activity_logits, count_logits, valid_mask, activity_th=args.activity_th
        )

        # count acc
        gt_count = target_count
        total_count_correct += (pred_count == gt_count).sum().item()
        total_count_num += gt_count.numel()

        # activity
        p, r, f1 = compute_activity_f1(pred_active, target_activity, valid_mask)
        act_p_sum += p
        act_r_sum += r
        act_f1_sum += f1
        act_n += 1

        # 简单 slot acc
        sacc = compute_simple_slot_accuracy(slot_logits, target_matrix, valid_mask)
        slot_acc_sum += sacc
        slot_acc_n += 1

        # identity + dominant
        slot_pack = build_slot_prototypes(frame_embeds, pred_slot_id, pred_active, pred_count)
        identity_results = identify_slots(slot_pack, bank, sim_th=args.sim_th)

        for i in range(min(len(identity_results) if identity_results is not None else 0, args.show_examples)):
            dom = compute_dominant_speaker(identity_results[i])
            examples.append({
                "pred_count": int(pred_count[i].item()),
                "identified": identity_results[i],
                "dominant_speaker": dom,
                "meta": batch["metas"][i],
            })

    print("=" * 100)
    print("Verification Summary")
    print(f"Count Accuracy        : {total_count_correct / max(total_count_num, 1):.4f}")
    print(f"Activity Precision    : {act_p_sum / max(act_n, 1):.4f}")
    print(f"Activity Recall       : {act_r_sum / max(act_n, 1):.4f}")
    print(f"Activity F1           : {act_f1_sum / max(act_n, 1):.4f}")
    print(f"Simple Slot Accuracy  : {slot_acc_sum / max(slot_acc_n, 1):.4f}")
    print("=" * 100)

    if len(examples) > 0:
        print("Examples:")
        for i, ex in enumerate(examples[:args.show_examples]):
            print(f"[Example {i}] pred_count={ex['pred_count']}, dominant={ex['dominant_speaker']}")
            print(f"  identified={ex['identified']}")
            print(f"  meta={ex['meta']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)

    parser.add_argument("--speaker_bank", type=str, default=None)
    parser.add_argument("--sim_th", type=float, default=0.45)
    parser.add_argument("--activity_th", type=float, default=0.5)

    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--embd_dim", type=int, default=192)
    parser.add_argument("--max_mix_speakers", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--show_examples", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify(args)