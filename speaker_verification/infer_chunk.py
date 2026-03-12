import os
import json
import argparse

import torch
import torch.nn.functional as F

from speaker_verification.models.resowave import ResoWave


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
    if bank_path is None:
        return None

    bank = torch.load(bank_path, map_location=device)
    names = bank["names"]
    embs = torch.as_tensor(bank["embeddings"], dtype=torch.float32, device=device)
    embs = F.normalize(embs, p=2, dim=-1)
    return {"names": names, "embeddings": embs}


def identify_prototypes(protos, bank, sim_th=0.45):
    if bank is None or protos is None or protos.numel() == 0:
        return []

    sims = cosine_sim(protos, bank["embeddings"])   # [S,N]
    vals, idxs = sims.max(dim=-1)

    out = []
    for i in range(protos.size(0)):
        name = bank["names"][idxs[i].item()] if vals[i].item() >= sim_th else "unknown"
        out.append({
            "name": name,
            "score": float(vals[i].item()),
        })
    return out


@torch.no_grad()
def infer_one(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(
        args.ckpt,
        device=device,
        channels=args.channels,
        embd_dim=args.embd_dim,
        max_mix_speakers=args.max_mix_speakers,
    )
    bank = load_speaker_bank(args.speaker_bank, device=device)

    sample = torch.load(args.input_pt, map_location="cpu")
    feat = sample["feat"].float().unsqueeze(0).to(device)  # [1,T,80]

    emb, frame_embeds, slot_logits, activity_logits, count_logits = model(feat, return_diarization=True)

    frame_embeds = frame_embeds[0]            # [T,D]
    slot_logits = slot_logits[0]              # [T,K]
    activity_logits = activity_logits[0]      # [T]
    count_logits = count_logits[0]            # [K]

    pred_count = int(count_logits.argmax().item() + 1)
    pred_slot = slot_logits.argmax(dim=-1)    # [T]
    pred_active = (torch.sigmoid(activity_logits) >= args.activity_th)

    # 只保留前 pred_count 个槽位
    slot_results = []
    protos = []

    for k in range(pred_count):
        mask = (pred_slot == k) & pred_active
        dur = int(mask.sum().item())
        if dur <= 0:
            continue

        proto = frame_embeds[mask].mean(dim=0)
        proto = F.normalize(proto, p=2, dim=-1)
        protos.append(proto)
        slot_results.append({
            "slot": k,
            "duration_frames": dur,
        })

    if len(protos) > 0:
        protos = torch.stack(protos, dim=0)
    else:
        protos = None

    identity_results = identify_prototypes(protos, bank, sim_th=args.sim_th)

    for i in range(len(slot_results)):
        if i < len(identity_results):
            slot_results[i]["name"] = identity_results[i]["name"]
            slot_results[i]["score"] = identity_results[i]["score"]
        else:
            slot_results[i]["name"] = f"slot_{slot_results[i]['slot']}"
            slot_results[i]["score"] = None

    dominant = None
    if len(slot_results) > 0:
        dominant = max(slot_results, key=lambda x: x["duration_frames"])["name"]

    result = {
        "num_speakers": pred_count,
        "speakers": slot_results,
        "dominant_speaker": dominant,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_pt", type=str, required=True)
    parser.add_argument("--speaker_bank", type=str, default=None)

    parser.add_argument("--sim_th", type=float, default=0.45)
    parser.add_argument("--activity_th", type=float, default=0.5)

    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--embd_dim", type=int, default=192)
    parser.add_argument("--max_mix_speakers", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer_one(args)