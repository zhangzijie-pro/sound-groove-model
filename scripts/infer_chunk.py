import json
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from speaker_verification.models.resowave import ResoWave


class SpeakerBankBase:
    def identify(self, embedding: torch.Tensor) -> Dict[str, Any]:
        """
        输入:
            embedding: [D]
        返回:
            {
                "name": "zhangsan",
                "score": 0.87,
                "is_known": True
            }
        """
        raise NotImplementedError


class DummySpeakerBank(SpeakerBankBase):
    def identify(self, embedding: torch.Tensor) -> Dict[str, Any]:
        return {
            "name": "unknown",
            "score": None,
            "is_known": False,
        }


def load_model(
    ckpt_path: str,
    device: str = "cuda",
    feat_dim: int = 80,
    channels: int = 512,
    emb_dim: int = 192,
    max_mix_speakers: int = 4,
) -> ResoWave:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    model = ResoWave(
        in_channels=feat_dim,
        channels=channels,
        embd_dim=emb_dim,
        max_mix_speakers=max_mix_speakers,
    ).to(dev)

    ckpt = torch.load(ckpt_path, map_location=dev)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def smooth_sequence(seq: torch.Tensor, min_run: int = 3) -> torch.Tensor:
    """
    seq: [T] int64
    """
    if seq.numel() == 0:
        return seq

    seq = seq.clone()
    T = seq.numel()

    start = 0
    while start < T:
        end = start + 1
        while end < T and seq[end] == seq[start]:
            end += 1

        run_len = end - start
        if run_len < min_run:
            left_val = seq[start - 1] if start > 0 else None
            right_val = seq[end] if end < T else None

            if left_val is not None and right_val is not None:
                fill_val = left_val if run_len <= 1 else right_val
            elif left_val is not None:
                fill_val = left_val
            elif right_val is not None:
                fill_val = right_val
            else:
                fill_val = seq[start]

            seq[start:end] = fill_val

        start = end

    return seq


def mask_short_active_runs(active_mask: torch.Tensor, min_active_frames: int = 3) -> torch.Tensor:
    """
    去掉极短 active burst
    active_mask: [T] bool
    """
    if active_mask.numel() == 0:
        return active_mask

    active_mask = active_mask.clone()
    T = active_mask.numel()

    start = 0
    while start < T:
        val = active_mask[start].item()
        end = start + 1
        while end < T and active_mask[end].item() == val:
            end += 1

        run_len = end - start
        if val and run_len < min_active_frames:
            active_mask[start:end] = False

        start = end

    return active_mask


def frames_to_segments(
    slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
    frame_shift_sec: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    把逐帧 slot + active 转成时间段
    """
    segments: List[Dict[str, Any]] = []
    T = slot_ids.numel()

    t = 0
    while t < T:
        if not active_mask[t]:
            t += 1
            continue

        spk = int(slot_ids[t].item())
        start = t
        t += 1
        while t < T and active_mask[t] and int(slot_ids[t].item()) == spk:
            t += 1
        end = t

        segments.append({
            "slot": spk,
            "start_sec": round(start * frame_shift_sec, 4),
            "end_sec": round(end * frame_shift_sec, 4),
            "duration_sec": round((end - start) * frame_shift_sec, 4),
        })

    return segments


def build_slot_prototypes(
    frame_embeds: torch.Tensor,
    slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
    pred_count: int,
) -> List[Dict[str, Any]]:
    """
    对每个 slot 聚合 prototype
    frame_embeds: [T,D]
    slot_ids: [T]
    active_mask: [T]
    """
    results: List[Dict[str, Any]] = []

    for k in range(pred_count):
        mask = (slot_ids == k) & active_mask
        num_frames = int(mask.sum().item())
        if num_frames <= 0:
            continue

        proto = frame_embeds[mask].mean(dim=0)
        proto = F.normalize(proto, p=2, dim=-1)

        results.append({
            "slot": k,
            "num_frames": num_frames,
            "prototype": proto,
        })

    return results


@torch.no_grad()
def infer_chunk(
    model: ResoWave,
    fbank: torch.Tensor,
    device: str = "cuda",
    activity_threshold: float = 0.5,
    frame_shift_sec: float = 0.01,
    min_active_frames: int = 3,
    min_slot_run: int = 3,
    speaker_bank: Optional[SpeakerBankBase] = None,
) -> Dict[str, Any]:
    """
    输入:
        fbank: [T,80] 或 [1,T,80]

    输出:
        {
            "num_speakers": int,
            "dominant_speaker": str,
            "slots": [...],
            "segments": [...],
            "activity_ratio": float,
        }
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    speaker_bank = speaker_bank or DummySpeakerBank()

    if fbank.dim() == 2:
        fbank = fbank.unsqueeze(0)  # [1,T,80]
    elif fbank.dim() != 3:
        raise ValueError(f"fbank shape must be [T,F] or [B,T,F], got {tuple(fbank.shape)}")

    fbank = fbank.to(dev)

    emb, frame_embeds, slot_logits, activity_logits, count_logits = model(fbank, return_diarization=True)

    frame_embeds = frame_embeds[0]          # [T,D]
    slot_logits = slot_logits[0]            # [T,K]
    activity_logits = activity_logits[0]    # [T]
    count_logits = count_logits[0]          # [K]

    pred_count = int(count_logits.argmax(dim=0).item() + 1)

    active_prob = torch.sigmoid(activity_logits)
    active_mask = active_prob >= activity_threshold
    active_mask = mask_short_active_runs(active_mask, min_active_frames=min_active_frames)

    slot_ids = slot_logits.argmax(dim=-1)   # [T]
    slot_ids = smooth_sequence(slot_ids, min_run=min_slot_run)

    segments = frames_to_segments(
        slot_ids=slot_ids,
        active_mask=active_mask,
        frame_shift_sec=frame_shift_sec,
    )

    slot_infos = build_slot_prototypes(
        frame_embeds=frame_embeds,
        slot_ids=slot_ids,
        active_mask=active_mask,
        pred_count=pred_count,
    )

    output_slots: List[Dict[str, Any]] = []
    dominant_name = "unknown"
    max_frames = -1

    for info in slot_infos:
        spk_ret = speaker_bank.identify(info["prototype"])
        duration_sec = info["num_frames"] * frame_shift_sec

        slot_item = {
            "slot": info["slot"],
            "name": spk_ret.get("name", f"slot_{info['slot']}"),
            "score": spk_ret.get("score", None),
            "is_known": spk_ret.get("is_known", False),
            "num_frames": info["num_frames"],
            "duration_sec": round(duration_sec, 4),
            "prototype": info["prototype"],
        }
        output_slots.append(slot_item)

        if info["num_frames"] > max_frames:
            max_frames = info["num_frames"]
            dominant_name = slot_item["name"]

    slot_to_name = {x["slot"]: x["name"] for x in output_slots}
    for seg in segments:
        seg["name"] = slot_to_name.get(seg["slot"], f"slot_{seg['slot']}")

    activity_ratio = float(active_mask.float().mean().item())

    return {
        "num_speakers": pred_count,
        "dominant_speaker": dominant_name,
        "activity_ratio": round(activity_ratio, 6),
        "slots": output_slots,
        "segments": segments,
        "frame_activity_prob": active_prob.detach().cpu(),
        "frame_slot_ids": slot_ids.detach().cpu(),
    }


def export_result_to_jsonable(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    去掉 tensor，方便存 json
    """
    jsonable = {
        "num_speakers": result["num_speakers"],
        "dominant_speaker": result["dominant_speaker"],
        "activity_ratio": result["activity_ratio"],
        "slots": [],
        "segments": result["segments"],
    }

    for slot in result["slots"]:
        jsonable["slots"].append({
            "slot": slot["slot"],
            "name": slot["name"],
            "score": slot["score"],
            "is_known": slot["is_known"],
            "num_frames": slot["num_frames"],
            "duration_sec": slot["duration_sec"],
        })

    return jsonable


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_pt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feat_dim", type=int, default=80)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--emb_dim", type=int, default=192)
    parser.add_argument("--max_mix_speakers", type=int, default=4)
    parser.add_argument("--activity_threshold", type=float, default=0.5)
    parser.add_argument("--frame_shift_sec", type=float, default=0.01)
    args = parser.parse_args()

    model = load_model(
        ckpt_path=args.ckpt,
        device=args.device,
        feat_dim=args.feat_dim,
        channels=args.channels,
        emb_dim=args.emb_dim,
        max_mix_speakers=args.max_mix_speakers,
    )

    sample = torch.load(args.input_pt, map_location="cpu")
    if "fbank" in sample:
        fbank = sample["fbank"]
    elif "feat" in sample:
        fbank = sample["feat"]
    else:
        raise KeyError("input_pt must contain 'fbank' or 'feat'")

    result = infer_chunk(
        model=model,
        fbank=fbank,
        device=args.device,
        activity_threshold=args.activity_threshold,
        frame_shift_sec=args.frame_shift_sec,
        speaker_bank=None,
    )

    print(json.dumps(export_result_to_jsonable(result), ensure_ascii=False, indent=2))