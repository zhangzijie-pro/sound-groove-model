import os
import json
import argparse
from typing import List, Dict, Any

import torch
from tqdm import tqdm
import sys


from speaker_verification.models.resowave import ResoWave


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def smooth_sequence(seq: torch.Tensor, min_run: int = 3) -> torch.Tensor:
    if seq.numel() == 0:
        return seq
    seq = seq.clone()
    t = 0
    T = seq.numel()
    while t < T:
        end = t + 1
        while end < T and seq[end] == seq[t]:
            end += 1
        run_len = end - t
        if run_len < min_run:
            left_val = seq[t - 1] if t > 0 else None
            right_val = seq[end] if end < T else None
            if left_val is not None:
                seq[t:end] = left_val
            elif right_val is not None:
                seq[t:end] = right_val
        t = end
    return seq


def remove_short_active_runs(active_mask: torch.Tensor, min_active_frames: int = 3) -> torch.Tensor:
    if active_mask.numel() == 0:
        return active_mask
    active_mask = active_mask.clone()
    t = 0
    T = active_mask.numel()
    while t < T:
        val = bool(active_mask[t].item())
        end = t + 1
        while end < T and bool(active_mask[end].item()) == val:
            end += 1
        run_len = end - t
        if val and run_len < min_active_frames:
            active_mask[t:end] = False
        t = end
    return active_mask


def frames_to_segments(slot_ids: torch.Tensor, active_mask: torch.Tensor, frame_shift_sec: float):
    segments = []
    T = slot_ids.numel()
    t = 0
    while t < T:
        if not active_mask[t]:
            t += 1
            continue
        slot = int(slot_ids[t].item())
        start = t
        t += 1
        while t < T and active_mask[t] and int(slot_ids[t].item()) == slot:
            t += 1
        end = t
        segments.append({
            "slot": slot,
            "start_sec": start * frame_shift_sec,
            "end_sec": end * frame_shift_sec,
            "duration_sec": (end - start) * frame_shift_sec,
        })
    return segments


def write_rttm_line(recording_id: str, start: float, dur: float, speaker: str) -> str:
    # Standard RTTM style:
    # SPEAKER <file-id> <chnl> <tbeg> <tdur> <ortho> <stype> <name> <conf> <slat>
    return f"SPEAKER {recording_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--output_rttm", type=str, required=True)
    parser.add_argument("--output_segments_jsonl", type=str, default=None)

    parser.add_argument("--feat_dim", type=int, default=80)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--emb_dim", type=int, default=192)
    parser.add_argument("--max_mix_speakers", type=int, default=4)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--activity_threshold", type=float, default=0.5)
    parser.add_argument("--frame_shift_sec", type=float, default=0.01)
    parser.add_argument("--min_active_frames", type=int, default=3)
    parser.add_argument("--min_slot_run", type=int, default=3)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = ResoWave(
        in_channels=args.feat_dim,
        channels=args.channels,
        embd_dim=args.emb_dim,
        max_mix_speakers=args.max_mix_speakers,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt.get("model_state") or ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    items = load_jsonl(args.manifest)

    os.makedirs(os.path.dirname(args.output_rttm), exist_ok=True) if os.path.dirname(args.output_rttm) else None
    if args.output_segments_jsonl:
        os.makedirs(os.path.dirname(args.output_segments_jsonl), exist_ok=True) if os.path.dirname(args.output_segments_jsonl) else None

    rttm_lines = []
    debug_segments = []

    for item in tqdm(items, desc="Export RTTM"):
        recording_id = item.get("recording_id") or item.get("utt_id") or item.get("id")
        if recording_id is None:
            raise KeyError(f"manifest item missing recording_id/utt_id/id: {item}")

        feat_path = item.get("feat_path") or item.get("pt_path") or item.get("path")
        if feat_path is None:
            raise KeyError(f"manifest item missing feat_path/pt_path/path: {item}")

        if not os.path.isabs(feat_path):
            feat_path = os.path.join(args.data_root, feat_path)

        sample = torch.load(feat_path, map_location="cpu")
        feat = sample.get("feat") or sample.get("fbank")
        if feat is None:
            raise KeyError(f"{feat_path} missing feat/fbank")

        if feat.dim() != 2:
            raise ValueError(f"Expected [T,F], got {tuple(feat.shape)} from {feat_path}")

        feat = feat.unsqueeze(0).to(device)  # [1,T,F]

        _, frame_embeds, slot_logits, activity_logits, count_logits = model(feat, return_diarization=True)

        slot_logits = slot_logits[0]        # [T,K]
        activity_logits = activity_logits[0]
        count_logits = count_logits[0]

        pred_count = int(count_logits.argmax().item() + 1)

        active_mask = (torch.sigmoid(activity_logits) >= args.activity_threshold)
        active_mask = remove_short_active_runs(active_mask, args.min_active_frames)

        slot_ids = slot_logits.argmax(dim=-1)
        slot_ids = smooth_sequence(slot_ids, args.min_slot_run)

        segments = frames_to_segments(slot_ids, active_mask, args.frame_shift_sec)

        for seg in segments:
            if seg["slot"] >= pred_count:
                continue
            spk_name = f"spk{seg['slot']}"
            rttm_lines.append(
                write_rttm_line(
                    recording_id=recording_id,
                    start=seg["start_sec"],
                    dur=seg["duration_sec"],
                    speaker=spk_name,
                )
            )

        debug_segments.append({
            "recording_id": recording_id,
            "pred_count": pred_count,
            "segments": segments,
        })

    with open(args.output_rttm, "w", encoding="utf-8") as f:
        for line in rttm_lines:
            f.write(line + "\n")

    if args.output_segments_jsonl:
        with open(args.output_segments_jsonl, "w", encoding="utf-8") as f:
            for x in debug_segments:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print("=" * 80)
    print(f"[INFO] RTTM saved to: {args.output_rttm}")
    if args.output_segments_jsonl:
        print(f"[INFO] Segment JSONL saved to: {args.output_segments_jsonl}")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
python export_rttm.py --ckpt outputs3/best.pt --manifest eval_staticmix/val_eval_manifest.jsonl --data_root processed/static_mix_cnceleb2 --output_rttm eval_staticmix/sys.rttm --output_segments_jsonl eval_staticmix/sys_segments.jsonl
"""