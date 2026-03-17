import os
import json
import argparse
from typing import List, Dict, Any

import torch
from tqdm import tqdm


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def frames_to_ref_segments(
    target_matrix: torch.Tensor,   # [T,K]
    speaker_names: List[str],
    frame_shift_sec: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    从 target_matrix 生成参考 segments
    支持 overlap：每个 slot 单独拉段
    """
    assert target_matrix.dim() == 2, f"target_matrix must be [T,K], got {tuple(target_matrix.shape)}"

    T, K = target_matrix.shape
    segments = []

    for k in range(K):
        active = target_matrix[:, k] > 0.5
        if active.sum().item() == 0:
            continue

        spk_name = speaker_names[k] if k < len(speaker_names) else f"spk{k}"

        t = 0
        while t < T:
            if not active[t]:
                t += 1
                continue

            start = t
            t += 1
            while t < T and active[t]:
                t += 1
            end = t

            segments.append({
                "speaker": spk_name,
                "slot": k,
                "start_sec": round(start * frame_shift_sec, 6),
                "end_sec": round(end * frame_shift_sec, 6),
                "duration_sec": round((end - start) * frame_shift_sec, 6),
            })

    return segments


def write_rttm_line(recording_id: str, start: float, dur: float, speaker: str) -> str:
    return f"SPEAKER {recording_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_out_dir", type=str, required=True, help="e.g. processed/static_mix_cnceleb2")
    parser.add_argument("--manifest", type=str, required=True, help="e.g. val_manifest.jsonl")
    parser.add_argument("--split_name", type=str, default="val")
    parser.add_argument("--frame_shift_sec", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, required=True, help="directory to save eval files")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    manifest_path = args.manifest
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(args.data_out_dir, manifest_path)

    items = load_jsonl(manifest_path)

    eval_manifest_path = os.path.join(args.output_dir, f"{args.split_name}_eval_manifest.jsonl")
    ref_rttm_path = os.path.join(args.output_dir, f"{args.split_name}_ref.rttm")
    uem_path = os.path.join(args.output_dir, f"{args.split_name}.uem")
    segments_jsonl_path = os.path.join(args.output_dir, f"{args.split_name}_ref_segments.jsonl")

    n_ok = 0
    n_skip = 0

    with open(eval_manifest_path, "w", encoding="utf-8") as mf, \
         open(ref_rttm_path, "w", encoding="utf-8") as rf, \
         open(uem_path, "w", encoding="utf-8") as uf, \
         open(segments_jsonl_path, "w", encoding="utf-8") as sf:

        for idx, item in enumerate(tqdm(items, desc="Preparing staticmix eval")):
            rel_pt = item.get("pt") or item.get("pt_path") or item.get("path")
            if rel_pt is None:
                n_skip += 1
                continue

            abs_pt = rel_pt
            if not os.path.isabs(abs_pt):
                abs_pt = os.path.join(args.data_out_dir, rel_pt)

            if not os.path.isfile(abs_pt):
                print(f"[WARN] Missing file: {abs_pt}")
                n_skip += 1
                continue

            pack = torch.load(abs_pt, map_location="cpu")

            target_matrix = pack.get("target_matrix", None)
            fbank = pack.get("fbank", None)
            speaker_names = pack.get("speaker_names", None)

            if target_matrix is None or fbank is None:
                print(f"[WARN] Missing target_matrix/fbank in: {abs_pt}")
                n_skip += 1
                continue

            if not torch.is_tensor(target_matrix):
                target_matrix = torch.tensor(target_matrix)
            if not torch.is_tensor(fbank):
                fbank = torch.tensor(fbank)

            target_matrix = target_matrix.float()
            fbank = fbank.float()

            if target_matrix.dim() != 2:
                print(f"[WARN] Unexpected target_matrix shape: {tuple(target_matrix.shape)} in {abs_pt}")
                n_skip += 1
                continue

            T = fbank.size(0)
            dur_sec = T * args.frame_shift_sec

            if speaker_names is None:
                # 兜底：如果没有 speaker_names，就用 slot 名
                speaker_names = [f"spk{k}" for k in range(target_matrix.size(1))]

            recording_id = f"staticmix_{args.split_name}_{idx:08d}"

            # 1) eval manifest
            out_item = {
                "recording_id": recording_id,
                "pt_path": rel_pt.replace("\\", "/"),
                "num_frames": int(T),
                "duration_sec": round(dur_sec, 6),
            }
            mf.write(json.dumps(out_item, ensure_ascii=False) + "\n")

            # 2) UEM
            uf.write(f"{recording_id} 1 0.000 {dur_sec:.3f}\n")

            # 3) reference segments
            segments = frames_to_ref_segments(
                target_matrix=target_matrix,
                speaker_names=speaker_names,
                frame_shift_sec=args.frame_shift_sec,
            )

            for seg in segments:
                rf.write(
                    write_rttm_line(
                        recording_id=recording_id,
                        start=seg["start_sec"],
                        dur=seg["duration_sec"],
                        speaker=seg["speaker"],
                    ) + "\n"
                )

            sf.write(json.dumps({
                "recording_id": recording_id,
                "pt_path": rel_pt.replace("\\", "/"),
                "segments": segments,
            }, ensure_ascii=False) + "\n")

            n_ok += 1

    print("=" * 80)
    print(f"[INFO] Done.")
    print(f"[INFO] OK items         : {n_ok}")
    print(f"[INFO] Skipped items    : {n_skip}")
    print(f"[INFO] Eval manifest    : {eval_manifest_path}")
    print(f"[INFO] Reference RTTM   : {ref_rttm_path}")
    print(f"[INFO] UEM             : {uem_path}")
    print(f"[INFO] Debug segments   : {segments_jsonl_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()


"""
python scripts/prepare_mix_eval.py --data_out_dir processed/static_mix_cnceleb2 --manifest val_manifest.jsonl --split_name val  --output_dir eval_staticmix
"""