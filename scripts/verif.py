import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from scipy.optimize import linear_sum_assignment

from speaker_verification.audio.features import TARGET_SR, load_wav_mono
from speaker_verification.interfaces.diar_interface import SpeakerAwareDiarizationInterface
from speaker_verification.interfaces.global_tracker import GlobalSpeakerTracker


@dataclass
class EvalCase:
    wav_path: Path
    rttm_path: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a trained diarization model on one wav or a manifest of wavs."
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path, usually outputs/.../best.pt")
    parser.add_argument("--wav", type=Path, help="Single wav/flac file to verify")
    parser.add_argument("--rttm", type=Path, help="Optional RTTM reference for the single wav")
    parser.add_argument("--manifest", type=Path, help="Optional JSONL manifest with wav/rttm fields for batch verify")
    parser.add_argument("--output", type=Path, default=Path("outputs/verif_results.json"), help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--chunk-sec", type=float, default=3.0, help="Chunk size for long-audio verification")
    parser.add_argument("--step-sec", type=float, default=1.5, help="Chunk step for long-audio verification")
    parser.add_argument("--merge-gap-sec", type=float, default=0.15, help="Merge adjacent same-speaker segments within this gap")
    parser.add_argument("--activity-threshold", type=float, default=None, help="Override activity threshold from checkpoint config")
    parser.add_argument("--slot-threshold", type=float, default=None, help="Override slot threshold from checkpoint config")
    parser.add_argument("--min-active-frames", type=int, default=None, help="Override minimum active-frame run")
    parser.add_argument("--min-slot-run", type=int, default=None, help="Override dominant slot smoothing run")
    parser.add_argument("--slot-presence-frames", type=int, default=None, help="Override minimum frames needed to keep a speaker slot")
    parser.add_argument("--fill-gap-frames", type=int, default=None, help="Override short-gap filling length")
    parser.add_argument("--slot-merge-threshold", type=float, default=0.92, help="Prototype cosine threshold to merge local slots")
    parser.add_argument("--tracker-match-threshold", type=float, default=0.68, help="Cross-chunk tracker match threshold")
    parser.add_argument("--tracker-momentum", type=float, default=0.92, help="Cross-chunk tracker momentum")
    parser.add_argument("--tracker-max-misses", type=int, default=40, help="How many unmatched chunks to keep a global track")
    parser.add_argument("--normalize", dest="normalize", action="store_true", help="Peak-normalize the input wav before inference")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable peak normalization")
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    if args.wav is None and args.manifest is None:
        raise ValueError("Provide either --wav or --manifest.")
    if args.wav is not None and args.manifest is not None:
        raise ValueError("Use either --wav or --manifest, not both.")
    if args.chunk_sec <= 0 or args.step_sec <= 0:
        raise ValueError("--chunk-sec and --step-sec must be positive.")
    return args


def load_cases(args: argparse.Namespace) -> List[EvalCase]:
    if args.wav is not None:
        return [EvalCase(wav_path=args.wav.expanduser().resolve(), rttm_path=args.rttm.expanduser().resolve() if args.rttm else None)]

    cases: List[EvalCase] = []
    with args.manifest.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            wav_path = item.get("wav") or item.get("wav_path")
            if not wav_path:
                raise ValueError(f"Missing wav path in manifest item: {item}")
            rttm_path = item.get("rttm") or item.get("rttm_path")
            cases.append(
                EvalCase(
                    wav_path=Path(wav_path).expanduser().resolve(),
                    rttm_path=Path(rttm_path).expanduser().resolve() if rttm_path else None,
                )
            )
    return cases


def parse_rttm(rttm_path: Path) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    with rttm_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue
            start_sec = float(parts[3])
            duration_sec = float(parts[4])
            speaker = parts[7]
            segments.append(
                {
                    "speaker": speaker,
                    "start_sec": start_sec,
                    "end_sec": start_sec + duration_sec,
                    "duration_sec": duration_sec,
                }
            )
    return segments


def build_windows(total_samples: int, chunk_samples: int, step_samples: int) -> List[tuple[int, int, float, float]]:
    if total_samples <= chunk_samples:
        total_sec = total_samples / TARGET_SR
        return [(0, total_samples, 0.0, total_sec)]

    starts = list(range(0, max(total_samples - chunk_samples + 1, 1), step_samples))
    last_start = max(0, total_samples - chunk_samples)
    if starts[-1] != last_start:
        starts.append(last_start)

    overlap = max(0.0, float(chunk_samples - step_samples) / TARGET_SR)
    half_overlap = overlap / 2.0
    total_sec = total_samples / TARGET_SR
    windows = []
    for idx, start in enumerate(starts):
        start_sec = start / TARGET_SR
        end_sec = min(total_sec, start_sec + chunk_samples / TARGET_SR)
        emit_start = start_sec if idx == 0 else start_sec + half_overlap
        emit_end = end_sec if idx == len(starts) - 1 else end_sec - half_overlap
        if emit_end <= emit_start:
            emit_start = start_sec
            emit_end = end_sec
        windows.append((start, min(total_samples, start + chunk_samples), emit_start, emit_end))
    return windows


def clip_segments(segments: List[Dict[str, Any]], emit_start: float, emit_end: float) -> List[Dict[str, Any]]:
    clipped: List[Dict[str, Any]] = []
    for seg in segments:
        start_sec = max(float(seg["start_sec"]), emit_start)
        end_sec = min(float(seg["end_sec"]), emit_end)
        if end_sec <= start_sec:
            continue
        out = dict(seg)
        out["start_sec"] = round(start_sec, 4)
        out["end_sec"] = round(end_sec, 4)
        out["duration_sec"] = round(end_sec - start_sec, 4)
        clipped.append(out)
    return clipped


def merge_segments(segments: List[Dict[str, Any]], max_gap_sec: float) -> List[Dict[str, Any]]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: (item["start_sec"], item["global_id"]))
    merged: List[Dict[str, Any]] = [dict(ordered[0])]
    for seg in ordered[1:]:
        prev = merged[-1]
        same_speaker = int(prev["global_id"]) == int(seg["global_id"])
        close_enough = float(seg["start_sec"]) - float(prev["end_sec"]) <= max_gap_sec
        if same_speaker and close_enough:
            prev["end_sec"] = round(max(float(prev["end_sec"]), float(seg["end_sec"])), 4)
            prev["duration_sec"] = round(float(prev["end_sec"]) - float(prev["start_sec"]), 4)
            continue
        merged.append(dict(seg))
    return merged


def build_frame_matrix(
    segments: List[Dict[str, Any]],
    duration_sec: float,
    frame_shift_sec: float,
    speaker_key: str,
) -> tuple[torch.Tensor, List[Any]]:
    total_frames = max(1, int(math.ceil(duration_sec / frame_shift_sec)))
    speaker_ids = sorted({seg[speaker_key] for seg in segments})
    matrix = torch.zeros(total_frames, len(speaker_ids), dtype=torch.bool)
    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(speaker_ids)}

    for seg in segments:
        start = max(0, int(math.floor(float(seg["start_sec"]) / frame_shift_sec)))
        end = min(total_frames, int(math.ceil(float(seg["end_sec"]) / frame_shift_sec)))
        if end <= start:
            continue
        matrix[start:end, speaker_to_idx[seg[speaker_key]]] = True
    return matrix, speaker_ids


def compute_framewise_der(
    pred_segments: List[Dict[str, Any]],
    ref_segments: List[Dict[str, Any]],
    duration_sec: float,
    frame_shift_sec: float = 0.01,
) -> Dict[str, float]:
    pred_matrix, pred_ids = build_frame_matrix(pred_segments, duration_sec, frame_shift_sec, speaker_key="global_id")
    ref_matrix, ref_ids = build_frame_matrix(ref_segments, duration_sec, frame_shift_sec, speaker_key="speaker")

    pred_activity = pred_matrix.any(dim=-1) if pred_matrix.numel() > 0 else torch.zeros(pred_matrix.size(0), dtype=torch.bool)
    ref_activity = ref_matrix.any(dim=-1) if ref_matrix.numel() > 0 else torch.zeros(ref_matrix.size(0), dtype=torch.bool)

    fa = float((pred_activity & ~ref_activity).sum().item())
    miss = float((~pred_activity & ref_activity).sum().item())
    gt_active = float(ref_activity.sum().item())
    pred_active = float(pred_activity.sum().item())

    if pred_matrix.size(1) == 0 or ref_matrix.size(1) == 0:
        conf = 0.0 if pred_matrix.size(1) == 0 and ref_matrix.size(1) == 0 else float((pred_activity & ref_activity).sum().item())
        der = (fa + miss + conf) / max(gt_active, 1.0)
        return {"der": der, "fa": fa, "miss": miss, "conf": conf, "gt_active": gt_active, "pred_active": pred_active}

    cost = torch.zeros(pred_matrix.size(1), ref_matrix.size(1), dtype=torch.float32)
    for pred_idx in range(pred_matrix.size(1)):
        for ref_idx in range(ref_matrix.size(1)):
            cost[pred_idx, ref_idx] = (pred_matrix[:, pred_idx] ^ ref_matrix[:, ref_idx]).float().sum()

    row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
    aligned_pred = torch.zeros_like(ref_matrix)
    matched_pred = set()
    for pred_idx, ref_idx in zip(row_ind.tolist(), col_ind.tolist()):
        aligned_pred[:, ref_idx] = pred_matrix[:, pred_idx]
        matched_pred.add(pred_idx)

    unmatched_pred = [idx for idx in range(pred_matrix.size(1)) if idx not in matched_pred]
    if unmatched_pred:
        unmatched_pred_activity = pred_matrix[:, unmatched_pred].any(dim=-1)
    else:
        unmatched_pred_activity = torch.zeros(ref_matrix.size(0), dtype=torch.bool)

    conf_mask = (((aligned_pred != ref_matrix).any(dim=-1)) | unmatched_pred_activity) & ref_activity
    conf = float(conf_mask.sum().item())
    der = (fa + miss + conf) / max(gt_active, 1.0)
    return {"der": der, "fa": fa, "miss": miss, "conf": conf, "gt_active": gt_active, "pred_active": pred_active}


def summarize_segments(segments: List[Dict[str, Any]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for seg in segments:
        key = f"speaker_{int(seg['global_id'])}"
        summary[key] = summary.get(key, 0.0) + float(seg["duration_sec"])
    return {key: round(value, 4) for key, value in sorted(summary.items())}


def build_interface(args: argparse.Namespace) -> SpeakerAwareDiarizationInterface:
    return SpeakerAwareDiarizationInterface(
        ckpt_path=str(args.ckpt),
        device=args.device,
        activity_threshold=args.activity_threshold if args.activity_threshold is not None else 0.5,
        slot_threshold=args.slot_threshold,
        min_active_frames=args.min_active_frames if args.min_active_frames is not None else 3,
        min_slot_run=args.min_slot_run if args.min_slot_run is not None else 3,
        slot_presence_frames=args.slot_presence_frames if args.slot_presence_frames is not None else 8,
        fill_gap_frames=args.fill_gap_frames if args.fill_gap_frames is not None else 3,
        cluster_merge_threshold=args.slot_merge_threshold,
    )


def run_case(
    case: EvalCase,
    args: argparse.Namespace,
    interface: SpeakerAwareDiarizationInterface,
) -> Dict[str, Any]:
    tracker = GlobalSpeakerTracker(
        match_threshold=args.tracker_match_threshold,
        momentum=args.tracker_momentum,
        max_misses=args.tracker_max_misses,
        device="cpu",
    )

    wav = load_wav_mono(str(case.wav_path), target_sr=TARGET_SR)
    total_samples = int(wav.numel())
    duration_sec = total_samples / TARGET_SR
    chunk_samples = max(1, int(round(args.chunk_sec * TARGET_SR)))
    step_samples = max(1, int(round(args.step_sec * TARGET_SR)))
    windows = build_windows(total_samples, chunk_samples, step_samples)

    all_segments: List[Dict[str, Any]] = []
    for start_sample, end_sample, emit_start, emit_end in windows:
        chunk = wav[start_sample:end_sample]
        offset_sec = start_sample / TARGET_SR
        result = interface.infer_wav(
            chunk,
            sample_rate=TARGET_SR,
            crop_sec=None,
            crop_mode="none",
            normalize=args.normalize,
            offset_sec=offset_sec,
        )
        tracker_result = tracker.update(result)
        chunk_segments = []
        for seg in result.segments:
            global_id = int(tracker_result.local_to_global.get(int(seg.slot), 0))
            if global_id <= 0:
                continue
            chunk_segments.append(
                {
                    "slot": int(seg.slot),
                    "global_id": global_id,
                    "name": f"speaker_{global_id}",
                    "start_sec": float(seg.start_sec),
                    "end_sec": float(seg.end_sec),
                    "duration_sec": float(seg.duration_sec),
                }
            )
        all_segments.extend(clip_segments(chunk_segments, emit_start, emit_end))

    merged_segments = merge_segments(all_segments, max_gap_sec=args.merge_gap_sec)
    merged_segments = clip_segments(merged_segments, 0.0, duration_sec)

    case_result: Dict[str, Any] = {
        "wav_path": str(case.wav_path),
        "duration_sec": round(duration_sec, 4),
        "num_speakers": len({int(seg["global_id"]) for seg in merged_segments}),
        "speaker_durations": summarize_segments(merged_segments),
        "segments": merged_segments,
    }

    if case.rttm_path is not None and case.rttm_path.is_file():
        ref_segments = parse_rttm(case.rttm_path)
        der_detail = compute_framewise_der(
            pred_segments=merged_segments,
            ref_segments=ref_segments,
            duration_sec=max(duration_sec, max((seg["end_sec"] for seg in ref_segments), default=duration_sec)),
        )
        case_result["reference_rttm"] = str(case.rttm_path)
        case_result["der_detail"] = {key: round(value, 6) for key, value in der_detail.items()}

    return case_result


def main() -> None:
    args = parse_args()
    cases = load_cases(args)
    interface = build_interface(args)

    results = [run_case(case, args, interface) for case in cases]
    payload: Dict[str, Any] = {"results": results}

    der_cases = [item["der_detail"] for item in results if "der_detail" in item]
    if der_cases:
        total_fa = sum(item["fa"] for item in der_cases)
        total_miss = sum(item["miss"] for item in der_cases)
        total_conf = sum(item["conf"] for item in der_cases)
        total_gt = sum(item["gt_active"] for item in der_cases)
        payload["summary"] = {
            "num_cases": len(results),
            "num_cases_with_reference": len(der_cases),
            "micro_der": round((total_fa + total_miss + total_conf) / max(total_gt, 1.0), 6),
            "micro_fa": round(total_fa, 6),
            "micro_miss": round(total_miss, 6),
            "micro_conf": round(total_conf, 6),
        }
    else:
        payload["summary"] = {"num_cases": len(results), "num_cases_with_reference": 0}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"[verif] wrote results to {args.output}")


if __name__ == "__main__":
    main()
