import argparse
import json
import os
import queue
import sys
import time
from collections import Counter, deque

import numpy as np
import sounddevice as sd
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.dirname(current_dir)
sys.path.append(proj_root)

from speaker_verification.audio.features import TARGET_SR
from speaker_verification.interfaces.diar_interface import (
    EmptySpeakerBank,
    SpeakerAwareDiarizationInterface,
)
from speaker_verification.interfaces.global_tracker import GlobalSpeakerTracker


class AudioRingBuffer:
    def __init__(self, max_seconds: float, sr: int):
        self.max_samples = int(max_seconds * sr)
        self.buf = deque(maxlen=self.max_samples)

    def append(self, x: np.ndarray):
        self.buf.extend(x.astype(np.float32).tolist())

    def get_array(self) -> np.ndarray:
        if not self.buf:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.buf, dtype=np.float32)


def compute_rms_np(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def majority_vote(values):
    if not values:
        return 0
    return Counter(values).most_common(1)[0][0]


def segments_with_global_ids(result, local_to_global):
    out = []
    for seg in result.segments:
        gid = local_to_global.get(int(seg.slot), 0)
        out.append(
            {
                "slot": int(seg.slot),
                "global_id": int(gid),
                "name": seg.name,
                "start_sec": float(seg.start_sec),
                "end_sec": float(seg.end_sec),
                "duration_sec": float(seg.duration_sec),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feat_dim", type=int, default=80)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--max_mix_speakers", type=int, default=4)

    parser.add_argument("--sr", type=int, default=TARGET_SR)
    parser.add_argument("--chunk_sec", type=float, default=2.0)
    parser.add_argument("--step_sec", type=float, default=1.0)

    parser.add_argument("--activity_threshold", type=float, default=0.55)
    parser.add_argument("--min_active_frames", type=int, default=3)
    parser.add_argument("--min_slot_run", type=int, default=3)

    parser.add_argument("--wav_rms_threshold", type=float, default=0.008)
    parser.add_argument("--fbank_energy_threshold", type=float, default=0.020)
    parser.add_argument("--min_activity_ratio", type=float, default=0.15)
    parser.add_argument("--min_mean_activity_prob", type=float, default=0.40)

    parser.add_argument("--tracker_match_threshold", type=float, default=0.72)
    parser.add_argument("--tracker_momentum", type=float, default=0.90)
    parser.add_argument("--tracker_max_misses", type=int, default=30)

    parser.add_argument("--display_top_segments", type=int, default=10)
    parser.add_argument("--smooth_windows", type=int, default=5)
    parser.add_argument("--print_json", action="store_true")

    args = parser.parse_args()

    if args.sr != TARGET_SR:
        raise ValueError(f"Current demo expects sr={TARGET_SR}, got {args.sr}")

    monitor = SpeakerAwareDiarizationInterface(
        ckpt_path=args.ckpt,
        device=args.device,
        feat_dim=args.feat_dim,
        channels=args.channels,
        emb_dim=args.emb_dim,
        max_mix_speakers=args.max_mix_speakers,
        activity_threshold=args.activity_threshold,
        frame_shift_sec=0.01,
        min_active_frames=args.min_active_frames,
        min_slot_run=args.min_slot_run,
        speaker_bank=EmptySpeakerBank(),
        fbank_energy_threshold=args.fbank_energy_threshold,
        min_activity_ratio=args.min_activity_ratio,
        min_mean_activity_prob=args.min_mean_activity_prob,
    )

    tracker = GlobalSpeakerTracker(
        match_threshold=args.tracker_match_threshold,
        momentum=args.tracker_momentum,
        max_misses=args.tracker_max_misses,
        device="cpu",
    )

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    ring = AudioRingBuffer(max_seconds=args.chunk_sec, sr=args.sr)

    smooth_num_speakers = deque(maxlen=max(1, args.smooth_windows))
    last_infer_time = 0.0

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[AudioStatus] {status}", file=sys.stderr)
        mono = indata[:, 0].astype(np.float32)
        audio_q.put(mono.copy())

    print("=" * 80)
    print("Realtime diarization + global tracking started")
    print("Press Ctrl+C to stop")
    print("=" * 80)

    with sd.InputStream(
        samplerate=args.sr,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=int(args.sr * 0.2),
    ):
        try:
            while True:
                try:
                    chunk = audio_q.get(timeout=0.2)
                    ring.append(chunk)
                except queue.Empty:
                    pass

                now = time.time()
                if now - last_infer_time < args.step_sec:
                    continue

                wav = ring.get_array()
                if wav.shape[0] < int(args.chunk_sec * args.sr * 0.8):
                    continue

                wav_rms = compute_rms_np(wav)

                if wav_rms < args.wav_rms_threshold:
                    tracker_result = None
                    result_json = {
                        "num_speakers": 0,
                        "num_speakers_stable": 0,
                        "dominant_speaker": None,
                        "dominant_speaker_slot": None,
                        "dominant_global_id": None,
                        "activity_ratio": 0.0,
                        "slots": [],
                        "segments": [],
                        "active_global_ids": [],
                        "local_to_global": {},
                    }
                else:
                    wav_tensor = torch.from_numpy(wav).float()
                    result = monitor.infer_wav(
                        wav_tensor,
                        sample_rate=args.sr,
                        crop_sec=args.chunk_sec,
                        crop_mode="tail",
                        normalize=True,
                    )

                    tracker_result = tracker.update(result)
                    result.global_frame_ids = tracker_result.global_frame_ids

                    result_json = monitor.to_jsonable(result)
                    result_json["local_to_global"] = {
                        int(k): int(v) for k, v in tracker_result.local_to_global.items()
                    }
                    result_json["active_global_ids"] = tracker_result.active_global_ids
                    result_json["dominant_global_id"] = tracker_result.dominant_global_id
                    result_json["segments"] = segments_with_global_ids(
                        result, tracker_result.local_to_global
                    )

                smooth_num_speakers.append(int(result_json["num_speakers"]))
                result_json["num_speakers_stable"] = majority_vote(list(smooth_num_speakers))

                print("\n" + "-" * 80)
                print(
                    f"[Realtime] "
                    f"num_speakers={result_json['num_speakers']} "
                    f"(stable={result_json['num_speakers_stable']}) | "
                    f"dominant_speaker={result_json['dominant_speaker']} | "
                    f"dominant_global_id={result_json['dominant_global_id']} | "
                    f"activity_ratio={result_json['activity_ratio']:.3f} | "
                    f"wav_rms={wav_rms:.5f}"
                )

                if not result_json["slots"]:
                    print("  no active speaker slots")
                else:
                    for slot in result_json["slots"]:
                        gid = result_json["local_to_global"].get(int(slot["slot"]), 0)
                        print(
                            f"  local_slot={slot['slot']} "
                            f"global_id={gid} "
                            f"name={slot['name']} "
                            f"known={slot['is_known']} "
                            f"dur={slot['duration_sec']:.2f}s "
                            f"frames={slot['num_frames']} "
                            f"score={slot['score']}"
                        )

                print(f"  active_global_ids={result_json['active_global_ids']}")
                print("  segments:")
                if not result_json["segments"]:
                    print("    []")
                else:
                    for seg in result_json["segments"][: args.display_top_segments]:
                        print(
                            f"    [{seg['start_sec']:.2f}s - {seg['end_sec']:.2f}s] "
                            f"local_slot={seg['slot']} "
                            f"global_id={seg['global_id']} "
                            f"name={seg['name']}"
                        )

                if args.print_json:
                    print(json.dumps(result_json, ensure_ascii=False, indent=2))

                last_infer_time = now

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()