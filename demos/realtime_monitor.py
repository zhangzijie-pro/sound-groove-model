import argparse
import queue
import sys
import time
from collections import deque

import json
import numpy as np
import sounddevice as sd
import torch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.dirname(current_dir)

sys.path.append(proj_root)

from speaker_verification.audio.features import TARGET_SR
from speaker_verification.interfaces.diar_interface import *

class AudioRingBuffer:
    def __init__(self, max_seconds: float, sr: int):
        self.sr = sr
        self.max_samples = int(max_seconds * sr)
        self.buf = deque(maxlen=self.max_samples)

    def append(self, x: np.ndarray):
        for v in x.tolist():
            self.buf.append(v)

    def get_array(self) -> np.ndarray:
        if len(self.buf) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.buf, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feat_dim", type=int, default=80)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--max_mix_speakers", type=int, default=4)

    parser.add_argument("--sr", type=int, default=TARGET_SR)
    parser.add_argument("--chunk_sec", type=float, default=4.0)
    parser.add_argument("--step_sec", type=float, default=1.0)
    parser.add_argument("--activity_threshold", type=float, default=0.5)

    parser.add_argument("--min_active_frames", type=int, default=3)
    parser.add_argument("--min_slot_run", type=int, default=3)
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
    )

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    ring = AudioRingBuffer(max_seconds=args.chunk_sec, sr=args.sr)

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[AudioStatus] {status}", file=sys.stderr)
        mono = indata[:, 0].astype(np.float32)
        audio_q.put(mono.copy())

    print("=" * 80)
    print("Realtime diarization monitor started")
    print("Press Ctrl+C to stop")
    print("=" * 80)

    last_infer_time = 0.0

    with sd.InputStream(
        samplerate=args.sr,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=int(args.sr * 0.2),  # 200 ms
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

                wav_tensor = torch.from_numpy(wav).float()
                result = monitor.infer_wav(wav_tensor, crop_sec=args.chunk_sec)
                result_json = monitor.to_jsonable(result)

                print("\n" + "-" * 80)
                print(
                    f"[Realtime] "
                    f"num_speakers={result_json['num_speakers']} | "
                    f"dominant_speaker={result_json['dominant_speaker']} | "
                    f"activity_ratio={result_json['activity_ratio']:.3f}"
                )

                for slot in result_json["slots"]:
                    print(
                        f"  slot={slot['slot']} "
                        f"name={slot['name']} "
                        f"known={slot['is_known']} "
                        f"dur={slot['duration_sec']:.2f}s "
                        f"frames={slot['num_frames']} "
                        f"score={slot['score']}"
                    )

                print("  segments:")
                for seg in result_json["segments"][:10]:
                    print(
                        f"    [{seg['start_sec']:.2f}s - {seg['end_sec']:.2f}s] "
                        f"slot={seg['slot']} name={seg['name']}"
                    )

                if args.print_json:
                    print(json.dumps(result_json, ensure_ascii=False, indent=2))

                last_infer_time = now

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()