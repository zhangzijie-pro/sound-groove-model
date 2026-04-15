import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speaker_verification.audio.features import TARGET_SR
from speaker_verification.inference import SpeakerDiarizationPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime microphone diarization test with outputs_trainali_synth/best.pt."
    )
    parser.add_argument("--ckpt", default="outputs_trainali_synth/best.pt", help="Checkpoint path.")
    parser.add_argument("--compute_device", default=None, help="cuda, cpu, or leave empty for auto.")
    parser.add_argument("--input_device", default=None, help="Optional sounddevice input device id/name.")
    parser.add_argument("--seconds", type=float, default=30.0, help="Total capture seconds. Use 0 for infinite.")
    parser.add_argument("--chunk_sec", type=float, default=4.0, help="Realtime chunk length.")
    parser.add_argument("--activity_threshold", type=float, default=0.55, help="Speaker activity threshold.")
    parser.add_argument("--exist_threshold", type=float, default=0.5, help="Speaker existence threshold.")
    parser.add_argument("--min_active_frames", type=int, default=5, help="Remove shorter active runs.")
    parser.add_argument("--keep_raw_slots", action="store_true", help="Use raw query slot names instead of compact labels.")
    parser.add_argument("--list_devices", action="store_true", help="Print audio devices and exit.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "sounddevice is required for realtime microphone testing. "
            "Install it with: pip install sounddevice"
        ) from exc

    if args.list_devices:
        print(sd.query_devices())
        return

    pipeline = SpeakerDiarizationPipeline.from_pretrained(
        args.ckpt,
        device=args.compute_device,
        chunk_sec=args.chunk_sec,
        hop_sec=args.chunk_sec,
        speaker_activity_threshold=args.activity_threshold,
        exist_threshold=args.exist_threshold,
        min_active_frames=args.min_active_frames,
        merge_gap_sec=0.0,
        relabel_speakers=not args.keep_raw_slots,
    )

    frames_per_chunk = int(round(args.chunk_sec * TARGET_SR))
    max_chunks = None if args.seconds <= 0 else int(math.ceil(args.seconds / args.chunk_sec))
    chunk_index = 0

    print("Realtime diarization started. Press Ctrl+C to stop.")
    print(f"sample_rate={TARGET_SR}, chunk_sec={args.chunk_sec}, device={pipeline.device}")

    try:
        while max_chunks is None or chunk_index < max_chunks:
            offset = chunk_index * args.chunk_sec
            print(f"\n[{time.strftime('%H:%M:%S')}] recording chunk {chunk_index} ...")
            audio = sd.rec(
                frames_per_chunk,
                samplerate=TARGET_SR,
                channels=1,
                dtype="float32",
                device=args.input_device,
            )
            sd.wait()

            wav = torch.from_numpy(audio.reshape(-1).copy())
            result = pipeline.predict_wav(wav, recording_id=f"live_{chunk_index:04d}")

            if not result.segments:
                print(f"{offset:8.3f}s - {offset + args.chunk_sec:8.3f}s no active speaker")
            else:
                for seg in result.segments:
                    start = offset + float(seg["start"])
                    end = offset + float(seg["end"])
                    print(
                        f"{start:8.3f}s - {end:8.3f}s "
                        f"{seg['speaker']} raw={seg['raw_speaker']} "
                        f"score={float(seg['score']):.3f}"
                    )

            chunk_index += 1

    except KeyboardInterrupt:
        print("\nRealtime diarization stopped.")


if __name__ == "__main__":
    main()
