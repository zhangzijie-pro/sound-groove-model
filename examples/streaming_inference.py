"""Low-latency streaming inference entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from speaker_verification.interfaces.diar_interface import SpeakerAwareDiarizationInterface
from speaker_verification.interfaces.global_tracker import GlobalSpeakerTracker
from speaker_verification.speaker_bank import JsonSpeakerBank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming speaker-aware inference demo")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--speaker-bank", default="artifacts/speaker_bank.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--max-mix-speakers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    speaker_bank = JsonSpeakerBank(Path(args.speaker_bank))
    _interface = SpeakerAwareDiarizationInterface(
        ckpt_path=args.ckpt,
        device=args.device,
        speaker_bank=speaker_bank,
        max_mix_speakers=args.max_mix_speakers,
    )
    _tracker = GlobalSpeakerTracker()

    print("Streaming inference entrypoint ready.")
    print("Integrate this script with a microphone ring buffer or websocket audio stream.")
    print(f"chunk_sec={args.chunk_sec} step_sec={args.step_sec}")


if __name__ == "__main__":
    main()

