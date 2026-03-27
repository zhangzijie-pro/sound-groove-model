"""Long-audio monitoring entrypoint for unknown-speaker and activity alerts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio

from speaker_verification.interfaces.diar_interface import SpeakerAwareDiarizationInterface
from speaker_verification.speaker_bank import JsonSpeakerBank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-audio monitoring demo")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--audio", required=True, help="Input audio path")
    parser.add_argument("--speaker-bank", default="speaker_bank.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alert-threshold", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    wav, sample_rate = torchaudio.load(args.audio)
    wav = wav.mean(dim=0).to(torch.float32)

    diar = SpeakerAwareDiarizationInterface(
        ckpt_path=args.ckpt,
        device=args.device,
        speaker_bank=JsonSpeakerBank(Path(args.speaker_bank)),
    )
    result = diar.infer_wav(wav, sample_rate=sample_rate)

    unknown_ratio = 0.0
    if result.slots:
        unknown_slots = [slot for slot in result.slots if not slot.is_known]
        unknown_ratio = len(unknown_slots) / len(result.slots)

    print(f"num_speakers={result.num_speakers} activity_ratio={result.activity_ratio:.4f}")
    print(f"unknown_ratio={unknown_ratio:.4f}")
    if unknown_ratio >= args.alert_threshold:
        print("ALERT: unknown speaker ratio exceeded threshold")


if __name__ == "__main__":
    main()

