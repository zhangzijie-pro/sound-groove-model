"""Meeting diarization and timeline export entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio

from speaker_verification.interfaces.diar_interface import SpeakerAwareDiarizationInterface
from speaker_verification.interfaces.global_tracker import GlobalSpeakerTracker
from speaker_verification.speaker_bank import JsonSpeakerBank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meeting analysis demo")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--audio", required=True, help="Meeting audio path")
    parser.add_argument("--output", default="artifacts/meeting_analysis.json")
    parser.add_argument("--speaker-bank", default="artifacts/speaker_bank.json")
    parser.add_argument("--device", default="cuda")
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
    tracker = GlobalSpeakerTracker()

    result = diar.infer_wav(wav, sample_rate=sample_rate)
    tracker_result = tracker.update(result)

    payload = diar.to_jsonable(result)
    payload["local_to_global"] = tracker_result.local_to_global
    payload["active_global_ids"] = tracker_result.active_global_ids
    payload["dominant_global_id"] = tracker_result.dominant_global_id

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Meeting analysis written to {output_path}")


if __name__ == "__main__":
    main()

