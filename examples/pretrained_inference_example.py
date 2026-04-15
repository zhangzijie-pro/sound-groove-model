import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speaker_verification.inference import SpeakerDiarizationPipeline, write_json, write_rttm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run diarization with outputs_trainali_synth/best.pt."
    )
    parser.add_argument("--audio", required=True, help="Input wav/flac/mp3 path readable by torchaudio.")
    parser.add_argument("--ckpt", default="outputs_trainali_synth/best.pt", help="Checkpoint path.")
    parser.add_argument("--out_dir", default="outputs_demo", help="Directory for JSON and RTTM outputs.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto.")
    parser.add_argument("--chunk_sec", type=float, default=4.0, help="Inference chunk length.")
    parser.add_argument("--hop_sec", type=float, default=2.0, help="Chunk hop for long audio.")
    parser.add_argument("--activity_threshold", type=float, default=0.55, help="Speaker activity threshold.")
    parser.add_argument("--exist_threshold", type=float, default=0.5, help="Speaker existence threshold.")
    parser.add_argument("--min_active_frames", type=int, default=5, help="Remove shorter active runs.")
    parser.add_argument("--merge_gap_sec", type=float, default=0.0, help="Merge short gaps; 0 keeps pretrain behavior.")
    parser.add_argument("--keep_raw_slots", action="store_true", help="Use raw query slot names instead of compact labels.")
    parser.add_argument("--include_frames", action="store_true", help="Also dump frame-level arrays to JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    audio_path = Path(args.audio)
    out_dir = Path(args.out_dir)
    recording_id = audio_path.stem

    pipeline = SpeakerDiarizationPipeline.from_pretrained(
        args.ckpt,
        device=args.device,
        chunk_sec=args.chunk_sec,
        hop_sec=args.hop_sec,
        speaker_activity_threshold=args.activity_threshold,
        exist_threshold=args.exist_threshold,
        min_active_frames=args.min_active_frames,
        merge_gap_sec=args.merge_gap_sec,
        relabel_speakers=not args.keep_raw_slots,
    )

    result = pipeline.predict_file(audio_path, recording_id=recording_id)
    json_path = out_dir / f"{recording_id}.json"
    rttm_path = out_dir / f"{recording_id}.rttm"
    write_json(result, json_path, include_frame_outputs=args.include_frames)
    write_rttm(result, rttm_path)

    print(f"recording_id: {result.recording_id}")
    print(f"segments: {len(result.segments)}")
    for seg in result.segments:
        print(
            f"{seg['start']:>8.3f}s - {seg['end']:>8.3f}s "
            f"{seg['speaker']} raw={seg['raw_speaker']} score={seg['score']:.3f}"
        )
    print(f"json: {json_path}")
    print(f"rttm: {rttm_path}")


if __name__ == "__main__":
    main()
