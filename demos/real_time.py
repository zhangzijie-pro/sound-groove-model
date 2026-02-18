import argparse
import os
import queue
import sys
import tempfile
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) != "Speaker-Verification" else current_dir
if project_root not in sys.path:
    sys.path.append(project_root)

from speaker_verification.inference import load_sv, extract_embedding


def rms_db(x: np.ndarray, eps: float = 1e-12) -> float:
    x = x.astype(np.float32)
    r = np.sqrt(np.mean(x * x) + eps)
    return 20.0 * np.log10(r + eps)


def meter_bar(db: float, min_db=-60.0, max_db=-10.0, width=30) -> str:
    db = float(np.clip(db, min_db, max_db))
    p = (db - min_db) / (max_db - min_db + 1e-9)
    n = int(p * width)
    return "[" + ("#" * n) + ("-" * (width - n)) + f"] {db:6.1f} dB"


def to_mono_float32(indata: np.ndarray) -> np.ndarray:
    x = np.asarray(indata)
    if x.ndim == 2:
        x = x.mean(axis=1)  # [T, C] -> [T]
    x = x.astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    return x


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def emb_to_np(emb):
    if isinstance(emb, torch.Tensor):
        return emb.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return np.asarray(emb, dtype=np.float32).reshape(-1)


def save_tmp_wav(y: np.ndarray, sr: int) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = f.name
    f.close()
    sf.write(path, y, sr, subtype="PCM_16")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="scripts/outputs/export/model.onnx")
    ap.add_argument("--onnx", action="store_true", help="force ONNX backend")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--chunk_ms", type=int, default=30)
    ap.add_argument("--vad_db", type=float, default=-35.0, help="VAD threshold in dB (higher = stricter)")
    ap.add_argument("--silence_ms", type=int, default=450, help="how long silence ends an utterance")
    ap.add_argument("--min_speech_ms", type=int, default=800, help="minimum utterance length to run SV")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--num_crops", type=int, default=5)
    ap.add_argument("--crop_sec", type=float, default=3.0)

    ap.add_argument(
        "--ref",
        action="append",
        default=[],
        help="reference speaker, format: name=path.wav ; can be used multiple times",
    )
    args = ap.parse_args()

    if not args.ref:
        print("你需要至少提供一个参考说话人：--ref name=path.wav")
        print("例如：--ref me=000.wav --ref friend=003.wav")
        sys.exit(1)

    print(f"Loading model: {args.model} (onnx={args.onnx})")
    sv, meta = load_sv(args.model, device=args.device, use_onnx=args.onnx)
    print("Meta:", meta)

    ref_embs = {}
    for item in args.ref:
        if "=" not in item:
            raise ValueError(f"--ref format must be name=path.wav, got {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        emb = extract_embedding(sv, path, num_crops=args.num_crops, crop_sec=args.crop_sec)
        ref_embs[name] = emb_to_np(emb)
        print(f"Enrolled: {name:>10s} <- {path}  | emb_dim={ref_embs[name].shape[0]}")

    q = queue.Queue()

    blocksize = int(args.sr * args.chunk_ms / 1000)

    def callback(indata, frames, time_info, status):
        if status:
            pass
        q.put(indata.copy())

    # VAD state machine
    in_speech = False
    speech_buf = []
    silence_count = 0
    silence_need = max(1, int(args.silence_ms / args.chunk_ms))
    min_need = max(1, int(args.min_speech_ms / args.chunk_ms))

    print("\n--- Realtime Speaker Verification CLI ---")
    print("Speak into mic. Ctrl+C to stop.")
    print(f"VAD: start if dB > {args.vad_db}, end after {args.silence_ms}ms silence")
    print(f"SV threshold: {args.threshold}\n")

    with sd.InputStream(
        samplerate=args.sr,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    ):
        try:
            while True:
                indata = q.get()
                x = to_mono_float32(indata)
                db = rms_db(x)

                line = meter_bar(db)
                if in_speech:
                    line += "  🎙️ recording..."
                else:
                    line += "  (idle)"
                print("\r" + line + " " * 10, end="", flush=True)

                is_voice = db > args.vad_db

                if not in_speech:
                    if is_voice:
                        in_speech = True
                        speech_buf = [x]
                        silence_count = 0
                else:
                    speech_buf.append(x)
                    if is_voice:
                        silence_count = 0
                    else:
                        silence_count += 1

                    if silence_count >= silence_need:
                        print()  # 换行输出结果
                        in_speech = False

                        if len(speech_buf) < min_need:
                            print(f"Utterance too short ({len(speech_buf)*args.chunk_ms} ms), skip.\n")
                            speech_buf = []
                            continue

                        y = np.concatenate(speech_buf, axis=0)
                        peak = np.max(np.abs(y)) + 1e-9
                        if peak > 0.99:
                            y = y / peak * 0.99

                        tmp_path = None
                        try:
                            tmp_path = save_tmp_wav(y, args.sr)
                            emb = extract_embedding(sv, tmp_path, num_crops=args.num_crops, crop_sec=args.crop_sec)
                            e = emb_to_np(emb)

                            best_name, best_score = None, -1e9
                            for name, r in ref_embs.items():
                                s = cosine_np(e, r)
                                if s > best_score:
                                    best_score = s
                                    best_name = name

                            ok = best_score >= args.threshold
                            tag = "✅ MATCH" if ok else "❌ NO-MATCH"
                            print(f"{tag}  best={best_name}  score={best_score:.4f}  thr={args.threshold}")
                            print()
                        finally:
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    os.remove(tmp_path)
                                except Exception:
                                    pass

                        speech_buf = []

        except KeyboardInterrupt:
            print("\nBye.")


if __name__ == "__main__":
    main()
    
"""
python real_time.py \
  --model ../scripts/outputs/export/model.onnx --onnx \
  --ref zzj=zzj.wav \
  --ref zzx=zzx.wav \
  --threshold 0.55
"""