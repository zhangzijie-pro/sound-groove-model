import argparse
import torch
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False
    print("⚠️ onnxruntime 未安装，ONNX 模式不可用。")
    print("   pip install onnxruntime  或  onnxruntime-gpu")

from models.ecapa import ECAPA_TDNN
from utils.audio import load_wav_mono, wav_to_fbank


@torch.no_grad()
def embed_wav_pt(model, wav_path: str, device: torch.device) -> torch.Tensor:
    wav = load_wav_mono(wav_path, target_sr=16000)      # [T]
    feat = wav_to_fbank(wav, n_mels=80)                 # [T_frames, 80]
    x = feat.unsqueeze(0).to(device)                    # [1, T, 80]
    emb = model(x).squeeze(0).cpu()                     # [D]
    emb = emb / (emb.norm() + 1e-12)                    # L2 normalize
    return emb


def embed_wav_onnx(session: ort.InferenceSession, wav_path: str) -> np.ndarray:
    wav = load_wav_mono(wav_path, target_sr=16000)
    feat = wav_to_fbank(wav, n_mels=80)                 # [T, 80]
    feat = feat.unsqueeze(0).numpy().astype(np.float32) # [1, T, 80]

    # ONNX 输入名通常是 "fbank"（你在 export.py 里设置的）
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: feat})

    emb = outputs[0][0]                                 # [D]
    emb = emb / (np.linalg.norm(emb) + 1e-12)           # L2 normalize
    return emb


def main():
    parser = argparse.ArgumentParser(description="两个音频说话人对比（PyTorch / ONNX）")
    
    parser.add_argument("--wav1", type=str, required=True, help="第一个音频路径")
    parser.add_argument("--wav2", type=str, required=True, help="第二个音频路径")
    
    parser.add_argument("--ckpt", type=str, default="outputs/export/model.onnx",
                        help="模型路径：.pt (PyTorch) 或 .onnx (ONNX)")
    
    parser.add_argument("--onnx", action="store_true", default=False,
                        help="使用 ONNX 推理（默认自动根据文件后缀判断）")
    
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="判断同一人的余弦相似度阈值（建议通过 verify.py 得到最佳阈值）")
    
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="PyTorch 模式使用的设备（ONNX 默认 CPU）")

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    use_onnx = args.onnx or ckpt_path.suffix.lower() == ".onnx"

    print("=" * 70)
    print("🎙️  Speaker Verification - Two Wavs Comparison")
    print("=" * 70)
    print(f"Audio 1 : {args.wav1}")
    print(f"Audio 2 : {args.wav2}")
    print(f"Model   : {ckpt_path}  ({'ONNX' if use_onnx else 'PyTorch'})")
    print(f"Threshold: {args.threshold}")
    print("=" * 70)

    if use_onnx:
        if not _HAS_ONNX:
            raise ImportError("请先安装 onnxruntime: pip install onnxruntime")

        print("使用 ONNX Runtime 推理...")
        session = ort.InferenceSession(
            str(ckpt_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        )

        e1 = embed_wav_onnx(session, args.wav1)
        e2 = embed_wav_onnx(session, args.wav2)
        score = float(np.dot(e1, e2))

    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"使用 PyTorch 推理 (device: {device})")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = ECAPA_TDNN(
            in_channels=80,
            channels=512,
            embd_dim=256
        ).to(device)

        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        e1 = embed_wav_pt(model, args.wav1, device)
        e2 = embed_wav_pt(model, args.wav2, device)
        score = float(torch.sum(e1 * e2).item())

    # ==================== 输出 ====================
    same = score >= args.threshold

    print(f"\n🔍 Cosine Similarity = {score:.4f}")
    print(f"   Threshold        = {args.threshold}")
    print(f"   → {'同一说话人' if same else '不同说话人'}")

    color = "\033[92m" if same else "\033[91m"
    print(f"\n{color}【最终判定】{'✅ 同一人' if same else '❌ 不同人'}\033[0m")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()