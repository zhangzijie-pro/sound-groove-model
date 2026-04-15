from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from speaker_verification.engine.checkpoint import get_model_state_from_ckpt
from speaker_verification.inference import build_model_from_config


def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def default_config() -> dict[str, Any]:
    return {
        "model": {
            "in_channels": 80,
            "channels": 512,
            "d_model": 256,
            "max_mix_speakers": 3,
            "assign_scale": 8.0,
            "decoder_type": "query",
            "decoder_layers": 4,
            "decoder_heads": 8,
            "decoder_ffn": 512,
            "dropout": 0.1,
            "post_ffn_hidden_dim": 512,
            "post_ffn_dropout": 0.2,
        },
        "data": {"crop_sec": 4.0},
        "validate": {
            "speaker_activity_threshold": 0.55,
            "exist_threshold": 0.5,
            "min_active_frames": 5,
        },
    }


class EENDExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, fbank: torch.Tensor, valid_mask: torch.Tensor):
        frame_embeds, _, exist_logits, diar_logits, activity_logits = self.model(
            fbank,
            valid_mask=valid_mask.bool(),
        )
        return frame_embeds, exist_logits, diar_logits, activity_logits


class EENDModelExporter:
    def __init__(self, ckpt_path: str, out_dir: str, device: str = "auto"):
        self.start_time = time.time()
        self.ckpt_path = Path(ckpt_path).expanduser().resolve()
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = self.select_device(device)

        self.ckpt = torch_load(self.ckpt_path, map_location=self.device)
        self.config = self.load_config(self.ckpt)
        self.model = build_model_from_config(self.config, self.device)
        self.model.load_state_dict(get_model_state_from_ckpt(self.ckpt), strict=True)
        self.model.eval()
        self.wrapper = EENDExportWrapper(self.model).to(self.device).eval()

        print(f"[export] model=EENDQueryModel")
        print(f"[export] checkpoint={self.ckpt_path}")
        print(f"[export] output={self.out_dir}")
        print(f"[export] device={self.device}")

    @staticmethod
    def select_device(device: str) -> torch.device:
        if device and device != "auto":
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def load_config(ckpt: Any) -> Mapping[str, Any]:
        if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), Mapping):
            return ckpt["config"]
        return default_config()

    def elapsed(self) -> str:
        seconds = int(time.time() - self.start_time)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}h {m:02d}m {s:02d}s"

    def dummy_inputs(self, batch_size: int, frames: int):
        in_channels = int(self.config.get("model", {}).get("in_channels", 80))
        fbank = torch.randn(batch_size, frames, in_channels, device=self.device)
        valid_mask = torch.ones(batch_size, frames, dtype=torch.bool, device=self.device)
        return fbank, valid_mask

    def split_checkpoint(self) -> None:
        model_state = get_model_state_from_ckpt(self.ckpt)
        torch.save(model_state, self.out_dir / "model_state.pt")

        with (self.out_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(self.config, handle, ensure_ascii=False, indent=2)

        meta = {
            "model_type": "EENDQueryModel",
            "task": "speaker_diarization",
            "original_checkpoint": str(self.ckpt_path),
            "epoch": self.ckpt.get("epoch") if isinstance(self.ckpt, dict) else None,
            "best_metric": self.ckpt.get("best_metric") if isinstance(self.ckpt, dict) else None,
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": ["fbank", "valid_mask"],
            "outputs": ["frame_embeds", "exist_logits", "diar_logits", "activity_logits"],
        }
        with (self.out_dir / "export_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)

        print("[export] wrote model_state.pt, config.json, export_meta.json")

    def export_torchscript(self, batch_size: int, frames: int) -> None:
        fbank, valid_mask = self.dummy_inputs(batch_size, frames)
        traced = torch.jit.trace(
            self.wrapper,
            (fbank, valid_mask),
            strict=False,
            check_trace=False,
        )
        ts_path = self.out_dir / "model.ts"
        traced.save(str(ts_path))
        print(f"[export] torchscript saved: {ts_path}")

    def export_onnx(self, batch_size: int, frames: int, opset: int) -> None:
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError("ONNX export requires `pip install onnx`.") from exc

        fbank, valid_mask = self.dummy_inputs(batch_size, frames)
        onnx_path = self.out_dir / "model.onnx"
        torch.onnx.export(
            self.wrapper,
            (fbank, valid_mask),
            str(onnx_path),
            input_names=["fbank", "valid_mask"],
            output_names=["frame_embeds", "exist_logits", "diar_logits", "activity_logits"],
            dynamic_axes={
                "fbank": {0: "batch", 1: "frames"},
                "valid_mask": {0: "batch", 1: "frames"},
                "frame_embeds": {0: "batch", 1: "frames"},
                "exist_logits": {0: "batch"},
                "diar_logits": {0: "batch", 1: "frames"},
                "activity_logits": {0: "batch", 1: "frames"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"[export] onnx saved: {onnx_path}")

    def export_mnn(self, fp16: bool = True, optimize: int = 1) -> None:
        onnx_path = self.out_dir / "model.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError("MNN export requires model.onnx. Run with --onnx first.")

        mnn_path = self.out_dir / "model.mnn"
        cmd = [
            "mnnconvert",
            "-f",
            "ONNX",
            "--modelFile",
            str(onnx_path),
            "--MNNModel",
            str(mnn_path),
            "--optimizePrefer",
            str(optimize),
        ]
        if fp16:
            cmd.append("--fp16")
        subprocess.run(cmd, check=True)
        print(f"[export] mnn saved: {mnn_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export EENDQueryModel checkpoint artifacts.")
    parser.add_argument("--ckpt", default="outputs_trainali_synth/best.pt", help="Checkpoint path.")
    parser.add_argument("--out_dir", default="outputs/export", help="Export output directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torchscript", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--onnx", action="store_true", default=False)
    parser.add_argument("--mnn", action="store_true", default=False)
    parser.add_argument("--dummy_batch", type=int, default=1)
    parser.add_argument("--dummy_frames", type=int, default=398)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exporter = EENDModelExporter(args.ckpt, args.out_dir, device=args.device)

    if args.split:
        exporter.split_checkpoint()
    if args.torchscript:
        exporter.export_torchscript(args.dummy_batch, args.dummy_frames)
    if args.onnx:
        exporter.export_onnx(args.dummy_batch, args.dummy_frames, args.opset)
    if args.mnn:
        exporter.export_mnn(fp16=args.fp16)

    print("=" * 60)
    print(f"Export finished in {exporter.elapsed()}")
    print(f"Output directory: {exporter.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
