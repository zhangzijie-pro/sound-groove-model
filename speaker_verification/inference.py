# speaker_verification/inference.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from .checkpointing import ModelCfg, load_ckpt
from .models.ecapa import ECAPA_TDNN
from .audio.features import load_wav_mono, wav_to_fbank

try:
    import onnxruntime as ort

    _HAS_ONNX = True
except ImportError:
    ort = None
    _HAS_ONNX = False


@dataclass
class SVModelPT:
    model: torch.nn.Module
    model_cfg: ModelCfg
    device: torch.device


@dataclass
class SVModelONNX:
    session: Any  # ort.InferenceSession
    input_name: str
    model_cfg: ModelCfg
    providers: Tuple[str, ...]


SVModel = Union[SVModelPT, SVModelONNX]


def _l2norm_t(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def _l2norm_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def build_model_pt(model_cfg: ModelCfg, device: torch.device) -> torch.nn.Module:
    model = ECAPA_TDNN(
        in_channels=model_cfg.feat_dim,
        channels=model_cfg.channels,
        embd_dim=model_cfg.emb_dim,
    )
    model.to(device)
    model.eval()
    return model


def load_sv(
    model_path: str,
    *,
    device: str = "cpu",
    use_onnx: Optional[bool] = None,
    providers: Optional[list[str]] = None,
) -> Tuple[SVModel, Dict]:
    """
    统一加载入口：
    - .pt: 返回 SVModelPT
    - .onnx: 返回 SVModelONNX
    """
    p = Path(model_path)
    suffix = p.suffix.lower()
    if use_onnx is None:
        use_onnx = suffix == ".onnx"

    if use_onnx:
        if not _HAS_ONNX:
            raise ImportError(
                "onnxruntime 未安装。请安装：pip install onnxruntime  或  onnxruntime-gpu"
            )
        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )

        session = ort.InferenceSession(str(p), providers=providers)
        input_name = session.get_inputs()[0].name

        # NOTE:
        # 你现在的代码这里 model_cfg = ModelCfg() 是默认值。
        # 如果你导出 onnx 时没有把 cfg 一并保存，那只能用默认值。
        # 如果你希望严格一致，建议你在导出时把 model_cfg 写进 json/pt，然后这里读取。
        model_cfg = ModelCfg()

        sv = SVModelONNX(
            session=session,
            input_name=input_name,
            model_cfg=model_cfg,
            providers=tuple(providers),
        )
        return sv, {"path": str(p), "backend": "onnx", "providers": providers}

    # PyTorch
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    ckpt = load_ckpt(str(p), map_location="cpu")
    model_cfg = ModelCfg(**ckpt["model_cfg"])
    model = build_model_pt(model_cfg, dev)
    model.load_state_dict(ckpt["model"], strict=True)

    sv = SVModelPT(model=model, model_cfg=model_cfg, device=dev)
    return sv, ckpt


def _ensure_3d_fbank(feat: torch.Tensor) -> torch.Tensor:
    """
    wav_to_fbank 可能返回：
      - [T, F]
      - [1, T, F]
      - [N, T, F]
    这里统一成 [N, T, F]，保证 ONNX/PT 都一致。
    """
    if not isinstance(feat, torch.Tensor):
        feat = torch.as_tensor(feat)

    if feat.dim() == 2:
        # [T, F] -> [1, T, F]
        feat = feat.unsqueeze(0)
    elif feat.dim() == 3:
        # [N, T, F] OK
        pass
    else:
        raise ValueError(f"Unexpected fbank dims: {feat.dim()}, shape={tuple(feat.shape)}")

    return feat


def _adapt_onnx_input_rank(session: Any, input_name: str, x3: np.ndarray) -> np.ndarray:
    """
    ONNX 模型输入 rank 可能是 2 或 3（绝大部分是 3: [B,T,F]）。
    这里根据 session 的 input shape 尝试适配。
    """
    inp = session.get_inputs()[0]
    shp = inp.shape  # e.g. ['batch', 'time', 80] or [None, None, 80] etc.
    # 有些 onnxruntime 返回 None / 'batch' 这种动态维度
    expected_rank = len(shp) if shp is not None else 3

    if expected_rank == 3:
        return x3  # [N,T,F]
    if expected_rank == 2:
        # 模型吃 [T,F]，那我们只能喂单条 crop（取第一个或平均）
        # 更合理：平均多个 crop 的特征，再喂
        x2 = x3.mean(axis=0)  # [T,F]
        return x2.astype(np.float32)

    # 其他 rank 极少见，直接抛错让你看清楚
    raise ValueError(
        f"Unsupported ONNX input rank={expected_rank}, input_shape={shp}, x3_shape={x3.shape}"
    )


@torch.no_grad()
def extract_embedding_pt(
    sv: SVModelPT,
    wav_path: str,
    *,
    num_crops: int = 5,
    crop_sec: float = 3.0,
) -> torch.Tensor:
    wav = load_wav_mono(wav_path, target_sr=sv.model_cfg.sample_rate)  # [T]
    feat = wav_to_fbank(
        wav,
        n_mels=sv.model_cfg.feat_dim,
        num_crops=num_crops,
        crop_sec=crop_sec,
    )  # [N,T,F] or [T,F] depending your impl
    feat = _ensure_3d_fbank(feat)  # -> [N,T,F]
    x = feat.to(sv.device)

    emb = sv.model(x)  # [N,D] or [D] depending model
    if emb.dim() == 2:
        emb = emb.mean(dim=0)  # crop average -> [D]
    emb = _l2norm_t(emb).reshape(-1).detach().cpu()
    return emb


def extract_embedding_onnx(
    sv: SVModelONNX,
    wav_path: str,
    *,
    num_crops: int = 5,
    crop_sec: float = 3.0,
) -> np.ndarray:
    wav = load_wav_mono(wav_path, target_sr=sv.model_cfg.sample_rate)  # [T]

    feat = wav_to_fbank(
        wav,
        n_mels=sv.model_cfg.feat_dim,
        num_crops=num_crops,
        crop_sec=crop_sec,
    )
    feat = _ensure_3d_fbank(feat)  # [N,T,F]

    x3 = feat.detach().cpu().numpy().astype(np.float32)  # [N,T,F]
    xin = _adapt_onnx_input_rank(sv.session, sv.input_name, x3)

    outs = sv.session.run(None, {sv.input_name: xin})

    emb = outs[0]
    # 可能是：
    #   [N,D]  (batch 输出)
    #   [1,D]
    #   [D]
    if emb is None:
        raise RuntimeError("ONNX session returned None for embedding output.")

    emb = np.asarray(emb)

    if emb.ndim == 2:
        # [N,D] -> 平均
        emb = emb.mean(axis=0)
    elif emb.ndim == 1:
        pass
    else:
        # 极少数模型输出 [1,1,D] 之类
        emb = emb.reshape(-1)

    emb = _l2norm_np(emb).reshape(-1).astype(np.float32)
    return emb


def extract_embedding(
    sv: SVModel,
    wav_path: str,
    *,
    num_crops: int = 5,
    crop_sec: float = 3.0,
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(sv, SVModelPT):
        return extract_embedding_pt(sv, wav_path, num_crops=num_crops, crop_sec=crop_sec)
    return extract_embedding_onnx(sv, wav_path, num_crops=num_crops, crop_sec=crop_sec)


def cosine_score(
    sv: SVModel,
    wav1: str,
    wav2: str,
    *,
    num_crops: int = 5,
    crop_sec: float = 3.0,
) -> float:
    e1 = extract_embedding(sv, wav1, num_crops=num_crops, crop_sec=crop_sec)
    e2 = extract_embedding(sv, wav2, num_crops=num_crops, crop_sec=crop_sec)

    if isinstance(e1, torch.Tensor):
        # e1,e2: [D]
        return float(torch.sum(e1 * e2).item())

    # numpy: [D]
    return float(np.dot(e1, e2))