from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorLike = Union[torch.Tensor, None]


@dataclass
class QuantizedTensorState:
    shape: Tuple[int, ...]
    idx: torch.Tensor
    orig_norm: torch.Tensor
    residual_sign: Optional[torch.Tensor] = None
    residual_norm: Optional[torch.Tensor] = None


def _make_random_orthogonal(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    a = torch.randn(d, d, device=device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    diag = torch.sign(torch.diag(r))
    diag[diag == 0] = 1.0
    q = q @ torch.diag(diag)
    return q


def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm, norm


class _BaseTurboQuant(nn.Module):
    def __init__(
        self,
        dim: int,
        bits: int,
        clip_sigma: float = 3.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 1234,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if bits <= 0:
            raise ValueError(f"bits must be positive, got {bits}")

        self.dim = int(dim)
        self.bits = int(bits)
        self.clip_sigma = float(clip_sigma)
        self.device_ = torch.device(device)
        self.dtype_ = dtype

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        with torch.random.fork_rng():
            torch.manual_seed(seed)
            rot = _make_random_orthogonal(self.dim, self.device_, self.dtype_)
        self.register_buffer("R", rot, persistent=True)
        self.register_buffer("RT", rot.transpose(0, 1).contiguous(), persistent=True)

    @property
    def levels(self) -> int:
        return 2 ** self.bits

    def _coord_clip(self) -> float:
        return self.clip_sigma / (self.dim ** 0.5)

    def _codebook(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        clip = self._coord_clip()
        return torch.linspace(-clip, clip, self.levels, device=device, dtype=dtype)

    def _quantize_rotated(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [N, D]
        返回 idx: [N, D]
        """
        codebook = self._codebook(y.device, y.dtype)  # [L]
        boundaries = (codebook[:-1] + codebook[1:]) * 0.5  # [L-1]
        y_clamped = y.clamp(min=codebook[0].item(), max=codebook[-1].item())
        idx = torch.bucketize(y_clamped, boundaries)
        return idx.long()

    def _dequantize_rotated(self, idx: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        codebook = self._codebook(device, dtype)
        y_hat = codebook[idx.long()]
        return y_hat

    def _flatten_last_dim(self, x: torch.Tensor) -> tuple[torch.Tensor, Tuple[int, ...]]:
        if x.size(-1) != self.dim:
            raise ValueError(f"Expected last dim == {self.dim}, got {tuple(x.shape)}")
        shape = tuple(x.shape)
        flat = x.reshape(-1, self.dim)
        return flat, shape

    def _restore_shape(self, x: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        return x.reshape(shape)

    def quant_dequant(self, x: torch.Tensor) -> torch.Tensor:
        state = self.quantize(x)
        return self.dequantize(state)

    def quantize(self, x: torch.Tensor) -> QuantizedTensorState:
        raise NotImplementedError

    def dequantize(self, state: QuantizedTensorState) -> torch.Tensor:
        raise NotImplementedError


class TurboQuantMSE(_BaseTurboQuant):
    def quantize(self, x: torch.Tensor) -> QuantizedTensorState:
        flat, shape = self._flatten_last_dim(x.float().to(self.device_))
        x_unit, orig_norm = _safe_normalize(flat)
        y = x_unit @ self.RT
        idx = self._quantize_rotated(y)
        return QuantizedTensorState(
            shape=shape,
            idx=idx,
            orig_norm=orig_norm,
            residual_sign=None,
            residual_norm=None,
        )

    def dequantize(self, state: QuantizedTensorState) -> torch.Tensor:
        y_hat = self._dequantize_rotated(
            state.idx.to(self.device_),
            device=self.device_,
            dtype=self.dtype_,
        )
        x_hat = y_hat @ self.R
        x_hat = F.normalize(x_hat, p=2, dim=-1) * state.orig_norm.to(self.device_)
        return self._restore_shape(x_hat, state.shape)


class TurboQuantProd(_BaseTurboQuant):
    def __init__(
        self,
        dim: int,
        bits: int,
        clip_sigma: float = 3.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 1234,
    ):
        if bits < 2:
            raise ValueError("TurboQuantProd requires bits >= 2")
        super().__init__(dim=dim, bits=bits, clip_sigma=clip_sigma, device=device, dtype=dtype, seed=seed)

        self.mse_branch = TurboQuantMSE(
            dim=dim,
            bits=bits - 1,
            clip_sigma=clip_sigma,
            device=device,
            dtype=dtype,
            seed=seed,
        )

        with torch.random.fork_rng():
            torch.manual_seed(seed + 17)
            s = torch.randn(self.dim, self.dim, device=self.device_, dtype=self.dtype_)
        self.register_buffer("S", s, persistent=True)
        self.register_buffer("ST", s.transpose(0, 1).contiguous(), persistent=True)

    def quantize(self, x: torch.Tensor) -> QuantizedTensorState:
        flat, shape = self._flatten_last_dim(x.float().to(self.device_))

        mse_state = self.mse_branch.quantize(flat)
        x_mse = self.mse_branch.dequantize(mse_state).reshape(-1, self.dim)

        residual = flat - x_mse
        residual_norm = residual.norm(p=2, dim=-1, keepdim=True)

        proj = residual @ self.ST
        sign = torch.sign(proj)
        sign[sign == 0] = 1.0

        return QuantizedTensorState(
            shape=shape,
            idx=mse_state.idx,
            orig_norm=mse_state.orig_norm,
            residual_sign=sign.to(torch.int8),
            residual_norm=residual_norm,
        )

    def dequantize(self, state: QuantizedTensorState) -> torch.Tensor:
        mse_state = QuantizedTensorState(
            shape=state.shape,
            idx=state.idx,
            orig_norm=state.orig_norm,
            residual_sign=None,
            residual_norm=None,
        )
        x_mse = self.mse_branch.dequantize(mse_state).reshape(-1, self.dim)

        if state.residual_sign is None or state.residual_norm is None:
            return self._restore_shape(x_mse, state.shape)

        sign = state.residual_sign.to(self.device_).float()
        gamma = state.residual_norm.to(self.device_)

        # x_qjl ≈ sqrt(pi/2)/d * gamma * S^T sign(Sr)
        coeff = (torch.tensor(torch.pi / 2.0, device=self.device_, dtype=self.dtype_).sqrt() / float(self.dim))
        x_qjl = coeff * (sign @ self.S)  # [N, D]
        x_qjl = F.normalize(x_qjl, p=2, dim=-1) * gamma.clamp_min(1e-8)

        x_hat = x_mse + x_qjl
        return self._restore_shape(x_hat, state.shape)


def build_turboquant(
    mode: str,
    dim: int,
    bits: int,
    clip_sigma: float = 3.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 1234,
):
    mode = str(mode).lower()
    if mode == "mse":
        return TurboQuantMSE(
            dim=dim,
            bits=bits,
            clip_sigma=clip_sigma,
            device=device,
            dtype=dtype,
            seed=seed,
        )
    if mode == "prod":
        return TurboQuantProd(
            dim=dim,
            bits=bits,
            clip_sigma=clip_sigma,
            device=device,
            dtype=dtype,
            seed=seed,
        )
    raise ValueError(f"Unsupported turboquant mode: {mode}")