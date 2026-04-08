import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.res_proj = None
        if in_channels != out_channels:
            self.res_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.res_proj is None else self.res_proj(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return out + residual


class Res2SEBlock(nn.Module):
    def __init__(self, channels: int, scale: int = 4, se_ratio: int = 8, dropout: float = 0.1):
        super().__init__()
        if channels % scale != 0:
            raise ValueError("channels must be divisible by scale")
        if channels % se_ratio != 0:
            raise ValueError("channels must be divisible by se_ratio")

        width = channels // scale
        self.scale = scale
        self.pre = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.pre_bn = nn.BatchNorm1d(channels)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(width, width, kernel_size=3, padding=1, bias=False)
                for _ in range(scale - 1)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale - 1)])
        self.post = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.post_bn = nn.BatchNorm1d(channels)
        self.se_fc1 = nn.Linear(channels, channels // se_ratio)
        self.se_fc2 = nn.Linear(channels // se_ratio, channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.pre_bn(self.pre(x)))

        splits = torch.chunk(x, self.scale, dim=1)
        running = splits[0]
        outputs = [running]

        for idx in range(self.scale - 1):
            running = running + splits[idx + 1]
            running = self.act(self.bns[idx](self.convs[idx](running)))
            outputs.append(running)

        x = torch.cat(outputs, dim=1)
        x = self.post_bn(self.post(x))

        se = x.mean(dim=-1)
        se = self.act(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se)).unsqueeze(-1)
        x = x * se
        x = self.drop(x)
        return self.act(x + residual)


class LightweightSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, ffn_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class FeedForwardAdapter(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class AcousticEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        channels: int = 512,
        out_dim: int = 256,
        post_ffn_hidden_dim: int = 1024,
        post_ffn_dropout: float = 0.1,
    ):
        super().__init__()
        attn_heads = 8 if out_dim >= 256 else 4
        self.stem = ConvBlock(in_channels, channels, kernel_size=5, dropout=post_ffn_dropout)
        self.block1 = Res2SEBlock(channels, scale=8, dropout=post_ffn_dropout)
        self.block2 = Res2SEBlock(channels, scale=8, dropout=post_ffn_dropout)
        self.block3 = Res2SEBlock(channels, scale=8, dropout=post_ffn_dropout)
        self.proj = nn.Linear(channels, out_dim)
        self.attn1 = LightweightSelfAttentionBlock(
            d_model=out_dim,
            nhead=attn_heads,
            ffn_dim=out_dim * 4,
            dropout=post_ffn_dropout,
        )
        self.attn2 = LightweightSelfAttentionBlock(
            d_model=out_dim,
            nhead=attn_heads,
            ffn_dim=out_dim * 4,
            dropout=post_ffn_dropout,
        )
        self.post_ffn = FeedForwardAdapter(
            dim=out_dim,
            hidden_dim=post_ffn_hidden_dim,
            dropout=post_ffn_dropout,
        )
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, F]
            valid_mask: [B, T]

        Returns:
            z: [B, T, D]
        """
        x = x.transpose(1, 2).contiguous()  # [B, F, T]
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.transpose(1, 2).contiguous()  # [B, T, C]
        x = self.proj(x)

        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask.bool()

        x = self.attn1(x, key_padding_mask=key_padding_mask)
        x = self.attn2(x, key_padding_mask=key_padding_mask)
        x = self.post_ffn(x)
        x = self.out_norm(x)

        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).float()
        return x
