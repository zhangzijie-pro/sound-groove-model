import torch
import torch.nn as nn

from speaker_verification.models.resowave import (
    Conv1dReluBn,
    SE_Res2Block,
    HybridEPA_WRR_Block,
    FeedForwardAdapter,
)

class AcousticEncoder(nn.Module):
    def __init__(
        self,
        in_channels=80,
        channels=512,
        out_dim=256,
        post_ffn_hidden_dim=1024,
        post_ffn_dropout=0.1,
    ):
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = HybridEPA_WRR_Block(channels, channels, wt_levels=3)
        self.layer4 = HybridEPA_WRR_Block(channels, channels, wt_levels=3)

        self.post_ffn = FeedForwardAdapter(
            dim=channels,
            hidden_dim=post_ffn_hidden_dim,
            dropout=post_ffn_dropout,
        )
        self.proj = nn.Linear(channels, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        return:
            z: [B, T, D]
        """
        x = x.transpose(1, 2)              # [B,F,T]
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out1 + out2)
        out4 = self.layer4(out1 + out2 + out3)

        h = out4.transpose(1, 2).contiguous()   # [B,T,C]
        h = self.post_ffn(h)
        z = self.norm(self.proj(h))
        return z