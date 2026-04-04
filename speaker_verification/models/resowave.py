import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.models.wrr_mixer import HybridEPA_WRR_Block
from speaker_verification.models.head.reat import REAT_DiarizationHead


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)
    )


class FeedForwardAdapter(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class ResoWave(nn.Module):
    def __init__(
        self,
        in_channels=80,
        channels=512,
        embedding_dim=192,
        max_mix_speakers=5,
        post_ffn_hidden_dim=None,
        post_ffn_dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        legacy_embedding_dim = kwargs.pop("embd_dim", None)
        alias_embedding_dim = kwargs.pop("emb_dim", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        if legacy_embedding_dim is not None and embedding_dim != 192:
            raise TypeError("Use only one of `embedding_dim`, `emb_dim`, or `embd_dim`.")
        if alias_embedding_dim is not None and embedding_dim != 192:
            raise TypeError("Use only one of `embedding_dim`, `emb_dim`, or `embd_dim`.")

        if legacy_embedding_dim is not None:
            embedding_dim = legacy_embedding_dim
        if alias_embedding_dim is not None:
            embedding_dim = alias_embedding_dim

        embedding_dim = int(embedding_dim)
        post_ffn_hidden_dim = int(post_ffn_hidden_dim or channels * 2)
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = HybridEPA_WRR_Block(channels, channels, wt_levels=3)
        self.layer4 = HybridEPA_WRR_Block(channels, channels, wt_levels=3)
        self.post_out4_ffn = FeedForwardAdapter(
            dim=channels,
            hidden_dim=post_ffn_hidden_dim,
            dropout=post_ffn_dropout,
        )

        self.diar_head = REAT_DiarizationHead(
            in_dim=channels,
            emb_dim=embedding_dim,
            num_speakers_max=max_mix_speakers,
        )

    def forward(self, x, return_diarization=False):
        """
        x: [B,T,80]
        """
        x = x.transpose(1, 2)  # [B,80,T]

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out1 + out2)
        out4 = self.layer4(out1 + out2 + out3)  # [B,512,T]

        frame_feat = out4.transpose(1, 2).contiguous()  # [B,T,512]
        frame_feat = self.post_out4_ffn(frame_feat)
        frame_embeds, slot_logits, activity_logits, count_logits = self.diar_head(frame_feat)
        global_emb = F.normalize(frame_embeds.mean(dim=1), p=2, dim=-1)

        if return_diarization:
            return global_emb, frame_embeds, slot_logits, activity_logits, count_logits
        return global_emb
