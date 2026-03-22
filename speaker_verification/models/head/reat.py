import torch
import torch.nn as nn
import torch.nn.functional as F


class REAT_DiarizationHead(nn.Module):
    """
    LSTM + Silence Prototype + Energy Gate + LayerNorm

    输出定义：
    - slot_logits[..., 0]   -> silence slot
    - slot_logits[..., 1:]  -> speaker slots
    - count_logits 类别定义：
        0 -> 0 speakers
        1 -> 1 speaker
        ...
        K -> K speakers
    """

    def __init__(self, in_dim=512, emb_dim=192, num_speakers_max=5):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_spk = num_speakers_max

        self.frame_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, emb_dim),
        )

        self.norm = nn.LayerNorm(emb_dim)

        self.temporal = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.residual = nn.Linear(emb_dim, emb_dim)

        self.silence_proto = nn.Parameter(torch.randn(1, emb_dim))
        self.speaker_protos = nn.Parameter(torch.randn(num_speakers_max, emb_dim))
        self.slot_scale = nn.Parameter(torch.tensor(30.0))

        self.activity_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim // 2, 1),
        )

        # 0..K，共 K+1 类
        self.count_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, num_speakers_max + 1),
        )

    def forward(self, frame_feat):
        """
        frame_feat: [B, T, C]
        """
        embeds = self.frame_proj(frame_feat)  # [B, T, D]

        energy = frame_feat.abs().mean(dim=-1, keepdim=True) + 1e-8
        embeds = embeds * torch.sigmoid(energy * 5.0)

        embeds = F.normalize(embeds, dim=-1)
        embeds = self.norm(embeds)

        if self.training:
            embeds = embeds + torch.randn_like(embeds) * 0.005

        temporal_out, _ = self.temporal(embeds)
        temporal_out = temporal_out + self.residual(embeds)
        frame_embeds = F.normalize(temporal_out, dim=-1)

        # slot logits: [silence | speakers]
        all_protos = torch.cat([self.silence_proto, self.speaker_protos], dim=0)  # [1+K, D]
        slot_logits = torch.matmul(frame_embeds, all_protos.T) * self.slot_scale   # [B,T,1+K]

        activity_logits = self.activity_head(frame_embeds).squeeze(-1)  # [B,T]

        utt_feat = frame_embeds.mean(dim=1)   # [B,D]
        count_logits = self.count_head(utt_feat)  # [B,K+1]

        return frame_embeds, slot_logits, activity_logits, count_logits