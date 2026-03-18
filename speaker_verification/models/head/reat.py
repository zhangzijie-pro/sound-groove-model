import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class REAT_DiarizationHead(nn.Module):
    def __init__(self, in_dim=512, emb_dim=192, num_speakers_max=5):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.max_spk = num_speakers_max

        self.frame_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, emb_dim),
        )

        # self.pos_encoding = PositionalEncoding(emb_dim)
        # encoder = nn.TransformerEncoderLayer(
        #     emb_dim,
        #     4,
        #     dim_feedforward=384,
        #     dropout=0.0,
        #     activation="relu",
        #     batch_first=True,
        #     norm_first=True
        # )
        # self.temporal = nn.TransformerEncoder(encoder, 1)

        self.temporal = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim // 2,   # 96
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.0
        )

        self.prototypes = nn.Parameter(torch.randn(num_speakers_max, emb_dim))
        self.register_buffer("prototype_momentum", torch.tensor(0.99))

        self.slot_scale = nn.Parameter(torch.tensor(30.0))  # learnable temperature

        self.activity_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim // 2, 1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, num_speakers_max),
        )

    def forward(self, frame_feat):
        """
        frame_feat: [B, T, 512]
        """
        embeds = self.frame_proj(frame_feat)          # [B,T,192]
        embeds = F.normalize(embeds, dim=-1)

        # embeds = self.pos_encoding(embeds)
        # temporal_out = self.temporal(embeds)          # [B,T,192]
        # frame_embeds = F.normalize(temporal_out, dim=-1)

        # LSTM
        temporal_out, _ = self.temporal(embeds)       # [B,T,192]
        frame_embeds = F.normalize(temporal_out, dim=-1)

        sim = torch.matmul(frame_embeds, self.prototypes.T) * self.slot_scale
        slot_logits = sim  # [B,T,K]

        activity_logits = self.activity_head(frame_embeds).squeeze(-1)

        utt_feat = frame_embeds.mean(dim=1)
        count_logits = self.count_head(utt_feat)

        return frame_embeds, slot_logits, activity_logits, count_logits