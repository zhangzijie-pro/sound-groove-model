import torch
import torch.nn as nn
import torch.nn.functional as F


class REAT_DiarizationHead(nn.Module):
    """
    End-to-end frame-wise diarization head.

    The head predicts K speaker activity logits for every frame:
    - diar_logits[..., k] -> activity logit of speaker slot k

    Speech activity and speaker count are derived from diarization masks instead of
    being predicted by separate auxiliary heads.
    """

    def __init__(self, in_dim=512, emb_dim=192, num_speakers_max=5, dropout=0.1):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.max_spk = int(num_speakers_max)
        if self.emb_dim % 2 != 0:
            raise ValueError("emb_dim must be even for the bidirectional LSTM head.")

        self.input_norm = nn.LayerNorm(in_dim)
        self.frame_proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.temporal_norm = nn.LayerNorm(emb_dim)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Dropout(dropout),
        )
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
        )
        self.mask_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, self.max_spk),
        )

    def forward(self, frame_feat):
        """
        frame_feat: [B, T, C]
        returns:
            frame_embeds: [B, T, D]
            diar_logits:  [B, T, K]
        """
        x = self.input_norm(frame_feat)
        x = self.frame_proj(x)

        temporal_out, _ = self.temporal(x)
        frame_state = self.temporal_norm(temporal_out + x)
        frame_state = frame_state + self.temporal_ffn(frame_state)
        frame_embeds = F.normalize(self.embedding_head(frame_state), dim=-1)
        diar_logits = self.mask_head(frame_state)
        return frame_embeds, diar_logits
