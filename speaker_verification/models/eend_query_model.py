import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.models.encoder.acoustic_encoder import AcousticEncoder
from speaker_verification.models.decoder.query_decoder import QueryDecoder
from speaker_verification.models.decoder.attractor_decoder_lstm import LSTMAttractorDecoder
from speaker_verification.models.heads.diar_assign import DotProductDiarizationHead

class EENDQueryModel(nn.Module):
    def __init__(
        self,
        in_channels=80,
        enc_channels=512,
        d_model=256,
        max_speakers=6,
        decoder_type="query",
        post_ffn_hidden_dim=1024,
        post_ffn_dropout=0.1,
        decoder_layers=4,
        decoder_heads=8,
        decoder_ffn=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = AcousticEncoder(
            in_channels=in_channels,
            channels=enc_channels,
            out_dim=d_model,
            post_ffn_hidden_dim=post_ffn_hidden_dim,
            post_ffn_dropout=post_ffn_dropout,
        )

        if decoder_type == "query":
            self.decoder = QueryDecoder(
                d_model=d_model,
                nhead=decoder_heads,
                num_layers=decoder_layers,
                dim_feedforward=decoder_ffn,
                dropout=dropout,
                max_speakers=max_speakers,
            )
        elif decoder_type == "eda_lstm":
            self.decoder = LSTMAttractorDecoder(
                d_model=d_model,
                hidden_size=d_model,
                max_speakers=max_speakers,
                num_layers=1,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

        self.assign_head = DotProductDiarizationHead(scale=d_model ** 0.5)
        self.activity_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None):
        """
        x: [B,T,F]
        returns:
            frame_embeds: [B,T,D]
            attractors: [B,N,D]
            exist_logits: [B,N]
            diar_logits: [B,T,N]
            activity_logits: [B,T]
        """
        frame_embeds = self.encoder(x, valid_mask=valid_mask)
        memory_key_padding_mask = None
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            memory_key_padding_mask = ~valid_mask
        attractors, exist_logits = self.decoder(
            frame_embeds,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        diar_logits = self.assign_head(frame_embeds, attractors)
        activity_logits = self.activity_head(frame_embeds).squeeze(-1)
        if valid_mask is not None:
            activity_logits = activity_logits.masked_fill(~valid_mask, -12.0)

        # Keep frame embeddings normalized for metric learning and prototype matching.
        frame_embeds = F.normalize(frame_embeds, dim=-1)
        return frame_embeds, attractors, exist_logits, diar_logits, activity_logits
