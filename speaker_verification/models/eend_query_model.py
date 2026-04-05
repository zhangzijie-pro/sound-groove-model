import torch
import torch.nn as nn

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

    def forward(self, x: torch.Tensor):
        """
        x: [B,T,F]
        returns:
            frame_embeds: [B,T,D]
            attractors: [B,N,D]
            exist_logits: [B,N]
            diar_logits: [B,T,N]
        """
        frame_embeds = self.encoder(x)
        attractors, exist_logits = self.decoder(frame_embeds)
        diar_logits = self.assign_head(frame_embeds, attractors)
        return frame_embeds, attractors, exist_logits, diar_logits