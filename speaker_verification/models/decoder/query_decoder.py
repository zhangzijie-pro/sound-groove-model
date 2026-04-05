import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_speakers=6,
    ):
        super().__init__()
        self.max_speakers = max_speakers
        self.query_embed = nn.Parameter(torch.randn(max_speakers, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out_norm = nn.LayerNorm(d_model)
        self.exist_head = nn.Linear(d_model, 1)

    def forward(self, memory: torch.Tensor):
        """
        memory: [B, T, D]
        return:
            attractors: [B, N, D]
            exist_logits: [B, N]
        """
        B = memory.size(0)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)   # [B,N,D]

        attractors = self.decoder(
            tgt=queries,
            memory=memory,
        )  # [B,N,D]

        attractors = self.out_norm(attractors)
        attractors = F.normalize(attractors, dim=-1)

        exist_logits = self.exist_head(attractors).squeeze(-1)      # [B,N]
        return attractors, exist_logits