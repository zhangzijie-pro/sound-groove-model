import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttractorDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        hidden_size=256,
        max_speakers=6,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.max_speakers = max_speakers
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.init_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.attractor_proj = nn.Linear(hidden_size, d_model)
        self.exist_head = nn.Linear(d_model, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.init_token, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.attractor_proj.weight)
        nn.init.zeros_(self.attractor_proj.bias)
        nn.init.xavier_uniform_(self.exist_head.weight)
        nn.init.constant_(self.exist_head.bias, -1.0)

    def forward(self, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor | None = None):
        """
        memory: [B,T,D]
        """
        del memory_key_padding_mask
        B = memory.size(0)

        _, (h_n, c_n) = self.encoder(memory)
        dec_in = self.init_token.expand(B, self.max_speakers, -1)

        dec_out, _ = self.decoder(dec_in, (h_n, c_n))
        attractors = self.attractor_proj(dec_out)      # [B,N,D]
        attractors = F.normalize(attractors, dim=-1)
        exist_logits = self.exist_head(attractors).squeeze(-1)
        return attractors, exist_logits
