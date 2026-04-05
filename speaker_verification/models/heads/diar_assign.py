import torch
import torch.nn as nn
import torch.nn.functional as F

class DotProductDiarizationHead(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, frame_embeds: torch.Tensor, attractors: torch.Tensor) -> torch.Tensor:
        """
        frame_embeds: [B,T,D]
        attractors:   [B,N,D]
        return:
            diar_logits: [B,T,N]
        """
        frame_embeds = F.normalize(frame_embeds, dim=-1)
        attractors = F.normalize(attractors, dim=-1)
        logits = torch.matmul(frame_embeds, attractors.transpose(1, 2))
        return logits * self.scale