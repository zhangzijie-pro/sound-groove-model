import torch
import torch.nn as nn
import torch.nn.functional as F

class AttractorMetricLoss(nn.Module):
    def __init__(self, pull_weight=1.0, sep_margin=0.3):
        super().__init__()
        self.pull_weight = pull_weight
        self.sep_margin = sep_margin

    def forward(self, frame_embeds, attractors, aligned_targets, valid_mask=None):
        """
        frame_embeds:   [B,T,D]
        attractors:     [B,N,D]
        aligned_targets:[B,T,N]  # PIT对齐后的target
        """
        B, T, D = frame_embeds.shape
        N = attractors.size(1)

        frame_embeds = F.normalize(frame_embeds, dim=-1)
        attractors = F.normalize(attractors, dim=-1)

        # pull
        sim = torch.matmul(frame_embeds, attractors.transpose(1, 2))  # [B,T,N]
        pull_mask = aligned_targets > 0.5

        if valid_mask is not None:
            pull_mask = pull_mask & valid_mask.unsqueeze(-1).bool()

        if pull_mask.any():
            pull_loss = (1.0 - sim[pull_mask]).mean()
        else:
            pull_loss = sim.new_tensor(0.0)

        # separation
        aa = torch.matmul(attractors, attractors.transpose(1, 2))  # [B,N,N]
        eye = torch.eye(N, device=aa.device, dtype=torch.bool).unsqueeze(0)
        sep_vals = aa.masked_fill(eye, -1.0)

        sep_loss = torch.relu(sep_vals - self.sep_margin)
        valid_sep = sep_loss > 0
        sep_loss = sep_loss[valid_sep].mean() if valid_sep.any() else aa.new_tensor(0.0)

        return pull_loss * self.pull_weight, sep_loss