import torch
import torch.nn as nn
import torch.nn.functional as F

class ExistenceLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, exist_logits: torch.Tensor, target_count: torch.Tensor):
        """
        exist_logits: [B,N]
        target_count: [B]
        """
        B, N = exist_logits.shape
        target = torch.zeros_like(exist_logits)

        for b in range(B):
            k = int(target_count[b].item())
            k = max(0, min(k, N))
            if k > 0:
                target[b, :k] = 1.0

        if self.pos_weight is not None:
            pos_weight = torch.tensor(self.pos_weight, device=exist_logits.device)
            return F.binary_cross_entropy_with_logits(
                exist_logits, target, pos_weight=pos_weight
            )
        return F.binary_cross_entropy_with_logits(exist_logits, target)