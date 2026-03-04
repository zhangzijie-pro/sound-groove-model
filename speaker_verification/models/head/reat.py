import torch
import torch.nn as nn

class REAT_DiarizationHead(nn.Module):
    def __init__(self, num_speakers_max=10, dim=512):
        super().__init__()
        self.max_spk = num_speakers_max
        self.memory = nn.Parameter(torch.randn(1, num_speakers_max, dim))
        self.resonance_assign = nn.Linear(dim, num_speakers_max)
        self.activity_head = nn.Linear(dim, 1)
        self.count_head = nn.Linear(dim, 1)
        
    def forward(self, frame_feat):  # frame_feat: (B, T, dim) 
        scores = torch.einsum('btd,nsd->btsn', frame_feat, self.memory)  # (B,T,S,10)
        ids = scores.argmax(dim=-1)  # (B,T) → Speaker ID 1~10
        activity = torch.sigmoid(self.activity_head(frame_feat)).squeeze(-1)
        count = torch.sigmoid(self.count_head(frame_feat.mean(1))).squeeze(-1) * self.max_spk
        return ids, activity, count.round().int()
    
