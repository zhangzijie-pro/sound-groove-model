import torch
import torch.nn as nn
import torch.nn.functional as F


class REAT_DiarizationHead(nn.Module):
    """
    输出:
      frame_embeds:    [B, T, D]
      slot_logits:     [B, T, K]   # K个说话槽位的逐帧logits，给PIT用
      activity_logits: [B, T]
      count_logits:    [B, K]      # 说话人数辅助分类
    """

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

        # 每帧输出 K 个槽位 logits
        self.slot_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, num_speakers_max),
        )

        self.activity_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 1),
        )

        self.count_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, num_speakers_max),
        )

    def forward(self, frame_feat):
        """
        frame_feat: [B, T, in_dim]
        """
        frame_embeds = self.frame_proj(frame_feat)                 # [B,T,D]
        frame_embeds = F.normalize(frame_embeds, dim=-1)

        slot_logits = self.slot_head(frame_feat)                   # [B,T,K]
        activity_logits = self.activity_head(frame_feat).squeeze(-1)  # [B,T]

        utt_feat = frame_feat.mean(dim=1)                          # [B,in_dim]
        count_logits = self.count_head(utt_feat)                   # [B,K]

        return frame_embeds, slot_logits, activity_logits, count_logits