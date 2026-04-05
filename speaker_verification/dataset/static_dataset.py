import json
import random
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class StaticMixDataset(Dataset):
    def __init__(
        self,
        out_dir: str = "processed/static_mix_cnceleb2",
        manifest: str = "train_manifest.jsonl",
        crop_sec: float = 4.0,
        shuffle: bool = True,
    ):
        super().__init__()
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.manifest_path = self.out_dir / manifest
        assert self.manifest_path.is_file(), f"Missing manifest: {self.manifest_path}"

        spk2id_path = self.out_dir / "spk2id.json"
        assert spk2id_path.is_file(), f"Missing spk2id.json: {spk2id_path}"

        with spk2id_path.open("r", encoding="utf-8") as f:
            self.spk2id = json.load(f)

        self.num_classes = len(self.spk2id)
        self.items: List[str] = []

        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                self.items.append(j["pt"])

        if shuffle:
            random.shuffle(self.items)

        self.crop_frames = int(float(crop_sec) * 100)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Dict[str, Any]:
        rel_pt = self.items[idx]
        abs_pt = self.out_dir / rel_pt
        pack = torch.load(abs_pt, map_location="cpu", weights_only=False)

        fbank = pack["fbank"].float()                    # [T,80]
        target_matrix = pack["target_matrix"].float()    # [T,K]
        spk_label = int(pack.get("spk_label", -1))

        T = fbank.size(0)

        if T > self.crop_frames:
            s = random.randint(0, T - self.crop_frames)
            fbank = fbank[s:s + self.crop_frames]
            target_matrix = target_matrix[s:s + self.crop_frames]
            valid_mask = torch.ones(self.crop_frames, dtype=torch.bool)
        elif T < self.crop_frames:
            pad = self.crop_frames - T
            fbank = F.pad(fbank, (0, 0, 0, pad))
            target_matrix = F.pad(target_matrix, (0, 0, 0, pad))

            valid_mask = torch.zeros(self.crop_frames, dtype=torch.bool)
            valid_mask[:T] = True
        else:
            valid_mask = torch.ones(self.crop_frames, dtype=torch.bool)

        target_activity = (target_matrix.sum(dim=-1) > 0).float()
        speaker_presence = (target_matrix.sum(dim=0) > 0).float()
        target_count = int(speaker_presence.sum().item())

        return {
            "fbank": fbank,
            "spk_label": torch.tensor(spk_label, dtype=torch.long),
            "target_matrix": target_matrix,
            "target_activity": target_activity,
            "target_count": torch.tensor(target_count, dtype=torch.long),
            "speaker_presence": speaker_presence,
            "valid_mask": valid_mask,
        }
