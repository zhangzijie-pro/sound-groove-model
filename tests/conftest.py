import json
from pathlib import Path

import pytest
import torch


@pytest.fixture()
def synthetic_processed_dataset(tmp_path: Path) -> Path:
    out_dir = tmp_path / "processed_dataset"
    mix_dir = out_dir / "mix_pt"
    mix_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "spk2id.json").write_text(
        json.dumps({"speaker_a": 0, "speaker_b": 1}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest_lines = []
    for idx, frames in enumerate((240, 520)):
        target_matrix = torch.zeros(frames, 4, dtype=torch.float32)
        target_matrix[: frames // 2, 0] = 1.0
        target_matrix[frames // 2 :, 1] = 1.0
        payload = {
            "fbank": torch.randn(frames, 80),
            "target_matrix": target_matrix,
            "target_activity": torch.ones(frames, dtype=torch.float32),
            "spk_label": idx % 2,
            "target_count": 2,
        }
        rel_path = f"mix_pt/sample_{idx}.pt"
        torch.save(payload, mix_dir / f"sample_{idx}.pt")
        manifest_lines.append(json.dumps({"pt": rel_path}, ensure_ascii=False))

    manifest_body = "\n".join(manifest_lines) + "\n"
    (out_dir / "train_manifest.jsonl").write_text(manifest_body, encoding="utf-8")
    (out_dir / "val_manifest.jsonl").write_text(manifest_body, encoding="utf-8")
    return out_dir

