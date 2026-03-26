import pytest
from speaker_verification.dataset.static_dataset import StaticMixDataset


def test_dataset_init():
    ds = StaticMixDataset(
        out_dir="processed/static_mix_cnceleb2",
        manifest="train_manifest.jsonl",
        crop_sec=4.0,
        shuffle=False,
    )
    assert len(ds) > 0


def test_dataset_item_keys():
    ds = StaticMixDataset(
        out_dir="processed/static_mix_cnceleb2",
        manifest="train_manifest.jsonl",
        crop_sec=4.0,
        shuffle=False,
    )
    item = ds[0]
    assert "fbank" in item
    assert "target_matrix" in item
    assert "target_activity" in item
    assert "target_count" in item
    assert "valid_mask" in item