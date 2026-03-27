from speaker_verification.dataset.static_dataset import StaticMixDataset


def test_dataset_init(synthetic_processed_dataset):
    ds = StaticMixDataset(
        out_dir=str(synthetic_processed_dataset),
        manifest="train_manifest.jsonl",
        crop_sec=4.0,
        shuffle=False,
    )
    assert len(ds) == 2


def test_dataset_item_keys_and_shapes(synthetic_processed_dataset):
    ds = StaticMixDataset(
        out_dir=str(synthetic_processed_dataset),
        manifest="train_manifest.jsonl",
        crop_sec=4.0,
        shuffle=False,
    )
    short_item = ds[0]
    long_item = ds[1]

    for item in (short_item, long_item):
        assert "fbank" in item
        assert "target_matrix" in item
        assert "target_activity" in item
        assert "target_count" in item
        assert "valid_mask" in item
        assert item["fbank"].shape == (400, 80)
        assert item["target_matrix"].shape == (400, 4)
        assert item["target_activity"].shape == (400,)
        assert item["valid_mask"].shape == (400,)

    assert short_item["valid_mask"].sum().item() == 240
    assert long_item["valid_mask"].sum().item() == 400
