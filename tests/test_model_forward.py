import torch

from speaker_verification.models.resowave import ResoWave


@torch.no_grad()
def test_resowave_forward():
    model = ResoWave(
        in_channels=80,
        channels=64,
        embedding_dim=64,
        max_mix_speakers=4,
    )
    x = torch.randn(2, 120, 80)

    global_emb, frame_embeds, slot_logits, activity_logits, count_logits = model(
        x, return_diarization=True
    )

    assert global_emb.shape == (2, 64)
    assert frame_embeds.shape == (2, 120, 64)
    assert slot_logits.shape == (2, 120, 5)
    assert activity_logits.shape == (2, 120)
    assert count_logits.shape == (2, 5)


def test_resowave_accepts_legacy_embedding_aliases():
    model_from_alias = ResoWave(channels=64, emb_dim=32)
    model_from_legacy = ResoWave(channels=64, embd_dim=32)

    assert model_from_alias.linear.out_features == 32
    assert model_from_legacy.linear.out_features == 32
