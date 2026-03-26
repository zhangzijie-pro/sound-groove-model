import torch

from speaker_verification.models.resowave import ResoWave


def test_resowave_forward():
    model = ResoWave(
        in_channels=80,
        channels=128,
        embedding_dim=64,
        max_mix_speakers=4,
    )
    x = torch.randn(2, 400, 80)

    global_emb, frame_embeds, slot_logits, activity_logits, count_logits = model(
        x, return_diarization=True
    )

    assert global_emb.shape[0] == 2
    assert frame_embeds.shape[:2] == (2, 400)
    assert slot_logits.shape[:2] == (2, 400)
    assert activity_logits.shape[:2] == (2, 400)
    assert count_logits.shape[0] == 2