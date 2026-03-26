import torch

from speaker_verification.models.resowave import ResoWave


@torch.no_grad()
def test_resowave_forward():
    model = ResoWave(
        in_channels=80,
        channels=128,
        embd_dim=64,
        max_mix_speakers=4,
    )
    model.eval()

    x = torch.randn(2, 400, 80)

    global_emb, frame_embeds, slot_logits, activity_logits, count_logits = model(
        x, return_diarization=True
    )

    assert global_emb.shape == (2, 64)
    assert frame_embeds.shape == (2, 400, 64)
    assert slot_logits.shape == (2, 400, 5)      # silence + 4 speaker slots
    assert activity_logits.shape == (2, 400)
    assert count_logits.shape == (2, 5)          # 0..4 speakers