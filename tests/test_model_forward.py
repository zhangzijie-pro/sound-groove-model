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

    frame_embeds, diar_logits = model(x)

    assert frame_embeds.shape == (2, 400, 64)
    assert diar_logits.shape == (2, 400, 4)
