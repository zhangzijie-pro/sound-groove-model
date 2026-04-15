# Checkpoint Schema

Current checkpoints are PyTorch `.pt` dictionaries created by `speaker_verification.engine.checkpoint.save_checkpoint`.

## Current Format

```python
{
    "epoch": 140,
    "model_state": {...},
    "optimizer_state": {...},
    "scheduler_state": {...},
    "best_metric": 15.327921,
    "history": {...},
    "config": {...},
}
```

## Required For Inference

- `model_state`: EENDQueryModel state dict.
- `config`: resolved config used to rebuild the model.

If `config` is missing, inference falls back to the current default EENDQueryModel config, but this should only be used for old artifacts.

## Required For Resume Training

- `model_state`
- `optimizer_state`
- `scheduler_state`, when scheduler is enabled
- `epoch`
- `best_metric`
- `history`
- `config`

## Model Compatibility

These config values must match the checkpoint weights for strict loading:

```yaml
model.in_channels
model.channels
model.d_model
model.max_mix_speakers
model.decoder_type
model.decoder_layers
model.decoder_heads
model.decoder_ffn
model.post_ffn_hidden_dim
model.post_ffn_dropout
```

The recommended pretrained checkpoint is:

```text
outputs_trainali_synth/best.pt
```

## Export Artifacts

`scripts/export.py` writes:

```text
model_state.pt
config.json
export_meta.json
model.ts
model.onnx     # optional
model.mnn      # optional, requires model.onnx and mnnconvert
```

The exported model wrapper expects:

```text
fbank:      [B, T, 80]
valid_mask: [B, T]
```

It returns:

```text
frame_embeds:     [B, T, D]
exist_logits:     [B, N]
diar_logits:      [B, T, N]
activity_logits:  [B, T]
```
