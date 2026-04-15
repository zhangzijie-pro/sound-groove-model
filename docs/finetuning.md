# Finetuning Guide

This guide describes the current EENDQueryModel finetuning path.

## Base Checkpoint

Recommended pretrained checkpoint:

```text
outputs_trainali_synth/best.pt
```

The checkpoint contains both model weights and the resolved training config under the `config` key.

## Full-Model Finetuning

Use full-model finetuning when the new data is close to the Train_Ali-style diarization task and you want to adapt all layers:

```powershell
python train.py `
  run.mode=finetune `
  finetune.checkpoint_path=outputs_trainali_synth/best.pt `
  finetune.load_mode=full `
  finetune.strict_load=true `
  finetune.freeze_backbone=false `
  finetune.lr_scale=0.1 `
  output.save_dir=outputs/finetune_eend
```

The effective learning rate is:

```text
effective_lr = train.lr * finetune.lr_scale
```

With the default `train.lr=5e-5`, `finetune.lr_scale=0.1` gives `5e-6`.

## Head-Only Finetuning

Use this when the encoder is stable but you want to adapt query/assignment heads:

```powershell
python train.py `
  run.mode=finetune `
  finetune.checkpoint_path=outputs_trainali_synth/best.pt `
  finetune.load_mode=backbone_only `
  finetune.strict_load=false `
  finetune.freeze_backbone=true `
  output.save_dir=outputs/head_only_ft
```

Default trainable prefixes are:

```yaml
finetune.head_prefixes:
  - decoder.
  - assign_head.
```

## Loss

Current loss:

```text
total = PIT_BCE + lambda_exist * existence_BCE
```

Config keys:

```yaml
loss.pit_pos_weight: 1.2
loss.exist_pos_weight: 1.5
loss.lambda_exist: 0.1
```

There is no active dice, consistency, smoothness, metric-learning, speaker-bank, or contrastive loss in the current training path.

## Validation Thresholds

Default validation/inference thresholds:

```yaml
validate.speaker_activity_threshold: 0.55
validate.exist_threshold: 0.5
validate.min_active_frames: 5
```

If DER is high because of false alarm, raise `speaker_activity_threshold` slightly. If miss dominates, lower it slightly. Keep changes small, for example `0.03` to `0.05` at a time.

## Compatibility Checklist

Before loading a checkpoint, keep these fields compatible:

```yaml
model.in_channels: 80
model.channels: 512
model.d_model: 256
model.max_mix_speakers: 3
model.decoder_layers: 4
model.decoder_heads: 8
model.decoder_ffn: 512
model.post_ffn_hidden_dim: 512
model.post_ffn_dropout: 0.2
```

Changing these values requires retraining or non-strict partial loading.

## Metrics To Watch

Watch these together rather than optimizing a single number:

```text
DER, val_loss, count_acc, count_mae, act_f1, exist_acc
```

A useful checkpoint should reduce DER without collapsing predicted activity or speaker count.
