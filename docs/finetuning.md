# Finetuning Guide

This guide describes the current **E2E** finetuning workflow for `EENDQueryModel`.

## Base Checkpoint

Recommended pretrained checkpoint (trained on synthetic + real diarization data):

```text
outputs_trainali_synth/best.pt
```

The checkpoint contains both model weights and the full resolved Hydra config under the `config` key.

## Full-Model Finetuning

Use full-model finetuning when your new data is similar to the original training distribution and you want to adapt the entire model:

```bash
python train.py \
  run.mode=finetune \
  finetune.checkpoint_path=outputs_trainali_synth/best.pt \
  finetune.load_mode=full \
  finetune.strict_load=true \
  finetune.freeze_backbone=false \
  finetune.lr_scale=0.1 \
  output.save_dir=outputs/finetune_eend
```

The effective learning rate is calculated as:

```text
effective_lr = train.lr * finetune.lr_scale
```

With the default `train.lr=5e-5`, `finetune.lr_scale=0.1` gives `5e-6`.

## Head-Only Finetuning

Use this mode when the AcousticEncoder is already well-trained and you only want to adapt the decoder / diarization heads:

```bash
python train.py \
  run.mode=finetune \
  finetune.checkpoint_path=outputs_trainali_synth/best.pt \
  finetune.load_mode=backbone_only \
  finetune.strict_load=false \
  finetune.freeze_backbone=true \
  output.save_dir=outputs/head_only_ft
```

Default trainable parameter prefixes (you can override via config):

```yaml
finetune.head_prefixes:
  - decoder.
  - assign_head.
```

## Loss

Current loss function:

```text
total = PIT_BCE + lambda_exist * existence_BCE
```

Configurable weights:

```yaml
loss.pit_pos_weight: 1.2
loss.exist_pos_weight: 1.5
loss.lambda_exist: 0.1
```

No Dice loss, consistency loss, smoothness loss, metric-learning loss, speaker-bank loss, or contrastive loss is active in the current training pipeline.

## Validation Thresholds

Default thresholds used during validation and inference:

```yaml
validate.speaker_activity_threshold: 0.55
validate.exist_threshold: 0.5
validate.min_active_frames: 5
```

- If DER is dominated by false alarms → slightly increase `speaker_activity_threshold` (e.g. +0.03~0.05).
- If DER is dominated by misses → slightly decrease it.
- Always make small incremental changes and re-evaluate.

## Compatibility Checklist

Before loading a checkpoint, the following model hyperparameters **must match** (otherwise use `strict_load=false` or retrain from scratch):

```yaml
model:
  in_channels: 80
  channels: 512
  d_model: 256
  max_mix_speakers: 8
  decoder_type: "query"          # or "eda_lstm"
```

Other decoder internals (`decoder_layers`, `decoder_heads`, etc.) are currently fixed in the model implementation. Changing them requires retraining or non-strict partial loading.

## Metrics To Watch

Monitor these metrics together instead of optimizing a single number:

```text
DER, val_loss, count_acc, count_mae, act_f1, exist_acc
```

A good checkpoint should **reduce DER** while keeping speaker count and activity predictions stable (no collapse).

---

Training uses **Hydra** for configuration. All overrides shown above are passed directly to `train.py`. For more advanced settings (data, optimizer, scheduler, etc.), refer to `configs/experiment.yaml` and the Hydra config groups.
