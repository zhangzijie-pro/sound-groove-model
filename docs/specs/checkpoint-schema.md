# Checkpoint Schema

This repository treats checkpoints as versioned artifacts, not opaque `torch.save(...)` blobs.

## Canonical Format

Checkpoint files should be stored as `.pt` files containing a top-level dictionary:

```python
{
  "schema_version": "sv.ckpt/v1",
  "model_name": "resowave",
  "task": "speaker-aware-diarization",
  "created_at": "2026-03-27T10:00:00+00:00",
  "git_commit": "abc1234",
  "python_version": "3.11.2",
  "torch_version": "2.2.1",
  "config": {...},
  "metrics": {
    "val_loss": 0.1234,
    "der": 6.52,
    "act_f1": 0.9421,
    "count_acc": 0.9110
  },
  "model_state": {...},
  "optimizer_state": {...},
  "scheduler_state": {...},
  "epoch": 42,
  "best_metric": 0.1234,
  "history": {...}
}
```

## Required Fields

- `schema_version`: versioned identifier. Current value: `sv.ckpt/v1`.
- `model_name`: short, stable architecture identifier.
- `task`: training target and checkpoint intent.
- `created_at`: ISO-8601 timestamp with timezone.
- `config`: fully resolved training config.
- `model_state`: PyTorch `state_dict`.

## Optional but Strongly Recommended

- `metrics`: scalar validation metrics used for ranking.
- `git_commit`: source revision used to create the artifact.
- `optimizer_state`, `scheduler_state`, `epoch`, `best_metric`, `history`: resume-training metadata.
- `export`: export-oriented metadata such as ONNX opset or quantization recipe.

## Compatibility Rules

- Backward-compatible additions may add new top-level keys.
- Breaking changes must bump `schema_version`.
- Inference-only checkpoints may omit optimizer and scheduler state.
- Loaders must fail fast when `model_name`, `max_mix_speakers`, or embedding dimensions are incompatible with the runtime config.

## Layout

```text
outputs/
├── best.pt
├── last.pt
├── metadata.json
├── model.onnx
└── SHA256SUMS
```

