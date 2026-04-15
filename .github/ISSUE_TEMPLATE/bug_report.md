---
name: Bug Report
about: Report a reproducible defect in training, inference, export, or examples
title: "[Bug] "
labels: bug, needs-triage
assignees: ""
---

## Summary

Describe the bug in one or two sentences. Be concrete about what failed and where.

## Environment

- Project version / commit:
- Python version:
- OS / architecture:
- PyTorch / torchaudio version:
- CUDA / ROCm / CPU-only:
- Installation method (`pip`, editable install, Docker, etc.):

## Reproduction

1. State the exact command or code path.
2. Provide the config overrides, checkpoint path, and input artifact details.
3. Describe the smallest reproducible input if possible.

## Expected Behavior

What should have happened?

## Actual Behavior

What happened instead?

## Logs and Screenshots

Paste the full stack trace, stderr, or structured logs here. Screenshots are useful for UI / notebook / workflow issues.

```text
Paste logs here
```

## Artifact Details

- Checkpoint schema version:
- Input sample rate / duration:
- `max_mix_speakers`:
- Checkpoint path:
- Inference/export entrypoint:

## Additional Context

Anything else that narrows the search space: regression status, suspected PR, related issues, or hardware constraints.

