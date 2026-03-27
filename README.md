# 🎙️ Speaker-Aware Diarization and Verification


<div align="center">

  [![GitHub stars](https://img.shields.io/github/stars/zhangzijie-pro/Speaker-Verification.svg?style=social)](https://github.com/zhangzijie-pro/Speaker-Verification/stargazers)
  [![Hugging Face](https://img.shields.io/badge/HuggingFace-模型%20%26%20数据集-yellow.svg)](https://huggingface.co/zzj-pro)
  ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
  [![PyPI](https://img.shields.io/pypi/v/speaker-verification)](https://pypi.org/project/speaker-verification/)
  [![CI](https://img.shields.io/github/actions/workflow/status/zhangzijie-pro/Speaker-Verification/ci.yml?branch=main&label=CI)](https://github.com/zhangzijie-pro/Speaker-Verification/actions)
</div>

<div align="center">
  <a href="./Readme_ch.md">中文文档</a> •
  <a href="./docs/data_guide.md">Data Guide</a> •
  <a href="./docs/finetuning.md">Finetuning</a> •
  <a href="./CONTRIBUTING.md">Contributing</a>
</div>

![WT-EPA-WRR-REAT Model](docs/imgs/model.jpg)

## 🚀 Showcase

- `ResoWave` + `REAT`: joint modeling for activity, slot assignment, speaker count, and frame prototypes
- `GlobalSpeakerTracker`: stable global speaker IDs across chunked or streaming inference
- `JsonSpeakerBank`: CRUD speaker profiles plus open-set identification so low-confidence matches stay `unknown`

This is where it matters:

- **Streaming inference** for low-latency online recognition
- **Meeting analysis** for diarization timelines and structured summaries
- **Long-audio monitoring** for persistent listening, alerting, and unknown-speaker detection

## ⚡ Quick Start

Install the project:

```bash
git clone https://github.com/zhangzijie-pro/Speaker-Verification.git
cd Speaker-Verification
pip install -r requirements.txt
```

Run the core path in a few lines:

```python
import torch

from speaker_verification.models.resowave import ResoWave
from speaker_verification.speaker_bank import JsonSpeakerBank

model = ResoWave(
    in_channels=80,
    channels=64,
    embedding_dim=64,
    max_mix_speakers=4,
).eval()

bank = JsonSpeakerBank("artifacts/speaker_bank.json")
bank.add_speaker(
    "alice",
    torch.tensor([1.0, 0.0, 0.0]),
    display_name="Alice",
    overwrite=True,
)

fbank = torch.randn(1, 400, 80)
global_emb, frame_embeds, slot_logits, activity_logits, count_logits = model(
    fbank,
    return_diarization=True,
)

print("global_emb", tuple(global_emb.shape))
print("frame_embeds", tuple(frame_embeds.shape))
print("slot_logits", tuple(slot_logits.shape))
print("activity_logits", tuple(activity_logits.shape))
print("count_logits", tuple(count_logits.shape))
print("speaker_bank_size", len(bank.list_speakers()))
```

For development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 🧭 Advanced Scenarios

### Streaming Inference

Use [`examples/streaming_inference.py`](./examples/streaming_inference.py) as the low-latency entrypoint for microphone buffers, websocket audio streams, or real-time agent pipelines.

```bash
python3 examples/streaming_inference.py \
  --ckpt outputs/train_v1/best.pt \
  --speaker-bank artifacts/speaker_bank.json \
  --chunk-sec 2.0 \
  --step-sec 0.5
```

### Meeting Analysis

Use [`examples/meeting_analysis.py`](./examples/meeting_analysis.py) to produce diarization results and a machine-readable timeline from a meeting recording.

```bash
python3 examples/meeting_analysis.py \
  --ckpt outputs/train_v1/best.pt \
  --audio assets/meeting.wav \
  --output artifacts/meeting_analysis.json
```

### Long-audio Monitoring

Use [`examples/long_audio_monitoring.py`](./examples/long_audio_monitoring.py) when you need persistent monitoring, unknown-speaker ratios, and alert thresholds instead of one-off diarization output.

```bash
python3 examples/long_audio_monitoring.py \
  --ckpt outputs/train_v1/best.pt \
  --audio assets/monitor.wav \
  --alert-threshold 0.25
```

## 🗂️ Data and Finetuning

The documentation for data preparation and model adaptation lives under `docs/`:

- [`docs/data_guide.md`](./docs/data_guide.md): raw audio requirements, cleaning scripts, manifest schema, and the fast-track Hugging Face path
- [`docs/finetuning.md`](./docs/finetuning.md): YAML-driven finetuning, key hyperparameters, checkpoint loading, and practical tuning logic
- [`docs/specs/checkpoint-schema.md`](./docs/specs/checkpoint-schema.md): checkpoint metadata and compatibility rules
- [`docs/specs/data-schema.md`](./docs/specs/data-schema.md): processed pack and inference JSON schema

## 🤝 Community

If you want to contribute, start with [`CONTRIBUTING.md`](./CONTRIBUTING.md). If you hit a bug, use the issue templates. If you want to integrate the project into a product, open a discussion with the target workload, latency budget, and artifact constraints.

## License

Apache License 2.0. External datasets such as CN-Celeb remain governed by their original licenses and usage terms.

