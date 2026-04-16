# 🎙️ Speaker-Aware Diarization and Verification

<div align="center">

  [![GitHub stars](https://img.shields.io/github/stars/zhangzijie-pro/Speaker-Verification.svg?style=social)](https://github.com/zhangzijie-pro/Speaker-Verification/stargazers)
  [![Hugging Face](https://img.shields.io/badge/HuggingFace-Models%20%26%20Datasets-yellow.svg)](https://huggingface.co/zzj-pro)
  ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
  [![PyPI](https://img.shields.io/pypi/v/speaker-verification)](https://pypi.org/project/speaker-verification/)
  [![CI](https://img.shields.io/github/actions/workflow/status/zhangzijie-pro/Speaker-Verification/ci.yml?branch=main&label=CI)](https://github.com/zhangzijie-pro/Speaker-Verification/actions)
</div>

<div align="center">
  <a href="./Readme_ch.md">中文</a> •
  <a href="./docs/finetuning.md">Fine-tuning Guide</a> •
  <a href="./CONTRIBUTING.md">Contributing Guide</a>
</div>

## 🚀 Core

This project implements an **End-to-End (E2E)** speaker-aware speech processing toolkit. The core model is **`EENDQueryModel`**, built with the following jointly trained components:

- **AcousticEncoder**: A multi-scale acoustic encoder using Res2SEBlock + Lightweight Self-Attention + FeedForwardAdapter. It maps FBank features to normalized frame-level embeddings (frame embeddings) for speaker verification.
- **QueryDecoder** (or `LSTMAttractorDecoder` / EDA-LSTM): Generates speaker attractors and existence logits, enabling dynamic speaker counting and activity detection.
- **DotProductDiarizationHead**: Computes frame-to-speaker assignment logits via dot-product between frame embeddings and attractors.
- **Activity Head**: Experimental frame-level activity branch. The current lean training path optimizes diarization masks and existence scores; activity during validation/inference is derived from gated diarization masks.

**EENDQueryModel** outputs:
- `frame_embeds`: Normalized frame embeddings (for verification / clustering)
- `attractors`: Speaker prototype embeddings
- `exist_logits`: Speaker existence scores
- `diar_logits`: Frame-speaker assignment logits
- `activity_logits`: Auxiliary speech activity logits

**SpeakerDiarizationPipeline**: High-level inference wrapper with:
- Chunked processing with overlap (streaming-friendly)
- Post-processing (short segment removal, gap filling, speaker re-labeling)
- Structured diarization output (RTTM, JSON, segment list)

**Real-world use cases**:
- **Streaming / long-audio inference**: Chunked processing + post-processing for low-latency online diarization
- **Meeting / monitoring analysis**: Precise speaker timelines, activity segments, and structured summaries
- **Speaker verification foundation**: Normalized frame embeddings support downstream metric learning or open-set recognition (speaker bank functionality to be extended in future iterations)

## ⚡ Quick Start

Clone and install:

```bash
git clone https://github.com/zhangzijie-pro/Speaker-Verification.git
git checkout E2E
cd Speaker-Verification
pip install -r requirements.txt
```

### Low-level model usage (direct EENDQueryModel)

```python
import torch
from speaker_verification.models.eend_query_model import EENDQueryModel

model = EENDQueryModel(
    in_channels=80,
    enc_channels=512,
    d_model=256,
    max_speakers=6,
    decoder_type="query"   # or "eda_lstm"
).eval()

# Input: FBank [B, T, F] (example: T=400, F=80)
fbank = torch.randn(1, 400, 80)

frame_embeds, attractors, exist_logits, diar_logits, activity_logits = model(fbank)

print("frame_embeds :", tuple(frame_embeds.shape))      # [B, T, D]
print("attractors   :", tuple(attractors.shape))       # [B, N, D]
print("diar_logits  :", tuple(diar_logits.shape))      # [B, T, N]
print("exist_logits :", tuple(exist_logits.shape))     # [B, N]
print("activity_logits :", tuple(activity_logits.shape))  # [B, T]
```

### High-level Pipeline usage (recommended for real audio)

```python
from speaker_verification.inference import SpeakerDiarizationPipeline

# Load from checkpoint (supports Hydra config)
pipeline = SpeakerDiarizationPipeline.from_pretrained(
    checkpoint_path="path/to/best.pt",   # or Hugging Face path
    chunk_sec=5.0,          # chunk length
    hop_sec=2.0,            # overlap step
    speaker_activity_threshold=0.55,
    merge_gap_sec=0.2,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Process audio file
result = pipeline.predict_file(
    audio_path="meeting.wav",
    recording_id="meeting_001"
)

# Structured output
print(result.to_dict())          # contains segments, speaker_prob, etc.
pipeline.save_rttm(result, "output.rttm")   # save as standard RTTM format
```

**Development installation**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 🗂️ Data & Fine-tuning

All documentation related to data preparation and model adaptation is in the `docs/` folder:

- [`docs/finetuning.md`](./docs/finetuning.md): YAML-driven fine-tuning, key hyperparameters, checkpoint loading, and tuning logic
- [`docs/specs/checkpoint-schema.md`](./docs/specs/checkpoint-schema.md): Checkpoint metadata and compatibility rules

## 🤝 Community

To contribute code, please start with [`CONTRIBUTING.md`](./CONTRIBUTING.md). For issues, use the issue templates. If you plan to integrate this project into a real product, please clearly specify workload, latency budget, and artifact constraints in discussions.

## License

Apache License 2.0
