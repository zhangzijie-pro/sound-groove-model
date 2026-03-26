# Speaker-Aware Diarization and Verification

<div align="center">

  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![GitHub stars](https://img.shields.io/github/stars/zhangzijie-pro/Speaker-Verification.svg?style=social)](https://github.com/zhangzijie-pro/Speaker-Verification/stargazers)
  [![Hugging Face](https://img.shields.io/badge/HuggingFace-模型%20%26%20数据集-yellow.svg)](https://huggingface.co/zzj-pro)
  ![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
  ![Task](https://img.shields.io/badge/Task-Speaker%20Verification-green)

</div>

<div align="center">
  <a href="Readme_ch.md">中文文档</a> • 
  <a href="https://github.com/zhangzijie-pro/Speaker-Verification">GitHub</a> • 
  <a href="https://huggingface.co/zzj-pro">Hugging Face</a>
</div>

**PyTorch** implementation of a **speaker-aware** multi-speaker audio model, focused on advanced speaker diarization and verification tasks.

This repository supports:

- Speaker Diarization
- Speaker Counting
- Speech Activity Detection (VAD)
- Dominant Speaker Estimation
- Chunk-level Speaker Tracking
- Future speaker-bank identity matching

Originally started as a speaker verification project, it has now evolved into a powerful **speaker-aware diarization** system.

---

## ✨ Features

- **ResoWave Backbone**
  - Temporal convolution frontend
  - SE-Res2 blocks
  - Hybrid WRR / EPA blocks
  - Attentive statistics pooling

- **REAT Diarization Head**
  - Frame-level speaker embeddings
  - Slot assignment logits
  - Activity logits
  - Speaker count logits

- **Multi-task Training**
  - PIT (Permutation Invariant Training) diarization loss
  - Activity detection loss
  - Speaker count loss
  - Frame-level prototype supervision

- **Advanced Inference Pipeline**
  - Chunk-level diarization
  - Speaking segment detection
  - Dominant speaker identification
  - Local slot IDs → Global speaker IDs (with tracker)

---

## Model Output

For each input audio chunk, the model outputs:

- Estimated number of speakers
- Frame-level speech activity
- Frame-level speaker slot assignment
- Speaking segments
- Dominant speaker slot
- Slot-level speaker prototypes (embeddings)

When the **GlobalSpeakerTracker** is enabled, it additionally provides:

- Local-to-global speaker mapping
- Global frame-level speaker IDs
- Currently active global speaker IDs

---

## Repository Structure

```bash
Speaker-Verification/
├── configs/
│   └── experiment.yaml
├── demos/
├── processed/
│   └── build_processed_dataset.py
├── speaker_verification/
│   ├── audio/
│   ├── dataset/
│   │   └── staticdataset.py
│   ├── engine/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── checkpoint.py
│   ├── interfaces/
│   │   ├── diar_interface.py
│   │   └── global_tracker.py
│   ├── loss/
│   ├── models/
│   ├── factory.py
│   ├── logging_utils.py
│   └── utils/
├── train.py
├── README.md
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/zhangzijie-pro/Speaker-Verification.git
cd Speaker-Verification
pip install -r requirements.txt
```

---

## Data Preparation

Build the processed training dataset:

```bash
python processed/build_processed_dataset.py --stage all
```

Available stages:

```bash
python processed/build_processed_dataset.py --stage cn
python processed/build_processed_dataset.py --stage mix
python processed/build_processed_dataset.py --stage add_single
python processed/build_processed_dataset.py --stage add_negative
```

The processed data will be saved in:

```
processed/static_mix_cnceleb2/
├── mix_pt/
├── train_manifest.jsonl
├── val_manifest.jsonl
└── spk2id.json
```

---

## Training

**Default training:**

```bash
python train.py
```

**Custom training:**

```bash
python train.py run.mode=train train.epochs=100 train.lr=3e-4 output.save_dir=outputs/train_v1
```

---

## Finetuning

**Standard finetuning:**

```bash
python train.py \
  run.mode=finetune \
  finetune.checkpoint_path=outputs/train_v1/best.pt \
  finetune.load_mode=full \
  finetune.freeze_backbone=false \
  finetune.lr_scale=0.1 \
  output.save_dir=outputs/finetune_v1 \
  output.monitor=der
```

**Train only the diarization head:**

```bash
python train.py \
  run.mode=finetune \
  finetune.checkpoint_path=outputs/train_v1/best.pt \
  finetune.load_mode=backbone_only \
  finetune.strict_load=false \
  finetune.freeze_backbone=true \
  output.save_dir=outputs/head_only_ft
```

---

## Resume Training

```bash
python train.py \
  run.mode=train \
  resume.enabled=true \
  resume.checkpoint_path=outputs/train_v1/last.pt \
  output.save_dir=outputs/train_v1
```

---

## Configuration

Main configuration file: `configs/experiment.yaml`

Key sections:

- `run`
- `model`
- `loss`
- `data`
- `train`
- `validate`
- `output`
- `finetune`
- `resume`

---

## Training Outputs

After training, the following files will be generated:

- `best.pt` — Best model checkpoint
- `last.pt` — Latest checkpoint
- `history.json` — Training history
- `train_YYYYMMDD_HHMMSS.log` — Training log

Example output directory:

```
outputs/train_v1/
├── best.pt
├── last.pt
├── history.json
└── train_20250326_211400.log
```

---

## Inference Example

```python
import torch

from speaker_verification.interfaces.diar_interface import SpeakerAwareDiarizationInterface
from speaker_verification.interfaces.global_tracker import GlobalSpeakerTracker

# Initialize model interface
diar = SpeakerAwareDiarizationInterface(
    ckpt_path="outputs/train_v1/best.pt",
    device="cuda",
    feat_dim=80,
    channels=512,
    emb_dim=256,
    max_mix_speakers=4,
)

tracker = GlobalSpeakerTracker()

# Example inference
wav = torch.randn(16000 * 4)  # 4 seconds of random audio
result = diar.infer_wav(wav, sample_rate=16000)

# Apply global tracking
track_out = tracker.update(result)
result.global_frame_ids = track_out.global_frame_ids
```

---

## License

This project is licensed under the **Apache License 2.0**.

The **CN-Celeb** dataset follows its original license and usage terms.

---

**Repository**: [https://github.com/zhangzijie-pro/Speaker-Verification](https://github.com/zhangzijie-pro/Speaker-Verification)  
**Models & Datasets**: [https://huggingface.co/zzj-pro](https://huggingface.co/zzj-pro)