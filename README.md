# Speaker-Aware Diarization and Verification

<p align="center">
  <b>A PyTorch project for speaker-aware multi-speaker audio understanding.</b><br>
  From speaker verification to diarization, speaker counting, activity detection, and future speaker-bank integration.
</p>

<div align="center">

  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![GitHub stars](https://img.shields.io/github/stars/zhangzijie-pro/Speaker-Verification.svg?style=social)](https://github.com/zhangzijie-pro/Speaker-Verification/stargazers)
  [![Hugging Face](https://img.shields.io/badge/HuggingFace-模型%20%26%20数据集-yellow.svg)](https://huggingface.co/zzj-pro)
  ![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
  ![Task](https://img.shields.io/badge/Task-Speaker%20Verification-green)

</div>

<div align="center">
  <a href="README.md">English</a> • 
  <a href="https://github.com/zhangzijie-pro/Speaker-Verification">GitHub</a> • 
  <a href="https://huggingface.co/zzj-pro">Hugging Face</a>
</div>

--- 
<img src="./docs/imgs/model.jpg" alt="WALL·E" width="600"/>

---
## Overview

This repository began as a **speaker verification / voiceprint recognition** project and is now evolving toward a more practical **speaker-aware diarization pipeline**.

The current **dev branch** focuses on answering questions like:

- **How many speakers are active in this chunk?**
- **Who is speaking during each time region?**
- **Who is the dominant speaker in the chunk?**
- **How can anonymous speaker slots be linked to a future speaker bank?**

The system is designed for multi-speaker chunk understanding rather than only pairwise speaker verification.

---

## Highlights

- **ResoWave backbone**
  - temporal convolution front-end
  - SE-Res2 style block
  - hybrid WRR/EPA blocks
  - attentive statistics pooling

- **REAT diarization head**
  - frame-level speaker embeddings
  - slot logits for speaker assignment
  - activity logits for speech detection
  - count logits for speaker number estimation

- **Multi-task training**
  - PIT-based slot assignment loss
  - speech activity loss
  - speaker count loss
  - frame-level prototype / contrastive supervision

- **Validation pipeline**
  - DER
  - activity precision / recall / F1
  - speaker count accuracy
  - loss breakdown monitoring

- **Inference-ready design**
  - chunk-level inference
  - dominant speaker estimation
  - reserved interface for future speaker bank integration

---

## Repository Structure

```text
Speaker-Verification/
├── configs/                      # Training configs
├── dataset/                      # Dataset loading logic
├── docs/                         # Notes and design documents
├── processed/                    # Data preprocessing and generated manifests
├── scripts/                      # Utility scripts
├── speaker_verification/
│   ├── checkpointing.py
│   ├── loss/
│   ├── models/
│   │   ├── head/
│   │   └── ...
│   └── ...
├── utils/                        # meters, plotting, seed, utilities
├── train.py                      # Training entry
├── verify.py                     # Validation / verification entry
├── README.md
├── Readme_ch.md
└── requirements.txt
````

---

## Project Goal

The current goal is to build a speaker-aware system that can process a mixed-speaker chunk and produce:

* the **number of speakers**
* the **speaking segments**
* the **dominant speaker**
* speaker-slot prototypes that can later be matched against a **speaker bank**

This means the dev branch is no longer just a classic speaker verification demo.
It is moving toward a **multi-speaker chunk analysis system**.

---

## Model Architecture

### 1. Backbone: `ResoWave`

Input shape:

```python
[B, T, 80]
```

Outputs:

* global embedding
* frame-level feature sequence for diarization

Main modules:

* `Conv1dReluBn`
* `SE_Res2Block`
* `HybridEPA_WRR_Block`
* `AttentiveStatsPool`

---

### 2. Diarization Head: `REAT_DiarizationHead`

The diarization head predicts:

* `frame_embeds: [B, T, D]`
* `slot_logits: [B, T, K]`
* `activity_logits: [B, T]`
* `count_logits: [B, K]`

Where:

* `T` = number of frames
* `D` = frame embedding dimension
* `K` = maximum number of speakers in a chunk

---

### 3. Multi-Task Loss

The current training objective combines:

* **PIT loss** for speaker-slot assignment
* **activity loss** for speech/non-speech prediction
* **count loss** for speaker number estimation
* **frame-level prototype / contrastive loss** for speaker-discriminative frame embeddings

This design avoids forcing a mixed-speaker chunk into a single speaker label.

---

## Data Pipeline

The current training pipeline is based on **static mixed-speaker chunks**.

Typical fields used in a training sample:

* `fbank`
* `target_matrix`
* `target_activity`
* `target_count`
* `valid_mask`

The repository includes preprocessing and metadata generation logic under:

* `processed/`
* `dataset/`

---

## Installation

```bash
git clone https://github.com/zhangzijie-pro/Speaker-Verification.git
cd Speaker-Verification
pip install -r requirements.txt
```

---

## Training

Default training:

```bash
python train.py
```

Hydra override example:

```bash
python train.py train.epochs=100 train.lr=3e-4 model.max_mix_speakers=4
```

Example config:

```yaml
seed: 1234
device: "cuda"
out_dir: "outputs"

data:
  out_dir: "processed/static_mix_cnceleb2"
  train_manifest: "train_manifest.jsonl"
  val_manifest: "val_manifest.jsonl"
  crop_sec: 4.0

model:
  feat_dim: 80
  channels: 512
  emb_dim: 192
  max_mix_speakers: 4

loss:
  lambda_pit: 1.0
  lambda_act: 1.0
  lambda_cnt: 0.2
  lambda_frm: 0.5
  pos_weight: 2.0
  pit_pos_weight: 1.5

train:
  epochs: 100
  batch_size: 16
  num_workers: 0
  lr: 3.0e-4
  weight_decay: 3.0e-5
  grad_clip: 5.0
  amp: true
  val_batches: 100
  activity_threshold: 0.5
```

---

## Outputs

Training typically produces:

* `outputs/last.pt`
* `outputs/best.pt`
* `outputs/history.json`
* training logs like:

  * `outputs/train_YYYYMMDD_HHMMSS.log`

The current best checkpoint is selected by **lowest DER**.

---

## Validation

Example:

```bash
python verify.py \
  --ckpt outputs/best.pt \
  --data_out_dir processed/static_mix_cnceleb2 \
  --manifest val_manifest.jsonl \
  --crop_sec 4.0 \
  --feat_dim 80 \
  --channels 512 \
  --emb_dim 192 \
  --max_mix_speakers 4 \
  --lambda_pit 1.0 \
  --lambda_act 1.0 \
  --lambda_cnt 0.2 \
  --lambda_frm 0.5 \
  --pos_weight 2.0 \
  --pit_pos_weight 1.5 \
  --batch_size 16 \
  --num_workers 0 \
  --device cuda \
  --max_batches 100 \
  --activity_threshold 0.5
```

Typical reported metrics:

* `val_loss`
* `pit_loss`
* `act_loss`
* `cnt_loss`
* `frm_loss`
* `DER`
* `CountAcc`
* `ActPrecision`
* `ActRecall`
* `ActF1`

---

## Inference

The current inference target is a **single chunk** of acoustic features.

Expected outputs:

* predicted number of speakers
* frame-level speaker-slot assignment
* speech activity
* speaking segments
* dominant speaker
* slot-level speaker prototypes

Example usage:

```python
from infer_realtime import load_model, infer_chunk

model = load_model(
    ckpt_path="outputs/best.pt",
    device="cuda",
    feat_dim=80,
    channels=512,
    emb_dim=192,
    max_mix_speakers=4,
)

result = infer_chunk(
    model=model,
    fbank=fbank_tensor,   # [T,80] or [1,T,80]
    device="cuda",
    activity_threshold=0.5,
    frame_shift_sec=0.01,
)
```

Example output:

```python
{
    "num_speakers": 3,
    "dominant_speaker": "slot_0",
    "activity_ratio": 0.81,
    "slots": [
        {
            "slot": 0,
            "name": "unknown",
            "score": None,
            "is_known": False,
            "num_frames": 210,
            "duration_sec": 2.1
        }
    ],
    "segments": [
        {
            "slot": 0,
            "start_sec": 0.12,
            "end_sec": 1.54,
            "duration_sec": 1.42,
            "name": "unknown"
        }
    ]
}
```

---

## Speaker Bank Integration

The inference pipeline is intentionally designed to keep a clean interface for future speaker-bank integration.

Planned workflow:

1. use a separate speaker verification / voiceprint model
2. build a speaker bank from enrollment utterances
3. extract slot prototypes from diarization output
4. match slot prototypes against the speaker bank
5. convert anonymous slots into real identities

Recommended interface pattern:

```python
class SpeakerBankBase:
    def identify(self, embedding):
        return {
            "name": "zhangsan",
            "score": 0.87,
            "is_known": True,
        }
```

This means the current diarization model can already provide:

* **how many people spoke**
* **when each slot was active**
* **who was dominant**

and later be upgraded to provide:

* **which known person each slot corresponds to**

without redesigning the whole pipeline.

---

## Current Status

### Working well

* stable training
* DER-based checkpoint selection
* strong speech activity detection
* good speaker count estimation
* clean validation pipeline
* chunk-level inference structure

### Under active improvement

* stronger frame-level speaker discrimination
* lower DER
* more stable speaker-slot assignment
* better connection between diarization output and identity retrieval
* real speaker bank deployment

---

## Branch Notes

* **dev**: current diarization-oriented development branch
* **release**: earlier speaker verification oriented branch

If you are mainly interested in the earlier verification-style project presentation, check the `release` branch. The release branch homepage still presents the repository more explicitly as a speaker verification / voiceprint recognition project. ([GitHub][2])

---

## Roadmap

* [x] multi-task diarization training
* [x] DER-based validation
* [x] chunk-level inference
* [x] dominant speaker estimation
* [ ] speaker bank integration
* [ ] real identity output
* [ ] streaming microphone pipeline
* [ ] deployment-friendly downstream interface

---
## 📜 License

This project is released under the **Apache License 2.0**.  
The CN-Celeb dataset follows its original license and usage terms.

---

## 🙋 Notes

This project is being iterated toward practical speaker-aware AI / robotics audio understanding scenarios, including future integration with voiceprint libraries, real-time pipelines, and downstream intelligent systems.

This repository is intended for:

- Learning speaker verification systems

It is **not** an off-the-shelf commercial system.

