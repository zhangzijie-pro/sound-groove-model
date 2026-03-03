
---

# 🎙️ Speaker Verification and Voiceprint Recognition

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/zhangzijie-pro/Speaker-Verification.svg?style=social)](https://github.com/zhangzijie-pro/Speaker-Verification/stargazers)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model%20%26%20Dataset-yellow.svg)](https://huggingface.co/zzj-pro)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![Task](https://img.shields.io/badge/Task-Speaker%20Verification-green)

<div align="center">
  <a href="Readme_ch.md">中文文档</a> • 
  <a href="https://github.com/zhangzijie-pro/Speaker-Verification">GitHub</a> • 
  <a href="https://huggingface.co/zzj-pro">Hugging Face</a>
</div>

<img src="./docs/imgs/model.jpg" alt="WALL·E" width="600"/>

---

## 📂 Project Structure

```
Speaker-Verification/
│
├── processed/              # Preprocessed features & metadata
│   ├── preprocess_cnceleb2_train.py
│   └── cn_celeb2/          # outputs
│       ├── fbank_pt/       # Saved fbank features (*.pt)
│       ├── train_fbank_list.txt
│       ├── val_meta.jsonl  # Validation metadata (speaker, feature path)
│       └── spk2id.json
│
├── configs/
│   ├── train.yaml
│   └── train_config.py     # Training hyperparameters
│
├── demos/
│   └── real_time.py        # real time to listen audio and test
│
├── data/
│   ├── dataset.py          # Train / validation datasets
│   └── pk_sampler.py       # PK batch sampler (speaker-balanced)
│
├── speaker_verification/
│   ├── checkpointing.py    
│   ├── inference.py
│   ├── head/
│   │   └── aamsoftmax.py   # AAM-Softmax loss
│   ├── models/
│   │   └── epaca.py        # model
│   └── audio/
│       └── features.py     # extract features
│
├── utils/
│   ├── meters.py           # Accuracy, average meters
│   ├── seed.py             # Reproducibility
│   ├── plot.py             # Training curves
│   ├── export.py           # export onnx/mnn and split model, head
│   └── path_utils.py       # Deal to path error
│
├── outputs/                # Training outputs (checkpoints, curves)
├── outputs_eval/           # Verification results (EER, ROC, DET, t-SNE)
│
├── train.py                # Main training script
├── finetune.py             # Main finetune script
├── verify_pairs.py         # Pairwise speaker verification
├── compare_two_wavs.py     # Compare two audio files
│
├── README.md
├── README_ch.md
└── LICENSE
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/zhangzijie-pro/Speaker-Verification.git
cd Speaker-Verification
pip install -r requirements.txt
```

### 2. Data Preprocessing

```bash
python processed/preprocess_cnceleb2_train.py
```

### 3. Training

```bash
# Train with default config
python train.py

# Override parameters via command line
python train.py train.epochs=100 train.lr=5e-4 train.emb_dim=256
```

---

## 📈 Evaluation (Speaker Verification)

### Run full evaluation

```bash
python verify.py \
    --val_meta processed/cn_celeb2/val_meta.jsonl \
    --ckpt outputs/best.pt \
    --out_dir outputs_eval
```

**Outputs**:
- `roc.png`, `det.png`, `score_hist.png`
- `tsne.png` (speaker clustering)
- `metrics.txt` (EER, Recall@K, etc.)

---

## 🎯 Single Audio Comparison (Most Used)

```bash
python compare_two_wavs.py \
    --wav1 test1.wav \
    --wav2 test2.wav \
    --ckpt outputs/export/model.onnx   # Supports ONNX
```

---

## 🛠️ Model Export (Deployment)

```bash
# One-click export to ONNX + MNN
python scripts/export.py \
    --ckpt outputs/best.pt \
    --out_dir outputs/deploy \
    --onnx --mnn
```

**Supported deployment**:
- **ONNX Runtime** (Python / C++)
- **MNN** (Mobile / Edge)
- **TensorRT** (High-performance server)

---

## 🧠 Model Overview

### Backbone

- **ECAPA-TDNN**
  - Res2Net-style temporal convolutions
  - Squeeze-and-Excitation (SE)
  - Attentive Statistics Pooling
- Embedding dimension: **192 / 256**

### Loss

- **AAM-Softmax (Additive Angular Margin Softmax)**
  - Encourages large inter-speaker margins
  - Used only during training

### Embedding

- L2-normalized speaker embeddings
- Cosine similarity for verification

---

## 📊 Dataset

- **CN-Celeb**
  - ~1000 speakers
  - Highly diverse recording conditions
- Split:
  - `train`: speaker-disjoint
  - `val`: speaker-disjoint
- Features:
  - 80-dim log Mel-filterbank
  - 16kHz sampling rate

---

## 📌 Recommended Configuration (6GB GPU)

```yaml
# configs/train.yaml
emb_dim: 256
channels: 512
lr: 1e-3
epochs: 80
crop_frames: 200          # Training
crop_frames_val: 400      # Validation
num_crops: 6
p: 32
k: 4
```

---

## 🔮 Future Improvements

- [x] Hydra configuration
- [x] Parallel preprocessing
- [x] ONNX / MNN export
- [ ] Noise / RIR augmentation

---

## 📜 License

This project is released under the **Apache License 2.0**.  
The CN-Celeb dataset follows its original license and usage terms.

---

## 🙋 Notes

This repository is intended for:

- Learning speaker verification systems

It is **not** an off-the-shelf commercial system.
