# 🎙️ Speaker-Aware Diarization and Verification

<div align="center">

  [![GitHub stars](https://img.shields.io/github/stars/zhangzijie-pro/Speaker-Verification.svg?style=social)](https://github.com/zhangzijie-pro/Speaker-Verification/stargazers)
  [![Hugging Face](https://img.shields.io/badge/HuggingFace-模型%20%26%20数据集-yellow.svg)](https://huggingface.co/zzj-pro)
  ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
  [![PyPI](https://img.shields.io/pypi/v/speaker-verification)](https://pypi.org/project/speaker-verification/)
  [![CI](https://img.shields.io/github/actions/workflow/status/zhangzijie-pro/Speaker-Verification/ci.yml?branch=main&label=CI)](https://github.com/zhangzijie-pro/Speaker-Verification/actions)
  [![Coverage](https://img.shields.io/badge/coverage-%E2%89%A575%25-brightgreen)](./pytest.ini)
</div>

<div align="center">
  <a href="./README.md">English</a> •
  <a href="./docs/data_guide.md">数据指南</a> •
  <a href="./docs/finetuning.md">微调指南</a> •
  <a href="./CONTRIBUTING.md">贡献指南</a>
</div>

![WT-EPA-WRR-REAT Model](docs/imgs/model.jpg)

## 🚀 核心

- `ResoWave` + `REAT`：联合建模 activity、slot assignment、speaker count 和 frame prototype
- `GlobalSpeakerTracker`：在分块或流式推理中保持稳定的 global speaker ID
- `JsonSpeakerBank`：支持声纹库 CRUD 与开放集识别，低置信度时明确返回 `unknown`

这些能力对应的实际场景是：

- **流式推理**：低延迟在线识别
- **会议分析**：生成 diarization 时间线和结构化摘要
- **长音频监控**：面向长期监听、告警和未知说话人检测

## ⚡ Quick Start / 快速开始

安装项目：

```bash
git clone https://github.com/zhangzijie-pro/Speaker-Verification.git
cd Speaker-Verification
pip install -r requirements.txt
```

用最短路径跑通核心逻辑：

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

开发环境安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 其他场景

### 流式推理

[`examples/streaming_inference.py`](./examples/streaming_inference.py) 是低延迟入口，适合接麦克风缓冲区、websocket 音频流或实时 agent pipeline。

```bash
python3 examples/streaming_inference.py \
  --ckpt outputs/train_v1/best.pt \
  --speaker-bank artifacts/speaker_bank.json \
  --chunk-sec 2.0 \
  --step-sec 0.5
```

### 会议分析

[`examples/meeting_analysis.py`](./examples/meeting_analysis.py) 用于对会议录音做 diarization 并导出机器可读时间线。

```bash
python3 examples/meeting_analysis.py \
  --ckpt outputs/train_v1/best.pt \
  --audio assets/meeting.wav \
  --output artifacts/meeting_analysis.json
```

### 长音频监控

[`examples/long_audio_monitoring.py`](./examples/long_audio_monitoring.py) 适合长期监听、未知说话人占比统计和告警阈值控制。

```bash
python3 examples/long_audio_monitoring.py \
  --ckpt outputs/train_v1/best.pt \
  --audio assets/monitor.wav \
  --alert-threshold 0.25
```

## 🗂️ 数据与微调

和数据准备、模型适配相关的文档集中放在 `docs/`：

- [`docs/data_guide.md`](./docs/data_guide.md)：原始音频要求、清洗脚本、manifest schema，以及 Hugging Face 快速入口
- [`docs/finetuning.md`](./docs/finetuning.md)：YAML 驱动微调、关键超参、checkpoint 加载和调参逻辑
- [`docs/specs/checkpoint-schema.md`](./docs/specs/checkpoint-schema.md)：checkpoint 元数据与兼容性规则
- [`docs/specs/data-schema.md`](./docs/specs/data-schema.md)：processed pack 与推理 JSON schema

## 🤝 社区

如果你要贡献代码，请从 [`CONTRIBUTING.md`](./CONTRIBUTING.md) 开始。如果你遇到问题，请优先使用 issue 模板。如果你准备把项目接到真实产品里，请在讨论中明确 workload、延迟预算和 artifact 约束。

## License

Apache License 2.0。外部数据集如 CN-Celeb 仍遵循其原始许可证和使用条款。

