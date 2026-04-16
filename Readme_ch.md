# 🎙️ Speaker-Aware Diarization and Verification

<div align="center">

  [![GitHub stars](https://img.shields.io/github/stars/zhangzijie-pro/Speaker-Verification.svg?style=social)](https://github.com/zhangzijie-pro/Speaker-Verification/stargazers)
  [![Hugging Face](https://img.shields.io/badge/HuggingFace-模型%20%26%20数据集-yellow.svg)](https://huggingface.co/zzj-pro)
  ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
  [![PyPI](https://img.shields.io/pypi/v/speaker-verification)](https://pypi.org/project/speaker-verification/)
  <!-- [![CI](https://img.shields.io/github/actions/workflow/status/zhangzijie-pro/Speaker-Verification/ci.yml?branch=main&label=CI)](https://github.com/zhangzijie-pro/Speaker-Verification/actions) -->
</div>

<div align="center">
  <a href="./README.md">English</a> •
  <a href="./docs/finetuning.md">微调指南</a> •
  <a href="./CONTRIBUTING.md">贡献指南</a>
</div>

## 🚀 核心

本项目实现了一个端到端（E2E）说话人感知的语音处理工具包，核心模型为 **`EENDQueryModel`**，基于以下组件联合建模：

- **AcousticEncoder**：采用 Res2SEBlock + Lightweight Self-Attention + FeedForwardAdapter 的多尺度声学编码器，将 FBank 特征映射为归一化帧级嵌入（frame embeddings），支持说话人验证。
- **QueryDecoder**（或 `LSTMAttractorDecoder` / EDA-LSTM）：生成说话人吸引子（attractors）和存在性 logits（exist_logits），实现动态说话人计数和活动检测。
- **DotProductDiarizationHead**：通过帧嵌入与吸引子的点积计算说话人分配 logits，实现说话人 diarization。
- **Activity Head**：实验性的帧级活动分支。当前精简训练路径主要优化 diarization mask 和 speaker existence；验证和推理阶段的活动区间由 gated diarization mask 派生。

**EENDQueryModel** 输出：
- `frame_embeds`：归一化帧嵌入（用于验证/聚类）
- `attractors`：说话人原型嵌入
- `exist_logits`：说话人存在性
- `diar_logits`：帧-说话人分配
- `activity_logits`：辅助语音活动 logits

**SpeakerDiarizationPipeline**：高阶推理封装，支持：
- 长音频分块 + 重叠处理（流式友好）
- 后处理（短段移除、间隙填充、说话人重标签）
- 输出结构化 diarization 结果（RTTM、JSON、段列表）

**实际场景**：
- **流式/长音频推理**：分块处理 + 后处理，实现低延迟在线 diarization
- **会议/监控分析**：生成精确说话人时间线、活动段和结构化摘要
- **说话人验证基础**：归一化帧嵌入支持后续 metric learning 或开放集识别（speaker bank 功能在后续迭代中扩展）

## ⚡ Quick Start / 快速开始

安装项目：

```bash
git clone https://github.com/zhangzijie-pro/Speaker-Verification.git
git checkout E2E
cd Speaker-Verification
pip install -r requirements.txt
```

### 低阶模型使用（直接调用 EENDQueryModel）

```python
import torch
from speaker_verification.models.eend_query_model import EENDQueryModel

model = EENDQueryModel(
    in_channels=80,
    enc_channels=512,
    d_model=256,
    max_speakers=6,
    decoder_type="query"   # 或 "eda_lstm"
).eval()

# 输入：FBank [B, T, F]（示例 T=400, F=80）
fbank = torch.randn(1, 400, 80)

frame_embeds, attractors, exist_logits, diar_logits, activity_logits = model(fbank)

print("frame_embeds :", tuple(frame_embeds.shape))      # [B, T, D]
print("attractors   :", tuple(attractors.shape))       # [B, N, D]
print("diar_logits  :", tuple(diar_logits.shape))      # [B, T, N]
print("exist_logits :", tuple(exist_logits.shape))     # [B, N]
print("activity_logits :", tuple(activity_logits.shape))  # [B, T]
```

### 高阶 Pipeline 使用（推荐，用于实际音频处理）

```python
from speaker_verification.inference import SpeakerDiarizationPipeline

# 从 checkpoint 加载（支持 hydra 配置）
pipeline = SpeakerDiarizationPipeline.from_pretrained(
    checkpoint_path="path/to/best.pt",   # 或 Hugging Face 路径
    chunk_sec=5.0,          # 分块长度
    hop_sec=2.0,            # 重叠步长
    speaker_activity_threshold=0.55,
    merge_gap_sec=0.2,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 处理音频文件
result = pipeline.predict_file(
    audio_path="meeting.wav",
    recording_id="meeting_001"
)

# 输出结构化结果
print(result.to_dict())          # 包含 segments、speaker_prob 等
pipeline.save_rttm(result, "output.rttm")   # 保存为标准 RTTM 格式
```

**开发环境安装**：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 🗂️ 数据与微调

和数据准备、模型适配相关的文档集中在 `docs/`：

- [`docs/finetuning.md`](./docs/finetuning.md)：YAML 驱动微调、关键超参、checkpoint 加载和调参逻辑
- [`docs/specs/checkpoint-schema.md`](./docs/specs/checkpoint-schema.md)：checkpoint 元数据与兼容性规则

## 🤝 社区

如果你要贡献代码，请从 [`CONTRIBUTING.md`](./CONTRIBUTING.md) 开始。如果你遇到问题，请优先使用 issue 模板。如果你准备把项目接到真实产品里，请在讨论中明确 workload、延迟预算和 artifact 约束。

## License

Apache License 2.0
