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
  <a href="README.md">English</a> • 
  <a href="Readme_ch.md">中文文档</a> • 
  <a href="https://github.com/zhangzijie-pro/Speaker-Verification">GitHub</a> • 
  <a href="https://huggingface.co/zzj-pro">Hugging Face</a>
</div>

**PyTorch** 实现的 **说话人感知（Speaker-Aware）** 多说话人音频模型，专注于先进的说话人分离（Diarization）与验证任务。

本仓库支持以下功能：

- 说话人分离（Speaker Diarization）
- 说话人数量估计（Speaker Counting）
- 语音活动检测（Speech Activity Detection / VAD）
- 主说话人估计（Dominant Speaker Estimation）
- 分块级说话人跟踪（Chunk-level Speaker Tracking）
- 未来说话人库身份匹配

项目最初从说话人验证（Speaker Verification）开始，目前已专注于 **说话人感知的说话人分离** 任务。

---

## ✨ 主要特性

- **ResoWave 主干网络**
  - 时域卷积前端
  - SE-Res2 残差块
  - Hybrid WRR / EPA 模块
  - 注意力统计池化（Attentive Statistics Pooling）

- **REAT 说话人分离头**
  - 帧级说话人嵌入
  - 时隙分配 logits
  - 语音活动 logits
  - 说话人数量 logits

- **多任务联合训练**
  - PIT（置换不变训练）分离损失
  - 语音活动检测损失
  - 说话人数量估计损失
  - 帧级原型监督

- **完整推理流程**
  - 分块级说话人分离
  - 说话片段检测
  - 主说话人识别
  - 本地时隙ID → 全局说话人ID（配合 Tracker）

---

## 模型输出

对于输入的单段音频块，模型输出：

- 估计的说话人数量
- 帧级语音活动（Activity）
- 帧级说话人时隙分配（Slot Assignment）
- 说话片段（Speaking Segments）
- 主说话人时隙
- 时隙级说话人原型嵌入（Prototypes）

启用 **GlobalSpeakerTracker** 后，还会额外输出：

- 本地时隙 → 全局说话人映射
- 全局帧级说话人ID
- 当前活跃的全局说话人ID列表

---

## 仓库结构

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
├── Readme_ch.md
└── requirements.txt
```

---

## 安装

```bash
git clone https://github.com/zhangzijie-pro/Speaker-Verification.git
cd Speaker-Verification
pip install -r requirements.txt
```

---

## 数据准备

构建处理后的训练数据：

```bash
python processed/build_processed_dataset.py --stage all
```

可用阶段：

```bash
python processed/build_processed_dataset.py --stage cn
python processed/build_processed_dataset.py --stage mix
python processed/build_processed_dataset.py --stage add_single
python processed/build_processed_dataset.py --stage add_negative
```

处理后的数据通常保存在：

```
processed/static_mix_cnceleb2/
├── mix_pt/
├── train_manifest.jsonl
├── val_manifest.jsonl
└── spk2id.json
```

---

## 训练

**默认训练：**

```bash
python train.py
```

**自定义训练：**

```bash
python train.py run.mode=train train.epochs=100 train.lr=3e-4 output.save_dir=outputs/train_v1
```

---

## 微调（Finetune）

**常规微调：**

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

**仅训练分离头（Head Only）：**

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

## 恢复训练（Resume）

```bash
python train.py \
  run.mode=train \
  resume.enabled=true \
  resume.checkpoint_path=outputs/train_v1/last.pt \
  output.save_dir=outputs/train_v1
```

---

## 配置说明

主配置文件：`configs/experiment.yaml`

主要配置部分：

- `run`：运行模式
- `model`：模型结构
- `loss`：损失函数
- `data`：数据配置
- `train`：训练参数
- `validate`：验证参数
- `output`：输出路径
- `finetune`：微调设置
- `resume`：断点续训

---

## 训练输出

训练完成后会生成以下文件：

- `best.pt` —— 最佳模型权重
- `last.pt` —— 最新检查点
- `history.json` —— 训练历史记录
- `train_YYYYMMDD_HHMMSS.log` —— 训练日志

示例输出目录：

```
outputs/train_v1/
├── best.pt
├── last.pt
├── history.json
└── train_20250326_211400.log
```

---

## 推理示例

```python
import torch

from speaker_verification.interfaces.diar_interface import SpeakerAwareDiarizationInterface
from speaker_verification.interfaces.global_tracker import GlobalSpeakerTracker

# 初始化模型接口
diar = SpeakerAwareDiarizationInterface(
    ckpt_path="outputs/train_v1/best.pt",
    device="cuda",
    feat_dim=80,
    channels=512,
    emb_dim=256,
    max_mix_speakers=4,
)

tracker = GlobalSpeakerTracker()

# 示例推理
wav = torch.randn(16000 * 4)   # 4秒随机音频
result = diar.infer_wav(wav, sample_rate=16000)

# 应用全局说话人跟踪
track_out = tracker.update(result)
result.global_frame_ids = track_out.global_frame_ids
```

---

## 许可证

本项目采用 **Apache License 2.0** 开源协议。

**CN-Celeb** 数据集请遵循其原始许可协议和使用条款。

---

**项目地址**： [https://github.com/zhangzijie-pro/Speaker-Verification](https://github.com/zhangzijie-pro/Speaker-Verification)  
**模型与数据集**： [https://huggingface.co/zzj-pro](https://huggingface.co/zzj-pro)

欢迎 Star ⭐ 支持！