# Finetuning Guide / 微调指南

## 1. Start from a Pretrained Checkpoint / 从预训练权重开始

**EN**  
from Hugging Face first: [checkpoint](https://huggingface.co/zzj-pro).

**中文**  
从 Hugging Face 下载预训练 [checkpoint](https://huggingface.co/zzj-pro)。

## 2. Configuration-first / 配置

**EN**  
Finetuning is controlled through `configs/experiment.yaml` plus Hydra overrides. The main fields are:

- `run.mode=finetune`
- `finetune.checkpoint_path`
- `finetune.load_mode`
- `finetune.freeze_backbone`
- `finetune.lr_scale`
- `output.save_dir`

Minimal finetuning command:

```bash
python3 train.py \
  run.mode=finetune \
  finetune.checkpoint_path=artifacts/pretrained/best.pt \
  finetune.load_mode=full \
  finetune.freeze_backbone=false \
  finetune.lr_scale=0.1 \
  output.save_dir=outputs/finetune_v1
```

**中文**  
微调通过 `configs/experiment.yaml` 配合 Hydra override 完成控制。最关键的字段是：

- `run.mode=finetune`
- `finetune.checkpoint_path`
- `finetune.load_mode`
- `finetune.freeze_backbone`
- `finetune.lr_scale`
- `output.save_dir`

最小微调命令：

```bash
python3 train.py \
  run.mode=finetune \
  finetune.checkpoint_path=artifacts/pretrained/best.pt \
  finetune.load_mode=full \
  finetune.freeze_backbone=false \
  finetune.lr_scale=0.1 \
  output.save_dir=outputs/finetune_v1
```

## 3. YAML


```yaml
run:
  mode: "finetune"

train:
  lr: 3.0e-4
  epochs: 30
  batch_size: 16

finetune:
  checkpoint_path: "artifacts/pretrained/best.pt"
  load_mode: "full"
  strict_load: true
  freeze_backbone: false
  lr_scale: 0.1
```

## 4. Learning Rate Strategy / 学习率策略

**EN**  
Do not reuse the original pretraining LR blindly. Finetuning usually needs a lower effective LR, especially if the domain shift is small.

Recommended logic:

- same task, small domain shift: set `finetune.lr_scale` to `0.05 ~ 0.2`
- same architecture, larger domain shift: start with `0.1 ~ 0.3`
- head-only finetuning: slightly larger LR is acceptable
- full-model finetuning: stay conservative first

In the current training code, the effective LR is:

```text
effective_lr = train.lr * finetune.lr_scale
```

**中文**  
不要机械复用预训练阶段的学习率。微调通常需要更低的有效学习率，尤其是在任务相同、领域偏移较小时。

推荐逻辑：

- 相同任务、领域偏移小：`finetune.lr_scale` 设为 `0.05 ~ 0.2`
- 架构相同、领域偏移更大：先从 `0.1 ~ 0.3` 开始
- 只训练 head：学习率可以略大
- 全量微调：先保守，再逐步放开

在当前训练代码中，有效学习率为：

```text
effective_lr = train.lr * finetune.lr_scale
```

## 5. Freeze Layers / 冻结层策略

**EN**  
Use freezing when the pretrained backbone is already strong and your new data volume is limited.

- `freeze_backbone=true`: only train layers matched by `finetune.head_prefixes`
- `freeze_backbone=false`: train the full model
- `load_mode=backbone_only`: useful when you want to replace or reinitialize the diarization head

Head-only example:

```bash
python3 train.py \
  run.mode=finetune \
  finetune.checkpoint_path=artifacts/pretrained/best.pt \
  finetune.load_mode=backbone_only \
  finetune.strict_load=false \
  finetune.freeze_backbone=true \
  output.save_dir=outputs/head_only_ft
```

**中文**  
当预训练 backbone 已经足够强，而你的新数据量有限时，优先考虑冻结策略。

- `freeze_backbone=true`：只训练 `finetune.head_prefixes` 匹配到的层
- `freeze_backbone=false`：全模型参与训练
- `load_mode=backbone_only`：适合你想替换或重初始化 diarization head 的情况

仅训练 head 的示例：

```bash
python3 train.py \
  run.mode=finetune \
  finetune.checkpoint_path=artifacts/pretrained/best.pt \
  finetune.load_mode=backbone_only \
  finetune.strict_load=false \
  finetune.freeze_backbone=true \
  output.save_dir=outputs/head_only_ft
```

## 6. Loss Function Logic / Loss Function 设置逻辑

**EN**  
The current objective is multi-task. You are not tuning a single scalar loss; you are balancing diarization quality, activity detection, count prediction, and frame-prototype consistency.

Relevant YAML keys:

- `loss.lambda_pit`
- `loss.lambda_act`
- `loss.lambda_cnt`
- `loss.lambda_frm`
- `loss.pos_weight`
- `loss.pit_pos_weight`

Practical guidance:

- increase `lambda_pit` if slot assignment quality is the bottleneck
- increase `lambda_act` if voice activity boundaries are unstable
- increase `lambda_cnt` if speaker count is consistently off
- increase `lambda_frm` if identity separation is weak across adjacent speakers

**中文**  
当前训练目标是多任务损失，不是在调一个单一标量，而是在平衡 diarization、activity、speaker count 和 frame prototype 一致性。

相关 YAML 字段：

- `loss.lambda_pit`
- `loss.lambda_act`
- `loss.lambda_cnt`
- `loss.lambda_frm`
- `loss.pos_weight`
- `loss.pit_pos_weight`

实践建议：

- 如果 slot assignment 是瓶颈，就提高 `lambda_pit`
- 如果语音活动边界不稳，就提高 `lambda_act`
- 如果说话人数估计偏差大，就提高 `lambda_cnt`
- 如果相邻说话人的身份区分弱，就提高 `lambda_frm`

## 7. Loading Rules / 权重加载规则

**EN**  
Before finetuning, verify that these values match the checkpoint you downloaded:

- `model.in_channels`
- `model.channels`
- `model.embedding_dim`
- `model.max_mix_speakers`

If those do not match, strict loading will fail or, worse, partial loading will give you a misleading starting point.

**中文**  
开始微调前，先确认以下配置与下载到的 checkpoint 一致：

- `model.in_channels`
- `model.channels`
- `model.embedding_dim`
- `model.max_mix_speakers`

这些值如果不一致，严格加载会失败；更糟的是，部分加载可能让你得到一个具有误导性的起点。

## 8. Validation During Finetuning / 微调过程中的验证

**EN**

- watch `val_loss`, `der`, `count_acc`, and `act_f1` together
- do not optimize only one metric if the others collapse
- save outputs to a separate `output.save_dir`
- keep `last.pt` and `best.pt` for rollback and comparison

**中文**

- 同时观察 `val_loss`、`der`、`count_acc` 和 `act_f1`
- 不要只盯单一指标，避免其它指标明显塌陷
- 使用独立的 `output.save_dir`
- 同时保留 `last.pt` 与 `best.pt`，便于回滚和对比

## 9. Related Docs / 相关文档

- Data guide: [`docs/data_guide.md`](./data_guide.md)
- Checkpoint schema: [`docs/specs/checkpoint-schema.md`](./specs/checkpoint-schema.md)
- Data schema: [`docs/specs/data-schema.md`](./specs/data-schema.md)

