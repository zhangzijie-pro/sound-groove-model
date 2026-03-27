# Data Guide / 数据指南

## 1. Fast Track / 快速通道

[Origin dataset](https://huggingface.co/datasets/zzj-pro/CN_Celeb_v2)

[Pretrained dataset](https://huggingface.co/datasets/zzj-pro/CN_Celeb_processed)

## 2. Pipeline

**EN**  
If you are following the built-in preprocessing path, `processed/build_processed_dataset.py` handles:

- single-speaker feature extraction,
- static multi-speaker mixture generation,
- augmentation with single-speaker and negative samples.

Typical commands:

```bash
python3 processed/build_processed_dataset.py --stage cn
python3 processed/build_processed_dataset.py --stage mix
python3 processed/build_processed_dataset.py --stage add_single
python3 processed/build_processed_dataset.py --stage add_negative

python3 processed/build_processed_dataset.py --stage all
```

**中文**  
如果你沿用仓库内置的预处理路径，`processed/build_processed_dataset.py` 会负责：

- 单说话人特征抽取
- 静态多说话人混合样本生成
- 单说话人样本与负样本增强

常用命令：

```bash
python3 processed/build_processed_dataset.py --stage cn
python3 processed/build_processed_dataset.py --stage mix
python3 processed/build_processed_dataset.py --stage add_single
python3 processed/build_processed_dataset.py --stage add_negative

python3 processed/build_processed_dataset.py --stage all
```

## 3. Related Specs / 相关规范

- Checkpoint schema: [`docs/specs/checkpoint-schema.md`](./specs/checkpoint-schema.md)
- Data schema: [`docs/specs/data-schema.md`](./specs/data-schema.md)

