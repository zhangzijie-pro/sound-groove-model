import os
import torch
from torch.utils.data import DataLoader

from speaker_verification.models.eend_query_model import EENDQueryModel
from speaker_verification.loss.multi_task import MultiTaskLoss
from speaker_verification.dataset.static_dataset import StaticMixDataset


def build_model(cfg, device):
    model = EENDQueryModel(
        in_channels=cfg.model.in_channels,
        enc_channels=cfg.model.channels,
        d_model=cfg.model.d_model,
        max_speakers=cfg.model.max_mix_speakers,
        assign_scale=getattr(cfg.model, "assign_scale", 8.0),
        decoder_type=cfg.model.decoder_type,
        post_ffn_hidden_dim=cfg.model.post_ffn_hidden_dim,
        post_ffn_dropout=cfg.model.post_ffn_dropout,
        decoder_layers=cfg.model.decoder_layers,
        decoder_heads=cfg.model.decoder_heads,
        decoder_ffn=cfg.model.decoder_ffn,
        dropout=cfg.model.dropout,
    )
    return model.to(device)


def build_loss(cfg, device):
    loss_fn = MultiTaskLoss(
        pit_pos_weight=cfg.loss.pit_pos_weight,
        activity_pos_weight=getattr(cfg.loss, "activity_pos_weight", None),
        exist_pos_weight=cfg.loss.exist_pos_weight,
        exist_focal_gamma=getattr(cfg.loss, "exist_focal_gamma", 2.0),
        lambda_exist=cfg.loss.lambda_exist,
        lambda_activity=getattr(cfg.loss, "lambda_activity", 0.3),
        lambda_diversity=getattr(cfg.loss, "lambda_diversity", 0.02),
        overlap_weight_2spk=getattr(cfg.loss, "overlap_weight_2spk", 2.0),
        overlap_weight_3spk=getattr(cfg.loss, "overlap_weight_3spk", 3.0),
    )
    return loss_fn.to(device)


def build_loaders(cfg, device):
    is_windows = os.name == "nt"
    train_num_workers = int(cfg.train.num_workers)
    val_num_workers = int(cfg.train.num_workers)

    # Windows DataLoader uses shared file mappings for worker batches and is
    # much easier to exhaust than Linux. Prefer conservative defaults there.
    if is_windows and train_num_workers > 0:
        train_num_workers = min(train_num_workers, 1)
        val_num_workers = min(val_num_workers, 1)

    train_set = StaticMixDataset(
        out_dir=cfg.data.out_dir,
        manifest=cfg.data.train_manifest,
        crop_sec=cfg.data.crop_sec,
        shuffle=True,
        crop_mode="random",
        max_speakers=cfg.model.max_mix_speakers,
        same_session_train_prob=getattr(cfg.data, "same_session_train_prob", None),
        same_session_val_prob=getattr(cfg.data, "same_session_val_prob", None),
    )

    val_set = StaticMixDataset(
        out_dir=cfg.data.out_dir,
        manifest=cfg.data.val_manifest,
        crop_sec=cfg.data.crop_sec,
        shuffle=False,
        crop_mode="center",
        max_speakers=cfg.model.max_mix_speakers,
        same_session_train_prob=getattr(cfg.data, "same_session_train_prob", None),
        same_session_val_prob=getattr(cfg.data, "same_session_val_prob", None),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=(device.type == "cuda" and not is_windows),
        drop_last=True,
        persistent_workers=bool(train_num_workers > 0 and not is_windows),
        prefetch_factor=(1 if train_num_workers > 0 else None),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.validate.batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=(device.type == "cuda" and not is_windows),
        drop_last=False,
        persistent_workers=bool(val_num_workers > 0 and not is_windows),
        prefetch_factor=(1 if val_num_workers > 0 else None),
    )

    return train_loader, val_loader
