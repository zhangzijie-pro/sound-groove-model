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
        exist_pos_weight=getattr(cfg.loss, "exist_pos_weight", None),
        activity_pos_weight=getattr(cfg.loss, "activity_pos_weight", None),
        lambda_activity=getattr(cfg.loss, "lambda_activity", 0.5),
        lambda_exist=cfg.loss.lambda_exist,
        lambda_pull=cfg.loss.lambda_pull,
        lambda_sep=cfg.loss.lambda_sep,
        lambda_smooth=cfg.loss.lambda_smooth,
    )
    return loss_fn.to(device)


def build_loaders(cfg, device):
    train_set = StaticMixDataset(
        out_dir=cfg.data.out_dir,
        manifest=cfg.data.train_manifest,
        crop_sec=cfg.data.crop_sec,
        shuffle=True,
        crop_mode="random",
    )

    val_set = StaticMixDataset(
        out_dir=cfg.data.out_dir,
        manifest=cfg.data.val_manifest,
        crop_sec=cfg.data.crop_sec,
        shuffle=False,
        crop_mode="center",
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.validate.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader
