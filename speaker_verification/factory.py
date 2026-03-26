import torch
from torch.utils.data import DataLoader

from speaker_verification.models.resowave import ResoWave
from speaker_verification.loss.multi_task import MultiTaskLoss
from speaker_verification.dataset.static_dataset import StaticMixDataset


def build_model(cfg, device):
    model = ResoWave(
        in_channels=cfg.model.in_channels,
        channels=cfg.model.channels,
        embd_dim=cfg.model.embedding_dim,
        max_mix_speakers=cfg.model.max_mix_speakers,
    )
    return model.to(device)


def build_loss(cfg, device):
    loss_fn = MultiTaskLoss(
        max_spk=cfg.model.max_mix_speakers,
        lambda_pit=cfg.loss.lambda_pit,
        lambda_act=cfg.loss.lambda_act,
        lambda_cnt=cfg.loss.lambda_cnt,
        lambda_frm=cfg.loss.lambda_frm,
        pos_weight=cfg.loss.pos_weight,
        pit_pos_weight=cfg.loss.pit_pos_weight,
        proto_eps=cfg.loss.proto_eps,
        proto_temperature=cfg.loss.proto_temperature,
    )
    return loss_fn.to(device)


def build_loaders(cfg, device):
    train_set = StaticMixDataset(
        out_dir=cfg.data.out_dir,
        manifest=cfg.data.train_manifest,
        crop_sec=cfg.data.crop_sec,
        shuffle=True,
    )
    val_set = StaticMixDataset(
        out_dir=cfg.data.out_dir,
        manifest=cfg.data.val_manifest,
        crop_sec=cfg.data.crop_sec,
        shuffle=False,
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