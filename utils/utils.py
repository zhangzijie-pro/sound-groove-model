from datetime import datetime
import os
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from speaker_verification.models.resowave import ResoWave
from speaker_verification.loss.mulit_task import MultiTaskLoss
from dataset.staticdataset import StaticMixDataset
from utils.seed import set_seed
from utils.meters import (
    AverageMeter,
    DERTracker,
    diarization_error_rate_pit,
    compute_count_acc,
    compute_activity_metrics,
)
from utils.utils import *

try:
    from utils.plot import plot_curves
    _HAS_PLOT = True
except Exception:
    _HAS_PLOT = False


def setup_logger(out_dir: str, logger_name: str = "train_logger"):
    os.makedirs(out_dir, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(out_dir, f"train_{log_time}.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger, log_file


def save_ckpt(path, model, optimizer, scheduler, epoch, best_metric, history, cfg):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "history": history,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(ckpt, path)


def build_model(cfg, device):
    model = ResoWave(
        in_channels=cfg.model.in_channels,
        channels=cfg.model.channels,
        embd_dim=cfg.model.embd_dim,
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
        # lambda_ortho=cfg.loss.lambda_ortho,
        pos_weight=cfg.loss.pos_weight,
        pit_pos_weight=cfg.loss.pit_pos_weight,
        proto_eps=cfg.loss.proto_eps,
        proto_temperature=cfg.loss.proto_temperature,
    )
    return loss_fn.to(device)


def build_loaders(cfg, device):
    train_set = StaticMixDataset(
        # out_dir=cfg.data.out_dir,
        # manifest=cfg.data.train_manifest,
        crop_sec=cfg.data.crop_sec,
        shuffle=True,
    )
    val_set = StaticMixDataset(
        # out_dir=cfg.data.out_dir,
        # manifest=cfg.data.val_manifest,
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


@torch.no_grad()
def validate(model, loss_fn, loader, device, max_batches=-1, activity_threshold=0.5):
    model.eval()

    val_loss_meter = AverageMeter()
    pit_meter = AverageMeter()
    act_meter = AverageMeter()
    cnt_meter = AverageMeter()
    frm_meter = AverageMeter()

    der_tracker = DERTracker()

    count_acc_sum = 0.0
    count_acc_n = 0

    act_prec_sum = 0.0
    act_rec_sum = 0.0
    act_f1_sum = 0.0
    act_n = 0

    total_steps = len(loader) if max_batches is None or max_batches < 0 else min(len(loader), max_batches)
    pbar = tqdm(loader, desc="VALID", total=total_steps, ncols=120)

    for bi, batch in enumerate(pbar):
        if max_batches is not None and max_batches >= 0 and bi >= max_batches:
            break

        fbank = batch["fbank"].to(device, non_blocking=True)
        target_matrix = batch["target_matrix"].to(device, non_blocking=True)
        target_act = batch["target_activity"].to(device, non_blocking=True)
        target_count = batch["target_count"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        _, frame_embeds, slot_logits, pred_act, pred_count = model(
            fbank, return_diarization=True
        )

        loss_dict = loss_fn(
            frame_embeds=frame_embeds,
            slot_logits=slot_logits,
            pred_activity=pred_act,
            pred_count=pred_count,
            target_matrix=target_matrix,
            target_activity=target_act,
            target_count=target_count,
            valid_mask=valid_mask,
            return_detail=True,
        )
        loss = loss_dict["total"]

        bs = fbank.size(0)
        val_loss_meter.update(float(loss.item()), bs)
        pit_meter.update(float(loss_dict["pit_loss"].item()), bs)
        act_meter.update(float(loss_dict["act_loss"].item()), bs)
        cnt_meter.update(float(loss_dict["cnt_loss"].item()), bs)
        frm_meter.update(float(loss_dict["frm_loss"].item()), bs)

        _, info = diarization_error_rate_pit(
            slot_logits,
            target_matrix,
            target_act,
            valid_mask=valid_mask,
            return_detail=True,
        )
        der_tracker.update(info)

        cacc = compute_count_acc(pred_count, target_count)
        count_acc_sum += cacc
        count_acc_n += 1

        ap, ar, af1 = compute_activity_metrics(
            pred_act, target_act, valid_mask, threshold=activity_threshold
        )
        act_prec_sum += ap
        act_rec_sum += ar
        act_f1_sum += af1
        act_n += 1

        global_der = der_tracker.value() * 100.0
        pbar.set_postfix(
            loss=f"{val_loss_meter.avg:.4f}",
            DER=f"{global_der:.2f}%",
            CAcc=f"{(count_acc_sum / max(count_acc_n, 1)) * 100:.1f}%",
            ActF1=f"{(act_f1_sum / max(act_n, 1)) * 100:.1f}%"
        )

    der_detail = der_tracker.detail()

    return {
        "val_loss": val_loss_meter.avg,
        "pit_loss": pit_meter.avg,
        "act_loss": act_meter.avg,
        "cnt_loss": cnt_meter.avg,
        "frm_loss": frm_meter.avg,
        "der": der_detail["der"] * 100.0,
        "der_detail": {
            "fa": der_detail["fa"],
            "miss": der_detail["miss"],
            "conf": der_detail["conf"],
            "gt_active": der_detail["gt_active"],
            "pred_active": der_detail["pred_active"],
        },
        "count_acc": count_acc_sum / max(1, count_acc_n),
        "act_prec": act_prec_sum / max(1, act_n),
        "act_rec": act_rec_sum / max(1, act_n),
        "act_f1": act_f1_sum / max(1, act_n),
    }


def train_one_epoch(model, loss_fn, loader, optimizer, device, grad_clip=None):
    model.train()

    loss_meter = AverageMeter()
    pit_meter = AverageMeter()
    act_meter = AverageMeter()
    cnt_meter = AverageMeter()
    frm_meter = AverageMeter()

    pbar = tqdm(loader, desc="TRAIN", ncols=120)
    for batch in pbar:
        fbank = batch["fbank"].to(device, non_blocking=True)
        target_matrix = batch["target_matrix"].to(device, non_blocking=True)
        target_act = batch["target_activity"].to(device, non_blocking=True)
        target_count = batch["target_count"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        _, frame_embeds, slot_logits, pred_act, pred_count = model(
            fbank, return_diarization=True
        )

        loss_dict = loss_fn(
            frame_embeds=frame_embeds,
            slot_logits=slot_logits,
            pred_activity=pred_act,
            pred_count=pred_count,
            target_matrix=target_matrix,
            target_activity=target_act,
            target_count=target_count,
            valid_mask=valid_mask,
            return_detail=True,
        )
        loss = loss_dict["total"]
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = fbank.size(0)
        loss_meter.update(float(loss.item()), bs)
        pit_meter.update(float(loss_dict["pit_loss"].item()), bs)
        act_meter.update(float(loss_dict["act_loss"].item()), bs)
        cnt_meter.update(float(loss_dict["cnt_loss"].item()), bs)
        frm_meter.update(float(loss_dict["frm_loss"].item()), bs)

        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            pit=f"{pit_meter.avg:.4f}",
            act=f"{act_meter.avg:.4f}",
            cnt=f"{cnt_meter.avg:.4f}",
            frm=f"{frm_meter.avg:.4f}",
        )

    return {
        "loss": loss_meter.avg,
        "pit_loss": pit_meter.avg,
        "act_loss": act_meter.avg,
        "cnt_loss": cnt_meter.avg,
        "frm_loss": frm_meter.avg,
    }
