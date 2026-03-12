import os
import json
import logging
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from speaker_verification.models.resowave import ResoWave
from speaker_verification.loss.mulit_task import MultiTaskLoss
from speaker_verification.checkpointing import ModelCfg, build_ckpt, save_ckpt

from dataset.staticdataset import StaticMixDataset
from utils.seed import set_seed
from utils.meters import AverageMeter, compute_eer, diarization_error_rate_pit

try:
    from utils.plot import plot_curves
    _HAS_PLOT = True
except Exception:
    _HAS_PLOT = False


def setup_logger(out_dir: str, logger_name: str = "train_logger"):
    """
    创建同时输出到控制台和文件的 logger
    """
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

    # 文件输出：保存全部 DEBUG 级别信息
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 控制台输出：INFO 及以上
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger, log_file


@torch.no_grad()
def validate(model, loader, device, max_batches=200):
    model.eval()
    eer_scores, eer_labels = [], []
    der_list = []
    der_detail_sum = {"fa": 0.0, "miss": 0.0, "conf": 0.0, "gt_active": 0.0, "pred_active": 0.0}
    der_n = 0

    pbar = tqdm(loader, desc="VALID", total=min(len(loader), max_batches))
    for bi, batch in enumerate(pbar):
        if bi >= max_batches:
            break

        fbank = batch["fbank"].to(device)
        spk_label = batch["spk_label"].to(device)
        target_matrix = batch["target_matrix"].to(device)
        target_act = batch["target_activity"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        emb, frame_embeds, slot_logits, pred_act, pred_count = model(fbank, return_diarization=True)

        # SV EER
        for i in range(len(emb)):
            for j in range(i + 1, len(emb)):
                sc = torch.cosine_similarity(emb[i], emb[j], dim=0).item()
                label = 1 if spk_label[i] == spk_label[j] else 0
                eer_scores.append(sc)
                eer_labels.append(label)

        der, info = diarization_error_rate_pit(
            slot_logits,
            target_matrix,
            target_act,
            valid_mask=valid_mask,
            return_detail=True,
        )
        der_list.append(der.item())

        for k in der_detail_sum:
            der_detail_sum[k] += info[k]
        der_n += 1

        pbar.set_postfix(
            DER=f"{sum(der_list)/len(der_list)*100:.2f}%",
            FA=f"{info['fa']:.0f}",
            MISS=f"{info['miss']:.0f}",
            CONF=f"{info['conf']:.0f}",
        )

    eer, _ = compute_eer(eer_labels, eer_scores)
    avg_der = sum(der_list) / max(1, len(der_list)) * 100.0

    der_detail_avg = {k: v / max(1, der_n) for k, v in der_detail_sum.items()}

    return {
        "eer": eer,
        "der": avg_der,
        "der_detail": der_detail_avg,
    }

def train_one_epoch(model, loss_fn, loader, device, optim, scaler, use_amp, grad_clip):
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(loader, desc="TRAIN", ncols=120)

    for batch in pbar:
        fbank = batch["fbank"].to(device, non_blocking=True)
        spk_label = batch["spk_label"].to(device, non_blocking=True)
        target_matrix = batch["target_matrix"].to(device, non_blocking=True)
        target_act = batch["target_activity"].to(device, non_blocking=True)
        target_count = batch["target_count"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            emb, frame_embeds, slot_logits, pred_act, pred_count = model(fbank, return_diarization=True)

            loss = loss_fn(
                emb,
                slot_logits,
                pred_act,
                pred_count,
                spk_label,
                target_matrix,
                target_act,
                target_count,
                valid_mask=valid_mask,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

        bs = fbank.size(0)
        loss_meter.update(float(loss.item()), bs)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", LR=f"{optim.param_groups[0]['lr']:.2e}")

    return loss_meter.avg


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))
    os.makedirs(cfg.out_dir, exist_ok=True)

    logger, log_file = setup_logger(cfg.out_dir)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)

    logger.info("Configuration:")
    logger.info(json.dumps(cfg_dict, ensure_ascii=False, indent=2))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_dataset = StaticMixDataset(
        out_dir=cfg.data.out_dir,
        manifest=cfg.data.train_manifest,
        crop_sec=float(cfg.data.crop_sec),
        shuffle=True,
    )
    logger.info(
        f"Train dataset loaded | num_classes={train_dataset.num_classes} | size={len(train_dataset)}"
    )

    val_dataset = StaticMixDataset(
        out_dir=cfg.data.out_dir,
        manifest=cfg.data.val_manifest,
        crop_sec=float(cfg.data.crop_sec),
        shuffle=False,
    )
    logger.info(f"Val dataset loaded | size={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = ResoWave(
        in_channels=int(cfg.model.feat_dim),
        channels=int(cfg.model.channels),
        embd_dim=int(cfg.model.emb_dim),
        max_mix_speakers=int(cfg.model.max_mix_speakers),
    ).to(device)
    logger.info(f"Model initialized: {model.__class__.__name__}")

    loss_fn = MultiTaskLoss(
        embedding_dim=int(cfg.model.emb_dim),
        num_classes=int(train_dataset.num_classes),   # 这里只给 SV 分支用
        max_spk=int(cfg.model.max_mix_speakers),
        lambda_ver=float(cfg.loss.lambda_ver),
        lambda_pit=float(cfg.loss.lambda_pit),
        lambda_act=float(cfg.loss.lambda_act),
        lambda_cnt=float(cfg.loss.lambda_cnt),
        pos_weight=float(cfg.loss.pos_weight),
        pit_pos_weight=float(cfg.loss.pit_pos_weight),
    ).to(device)
    logger.info(f"Loss initialized: {loss_fn.__class__.__name__}")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=int(cfg.train.epochs)
    )

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info(f"AMP enabled: {use_amp}")

    history = {"train_loss": [], "val_eer": [], "val_der": []}
    best_eer = 1e9

    for epoch in range(1, int(cfg.train.epochs) + 1):
        logger.info("=" * 80)
        logger.info(f"Epoch {epoch}/{cfg.train.epochs} started")

        train_loss = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            loader=train_loader,
            device=device,
            optim=optim,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=float(cfg.train.grad_clip),
        )

        scheduler.step()
        logger.info(
            f"[Epoch {epoch}] Scheduler stepped | current_lr={optim.param_groups[0]['lr']:.6e}"
        )

        val_info = validate(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=int(cfg.train.val_batches),
        )

        history["train_loss"].append(train_loss)
        history["val_eer"].append(val_info["eer"])
        history["val_der"].append(val_info["der"])

        logger.info(
            f"[Epoch {epoch}] "
            f"Loss={train_loss:.4f} | "
            f"SV-EER={val_info['eer'] * 100:.2f}% | "
            f"DER={val_info['der']:.2f}% | "
            f"FA={val_info['der_detail']['fa']:.1f} | "
            f"MISS={val_info['der_detail']['miss']:.1f} | "
            f"CONF={val_info['der_detail']['conf']:.1f}"
        )

        ckpt = build_ckpt(
            model=model,
            optim=optim,
            scheduler=scheduler,
            epoch=epoch,
            best_eer=val_info["eer"],
            model_cfg=ModelCfg(
                channels=int(cfg.model.channels),
                emb_dim=int(cfg.model.emb_dim),
                feat_dim=int(cfg.model.feat_dim),
                sample_rate=16000,
            ),
        )

        last_ckpt_path = os.path.join(cfg.out_dir, "last.pt")
        save_ckpt(last_ckpt_path, ckpt)
        logger.info(f"[Epoch {epoch}] Saved checkpoint: {last_ckpt_path}")

        if val_info["eer"] < best_eer:
            best_eer = val_info["eer"]
            best_ckpt_path = os.path.join(cfg.out_dir, "best.pt")
            save_ckpt(best_ckpt_path, ckpt)
            logger.info(
                f"[Epoch {epoch}] New best EER={best_eer * 100:.2f}% | Saved best checkpoint: {best_ckpt_path}"
            )

        if _HAS_PLOT:
            try:
                plot_curves(cfg.out_dir, history)
                logger.info(f"[Epoch {epoch}] Curves plotted successfully")
            except Exception as e:
                logger.warning(f"[Epoch {epoch}] plot failed: {e}")

        history_path = os.path.join(cfg.out_dir, "history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f"[Epoch {epoch}] History saved: {history_path}")

    logger.info(
        f"Training finished! Best EER: {best_eer * 100:.2f}% | Output: {cfg.out_dir} | Log: {log_file}"
    )


if __name__ == "__main__":
    main()