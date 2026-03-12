import os
import json
import logging
from datetime import datetime

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
from utils.meters import AverageMeter, diarization_error_rate_pit, compute_count_acc, compute_activity_metrics

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


@torch.no_grad()
def validate(model, loss_fn, loader, device, max_batches=200, activity_threshold=0.5):
    model.eval()

    val_loss_meter = AverageMeter()
    pit_meter = AverageMeter()
    act_meter = AverageMeter()
    cnt_meter = AverageMeter()
    frm_meter = AverageMeter()

    der_list = []
    der_detail_sum = {"fa": 0.0, "miss": 0.0, "conf": 0.0, "gt_active": 0.0, "pred_active": 0.0}
    der_n = 0

    count_acc_sum = 0.0
    count_acc_n = 0

    act_prec_sum = 0.0
    act_rec_sum = 0.0
    act_f1_sum = 0.0
    act_n = 0

    pbar = tqdm(loader, desc="VALID", total=min(len(loader), max_batches), ncols=120)

    for bi, batch in enumerate(pbar):
        if bi >= max_batches:
            break

        fbank = batch["fbank"].to(device, non_blocking=True)
        target_matrix = batch["target_matrix"].to(device, non_blocking=True)
        target_act = batch["target_activity"].to(device, non_blocking=True)
        target_count = batch["target_count"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        _, frame_embeds, slot_logits, pred_act, pred_count = model(fbank, return_diarization=True)

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

        der, info = diarization_error_rate_pit(
            slot_logits,
            target_matrix,
            target_act,
            valid_mask=valid_mask,
            return_detail=True,
        )
        der_list.append(float(der.item()))

        for k in der_detail_sum:
            der_detail_sum[k] += float(info[k])
        der_n += 1

        cacc = compute_count_acc(pred_count, target_count)
        count_acc_sum += cacc
        count_acc_n += 1

        ap, ar, af1 = compute_activity_metrics(pred_act, target_act, valid_mask, threshold=activity_threshold)
        act_prec_sum += ap
        act_rec_sum += ar
        act_f1_sum += af1
        act_n += 1

        pbar.set_postfix(
            loss=f"{val_loss_meter.avg:.4f}",
            DER=f"{(sum(der_list)/max(len(der_list),1))*100:.2f}%",
            CAcc=f"{(count_acc_sum/max(count_acc_n,1))*100:.1f}%",
            ActF1=f"{(act_f1_sum/max(act_n,1))*100:.1f}%"
        )

    avg_der = sum(der_list) / max(1, len(der_list)) * 100.0
    der_detail_avg = {k: v / max(1, der_n) for k, v in der_detail_sum.items()}

    return {
        "val_loss": val_loss_meter.avg,
        "pit_loss": pit_meter.avg,
        "act_loss": act_meter.avg,
        "cnt_loss": cnt_meter.avg,
        "frm_loss": frm_meter.avg,
        "der": avg_der,
        "der_detail": der_detail_avg,
        "count_acc": count_acc_sum / max(1, count_acc_n),
        "act_prec": act_prec_sum / max(1, act_n),
        "act_rec": act_rec_sum / max(1, act_n),
        "act_f1": act_f1_sum / max(1, act_n),
    }


def train_one_epoch(model, loss_fn, loader, device, optim, scaler, use_amp, grad_clip):
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

        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            _, frame_embeds, slot_logits, pred_act, pred_count = model(fbank, return_diarization=True)

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
            LR=f"{optim.param_groups[0]['lr']:.2e}",
        )

    return {
        "loss": loss_meter.avg,
        "pit_loss": pit_meter.avg,
        "act_loss": act_meter.avg,
        "cnt_loss": cnt_meter.avg,
        "frm_loss": frm_meter.avg,
    }


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
        f"Train dataset loaded | num_classes={getattr(train_dataset, 'num_classes', 'N/A')} | size={len(train_dataset)}"
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

    lambda_frm = float(getattr(cfg.loss, "lambda_frm", 0.5))

    loss_fn = MultiTaskLoss(
        max_spk=int(cfg.model.max_mix_speakers),
        lambda_pit=float(cfg.loss.lambda_pit),
        lambda_act=float(cfg.loss.lambda_act),
        lambda_cnt=float(cfg.loss.lambda_cnt),
        lambda_frm=lambda_frm,
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
        optim,
        T_max=int(cfg.train.epochs)
    )

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info(f"AMP enabled: {use_amp}")

    history = {
        "train_loss": [],
        "train_pit_loss": [],
        "train_act_loss": [],
        "train_cnt_loss": [],
        "train_frm_loss": [],
        "val_loss": [],
        "val_pit_loss": [],
        "val_act_loss": [],
        "val_cnt_loss": [],
        "val_frm_loss": [],
        "val_der": [],
        "val_count_acc": [],
        "val_act_f1": [],
    }

    best_der = 1e9

    for epoch in range(1, int(cfg.train.epochs) + 1):
        logger.info("=" * 80)
        logger.info(f"Epoch {epoch}/{cfg.train.epochs} started")

        train_info = train_one_epoch(
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
            loss_fn=loss_fn,
            loader=val_loader,
            device=device,
            max_batches=int(cfg.train.val_batches),
            activity_threshold=float(getattr(cfg.train, "activity_threshold", 0.5)),
        )

        history["train_loss"].append(train_info["loss"])
        history["train_pit_loss"].append(train_info["pit_loss"])
        history["train_act_loss"].append(train_info["act_loss"])
        history["train_cnt_loss"].append(train_info["cnt_loss"])
        history["train_frm_loss"].append(train_info["frm_loss"])

        history["val_loss"].append(val_info["val_loss"])
        history["val_pit_loss"].append(val_info["pit_loss"])
        history["val_act_loss"].append(val_info["act_loss"])
        history["val_cnt_loss"].append(val_info["cnt_loss"])
        history["val_frm_loss"].append(val_info["frm_loss"])
        history["val_der"].append(val_info["der"])
        history["val_count_acc"].append(val_info["count_acc"])
        history["val_act_f1"].append(val_info["act_f1"])

        logger.info(
            f"[Epoch {epoch}] "
            f"TrainLoss={train_info['loss']:.4f} | "
            f"Train(PIT/ACT/CNT/FRM)=({train_info['pit_loss']:.4f}/"
            f"{train_info['act_loss']:.4f}/"
            f"{train_info['cnt_loss']:.4f}/"
            f"{train_info['frm_loss']:.4f}) | "
            f"ValLoss={val_info['val_loss']:.4f} | "
            f"DER={val_info['der']:.2f}% | "
            f"CountAcc={val_info['count_acc'] * 100:.2f}% | "
            f"ActF1={val_info['act_f1'] * 100:.2f}% | "
            f"FA={val_info['der_detail']['fa']:.1f} | "
            f"MISS={val_info['der_detail']['miss']:.1f} | "
            f"CONF={val_info['der_detail']['conf']:.1f}"
        )

        ckpt = build_ckpt(
            model=model,
            optim=optim,
            scheduler=scheduler,
            epoch=epoch,
            best_eer=float(val_info["der"]),
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

        if val_info["der"] < best_der:
            best_der = val_info["der"]
            best_ckpt_path = os.path.join(cfg.out_dir, "best.pt")
            save_ckpt(best_ckpt_path, ckpt)
            logger.info(
                f"[Epoch {epoch}] New best DER={best_der:.2f}% | Saved best checkpoint: {best_ckpt_path}"
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
        f"Training finished! Best DER: {best_der:.2f}% | Output: {cfg.out_dir} | Log: {log_file}"
    )


if __name__ == "__main__":
    main()