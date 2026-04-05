import json
import os
from typing import Dict, Any, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from speaker_verification.engine.trainer import train_one_epoch
from speaker_verification.engine.evaluator import validate
from speaker_verification.engine.checkpoint import (
    save_checkpoint,
    get_model_state_from_ckpt,
)
from speaker_verification.factory import build_model, build_loss, build_loaders
from speaker_verification.logging_utils import setup_logger
from speaker_verification.utils.seed import set_seed
from speaker_verification.quantization.turboquant import build_turboquant


try:
    from speaker_verification.utils.plot import plot_curves
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


def filter_state_dict(state_dict: Dict[str, torch.Tensor], exclude_prefixes=None):
    exclude_prefixes = exclude_prefixes or []
    return {
        k: v
        for k, v in state_dict.items()
        if not any(k.startswith(prefix) for prefix in exclude_prefixes)
    }


def freeze_backbone_params(model, head_prefixes=None):
    head_prefixes = head_prefixes or ["diar_head."]
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in head_prefixes)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer):
    if not cfg.train.use_scheduler:
        return None

    warmup_epochs = int(getattr(cfg.train, "warmup_epochs", 0))
    warmup_epochs = max(0, min(warmup_epochs, int(cfg.train.epochs) - 1))
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.train.epochs),
            eta_min=float(cfg.train.min_lr),
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(cfg.train.warmup_start_factor),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg.train.epochs) - warmup_epochs),
        eta_min=float(cfg.train.min_lr),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def build_history() -> Dict[str, list]:
    return {
        "train_loss": [],
        "train_diar_loss": [],
        "train_smooth_loss": [],
        "val_loss": [],
        "val_diar_loss": [],
        "val_smooth_loss": [],
        "val_der": [],
        "val_count_acc": [],
        "val_act_prec": [],
        "val_act_rec": [],
        "val_act_f1": [],
    }


def append_history(history: Dict[str, list], train_stats: Dict[str, float], val_stats: Dict[str, float]):
    history["train_loss"].append(train_stats["loss"])
    history["train_diar_loss"].append(train_stats["diar_loss"])
    history["train_smooth_loss"].append(train_stats["smooth_loss"])

    history["val_loss"].append(val_stats["val_loss"])
    history["val_diar_loss"].append(val_stats["diar_loss"])
    history["val_smooth_loss"].append(val_stats["smooth_loss"])
    history["val_der"].append(val_stats["der"])
    history["val_count_acc"].append(val_stats["count_acc"])
    history["val_act_prec"].append(val_stats["act_prec"])
    history["val_act_rec"].append(val_stats["act_rec"])
    history["val_act_f1"].append(val_stats["act_f1"])


def log_epoch_stats(logger, epoch: int, train_stats: Dict[str, float], val_stats: Dict[str, float], lr: float):
    logger.info(
        "[Epoch %d] LR=%.6e | "
        "TrainLoss=%.4f | TrainDiar=%.4f | TrainSmooth=%.6f",
        epoch,
        lr,
        train_stats["loss"],
        train_stats["diar_loss"],
        train_stats["smooth_loss"],
    )

    logger.info(
        "[Epoch %d] "
        "ValLoss=%.4f | Diar=%.4f | Smooth=%.6f | "
        "DER=%.2f%% | CAcc=%.2f%% | ActF1=%.2f%%",
        epoch,
        val_stats["val_loss"],
        val_stats["diar_loss"],
        val_stats["smooth_loss"],
        val_stats["der"],
        val_stats["count_acc"] * 100.0,
        val_stats["act_f1"] * 100.0,
    )

    logger.info(
        "[Epoch %d] DER detail | FA=%.2f | MISS=%.2f | CONF=%.2f | GT=%.2f | PRED=%.2f",
        epoch,
        val_stats["der_detail"]["fa"],
        val_stats["der_detail"]["miss"],
        val_stats["der_detail"]["conf"],
        val_stats["der_detail"]["gt_active"],
        val_stats["der_detail"]["pred_active"],
    )

    if val_stats.get("proto_cos_q", None) is not None:
        logger.info(
            "[Epoch %d] QuantCheck | ProtoCosQ=%.4f | TrackerUpdatesQ=%d",
            epoch,
            val_stats["proto_cos_q"],
            int(val_stats.get("tracker_updates_q", 0)),
        )

def maybe_load_finetune_weights(cfg, model, device, logger) -> float:
    if cfg.run.mode != "finetune":
        return float(cfg.train.lr)

    ckpt_path = cfg.finetune.checkpoint_path
    if not ckpt_path:
        raise ValueError("run.mode=finetune but finetune.checkpoint_path is empty")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Finetune checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = get_model_state_from_ckpt(ckpt)

    if cfg.finetune.load_mode == "backbone_only":
        state_dict = filter_state_dict(
            state_dict,
            exclude_prefixes=list(cfg.finetune.head_prefixes),
        )
    elif cfg.finetune.load_mode != "full":
        raise ValueError(f"Unsupported finetune.load_mode: {cfg.finetune.load_mode}")

    load_result = model.load_state_dict(state_dict, strict=cfg.finetune.strict_load)
    logger.info("Finetune checkpoint loaded from: %s", ckpt_path)
    logger.info("Missing keys: %s", getattr(load_result, "missing_keys", []))
    logger.info("Unexpected keys: %s", getattr(load_result, "unexpected_keys", []))

    if cfg.finetune.freeze_backbone:
        freeze_backbone_params(model, head_prefixes=list(cfg.finetune.head_prefixes))
        logger.info("Backbone frozen. Train only head prefixes: %s", list(cfg.finetune.head_prefixes))
    else:
        logger.info("Backbone not frozen.")

    return float(cfg.train.lr) * float(cfg.finetune.lr_scale)


def maybe_resume_training(cfg, model, optimizer, scheduler, device, logger):
    if not cfg.resume.enabled:
        return 1, (float("inf") if cfg.output.monitor_mode == "min" else float("-inf")), build_history()

    ckpt_path = cfg.resume.checkpoint_path
    if not ckpt_path:
        raise ValueError("resume.enabled=true but resume.checkpoint_path is empty")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(get_model_state_from_ckpt(ckpt), strict=True)

    if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = ckpt.get(
        "best_metric",
        float("inf") if cfg.output.monitor_mode == "min" else float("-inf"),
    )
    loaded_history = ckpt.get("history", {})
    history = build_history()
    for key, value in loaded_history.items():
        if key in history and isinstance(value, list):
            history[key] = value

    logger.info("Resume checkpoint loaded from: %s", ckpt_path)
    logger.info("Resume start epoch: %d", start_epoch)
    logger.info("Resume best_metric: %s", str(best_metric))

    return start_epoch, best_metric, history


@hydra.main(version_base=None, config_path="configs", config_name="experiment")
def main(cfg: DictConfig):
    os.makedirs(cfg.output.save_dir, exist_ok=True)
    logger, _ = setup_logger(cfg.output.save_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.train.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu"
    )
    logger.info("Using device: %s", device)

    train_loader, val_loader = build_loaders(cfg, device)
    model = build_model(cfg, device)
    loss_fn = build_loss(cfg, device)
    frame_quantizer = None
    tracker_quantizer = None

    if getattr(cfg, "quant", None) is not None and cfg.quant.enable:
        frame_quantizer = build_turboquant(
            mode=cfg.quant.mode,
            dim=int(cfg.model.embedding_dim),
            bits=int(cfg.quant.bits),
            clip_sigma=float(cfg.quant.clip_sigma),
            device=str(device),
            seed=int(cfg.quant.seed),
        )

        if cfg.quant.apply_tracker_quant:
            tracker_quantizer = build_turboquant(
                mode=cfg.quant.mode,
                dim=int(cfg.model.embedding_dim),
                bits=int(cfg.quant.bits),
                clip_sigma=float(cfg.quant.clip_sigma),
                device=str(device),
                seed=int(cfg.quant.seed) + 99,
            )

    actual_lr = maybe_load_finetune_weights(cfg, model, device, logger)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=actual_lr,
        weight_decay=cfg.train.weight_decay,
    )

    scheduler = build_scheduler(cfg, optimizer)

    start_epoch, best_metric, history = maybe_resume_training(
        cfg, model, optimizer, scheduler, device, logger
    )

    total_params, trainable_params = count_parameters(model)
    logger.info("Model params | total=%d | trainable=%d", total_params, trainable_params)
    logger.info("Actual LR: %.6e", actual_lr)

    last_ckpt = os.path.join(cfg.output.save_dir, "last.pt")
    best_ckpt = os.path.join(cfg.output.save_dir, "best.pt")
    history_json = os.path.join(cfg.output.save_dir, "history.json")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        logger.info("=" * 80)
        logger.info("Epoch %d/%d started", epoch, cfg.train.epochs)

        train_stats = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=cfg.train.grad_clip,
        )

        val_stats = validate(
            model=model,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device,
            max_batches=cfg.validate.max_batches,
            activity_threshold=cfg.validate.activity_threshold,
            slot_threshold=cfg.validate.slot_threshold,
            min_active_frames=cfg.validate.min_active_frames,
            min_slot_run=cfg.validate.min_slot_run,
            fill_gap_frames=cfg.validate.fill_gap_frames,
            slot_presence_frames=cfg.validate.slot_presence_frames,
            frame_quantizer=frame_quantizer if cfg.quant.apply_frame_quant_in_validate else None,
            tracker_quantizer=tracker_quantizer if cfg.quant.apply_tracker_quant else None,
            run_quant_tracker=bool(cfg.quant.run_tracker_in_validate),
            tracker_match_threshold=cfg.validate.tracker_match_threshold,
            tracker_momentum=cfg.validate.tracker_momentum,
            tracker_max_misses=cfg.validate.tracker_max_misses,
        )

        if scheduler is not None:
            scheduler.step()

        append_history(history, train_stats, val_stats)

        current_lr = optimizer.param_groups[0]["lr"]
        log_epoch_stats(logger, epoch, train_stats, val_stats, current_lr)

        save_checkpoint(
            last_ckpt,
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric,
            history,
            cfg,
        )

        monitor_value = val_stats[cfg.output.monitor]
        improved = (
            monitor_value < best_metric
            if cfg.output.monitor_mode == "min"
            else monitor_value > best_metric
        )

        if improved:
            best_metric = monitor_value
            save_checkpoint(
                best_ckpt,
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                history,
                cfg,
            )
            logger.info(
                "[Epoch %d] New best checkpoint saved: %s | %s=%.6f",
                epoch,
                best_ckpt,
                cfg.output.monitor,
                monitor_value,
            )

        with open(history_json, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if HAS_PLOT and cfg.output.plot_history:
            try:
                plot_curves(cfg.output.save_dir, history)
            except Exception as e:
                logger.warning("plot failed: %s", e)

    logger.info("=" * 80)
    logger.info("Training finished. Best %s = %.6f", cfg.output.monitor, best_metric)
    logger.info("Best checkpoint: %s", best_ckpt)


if __name__ == "__main__":
    main()
