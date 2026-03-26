import json
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from speaker_verification.engine.trainer import train_one_epoch
from speaker_verification.engine.evaluator import validate
from speaker_verification.engine.checkpoint import save_checkpoint, get_model_state_from_ckpt
from speaker_verification.factory import build_model, build_loss, build_loaders
from speaker_verification.logging_utils import setup_logger
from speaker_verification.utils.seed import set_seed

try:
    from speaker_verification.utils.plot import plot_curves
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


def filter_state_dict(state_dict: dict, exclude_prefixes=None):
    exclude_prefixes = exclude_prefixes or []
    return {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(prefix) for prefix in exclude_prefixes)
    }


def freeze_backbone_params(model, head_prefixes=None):
    head_prefixes = head_prefixes or ["diar_head."]
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in head_prefixes)


@hydra.main(version_base=None, config_path="configs", config_name="experiment")
def main(cfg: DictConfig):
    os.makedirs(cfg.output.save_dir, exist_ok=True)
    logger, _ = setup_logger(cfg.output.save_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader = build_loaders(cfg, device)
    model = build_model(cfg, device)
    loss_fn = build_loss(cfg, device)

    if cfg.run.mode == "finetune":
        ckpt = torch.load(cfg.finetune.checkpoint_path, map_location=device)
        state_dict = get_model_state_from_ckpt(ckpt)

        if cfg.finetune.load_mode == "backbone_only":
            state_dict = filter_state_dict(state_dict, exclude_prefixes=list(cfg.finetune.head_prefixes))

        load_result = model.load_state_dict(state_dict, strict=cfg.finetune.strict_load)
        logger.info("Finetune checkpoint loaded from: %s", cfg.finetune.checkpoint_path)
        logger.info("Missing keys: %s", getattr(load_result, "missing_keys", []))
        logger.info("Unexpected keys: %s", getattr(load_result, "unexpected_keys", []))

        if cfg.finetune.freeze_backbone:
            freeze_backbone_params(model, head_prefixes=list(cfg.finetune.head_prefixes))

        actual_lr = cfg.train.lr * cfg.finetune.lr_scale
    else:
        actual_lr = cfg.train.lr

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=actual_lr,
        weight_decay=cfg.train.weight_decay,
    )

    scheduler = None
    if cfg.train.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.epochs,
            eta_min=cfg.train.min_lr,
        )

    best_metric = float("inf") if cfg.output.monitor_mode == "min" else float("-inf")
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

    last_ckpt = os.path.join(cfg.output.save_dir, "last.pt")
    best_ckpt = os.path.join(cfg.output.save_dir, "best.pt")
    history_json = os.path.join(cfg.output.save_dir, "history.json")

    for epoch in range(1, cfg.train.epochs + 1):
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
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_stats["loss"])
        history["train_pit_loss"].append(train_stats["pit_loss"])
        history["train_act_loss"].append(train_stats["act_loss"])
        history["train_cnt_loss"].append(train_stats["cnt_loss"])
        history["train_frm_loss"].append(train_stats["frm_loss"])
        history["val_loss"].append(val_stats["val_loss"])
        history["val_pit_loss"].append(val_stats["pit_loss"])
        history["val_act_loss"].append(val_stats["act_loss"])
        history["val_cnt_loss"].append(val_stats["cnt_loss"])
        history["val_frm_loss"].append(val_stats["frm_loss"])
        history["val_der"].append(val_stats["der"])
        history["val_count_acc"].append(val_stats["count_acc"])
        history["val_act_f1"].append(val_stats["act_f1"])

        save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch, best_metric, history, cfg)

        monitor_value = val_stats[cfg.output.monitor]
        improved = monitor_value < best_metric if cfg.output.monitor_mode == "min" else monitor_value > best_metric
        if improved:
            best_metric = monitor_value
            save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch, best_metric, history, cfg)

        with open(history_json, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if HAS_PLOT and cfg.output.plot_history:
            try:
                plot_curves(history, cfg.output.save_dir)
            except Exception as e:
                logger.warning("plot failed: %s", e)

    logger.info("Training finished. Best %s = %.6f", cfg.output.monitor, best_metric)


if __name__ == "__main__":
    main()