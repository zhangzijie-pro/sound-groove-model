import os
import json
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.utils import *

try:
    from utils.plot import plot_curves
    _HAS_PLOT = True
except Exception:
    _HAS_PLOT = False


def _get_model_state_from_ckpt(ckpt: dict):
    """
    兼容不同 checkpoint 存储格式:
    - ckpt["model_state"]
    - ckpt["state_dict"]
    - 直接就是 state_dict
    """
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dict.")

    if "model_state" in ckpt:
        return ckpt["model_state"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]

    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt

    raise KeyError("Cannot find model weights in checkpoint. Expected 'model_state' or 'state_dict'.")


def _filter_state_dict(state_dict: dict, exclude_prefixes=None):
    if exclude_prefixes is None:
        exclude_prefixes = []

    filtered = {}
    for k, v in state_dict.items():
        if any(k.startswith(prefix) for prefix in exclude_prefixes):
            continue
        filtered[k] = v
    return filtered


def _freeze_backbone_params(model, head_prefixes=None, logger=None):
    if head_prefixes is None:
        head_prefixes = ["diar_head."]

    total, trainable = 0, 0
    for name, param in model.named_parameters():
        total += 1
        if any(name.startswith(prefix) for prefix in head_prefixes):
            param.requires_grad = True
            trainable += 1
        else:
            param.requires_grad = False

    if logger is not None:
        logger.info(
            "Backbone frozen. Trainable params groups: %d / %d (by named parameter tensors)",
            trainable, total
        )


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@hydra.main(version_base=None, config_path="configs", config_name="finetune")
def main(cfg: DictConfig):
    os.makedirs(cfg.output.save_dir, exist_ok=True)

    logger, _ = setup_logger(cfg.output.save_dir)
    logger.info("Finetune config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.train.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu"
    )
    logger.info("Using device: %s", device)

    train_loader, val_loader = build_loaders(cfg, device)
    model = build_model(cfg, device)
    loss_fn = build_loss(cfg, device)

    ckpt_path = cfg.finetune.checkpoint_path
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    pretrained_state = _get_model_state_from_ckpt(ckpt)

    load_mode = cfg.finetune.load_mode
    head_prefixes = list(cfg.finetune.head_prefixes)

    if load_mode == "full":
        state_to_load = pretrained_state
        logger.info("Load mode = full: load all available weights from checkpoint.")
    elif load_mode == "backbone_only":
        state_to_load = _filter_state_dict(
            pretrained_state,
            exclude_prefixes=head_prefixes
        )
        logger.info(
            "Load mode = backbone_only: excluding head prefixes = %s",
            head_prefixes
        )
    else:
        raise ValueError(
            f"Unsupported finetune.load_mode={load_mode}, expected one of: ['full', 'backbone_only']"
        )

    load_result = model.load_state_dict(state_to_load, strict=cfg.finetune.strict_load)
    logger.info("Checkpoint loaded from: %s", ckpt_path)
    logger.info("Checkpoint epoch: %s", ckpt.get("epoch", "unknown"))
    logger.info("strict_load = %s", cfg.finetune.strict_load)

    if hasattr(load_result, "missing_keys") and hasattr(load_result, "unexpected_keys"):
        logger.info("Missing keys (%d): %s", len(load_result.missing_keys), load_result.missing_keys)
        logger.info("Unexpected keys (%d): %s", len(load_result.unexpected_keys), load_result.unexpected_keys)

    if cfg.finetune.freeze_backbone:
        _freeze_backbone_params(model, head_prefixes=head_prefixes, logger=logger)
    else:
        logger.info("Backbone is NOT frozen. All loaded params remain trainable unless already disabled.")

    total_params, trainable_params = _count_params(model)
    logger.info("Model params: total=%d | trainable=%d", total_params, trainable_params)

    finetune_lr = cfg.train.lr * cfg.finetune.lr_scale
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=finetune_lr,
        weight_decay=cfg.train.weight_decay,
    )
    logger.info("Optimizer: AdamW | lr=%.8f | weight_decay=%.8f", finetune_lr, cfg.train.weight_decay)

    scheduler = None
    if cfg.train.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.epochs,
            eta_min=cfg.train.min_lr,
        )
        logger.info("Scheduler: CosineAnnealingLR | T_max=%d | eta_min=%.8e", cfg.train.epochs, cfg.train.min_lr)

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
            logger.info(
                "[Epoch %d] Scheduler stepped | current_lr=%.6e",
                epoch,
                optimizer.param_groups[0]["lr"]
            )

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

        logger.info(
            "[Epoch %d] "
            "TrainLoss=%.4f | TrainPIT=%.4f | TrainACT=%.4f | TrainCNT=%.4f | TrainFRM=%.6f",
            epoch,
            train_stats["loss"],
            train_stats["pit_loss"],
            train_stats["act_loss"],
            train_stats["cnt_loss"],
            train_stats["frm_loss"],
        )

        logger.info(
            "[Epoch %d] "
            "ValLoss=%.4f | PIT=%.4f | ACT=%.4f | CNT=%.4f | FRM=%.6f | "
            "DER=%.2f%% | CAcc=%.2f%% | ActF1=%.2f%%",
            epoch,
            val_stats["val_loss"],
            val_stats["pit_loss"],
            val_stats["act_loss"],
            val_stats["cnt_loss"],
            val_stats["frm_loss"],
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

        save_ckpt(
            last_ckpt,
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric,
            history,
            cfg,
        )
        logger.info("[Epoch %d] Saved checkpoint: %s", epoch, last_ckpt)

        monitor_value = val_stats[cfg.output.monitor]
        improved = (
            monitor_value < best_metric
            if cfg.output.monitor_mode == "min"
            else monitor_value > best_metric
        )

        if improved:
            best_metric = monitor_value
            save_ckpt(
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
        logger.info("[Epoch %d] History saved: %s", epoch, history_json)

        if _HAS_PLOT and cfg.output.plot_history:
            try:
                plot_curves(cfg.output.save_dir, history)
            except Exception as e:
                logger.warning("[Epoch %d] plot failed: %s", epoch, e)

    logger.info("=" * 80)
    logger.info("Finetuning finished. Best %s = %.6f", cfg.output.monitor, best_metric)
    logger.info("Best checkpoint: %s", best_ckpt)


if __name__ == "__main__":
    main()