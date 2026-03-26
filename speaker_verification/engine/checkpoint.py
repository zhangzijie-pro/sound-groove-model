import torch
from omegaconf import OmegaConf


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, history, cfg):
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


def get_model_state_from_ckpt(ckpt: dict):
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dict.")
    if "model_state" in ckpt:
        return ckpt["model_state"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt
    raise KeyError("Cannot find model weights in checkpoint.")