import torch
from tqdm import tqdm

from speaker_verification.utils.meters import AverageMeter


def train_one_epoch(model, loss_fn, loader, optimizer, device, grad_clip=None):
    model.train()

    loss_meter = AverageMeter()
    diar_meter = AverageMeter()
    smooth_meter = AverageMeter()

    pbar = tqdm(loader, desc="TRAIN", ncols=120)
    for batch in pbar:
        fbank = batch["fbank"].to(device, non_blocking=True)
        target_matrix = batch["target_matrix"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        _, diar_logits = model(
            fbank
        )

        loss_dict = loss_fn(
            diar_logits=diar_logits,
            target_matrix=target_matrix,
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
        diar_meter.update(float(loss_dict["diar_loss"].item()), bs)
        smooth_meter.update(float(loss_dict["smooth_loss"].item()), bs)

        pbar.set_postfix(
            loss=f"{loss_meter.avg:.4f}",
            diar=f"{diar_meter.avg:.4f}",
            smooth=f"{smooth_meter.avg:.4f}",
        )

    return {
        "loss": loss_meter.avg,
        "diar_loss": diar_meter.avg,
        "smooth_loss": smooth_meter.avg,
    }
