import torch
from tqdm import tqdm

from speaker_verification.utils.meters import AverageMeter


def train_one_epoch(model, loss_fn, loader, optimizer, device, grad_clip=None):
    model.train()

    total_meter = AverageMeter()
    pit_meter = AverageMeter()
    activity_meter = AverageMeter()
    exist_meter = AverageMeter()
    diversity_meter = AverageMeter()

    pbar = tqdm(loader, desc="TRAIN", ncols=150)

    for batch in pbar:
        fbank = batch["fbank"].to(device, non_blocking=True)                  # [B,T,F]
        target_matrix = batch["target_matrix"].to(device, non_blocking=True)  # [B,T,K]
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)        # [B,T]

        optimizer.zero_grad(set_to_none=True)

        frame_embeds, attractors, exist_logits, diar_logits, activity_logits = model(
            fbank,
            valid_mask=valid_mask,
        )

        loss_dict = loss_fn(
            frame_embeds=frame_embeds,
            attractors=attractors,
            exist_logits=exist_logits,
            diar_logits=diar_logits,
            activity_logits=activity_logits,
            target_matrix=target_matrix,
            target_count=batch["target_count"].to(device, non_blocking=True),
            valid_mask=valid_mask,
            return_detail=True,
        )

        loss = loss_dict["total"]
        loss.backward()

        grad_norm = None
        if grad_clip is not None and grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if grad_norm is not None and not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()

        bs = fbank.size(0)
        total_meter.update(float(loss.item()), bs)
        pit_meter.update(float(loss_dict["pit_loss"].item()), bs)
        activity_meter.update(float(loss_dict["activity_loss"].item()), bs)
        exist_meter.update(float(loss_dict["exist_loss"].item()), bs)
        diversity_meter.update(float(loss_dict["diversity_loss"].item()), bs)

        pbar.set_postfix(
            total=f"{total_meter.avg:.4f}",
            pit=f"{pit_meter.avg:.4f}",
            act=f"{activity_meter.avg:.4f}",
            exist=f"{exist_meter.avg:.4f}",
            div=f"{diversity_meter.avg:.4f}",
        )

    return {
        "loss": total_meter.avg,
        "pit_loss": pit_meter.avg,
        "activity_loss": activity_meter.avg,
        "exist_loss": exist_meter.avg,
        "diversity_loss": diversity_meter.avg,
    }
