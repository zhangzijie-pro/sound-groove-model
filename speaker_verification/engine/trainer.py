import torch
from tqdm import tqdm

from speaker_verification.utils.meters import AverageMeter


def train_one_epoch(model, loss_fn, loader, optimizer, device, grad_clip=None):
    model.train()

    total_meter = AverageMeter()
    pit_meter = AverageMeter()
    dice_meter = AverageMeter()
    activity_meter = AverageMeter()
    exist_meter = AverageMeter()
    consistency_meter = AverageMeter()
    smooth_meter = AverageMeter()

    pbar = tqdm(loader, desc="TRAIN", ncols=150)

    for batch in pbar:
        fbank = batch["fbank"].to(device, non_blocking=True)                  # [B,T,F]
        target_matrix = batch["target_matrix"].to(device, non_blocking=True)  # [B,T,K]
        target_count = batch["target_count"].to(device, non_blocking=True)    # [B]
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
        total_meter.update(float(loss.item()), bs)
        pit_meter.update(float(loss_dict["pit_loss"].item()), bs)
        dice_meter.update(float(loss_dict["dice_loss"].item()), bs)
        activity_meter.update(float(loss_dict["activity_loss"].item()), bs)
        exist_meter.update(float(loss_dict["exist_loss"].item()), bs)
        consistency_meter.update(float(loss_dict["consistency_loss"].item()), bs)
        smooth_meter.update(float(loss_dict["smooth_loss"].item()), bs)

        pbar.set_postfix(
            total=f"{total_meter.avg:.4f}",
            pit=f"{pit_meter.avg:.4f}",
            dice=f"{dice_meter.avg:.4f}",
            act=f"{activity_meter.avg:.4f}",
            exist=f"{exist_meter.avg:.4f}",
            cons=f"{consistency_meter.avg:.4f}",
        )

    return {
        "loss": total_meter.avg,
        "pit_loss": pit_meter.avg,
        "dice_loss": dice_meter.avg,
        "activity_loss": activity_meter.avg,
        "exist_loss": exist_meter.avg,
        "consistency_loss": consistency_meter.avg,
        "smooth_loss": smooth_meter.avg,
    }
