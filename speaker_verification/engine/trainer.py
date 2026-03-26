import torch
from tqdm import tqdm

from speaker_verification.utils.meters import AverageMeter


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
        target_activity = batch["target_activity"].to(device, non_blocking=True)
        target_count = batch["target_count"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        _, frame_embeddings, slot_logits, pred_activity, pred_count = model(
            fbank, return_diarization=True
        )

        loss_dict = loss_fn(
            frame_embeds=frame_embeddings,
            slot_logits=slot_logits,
            pred_activity=pred_activity,
            pred_count=pred_count,
            target_matrix=target_matrix,
            target_activity=target_activity,
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