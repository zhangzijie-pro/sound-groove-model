import argparse
import torch
from torch.utils.data import DataLoader

from speaker_verification.models.resowave import ResoWave
from speaker_verification.loss.mulit_task import MultiTaskLoss
from dataset.staticdataset import StaticMixDataset
from train import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_out_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--crop_sec", type=float, default=4.0)

    parser.add_argument("--feat_dim", type=int, default=80)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--emb_dim", type=int, default=192)
    parser.add_argument("--max_mix_speakers", type=int, default=4)

    parser.add_argument("--lambda_pit", type=float, default=1.0)
    parser.add_argument("--lambda_act", type=float, default=1.0)
    parser.add_argument("--lambda_cnt", type=float, default=0.2)
    parser.add_argument("--lambda_frm", type=float, default=0.5)
    parser.add_argument("--pos_weight", type=float, default=2.0)
    parser.add_argument("--pit_pos_weight", type=float, default=1.5)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_batches", type=int, default=100)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    val_dataset = StaticMixDataset(
        out_dir=args.data_out_dir,
        manifest=args.manifest,
        crop_sec=float(args.crop_sec),
        shuffle=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = ResoWave(
        in_channels=args.feat_dim,
        channels=args.channels,
        embd_dim=args.emb_dim,
        max_mix_speakers=args.max_mix_speakers,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]

    model.load_state_dict(state_dict, strict=True)
    print(f"[INFO] Loaded checkpoint strictly: {args.ckpt}")

    loss_fn = MultiTaskLoss(
        max_spk=args.max_mix_speakers,
        lambda_pit=args.lambda_pit,
        lambda_act=args.lambda_act,
        lambda_cnt=args.lambda_cnt,
        lambda_frm=args.lambda_frm,
        pos_weight=args.pos_weight,
        pit_pos_weight=args.pit_pos_weight,
    ).to(device)

    result = validate(
        model=model,
        loss_fn=loss_fn,
        loader=val_loader,
        device=device,
        max_batches=args.max_batches,
        activity_threshold=0.5,
    )

    print("=" * 100)
    print("STRICT VERIFY RESULT")
    for k, v in result.items():
        print(f"{k}: {v:.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()