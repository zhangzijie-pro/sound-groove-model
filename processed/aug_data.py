import argparse
import json
from dataclasses import asdict

from build_processed_dataset import BuildCfg, build_augmented_cn


def parse_args() -> BuildCfg:
    defaults = BuildCfg()
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for CN-Celeb feature augmentation. "
        "The implementation now lives in processed/build_processed_dataset.py."
    )
    parser.add_argument("--cn_out_dir", type=str, default=defaults.cn_out_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--incremental", dest="incremental", action="store_true")
    parser.add_argument("--no-incremental", dest="incremental", action="store_false")
    parser.set_defaults(incremental=defaults.incremental)
    parser.add_argument("--backup_manifest", dest="backup_manifest", action="store_true")
    parser.add_argument("--no-backup_manifest", dest="backup_manifest", action="store_false")
    parser.set_defaults(backup_manifest=defaults.backup_manifest)
    parser.add_argument("--backup_dir_name", type=str, default=defaults.backup_dir_name)
    parser.add_argument("--num_aug_per_utt_train", type=int, default=defaults.num_aug_per_utt_train)
    parser.add_argument("--num_aug_per_utt_val", type=int, default=defaults.num_aug_per_utt_val)
    parser.add_argument("--aug_subdir", type=str, default=defaults.aug_subdir)
    parser.add_argument("--augmentation_version", type=str, default=defaults.augmentation_version)
    args = parser.parse_args()

    cfg = BuildCfg(
        stage="augment_cn",
        cn_out_dir=args.cn_out_dir,
        seed=args.seed,
        incremental=args.incremental,
        backup_manifest=args.backup_manifest,
        backup_dir_name=args.backup_dir_name,
        num_aug_per_utt_train=args.num_aug_per_utt_train,
        num_aug_per_utt_val=args.num_aug_per_utt_val,
        aug_subdir=args.aug_subdir,
        augmentation_version=args.augmentation_version,
    )
    return cfg


def main():
    cfg = parse_args()
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    build_augmented_cn(cfg)
    print("[AUG CN DONE]")


if __name__ == "__main__":
    main()
