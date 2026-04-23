import json
from dataclasses import asdict
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from speaker_verification.dataset.manifest_builder import BuildCfg, build_online_dataset


def main():
    cfg = BuildCfg()
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    output_root = build_online_dataset(cfg, project_root=PROJECT_ROOT)
    print(f"[DONE] online manifests saved to: {output_root}")


if __name__ == "__main__":
    main()
