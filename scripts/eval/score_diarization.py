import os
import re
import json
import argparse
import subprocess
import sys

def parse_metric(text: str, name: str):
    """
    Try to parse lines like:
      OVERALL DER: 12.34
      DER | 12.34
      JER | 45.67
    """
    patterns = [
        rf"{name}\s*[:|]\s*([0-9]+(?:\.[0-9]+)?)",
        rf"OVERALL\s+{name}\s*[:|]?\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dscore_repo", type=str, required=True, help="Path to cloned dscore repo")
    parser.add_argument("--ref_rttm", type=str, required=True)
    parser.add_argument("--sys_rttm", type=str, required=True)
    parser.add_argument("--uem", type=str, default=None)
    parser.add_argument("--collar", type=float, default=0.0)
    parser.add_argument("--ignore_overlap", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    score_py = os.path.join(args.dscore_repo, "score.py")
    if not os.path.isfile(score_py):
        raise FileNotFoundError(f"Cannot find dscore score.py at: {score_py}")

    cmd = [
        sys.executable,
        score_py,
        "-r", args.ref_rttm,
        "-s", args.sys_rttm,
    ]

    if args.uem is not None:
        cmd += ["-u", args.uem]

    # dscore uses md-eval style options underneath; different versions may vary.
    # We keep these optional and lightweight.
    if args.collar is not None:
        cmd += ["--collar", str(args.collar)]

    if args.ignore_overlap:
        cmd += ["--ignore_overlaps"]

    print("[INFO] Running command:")
    print(" ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("=" * 80)
    print(proc.stdout)
    if proc.stderr.strip():
        print("[STDERR]")
        print(proc.stderr)
    print("=" * 80)

    if proc.returncode != 0:
        raise RuntimeError(f"dscore failed with return code {proc.returncode}")

    der = parse_metric(proc.stdout, "DER")
    jer = parse_metric(proc.stdout, "JER")

    result = {
        "ref_rttm": args.ref_rttm,
        "sys_rttm": args.sys_rttm,
        "uem": args.uem,
        "collar": args.collar,
        "ignore_overlap": args.ignore_overlap,
        "DER": der,
        "JER": jer,
        "raw_stdout": proc.stdout,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True) if os.path.dirname(args.output_json) else None
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved score json to: {args.output_json}")


if __name__ == "__main__":
    main()

"""
python score_diarization.py --dscore_repo scripts/dscore --ref_rttm eval_staticmix/val_ref.rttm --sys_rttm eval_staticmix/sys.rttm --uem eval_staticmix/val.uem --collar 0.0 --output_json eval_staticmix/score.json
"""