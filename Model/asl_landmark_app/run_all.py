"""One command: download -> extract -> train -> report -> export.

Every stage is idempotent. The live app is NOT part of the pipeline.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402  (only for the profile table, so one name drives every stage)


def run_stage(name: str, script: str, extra: list[str] = []) -> float:
    cmd = [sys.executable, "-u", str(ROOT / script), *extra]
    bar = "=" * 74
    print(f"\n{bar}\n>>> {name}   ({' '.join(cmd[2:])})\n{bar}")
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"\n<<< {name}: {status} in {dt:.0f}s")
    if rc != 0:
        raise SystemExit(f"Stage '{name}' failed - stopping. Fix it and re-run with "
                         f"the --skip flags for the stages already done.")
    return dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--no-folds", action="store_true", help="skip cross-dataset folds")
    ap.add_argument("--quick", action="store_true",
                    help="max 400 images/class, 10 epochs - end-to-end smoke test")
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (half the cores)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--fold-epochs", type=int, default=30,
                    help="epoch cap for the ~15 evaluation models")
    ap.add_argument("--live", action="store_true", help="start src/live.py when finished")
    ap.add_argument("--profile", default=None, metavar="NAME",
                    help="which model profile to train and export "
                         f"({', '.join(sorted(config.PROFILES))}; default "
                         f"{config.PROFILE_DEFAULT}). Train the other one with "
                         "`python src/train.py --profile standard` afterwards.")
    args = ap.parse_args()
    prof = config.profile(args.profile)

    workers = args.workers or max(2, (os.cpu_count() or 4) // 2)

    max_per_class = 400 if args.quick else args.max_per_class
    epochs = 10 if args.quick else args.epochs
    fold_epochs = 4 if args.quick else args.fold_epochs

    t_total = time.time()
    done = []
    if not args.skip_download:
        run_stage("Step 1 download datasets", "src/download_data.py", [])
        done.append("download")
    if not args.skip_extract:
        extra = ["--max-per-class", str(max_per_class), "--workers", str(workers)]
        if args.force_extract:
            extra.append("--force")
        run_stage("Step 2 landmark extraction", "src/extract_landmarks.py", extra)
        done.append("extract")
    if not args.skip_train:
        extra = ["--epochs", str(epochs), "--fold-epochs", str(fold_epochs),
                 "--profile", prof]
        if args.quick:
            extra.append("--quick")
        if args.no_folds:
            extra.append("--no-folds")
        run_stage("Step 4 train + report", "src/train.py", extra)
        done.append("train")
    if not args.skip_export:
        run_stage("Step 6 ONNX export + parity", "src/export.py",
                  ["--samples", "500" if args.quick else "3000", "--profile", prof])
        done.append("export")

    print("\n" + "=" * 74)
    print(f"PIPELINE COMPLETE ({', '.join(done) or 'nothing - all stages skipped'}) "
          f"in {time.time() - t_total:.0f}s")
    print("=" * 74)
    print(f"  profile   : {prof} - {config.PROFILES[prof]['note']}")
    print(f"  model     : {config.model_path(prof)}")
    print(f"  meta      : {config.meta_path(prof)}")
    print(f"  report    : {config.profile_report(prof)}")
    print(f"  onnx      : {config.onnx_path(prof)}")
    print(f"\n  live app  : \"{sys.executable}\" \"{ROOT / 'src' / 'live.py'}\"")

    if args.live:
        run_stage("Step 5 live app", "src/live.py", [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
