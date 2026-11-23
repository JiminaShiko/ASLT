"""One resumable command from an empty cache to a finished report (extract -> train -> onnx -> report).

Every step checks what is on disk and skips itself; steps run as subprocesses
of this interpreter. See README for flags.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, features  # noqa: E402

STEPS = ["extract", "train", "onnx", "report"]
PYTHON = sys.executable


def _run(desc: str, cmd: list[str], dry: bool = False) -> float:
    print(f"\n{'=' * 70}\n{desc}\n  $ cd \"{config.ROOT}\" && "
          f"{' '.join(cmd)}\n{'=' * 70}", flush=True)
    if dry:
        return 0.0
    t0 = time.time()
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    rc = subprocess.call(cmd, cwd=str(config.ROOT), env=env)
    dt = time.time() - t0
    if rc != 0:
        print(f"\nSTEP FAILED ({desc}), exit {rc}, after {dt:.0f} s.\n"
              "Fix the cause and run this script again - the steps that "
              "already finished will be skipped.", file=sys.stderr)
        sys.exit(rc)
    return dt


def need_extract(workers: int, splits: tuple[str, ...]) -> tuple[bool, str]:
    """Run extract when any needed video is not cached."""
    from src import extract as E
    from src import vocab as V
    man = E.read_manifest()
    try:
        need = V.videos_to_process(splits)
    except Exception as exc:                       # noqa: BLE001
        return True, f"the video list cannot be built: {str(exc)[:120]}"
    done = set(man[man["status"].astype(str) == "ok"]["video"].astype(str))
    missing = int(len(set(need["video"].astype(str)) - done))
    cmd = [PYTHON, "-m", "src.extract", "--splits", ",".join(splits),
           "--workers", str(workers)]
    if missing == 0:
        return False, f"all {len(need)} videos are already extracted"
    secs = missing * 2.0 / max(workers, 1)
    return True, (f"{missing} of {len(need)} videos to extract. Measured 2.0 s "
                  f"per video in one process, so {missing * 2.0 / 60:.0f} min "
                  f"single-process and about {secs / 60:.0f} min if {workers} "
                  "workers scale perfectly (MediaPipe already uses several "
                  "cores, so expect somewhere in between)"
                  ) + " -> " + " ".join(cmd[1:])


def weights_are_stale() -> str | None:
    """'v2/170' when meta.json describes a different feature vector, else None.

    Missing weights are not stale - that is the first-train case.
    """
    if not config.META_JSON.is_file():
        return None
    try:
        m = json.loads(config.META_JSON.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return None
    dim = m.get("feature_dim")
    if dim is None:  # older meta: ask the arch instead
        dim = (m.get("arch") or {}).get("in_dim")
    ver = m.get("feature_version")
    if dim == features.FEATURE_DIM and ver == features.FEATURE_VERSION:
        return None
    return f"v{ver}/{dim}"


def need_train(epochs: int, extra: list[str]) -> tuple[bool, str]:
    cmd = [PYTHON, "-m", "src.train", "--epochs", str(epochs)] + extra
    ck = config.CKPT_DIR / "train_state.pt"
    # an old-width checkpoint cannot resume, and old weights make live say "uncertain" forever
    stale = weights_are_stale()
    if stale is not None:
        return True, (f"the weights on disk are {stale}, this code builds "
                      f"v{features.FEATURE_VERSION}/{features.FEATURE_DIM} - "
                      "retraining from scratch (the landmark cache is reused, "
                      "nothing is re-extracted) -> "
                      + " ".join(cmd[1:] + ["--fresh"]))
    if not config.FOR_RELEASE.is_file() and not ck.is_file():
        return True, "no weights and no checkpoint -> " + " ".join(cmd[1:])
    if ck.is_file():
        try:
            obj = torch_header(ck)
            ep = int(obj.get("epoch") or 0)
            if ep >= epochs:
                return False, f"checkpoint is at epoch {ep}, which is the " \
                              f"requested {epochs}"
            return True, f"resuming from epoch {ep} -> " + " ".join(cmd[1:])
        except Exception as exc:                   # noqa: BLE001
            return True, f"checkpoint unreadable ({str(exc)[:60]}) -> " + \
                " ".join(cmd[1:])
    return False, ("training already finished (weights on disk and no resume "
                   "state - src/train.py clears it only on a completed run). "
                   "To train again from scratch: python run_all.py --only "
                   "train --fresh")


def torch_header(path: Path) -> dict:
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj if isinstance(obj, dict) else {}


def need_onnx() -> tuple[bool, str]:
    cmd = [PYTHON, "-m", "src.export_onnx"]
    if not config.FOR_RELEASE.is_file():
        return False, "no trained weights to export"
    if not config.ONNX_PATH.is_file():
        return True, "no ONNX graph yet -> " + " ".join(cmd[1:])
    if config.ONNX_PATH.stat().st_mtime < config.FOR_RELEASE.stat().st_mtime:
        return True, "the weights are newer than the graph -> " + \
            " ".join(cmd[1:])
    return False, "the ONNX graph is already up to date with the weights"


def need_report() -> tuple[bool, str]:
    cmd = [PYTHON, "check_results.py", "--write"]
    md = ROOT / "REPORT_WORDS.md"
    if not (config.REPORTS_DIR / "words_summary.json").is_file():
        return False, "nothing to report on yet"
    if not md.is_file():
        return True, "writing REPORT_WORDS.md -> " + " ".join(cmd[1:])
    if md.stat().st_mtime < (config.REPORTS_DIR / "words_summary.json").stat().st_mtime:
        return True, "the report is older than the run -> " + " ".join(cmd[1:])
    return False, "REPORT_WORDS.md is newer than the last training run"


def plan(args) -> list[tuple[str, list[str], bool, str]]:
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    out = []
    want = set(args.only or STEPS)
    order = STEPS[STEPS.index(args.from_step or "extract"):] if (
        args.from_step) else STEPS
    want = [s for s in order if s in want]
    for step in want:
        if step == "extract":
            run, why = need_extract(args.workers, splits)
            cmd = [PYTHON, "-m", "src.extract", "--splits",
                   args.splits, "--workers", str(args.workers)]
        elif step == "train":
            run, why = need_train(args.epochs, args.train_arg)
            cmd = [PYTHON, "-m", "src.train", "--epochs", str(args.epochs)] + \
                args.train_arg
            if weights_are_stale() is not None:
                # never resume an old-width optimiser state into a new net
                cmd = [c for c in cmd if c != "--fresh"] + ["--fresh"]
        elif step == "onnx":
            run, why = need_onnx()
            cmd = [PYTHON, "-m", "src.export_onnx"]
        else:
            run, why = need_report()
            cmd = [PYTHON, "check_results.py", "--write"]
        out.append((step, cmd, run, why))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the whole pipeline, "
                                             "resumably.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--only", nargs="+", choices=STEPS)
    ap.add_argument("--from", dest="from_step", choices=STEPS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="train: ignore the checkpoint")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--blocks", default=None)
    a = ap.parse_args(argv)

    if a.status:
        import status as S
        st = S.collect()
        st["next"] = S.next_command(st)
        S.print_human(st)
        return 0

    if not config.dataset_available():
        print(f"The dataset is not at\n  {config.DATASET_DIR}\n"
              "It is read-only and never downloaded by this script. See the "
              "README, step 1.", file=sys.stderr)
        return 2

    extra = []
    if a.fresh:
        extra.append("--fresh")
    if a.limit:
        extra += ["--limit", str(a.limit)]
    if a.blocks:
        extra += ["--blocks", a.blocks]
    a.train_arg = extra

    print(__doc__.split("\n\n")[0])
    print(f"\nfolder   : {config.ROOT}")
    print(f"dataset  : {config.DATASET_DIR} (read only)")
    print(f"free disk: {config.free_bytes(config.ROOT) / 1e9:.1f} GB on this "
          f"drive")
    print(f"steps    : {' -> '.join(s for s, _, _, _ in plan(a))}")
    print("\nPlan:")
    for step, cmd, run, why in plan(a):
        print(f"  {step:<8s} {'RUN  ' if run else 'SKIP '} {why}")
    timings: dict[str, float] = {}
    for step, cmd, run, why in plan(a):
        if not run:
            continue
        label = {"extract": "extract landmarks", "train": "train the word "
                 "recognizer", "onnx": "export ONNX and prove parity",
                 "report": "write the results report"}[step]
        print(f"\n[{step}] {why}")
        timings[step] = _run(label, cmd, a.dry_run)
    if not timings:
        print("\nEverything is already done. Nothing was re-run.")
    print(f"\n{'-' * 70}\nsteps run: "
          f"{json.dumps({k: round(v, 1) for k, v in timings.items()})}")
    print("next: python check_results.py    (and the live demo in the README)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
