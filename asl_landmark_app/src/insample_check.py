"""In-sample per-letter check for a shipped model.

Answers one narrow question: when the deployed final model is run back over the
exact landmark vectors it was trained on, can it separate the letters that
collapse in the cross-dataset table? Nothing here is an accuracy claim - the
headline is the leave-one-dataset-out number in the profile's report. This
exists because "the model cannot do N" and "the model cannot do N on hands it
has never seen" are different failures with different fixes, and round 1 quoted
the first one from a check that could not be re-run.

Reads the cache and one profile's .pt/.meta.json. Writes nothing, opens no camera.

    python src/insample_check.py
    python src/insample_check.py --letters NMTP --csv
    python src/insample_check.py --profile standard --letters NMTP
    python src/insample_check.py --confuse T,N,M --by-path
    python src/insample_check.py --preds reports/oof_all.csv --confuse T,N,M

`--preds` reads a leave-one-dataset-out prediction dump from
`src/train.py --dump-preds`, which is the honest evaluation, and runs the same
confusion analysis on it. Without it everything below is in-sample. (The two
22 MB dumps that used to sit in `reports/` were pruned in the 2026-11-15 folder
cleanup; `python src/train.py --profile all --dump-preds reports/oof_all.csv`
rebuilds one, and it exits with that command if the file is missing.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

import config
import model as M
import train as T


def confusion_rows(df: pd.DataFrame, letters: list[str], ycol: str,
                   pcol: str, by: str) -> None:
    """For each true letter, what the model answers, split by one column.

    `by` is `dataset` (which signer) or `how` (which detection rung produced the
    landmarks). A letter that fails the same way everywhere is a feature problem;
    one that only fails on the ladder-rescued frames is a detection problem, and
    one that only fails for one dataset is a style conflict.
    """
    for lab in letters:
        sub = df[df[ycol] == lab]
        if not len(sub):
            continue
        acc = 100 * float(sub["correct"].mean())
        print(f"\n  true {lab}  n={len(sub):,}  correct {acc:.1f}%  "
              f"(<- what it is answered as, per {by})")
        for key, s in sub.groupby(by):
            top = s[pcol].value_counts().head(4)
            txt = ", ".join(f"{p} {100 * n / len(s):.0f}%" for p, n in top.items())
            print(f"    {str(key):<22} n={len(s):>6,}  "
                  f"correct {100 * float(s['correct'].mean()):5.1f}%  {txt}")


def path_groups(df: pd.DataFrame) -> pd.Series:
    """Collapse the five detection rungs into the three that mean something."""
    how = df["how"].astype(str)
    return np.where(how.str.startswith("flip"), "flipped",
                    np.where(how == "direct_0.5", "first try (0.5)", "ladder-rescued"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--letters", default="NMTGPKDCQOS",
                    help="letters to print (default: the weak ones plus controls)")
    ap.add_argument("--csv", action="store_true", help="print every letter as CSV")
    ap.add_argument("--all", action="store_true",
                    help="print every letter instead of the selected ones")
    ap.add_argument("--confuse", default="",
                    help="comma-separated letters: show, per dataset and per "
                         "detection path, what each one is answered as")
    ap.add_argument("--by-path", action="store_true",
                    help="accuracy split by detection rung (first try / "
                         "ladder-rescued / flipped) for the --confuse letters")
    ap.add_argument("--preds", default=None, metavar="CSV",
                    help="analyse an out-of-fold prediction dump written by "
                         "`src/train.py --dump-preds` instead of the shipped "
                         "model. Those rows are the honest numbers; everything "
                         "this script produces itself is in-sample.")
    ap.add_argument("--profile", default=None, choices=sorted(config.PROFILES),
                    help="which shipped model to probe (default: "
                         f"{config.PROFILE_DEFAULT}, override with ASL_PROFILE)")
    args = ap.parse_args()

    want = [c.strip().upper() for c in args.confuse.split(",") if c.strip()]
    if args.preds:
        preds_path = Path(args.preds)
        if not preds_path.is_file():
            raise SystemExit(
                f"{preds_path.name} is not in reports/ (the out-of-fold dumps "
                f"were pruned from this folder). Rebuild it with one retrain:\n"
                f"  python src/train.py --profile {config.profile(args.profile)} "
                f"--dump-preds {preds_path}")
        df = pd.read_csv(preds_path, dtype={"session": str})
        df["dataset"] = df["fold"]
        df = df.assign(how=path_groups(df))
        print(f"\nOUT-OF-FOLD predictions from {args.preds}: {len(df):,} images, "
              f"each scored by a model that never saw its dataset. "
              f"Overall accuracy {100 * float(df['correct'].mean()):.1f}%.")
        if want:
            confusion_rows(df, want, "letter", "predicted", "dataset")
            if args.by_path:
                confusion_rows(df, want, "letter", "predicted", "how")
        else:
            print("  (nothing to show: pass --confuse T,N,M)")
        return 0

    prof = config.profile(args.profile)
    meta = json.loads(config.meta_path(prof).read_text(encoding="utf-8"))
    in_dim = meta["model"]["in_dim"]
    if in_dim != config.feature_dim():
        print(f"REFUSING: {config.meta_path(prof).name} expects {in_dim} features, "
              f"the code builds {config.feature_dim()}. Re-train "
              f"(python src/train.py --profile {prof}).")
        return 2
    if list(meta["labels"]) != list(T.LABELS):
        print(f"REFUSING: {prof} was trained on a different label set: "
              f"{','.join(meta['labels'])}")
        return 2
    net = M.load_model(config.model_path(prof, must_exist=True), in_dim,
                       len(meta["labels"]), tuple(meta["model"]["hidden"]),
                       meta["model"]["dropout"])

    data = T.load_dataset()
    feats = T.feats_of(data["lm"])
    pred = np.empty(len(data["y"]), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(feats), 4096):
            x = torch.from_numpy(np.ascontiguousarray(feats[i:i + 4096])).float()
            pred[i:i + 4096] = net(x).argmax(1).numpy()

    pc = T.per_class(data["y"], pred)
    overall = 100 * float((pred == data["y"]).mean())
    print(f"\nIN-SAMPLE accuracy of {config.model_path(prof).name} "
          f"(profile `{prof}`) over the "
          f"{len(data['y']):,} cached hands it trained on "
          f"({in_dim}-dim features, meta seed "
          f"{meta['model'].get('seed', meta.get('seed'))}).")
    print("This is NOT an accuracy result. It separates 'the features cannot "
          "express this letter' from 'the features can, but only for the hands "
          "it was trained on'.\n")
    if args.csv:
        print("letter,n,correct,in_sample_pct")
        for _, r in pc.iterrows():
            print(f"{r['letter']},{int(r['n'])},{int(r['correct'])},"
                  f"{100 * float(r['accuracy']):.1f}")
    else:
        want = list(T.LABELS) if args.all else list(args.letters)
        print(f"{'letter':6s} {'in-sample':>10s} {'n':>8s}")
        for lab in want:
            row = pc.loc[pc["letter"] == lab]
            if len(row):
                print(f"{lab:6s} {100 * float(row['accuracy'].iloc[0]):9.1f}% "
                      f"{int(row['n'].iloc[0]):8,}")
    print(f"{'ALL':6s} {overall:9.1f}% {len(data['y']):8,}")
    if want:
        how = (data["df"]["how"].astype(str).to_numpy() if "how" in data["df"].columns
               else np.full(len(pred), "direct_0.5"))
        frame = pd.DataFrame({
            "letter": data["df"]["letter"].astype(str).to_numpy(),
            "predicted": [T.LABELS[i] for i in pred],
            "correct": pred == data["y"],
            "dataset": data["ds"],
            "how": path_groups(pd.DataFrame({"how": how})),
        })
        print("\n  IN-SAMPLE confusion rows: the model has seen these frames, so "
              "this is not an\n  error rate - it says which shape the network "
              "associates with each letter.")
        confusion_rows(frame, want, "letter", "predicted", "dataset")
        if args.by_path:
            print("\n  ... and the same question asked of the detection rung that "
                  "produced\n  the landmarks.")
            confusion_rows(frame, want, "letter", "predicted", "how")
    print(f"\nRead against {config.profile_report(prof).name}'s pooled "
          "cross-dataset column: a letter that "
          "is high here and low there is a transfer problem (more independent "
          "signers), not a feature or capacity problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
