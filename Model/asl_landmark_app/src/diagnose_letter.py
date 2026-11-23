"""Diagnose a weak letter from the out-of-fold dump and cached landmarks.

Separates detection-path artifacts, landmark-quality problems and style or
feature-space conflicts. No camera; writes only the `--out` file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
import normalize

# Ladder rungs, as recorded in the `how` column of the cache and pred dump.
FAMILIES = {
    "direct (first try)": ["direct_0.5", "direct_0.3"],
    "upscale+pad": ["upscale_pad_0.3", "flip_upscale_pad_0.3"],
    "flipped": ["flip_0.3"],
}


def load_preds(profile: str, preds: Path | None) -> pd.DataFrame:
    p = preds or (config.REPORTS_DIR / f"oof_{profile}.csv")
    if not p.is_file():
        raise SystemExit(
            f"{p} not found. Train with a dump first:\n"
            f"  python src/train.py --profile {profile} --dump-preds {p}")
    df = pd.read_csv(p)
    need = {"fold", "letter", "predicted", "correct", "how"}
    if not need <= set(df.columns):
        raise SystemExit(f"{p} is missing columns {sorted(need - set(df.columns))}")
    return df


def acc(df: pd.DataFrame) -> str:
    return f"{100 * df['correct'].mean():.1f}% (n={len(df):,})" if len(df) else "-"


def block_confusion(df: pd.DataFrame, letters: list[str]) -> str:
    out = ["## 1. What the letter actually gets called, per held-out dataset",
           "",
           "Out-of-fold predictions, so every row was produced by a model that "
           "never saw that dataset. `top confusions` is the share of the letter's "
           "images that landed on each wrong answer.",
           "",
           "| letter | held-out dataset | images | accuracy | top confusions |",
           "|---|---|---|---|---|"]
    for lab in letters:
        for fold, sub in df[df["letter"] == lab].groupby("fold"):
            wrong = (sub.loc[~sub["correct"], "predicted"].value_counts(normalize=True)
                     .head(3))
            wc = ", ".join(f"{k} {100 * v:.0f}%" for k, v in wrong.items()) or "-"
            out.append(f"| {lab} | {fold} | {len(sub):,} | {acc(sub)} | {wc} |")
    return "\n".join(out) + "\n"


def block_paths(df: pd.DataFrame, letters: list[str]) -> str:
    out = ["## 2. Accuracy split by how the hand was found",
           "",
           "`direct` is the plain first detection (0.5, then the 0.3 retry the "
           "extractor ladder runs before anything fancier). If a letter were "
           "failing because the ladder handed the model worse landmarks, its "
           "`direct` column would be healthy and the ladder columns would "
           "collapse. The reverse means the detection path is not the problem.",
           "",
           "| letter | all | " + " | ".join(FAMILIES) + " |",
           "|" + "---|" * (2 + len(FAMILIES))]
    for lab in letters:
        sub = df[df["letter"] == lab]
        cells = [acc(sub[sub["how"].isin(how)]) for how in FAMILIES.values()]
        out.append(f"| {lab} | {acc(sub)} | " + " | ".join(cells) + " |")
    allrows = [acc(df[df["how"].isin(how)]) for how in FAMILIES.values()]
    out.append(f"| *all letters* | {acc(df)} | " + " | ".join(allrows) + " |")
    return "\n".join(out) + "\n"


def _md_table(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows) + "\n"


def block_shapes(letters: list[str]) -> str:
    """Distance from each letter's mean hand to the nearest OTHER letter's mean,
    inside one dataset (`src/inspect_data.py --nearest`), plus the thumb landmark
    audit (`src/inspect_data.py --thumb-audit`)."""
    out = ["## 3. Is the shape even separable?",
           "",
           "`crowding` is the distance from this letter's mean hand to the closest "
           "other letter's mean, divided by the median distance to any other letter "
           "(units: summed joint distance, a hand is ~1 palm across). Near 0 means "
           "two letters share one region of feature space *within the same "
           "dataset*, which no re-weighting can separate - a different problem from "
           "one dataset disagreeing about how a letter looks. Rows below are the "
           "letters asked about, out of every letter in that dataset.",
           ""]
    near = config.REPORTS_DIR / "nearest_shapes.csv"
    if near.is_file():
        tbl = pd.read_csv(near)
        rows = tbl[tbl["letter"].isin(letters)].sort_values(["dataset", "crowding"])
        out += ["From `python src/inspect_data.py --nearest`:", "", _md_table(rows)]
    else:
        out += [f"`{near.name}` not found - reproduce with "
                "`python src/inspect_data.py --nearest`.", ""]

    out += ["### Is the thumb landmark garbage?",
            "",
            "`in_palm%` is the share of frames where the thumb tip lands inside the "
            "palm polygon (expected for a tucked thumb, so T and M should not be 0); "
            "`thumb_len` is tip-to-wrist in palm units (a real thumb is ~1-1.3, a "
            "hallucinated one wanders); `jitter` is how much the thumb moves between "
            "two frames of the *same* sign (a stable landmark is < 0.4).", ""]
    audit = config.REPORTS_DIR / "thumb_audit.csv"
    if audit.is_file():
        tbl = pd.read_csv(audit)
        rows = tbl[tbl["letter"].isin(letters)]
        out += ["From `python src/inspect_data.py --thumb-audit T,N,M`:", "",
                _md_table(rows)]
    else:
        out += [f"`{audit.name}` not found - reproduce with "
                "`python src/inspect_data.py --thumb-audit T,N,M`.", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--letters", default="T,N,M")
    ap.add_argument("--profile", default="all",
                    help="whose out-of-fold predictions to read (reports/oof_<profile>.csv)")
    ap.add_argument("--preds", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="write this markdown here (default: print only)")
    args = ap.parse_args()

    letters = [x.strip().upper() for x in args.letters.split(",") if x.strip()]
    prof = config.profile(args.profile)
    df = load_preds(prof, args.preds)
    labels = sorted(df["letter"].unique())
    bad = [l for l in letters if l not in labels]
    if bad:
        raise SystemExit(f"no class for {bad}; this model has {labels}")

    head = [f"# Letter diagnosis: {', '.join(letters)} (profile `{prof}`)",
            "",
            f"{len(df):,} out-of-fold predictions over {df['fold'].nunique()} "
            f"held-out datasets, 71-dim features "
            f"(`{'+'.join(normalize.FEATURE_BLOCKS)}`).",
            ""]
    md = "\n".join(head) + "\n" + "\n".join([
        block_confusion(df, letters), block_paths(df, letters),
        block_shapes(letters)])
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"[diagnose_letter] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
