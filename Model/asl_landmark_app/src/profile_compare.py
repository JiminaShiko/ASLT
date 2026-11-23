"""Compare model profiles from the leave-one-dataset-out CSVs that
`src/train.py --dump-preds` writes; re-trains nothing, paired by image."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DEFAULTS = {"all": "reports/oof_all.csv", "standard": "reports/oof_standard.csv"}
WEAK = ["M", "N", "P", "T", "C", "D", "G", "Q"]


def load(specs: dict[str, str]) -> dict[str, pd.DataFrame]:
    out = {}
    for prof, path in specs.items():
        p = Path(path)
        if not p.is_absolute():
            p = config.ROOT / p
        if not p.is_file():
            print(f"  [skip] {prof}: {p} does not exist - train it with "
                  f"`python src/train.py --profile {prof} --dump-preds {path}`")
            continue
        df = pd.read_csv(p)
        need = {"letter", "predicted", "correct", "fold"}
        if not need.issubset(df.columns):
            raise SystemExit(f"{p} is missing columns {sorted(need - set(df.columns))}; "
                             "re-dump it with --dump-preds")
        out[prof] = df
    return out


def per_letter(df: pd.DataFrame, letters: list[str]) -> dict[str, float]:
    sub = df[df["letter"].isin(letters)]
    return {l: 100 * float(g["correct"].mean()) if len(g) else float("nan")
            for l, g in sub.groupby("letter")}


def table(profs: dict[str, pd.DataFrame], letters: list[str]) -> None:
    names = list(profs)
    print("\n| measure | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    rows = []
    rows.append(("held-out images (pooled cross-dataset)",
                 [f"{len(p):,}" for p in profs.values()]))
    rows.append(("**pooled accuracy (headline)**",
                 [f"**{100 * p['correct'].mean():.1f}%**" for p in profs.values()]))
    for did in sorted(profs[names[0]]["fold"].unique()):
        rows.append((f"  {did}",
                     [f"{100 * p[p['fold'] == did]['correct'].mean():.1f}%"
                      f" ({(p['fold'] == did).sum():,})" for p in profs.values()]))
    acc = {n: per_letter(p, letters) for n, p in profs.items()}
    for lab in letters:
        rows.append((f"letter {lab}", [f"{acc[n][lab]:.1f}%" for n in names]))
    for row, cells in rows:
        print(f"| {row} | " + " | ".join(cells) + " |")

    print("\nPer letter x held-out dataset - the cells `standard` down-weights "
          "are the ones that can move here:\n")
    print("| letter | dataset | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 2) + "|")
    for lab in ["M", "N", "P", "T"]:
        for did in sorted(profs[names[0]]["fold"].unique()):
            cells = []
            for p in profs.values():
                s = p[(p["letter"] == lab) & (p["fold"] == did)]
                cells.append(f"{100 * s['correct'].mean():.1f}% (n={len(s):,})"
                             if len(s) else "-")
            print(f"| {lab} | {did} | " + " | ".join(cells) + " |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--letters", default=",".join(WEAK),
                    help="per-letter rows for the headline table")
    ap.add_argument("--preds", action="append", default=[], metavar="PROF=CSV",
                    help="override the profile/prediction-file pairs "
                         "(repeatable); default all + standard")
    args = ap.parse_args()
    specs = {}
    for kv in args.preds:
        prof, _, path = kv.partition("=")
        specs[prof.strip()] = path.strip()
    if not specs:
        specs = dict(DEFAULTS)
    profs = load(specs)
    if len(profs) < 2:
        print("  need two profiles to compare; printing what is available anyway")
    if not profs:
        return 1
    print("\n== Profile comparison (leave-one-dataset-out, paired by image) ==")
    for n, p in profs.items():
        note = config.PROFILES.get(n, {}).get("note", "")
        print(f"  {n:9}: {len(p):,} held-out images - {note}")
    table(profs, [x.strip().upper() for x in args.letters.split(",") if x.strip()])
    print("\n**Read this as two different questions, not a league table.** In "
          "`standard` the massey M/N/P cells are removed from training, so in the "
          "folds where massey is the *test* set the model was never shown a massey "
          "M, N or P at all - those rows are expected to look worse, and that is "
          "the design, not a regression. The rows that decide whether the profile "
          "is worth anything are the SAME letters in the OTHER datasets, which "
          "still have their own M/N/P in training. Pooled accuracy mixes the two "
          "effects, so it cannot be the only number quoted here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
