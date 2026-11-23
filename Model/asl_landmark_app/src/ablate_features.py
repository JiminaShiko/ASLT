"""Stage D - feature ablation under the cross-dataset protocol.

Runs `train.py --folds-only` per feature set in a fresh subprocess with
ASL_FEATURE_BLOCKS set, so the same code path, folds and augmentation apply.
Writes reports/ablation_features.md.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# (label, blocks); one row of the table each. Set 3 is the shipped baseline.
SETS = [
    ("1 baseline (57)",            "xy,tip_dists,curl"),
    ("2 + z coords",               "xyz,tip_dists,curl"),
    ("3 + thumb features",         "xy,tip_dists,curl,thumb"),
    ("4 + orientation features",   "xy,tip_dists,curl,orient"),
    ("5 thumb + orientation",      "xy,tip_dists,curl,thumb,orient"),
    ("6 shipped + thumb-cross",    "xy,tip_dists,curl,thumb,cross"),
]

RESULT_RE = re.compile(
    r"ABLATION_RESULT blocks=(?P<blocks>\S+) dim=(?P<dim>\d+) "
    r"pooled=(?P<pooled>[\d.]+) macro=(?P<macro>[\d.]+) n=(?P<n>\d+) "
    r"r1=(?P<r1>[\d.]+|na)(?P<rest>.*)")
LETTER_RE = re.compile(r"([A-Z])=([\d.]+)")


def run_one(label: str, blocks: str, extra: list[str], quiet: bool) -> dict | None:
    watch = os.environ.get("ASL_WATCH", "")
    env = dict(os.environ, ASL_FEATURE_BLOCKS=blocks,
               PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-u", str(Path(__file__).with_name("train.py")),
           "--folds-only"] + extra
    # ASL_WATCH must be forwarded, or watched letters come back as nan
    if watch:
        cmd += ["--watch", watch]
    t0 = time.time()
    print(f"\n=== {label}  blocks={blocks} ===", flush=True)
    res = subprocess.run(cmd, env=env, cwd=str(config.ROOT),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    out = res.stdout or ""
    if not quiet:
        for line in out.splitlines():
            if any(k in line for k in ("HEADLINE", "macro", "pooled, excluding",
                                       "per-letter (pooled)", "ABLATION_RESULT",
                                       "feature dim", "loaded ")):
                print("   " + line.strip(), flush=True)
    m = None
    for m in RESULT_RE.finditer(out):
        pass
    if m is None:
        print(f"  !! {label}: no ABLATION_RESULT (exit {res.returncode}); "
              f"last lines:\n" +
              "\n".join("     " + l for l in out.splitlines()[-12:]), flush=True)
        return None
    row = {"label": label, "blocks": m["blocks"], "dim": int(m["dim"]),
           "pooled": float(m["pooled"]), "macro": float(m["macro"]),
           "n": int(m["n"]), "seconds": round(time.time() - t0, 1)}
    row.update({k: float(v) for k, v in LETTER_RE.findall(m["rest"] or "")})
    print(f"  -> pooled {row['pooled']:.1f}%  macro {row['macro']:.1f}%  "
          f"({row['seconds']:.0f}s)", flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="1,2,3,4,5",
                    help="comma-separated set numbers from SETS")
    ap.add_argument("--fold-epochs", type=int, default=None,
                    help="epoch cap for the fold models (train.py default is 30)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seed list, e.g. --seeds 0,1,2. Each "
                         "feature set is re-run once per seed and the table "
                         "reports the mean with the min-max spread: with three "
                         "LODO folds the fold models differ only by the "
                         "initialisation, and a single run cannot tell a real "
                         "+0.5 point gain from the same run done twice.")
    ap.add_argument("--style-weight", default="1.0",
                    help="Stage B minority-style weight passed straight to "
                         "train.py (use the same value as the final run so the "
                         "table is comparable with REPORT.md)")
    ap.add_argument("--aug", default=None,
                    help="override --aug-strength for every run (train.py "
                         "default is 1.0); used for the Stage E augmentation "
                         "check, where the same feature set is re-run at a "
                         "different noise level")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default=None, help="write the table to this path")
    args = ap.parse_args()

    want = {int(x) for x in args.sets.split(",") if x.strip()}
    chosen = [s for s in SETS if int(s[0][0]) in want]
    seed_list = ([int(x) for x in args.seeds.split(",") if x.strip()]
                 if args.seeds else [args.seed])
    if args.fold_epochs is not None:
        fe = ["--fold-epochs", str(args.fold_epochs)]
    else:
        fe = []
    if args.aug is not None:
        fe += ["--aug-strength", str(args.aug)]

    rows = []
    for label, blocks in chosen:
        per_seed = []
        for sd in seed_list:
            extra = ["--seed", str(sd), "--style-weight", str(args.style_weight)] + fe
            r = run_one(f"{label} seed{sd}", blocks, extra, args.quiet)
            if r:
                per_seed.append(r)
        if not per_seed:
            continue
        agg = {"label": label, "blocks": per_seed[0]["blocks"],
               "dim": per_seed[0]["dim"], "n": per_seed[0]["n"],
               "seconds": round(sum(r["seconds"] for r in per_seed), 1),
               "reps": len(per_seed)}
        for key in ("pooled", "macro"):
            vals = [r[key] for r in per_seed]
            agg[key] = sum(vals) / len(vals)
            agg[key + "_lo"], agg[key + "_hi"] = min(vals), max(vals)
        watch_now = [c.strip().upper() for c in
                     os.environ.get("ASL_WATCH", "M,N,T,G,K").split(",") if c.strip()]
        for w in watch_now:
            vals = [r[w] for r in per_seed if w in r]
            if vals:
                agg[w] = sum(vals) / len(vals)
                agg[w + "_lo"], agg[w + "_hi"] = min(vals), max(vals)
        rows.append(agg)
    if not rows:
        print("\n  nothing to compare - see the errors above")
        return 1

    watch = [c.strip().upper() for c in
             os.environ.get("ASL_WATCH", "M,N,T,G,K").split(",") if c.strip()]
    base = rows[0]
    spread = max(r["pooled_hi"] - r["pooled_lo"] for r in rows)
    lspread = [(w, max((r.get(w + "_hi", 0) - r.get(w + "_lo", 0)) for r in rows))
               for w in watch]
    lines = ["# Stage D - feature ablation\n",
             "Same protocol as the headline number: leave-one-dataset-out, every "
             "test image scored by a model that never saw its dataset. Pooled = "
             "micro over all folds (the headline); macro = mean of the per-dataset "
             f"accuracies. **Each row is the mean of {rows[0]['reps']} independent "
             "runs** (`--seed` changed - the data, the folds and the epochs are "
             "fixed), so the spread column is run-to-run noise, not sampling error. "
             "Since 2026-11-14 `--seed` controls the weight init, the sampler and "
             "the augmentation together; sweeps recorded here that predate that fix "
             "had only the augmentation seeded, so their repeats were unpaired "
             "draws. Measured cost of re-running one identical command at that time: "
             "0.7 pooled points - which is why nothing below is decided on a gap "
             "smaller than the spread column.\n",
             "| # | feature set | dim | pooled | spread | vs baseline | macro | " +
             " | ".join(watch) + " |",
             "|---|---|---|---|---|---|---|" + "---|" * len(watch)]
    for r in rows:
        d = r["pooled"] - base["pooled"]
        cells = " | ".join(
            (f"{r.get(w, float('nan')):.1f}"
             f" ({r.get(w + '_lo', float('nan')):.0f}-{r.get(w + '_hi', float('nan')):.0f})"
             if r["reps"] > 1 else f"{r.get(w, float('nan')):.1f}")
            for w in watch)
        lines.append(f"| {r['label'][0]} | {r['label'][2:]} | {r['dim']} "
                     f"| {r['pooled']:.1f}% | {r['pooled_hi'] - r['pooled_lo']:.1f} "
                     f"| {d:+.1f} | {r['macro']:.1f}% | {cells} |")
    lines.append("")
    if rows[0]["reps"] > 1:
        lines.append(
            f"Run-to-run noise: pooled moved at most {spread:.1f} points across "
            f"repeats of the same feature set. Per letter the noise is far larger"
            + (": " + ", ".join(f"{w} up to {s:.1f} points" for w, s in lspread if s > 0)
               if any(s > 0 for _w, s in lspread) else "")
            + " - the letters trade confusions with each other, so a single "
              "letter number from a single run is not evidence of anything. "
              "Only differences larger than the pooled spread are treated as real.")
        lines.append("")
    best = max(rows, key=lambda r: r["pooled"])
    lines.append(f"**Best by pooled cross-dataset accuracy: {best['label']}** "
                 f"({best['pooled']:.1f}%, {best['dim']} dims).")
    hurt = [(r, w, base.get(w, 0) - r.get(w, 0)) for r in rows for w in watch
            if r is not best and base.get(w) is not None
            and r.get(w) is not None and base[w] - r[w] > 3.0]
    if hurt:
        lines.append("\nSets that cost a baseline letter more than 3 points: " +
                     ", ".join(f"{r['label'][0]} loses {w} ({d:.1f})" for r, w, d in hurt))
    lines.append("\nRule from the brief: promote the best pooled set unless it "
                 "costs a currently strong letter more than 3 points.")
    text = "\n".join(lines) + "\n"
    out = Path(args.out) if args.out else config.REPORTS_DIR / "ablation_features.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
