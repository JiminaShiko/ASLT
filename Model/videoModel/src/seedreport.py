"""
The seed-sweep table, kept free of torch on purpose.

train.py writes reports/seed_runs.csv (one row per seed per config) and this
module turns it into the two human files:

  reports/seed_runs.md   every run, mean +- spread per metric
  compare_table()        the baseline-vs-new block for REPORT_WORDS.md

check_results.py imports this instead of src.train, so reading the results
never pulls in torch, the model or the dataset. That is the whole reason it is
a separate file.

Spread is the half-range over the seeds of one config. It is printed next to
every mean because it is the only way to read them: a gain smaller than the
baseline's own spread is not a gain. Configs are compared on the VAL columns;
the TEST columns appear because they are what gets reported, not because they
choose anything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import config

SEED_RUN_COLUMNS = ["run_tag", "reg", "feature_version", "seed", "promoted",
                    "candidate", "label_set_id", "evaluation_id", "labels",
                    "epochs_ran", "best_ep", "sec_per_epoch", "best_val_top1"]


def identity(value) -> str:
    """Stable identity; missing historical identities must never be guessed."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()[:20]


def candidate_name() -> str | None:
    """Derive the output label from the paths selected by config.use_candidate."""
    try:
        parts = config.REPORTS_DIR.resolve().relative_to(
            (config.ROOT / "candidates").resolve()).parts
    except ValueError:
        return None
    return parts[0] if len(parts) >= 2 else None


def comparable(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Only explicit, uniform class AND evaluation identities are comparable."""
    for key in ("label_set_id", "evaluation_id"):
        if key not in a or key not in b or a.empty or b.empty:
            return False
        av, bv = a[key], b[key]
        if av.isna().any() or bv.isna().any():
            return False
        aa, bb = set(av.astype(str)), set(bv.astype(str))
        if len(aa) != 1 or aa != bb or aa & {"", "unknown", "None", "nan"}:
            return False
    return True

# The columns worth reading. seed_runs.csv keeps everything (including the n
# behind each number); the table shows only these, because a 40-row table is
# how a report stops being read.
SWEEP_METRICS = ["val_words_top1", "val_top1", "val_typed_share",
                 "val_typed_precision", "val_other_slip", "val_rest_slip",
                 "test_words_top1", "test_words_top5", "test_top1",
                 "test_macro_words", "test_typed_share",
                 "test_typed_precision", "test_typed_precision_words",
                 "test_other_slip", "test_rest_slip", "test_other_as_word",
                 "ps_test_words_top1", "ps_test_top1", "ps_test_typed_share",
                 "ps_test_typed_precision_words"]

# the fair-comparison table: exactly the columns the plan asked for
FAIR_COLUMNS = [
    ("val_top1", "val top-1 (DECIDES)"),
    ("test_words_top1", "words-only test top-1"),
    ("test_words_top5", "words-only test top-5"),
    ("test_typed_share", "typed share at the gate"),
    ("test_typed_precision_words", "typed precision, real words only"),
    ("test_typed_precision", "typed precision, everything typed"),
    ("test_other_slip", "OTHER slip"),
    ("test_rest_slip", "REST slip"),
    ("test_swap_good_thanks", "good/thanks swapped"),
    ("test_swap_happy_please", "happy/please swapped"),
]


def metric_columns(g: pd.DataFrame) -> list[str]:
    """The decision columns present in this frame, in reading order."""
    fixed = [c for c in SWEEP_METRICS if c in g.columns]
    groups = sorted(c for c in g.columns
                    if any(f"_{name}_" in c for name in ("core20", "weak6", "new_words")))
    commits = sorted(c for c in g.columns
                     if c.endswith(("correct_commit_recall", "n_hedge_refused")))
    swaps = sorted(c for c in g.columns if "_swap_" in c)
    return list(dict.fromkeys(fixed + groups + commits + swaps))


def seed_row_path() -> Path:
    return config.REPORTS_DIR / "seed_runs.csv"


def append_seed_rows(rows: list[dict], reports_dir: Path | None = None) -> pd.DataFrame:
    """One growing table per output scope (scratch smoke tables stay separate)."""
    reports_dir = reports_dir or config.REPORTS_DIR
    path = reports_dir / "seed_runs.csv"
    df = pd.DataFrame(rows)
    old = None
    if path.is_file():
        old = pd.read_csv(path)
        keep = old[old["run_tag"] != df["run_tag"].iloc[0]] if len(df) else old
        df = pd.concat([keep, df], ignore_index=True) if keep is not None else df
    cols = list(SEED_RUN_COLUMNS) + [c for c in df.columns
                                     if c not in SEED_RUN_COLUMNS]
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    write_seed_table(df, reports_dir=reports_dir)
    return df


def mean_spread(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """'0.583 +-0.021 (0.571..0.601)' per column: mean, half-range, extremes."""
    lines = []
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if not len(v):
            lines.append(f"    {c:26s}  -")
            continue
        spread = (float(v.max()) - float(v.min())) / 2.0
        lines.append(f"    {c:26s}  {float(v.mean()):.3f} "
                     f"+-{spread:.3f}   ({float(v.min()):.3f}.."
                     f"{float(v.max()):.3f})  n={len(v)}")
    return lines


def write_seed_table(df: pd.DataFrame, reports_dir: Path | None = None) -> Path:
    """The table check_results.py pastes into the report, in the active scope."""
    p = (reports_dir or config.REPORTS_DIR) / "seed_runs.md"
    out = ["# Seed runs (mean +- spread over seeds of the same config)", "",
           "Decided by the VAL columns. The TEST columns are printed because",
           "you asked for them, not because they choose anything: picking a",
           "config on test would make the test number meaningless.", ""]
    if candidate_name():
        out += [f"Candidate: {candidate_name()} (offline evaluation, not deployed).", ""]
    out += ["Core20 retention and new-word totals are separate. Different or unknown",
            "label/evaluation identities are not baseline improvements.", ""]
    for tag, grp in df.groupby("run_tag", sort=False):
        out.append(f"## {tag}   ({len(grp)} seed(s), reg={grp['reg'].iloc[0]}, "
                   f"feature v{grp['feature_version'].iloc[0]})")
        for key in ("label_set_id", "evaluation_id"):
            values = ", ".join(grp[key].fillna("unknown").astype(str).unique()) if key in grp else "unknown"
            out.append(f"{key}: {values}")
        out += mean_spread(grp, metric_columns(grp))
        out.append("")
    p.write_text("\n".join(out), encoding="utf-8")
    return p


def compare_table(path: Path | None = None) -> str:
    """The baseline-vs-new block for REPORT_WORDS.md, from seed_runs.csv."""
    p = path or seed_row_path()
    if not p.is_file():
        return ("(no seed runs measured yet - `python -m src.train --seeds 3 "
                "--tag baseline` first)")
    df = pd.read_csv(p)
    lines: list[str] = []
    tags = list(dict.fromkeys(df["run_tag"]))
    for tag in tags:
        g = df[df["run_tag"] == tag]
        lines.append(f"**{tag}** - {len(g)} seed(s), reg={g['reg'].iloc[0]}, "
                     f"feature v{int(g['feature_version'].iloc[0])}, "
                     f"promoted seed "
                     f"{int(g.loc[g['promoted'] == 1, 'seed'].iloc[0]) if (g['promoted'] == 1).any() else 'none'}")
        lines.append("")
        for key in ("label_set_id", "evaluation_id"):
            values = ", ".join(g[key].fillna("unknown").astype(str).unique()) if key in g else "unknown"
            lines.append(f"{key}: {values}")
        lines.append("")
        cols = metric_columns(g)
        head = ["metric", "mean", "spread (half-range)", "min..max"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        for c in cols:
            v = pd.to_numeric(g[c], errors="coerce").dropna()
            if not len(v):
                continue
            lines.append(f"| {c} | {float(v.mean()):.3f} | "
                         f"{(float(v.max()) - float(v.min())) / 2:.3f} | "
                         f"{float(v.min()):.3f}..{float(v.max()):.3f} |")
        lines.append("")
    if len(tags) > 1:
        base = df[df["run_tag"] == tags[0]]
        bkey = "val_top1"
        bv = pd.to_numeric(base.get(bkey, pd.Series(dtype=float)), errors="coerce").dropna()
        bspread = (float(bv.max()) - float(bv.min())) / 2 if len(bv) > 1 else 0.0
        lines.append(f"## Does it beat the baseline? (decided on {bkey})")
        lines.append("")
        lines.append("Adoption comparisons require matching explicit label-set and evaluation "
                     "identities. Historical rows without them are not comparable. "
                     "Expanded totals are not improvements over a smaller vocabulary.")
        lines.append("")
        lines.append("| run | val top-1 | vs baseline | bigger than the spread? |")
        lines.append("|---|---|---|---|")
        for tag in tags:
            g = df[df["run_tag"] == tag]
            gv = pd.to_numeric(g.get(bkey, pd.Series(dtype=float)), errors="coerce").dropna()
            if not len(gv):
                continue
            if not comparable(base, g) or not len(bv):
                lines.append(f"| {tag} | {float(gv.mean()):.3f} | n/a | "
                             "comparison suppressed: different or unknown identities |")
                continue
            d = float(gv.mean()) - float(bv.mean())
            verdict = ("baseline" if tag == tags[0] else
                       ("YES - adopt" if d > bspread else
                        ("no - worse than the baseline" if d < -bspread else
                         "no - inside the noise")))
            lines.append(f"| {tag} | {float(gv.mean()):.3f} | {d:+.3f} | "
                         f"{verdict} |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --fair: feature v3 vs the old v2 vector, same seeds, decided on VAL
# ---------------------------------------------------------------------------
def _stat(g: pd.DataFrame, col: str) -> tuple[str, str]:
    """'mean' and 'spread' strings for one column of one run."""
    if col not in g.columns:
        return "-", "-"
    v = pd.to_numeric(g[col], errors="coerce").dropna()
    if not len(v):
        return "-", "-"
    sp = (float(v.max()) - float(v.min())) / 2.0 if len(v) > 1 else 0.0
    return f"{float(v.mean()):.3f}", f"+-{sp:.3f}"


def fair_decision(df: pd.DataFrame, tag3: str, tag2: str) -> dict:
    """The VAL verdict: adopt v3 only above the pooled seed spread.

    The spread is the half-range over the POOLED val top-1 of both runs'
    seeds - the same seeds run twice, so together they are the most
    conservative estimate of run-to-run noise this comparison has. A diff
    inside it keeps the simpler v2 vector, per the plan.
    """
    g3, g2 = df[df["run_tag"] == tag3], df[df["run_tag"] == tag2]
    v3 = pd.to_numeric(g3["val_top1"], errors="coerce").dropna()
    v2 = pd.to_numeric(g2["val_top1"], errors="coerce").dropna()
    pooled = pd.concat([v3, v2])
    spread = (((float(pooled.max()) - float(pooled.min())) / 2.0)
              if len(pooled) > 1 else 0.0)
    mean3 = float(v3.mean()) if len(v3) else float("nan")
    mean2 = float(v2.mean()) if len(v2) else float("nan")
    diff = mean3 - mean2
    same = comparable(g3, g2) and bool(len(v3) and len(v2))
    return {"mean3": mean3, "mean2": mean2, "diff": diff, "spread": spread,
            "n3": int(len(v3)), "n2": int(len(v2)), "comparable": same,
            "winner": (3 if diff > spread else 2) if same else None}


def fair_table(tag3: str, tag2: str, path: Path | None = None) -> str:
    """The side-by-side the plan asked for, plus the VAL verdict line."""
    p = path or seed_row_path()
    if not p.is_file():
        return "(no seed runs measured yet)"
    df = pd.read_csv(p)
    d = fair_decision(df, tag3, tag2)
    lines = [f"## Fair comparison: feature v3 ({tag3}) vs v2 ({tag2})", ""]
    lines.append("| metric | v3 mean | v3 spread | v2 mean | v2 spread |")
    lines.append("|---|---|---|---|---|")
    for col, label in FAIR_COLUMNS:
        m3, s3 = _stat(df[df["run_tag"] == tag3], col)
        m2, s2 = _stat(df[df["run_tag"] == tag2], col)
        lines.append(f"| {label} | {m3} | {s3} | {m2} | {s2} |")
    lines.append("")
    if not d["comparable"]:
        lines += ["Decision suppressed: different or unknown label/evaluation identities; "
                  "no adoption recommendation.", ""]
        return "\n".join(lines)
    verdict = ("adopt v3 - the gain is bigger than the seed spread"
               if d["winner"] == 3 else
               "keep v2 - v3 is inside the seed spread or behind")
    lines.append(f"val decides: v3 {d['mean3']:.3f} vs v2 {d['mean2']:.3f} "
                 f"(diff {d['diff']:+.3f}, pooled seed spread "
                 f"+-{d['spread']:.3f}) -> {verdict}.")
    lines.append("")
    return "\n".join(lines)


def latest_fair_tags(path: Path | None = None) -> tuple[str, str] | None:
    """The most recent (v3_tag, v2_tag) pair, or None if none was run."""
    p = path or seed_row_path()
    if not p.is_file():
        return None
    tags = list(dict.fromkeys(pd.read_csv(p)["run_tag"].astype(str)))
    for t in reversed([t for t in tags if t.endswith("-v3")]):
        base = t[:-3]
        if f"{base}-v2" in tags:
            return t, f"{base}-v2"
    return None
