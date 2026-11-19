"""Stage E - is the bottleneck the model or the features?

Runs the same leave-one-dataset-out folds with three learners:

  mlp 128-64    the round-1 architecture
  mlp 256-128   twice as wide, identical regularisation and early stopping
  hgb           sklearn HistGradientBoostingClassifier, class_weight=balanced

and reports per learner the pooled (headline) and macro cross-dataset accuracy,
the watched letters, and the single-frame inference cost - a learner that wins
but cannot hold 20 FPS live is not a candidate.

    python src/ablate_models.py
    python src/ablate_models.py --models "mlp 128-64,hgb"
    python src/ablate_models.py --blocks xy,tip_dists,curl,thumb,orient

Writes reports/ablation_models.md. Nothing under models/ is touched: this is a
measurement script, and the real fit is still `python src/train.py`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
import model as M  # noqa: E402
import normalize  # noqa: E402
import train as T  # noqa: E402

MLPS = {"mlp 128-64": [128, 64], "mlp 256-128": [256, 128]}
WATCH_DEFAULT = "M,N,T,G,K,P,C"


def fold_rows(data, cfg: dict, seed: int):
    """Leave-one-dataset-out with the MLP, reusing the headline code path."""
    cross, py, pp = T.cross_dataset_folds(data, cfg, seed=seed)
    return cross, py, pp


def fold_rows_hgb(data, seed: int, xtr_cache: dict):
    """The same folds for HistGradientBoosting, split built identically."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    results, ys, ps = [], [], []
    for held in sorted(set(data["ds"])):
        tr = np.where(data["ds"] != held)[0]
        te = np.where(data["ds"] == held)[0]
        if len(te) < 20:
            continue
        xtr, ytr = xtr_cache["f"][tr], data["y"][tr]
        xte, yte = xtr_cache["f"][te], data["y"][te]
        w = data["w"][tr] if data.get("w") is not None else None
        t0 = time.time()
        print(f"\n  [cross-hgb] train on {sorted(set(data['ds'][tr]))} -> "
              f"test on {held} ({len(tr):,} / {len(te):,})")
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, random_state=seed, class_weight="balanced")
        clf.fit(xtr, ytr, sample_weight=w)
        p = clf.predict(xte)
        acc = float((yte == p).mean())
        results.append({"held_out_dataset": held, "n_test": int(len(te)),
                        "accuracy": acc, "y": yte, "p": p,
                        "carries_headline": len(te) >= config.HOLDOUT_MIN_FOR_HEADLINE,
                        "letters_covered": sorted(set(
                            data["df"]["letter"].to_numpy()[te])),
                        "seconds": time.time() - t0})
        ys.append(yte)
        ps.append(p)
        print(f"  [cross-hgb] {held}: {100 * acc:.1f}%  ({time.time() - t0:.0f}s)")
    return results, np.concatenate(ys), np.concatenate(ps)


def bench(kind, data, cfg, net_or_clf, n: int = 400) -> dict:
    """Single-frame latency and throughput, on real feature vectors."""
    lm = data["lm"][:n]
    x = T.feats_of(lm)
    if kind == "hgb":
        clf = net_or_clf
        one = x[:1]
        t0 = time.perf_counter()
        for i in range(min(n, 200)):
            clf.predict_proba(x[i:i + 1])
        lat = (time.perf_counter() - t0) / min(n, 200)
    else:
        net = net_or_clf
        with torch.no_grad():
            t0 = time.perf_counter()
            for i in range(min(n, 200)):
                M.predict_probs(net, x[i:i + 1])
            lat = (time.perf_counter() - t0) / min(n, 200)
    return {"latency_ms": round(1000 * lat, 2), "fps": round(1.0 / max(lat, 1e-9), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="mlp 128-64,mlp 256-128,hgb")
    ap.add_argument("--blocks", default=None,
                    help="ASL_FEATURE_BLOCKS to use (default: normalize's own)")
    ap.add_argument("--fold-epochs", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--watch", default=WATCH_DEFAULT)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--style-weight", type=float, default=1.0)
    ap.add_argument("--divergent-ratio", type=float, default=2.0,
                    help="minority-distance multiple that calls a letter "
                         "divergent (same meaning as in train.py)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.blocks:
        normalize.FEATURE_BLOCKS = tuple(b.strip() for b in args.blocks.split(","))
        config.FEATURE_BLOCKS = normalize.FEATURE_BLOCKS
    print(f"== Stage E: model capacity ==  blocks={'+'.join(normalize.FEATURE_BLOCKS)}")

    data = T.load_dataset(quick=args.quick)
    cells = (T.divergent_cells(args.divergent_ratio)
             if args.style_weight != 1.0 else set())
    data["w"] = T.style_weights(data, cells, args.style_weight)
    in_dim = T.feats_of(data["lm"][:8]).shape[1]
    print(f"  feature dim = {in_dim}")

    base = {"name": "stage_e", "in_dim": in_dim, "hidden": [128, 64],
            "dropout": 0.25, "label_smoothing": 0.05, "lr": 1e-3,
            "weight_decay": 1e-4, "batch": 256, "epochs": args.epochs,
            "fold_epochs": args.fold_epochs, "patience": 12, "aug_strength": 1.0}
    feats_cache = {"f": T.feats_of(data["lm"])}

    watch = [c.strip().upper() for c in args.watch.split(",") if c.strip()]
    rows = []
    for kind in [m.strip() for m in args.models.split(",") if m.strip()]:
        t0 = time.time()
        if kind == "hgb":
            cross, py, pp = fold_rows_hgb(data, args.seed, feats_cache)
            probe = None
            bench_obj = cross_probe_hgb(data, feats_cache) if cross else None
        elif kind in MLPS:
            cfg = dict(base, hidden=MLPS[kind])
            cross, py, pp = fold_rows(data, cfg, args.seed)
            bench_obj = None
            probe = cfg
        else:
            print(f"  ?? unknown model '{kind}'")
            continue
        hs = T.headline_stats(cross, py, pp)
        pc = T.per_class(py, pp).set_index("letter")
        row = {"model": kind, "dim": in_dim, "pooled": 100 * hs["pooled"],
               "macro": 100 * hs["macro"], "n": hs["n_pooled"],
               "per_ds": {r["held_out_dataset"]: 100 * r["accuracy"] for r in cross},
               "seconds": round(time.time() - t0, 1)}
        row.update({w: 100 * pc.loc[w, "accuracy"] for w in watch
                    if w in pc.index and pc.loc[w, "n"] > 0})
        if kind == "hgb" and bench_obj is not None:
            row.update(bench("hgb", data, None, bench_obj))
        else:
            net = T.train_one(data, np.arange(len(data["y"])),
                              dict(probe, epochs=1), seed=args.seed)
            row.update(bench(kind, data, probe, net))
        rows.append(row)
        print(f"\n  [{kind}] pooled {row['pooled']:.1f}%  macro {row['macro']:.1f}%  "
              f"fps {row.get('fps')}  ({row['seconds']:.0f}s fold time)")

    if not rows:
        return 1
    text = render(rows, watch, args)
    out = Path(args.out) if args.out else config.REPORTS_DIR / "ablation_models.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"  wrote {out}")
    return 0


def cross_probe_hgb(data, feats_cache):
    """A model fitted on everything, only used to measure inference cost."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    x, y = feats_cache["f"], data["y"]
    sub = np.random.default_rng(0).choice(len(y), 20000, replace=False)
    clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08,
                                         random_state=0)
    clf.fit(x[sub], y[sub])
    return clf


def render(rows, watch, args) -> str:
    best = max(rows, key=lambda r: r["pooled"])
    lines = ["# Stage E - model capacity check\n",
             f"Features: `{'+'.join(normalize.FEATURE_BLOCKS)}` "
             f"({rows[0]['dim']} dims). Protocol: leave-one-dataset-out, "
             f"{rows[0]['n']:,} test images.\n",
             "| learner | pooled | macro | " + " | ".join(watch) +
             " | 1-frame latency | live FPS | fold time |",
             "|---|---|---|" + "---|" * len(watch) + "---|---|---|"]
    for r in rows:
        cells = " | ".join(
            f"{r[w]:.1f}" if w in r else "–" for w in watch)
        lines.append(f"| {r['model']} | {r['pooled']:.1f}% | {r['macro']:.1f}% | "
                     f"{cells} | {r.get('latency_ms', float('nan')):.1f} ms | "
                     f"{r.get('fps', float('nan')):.0f} | {r['seconds']:.0f}s |")
    lines.append("")
    lines.append("| learner | " + " | ".join(sorted(rows[0]["per_ds"])) + " |")
    lines.append("|---|" + "---|" * len(rows[0]["per_ds"]))
    for r in rows:
        lines.append(f"| {r['model']} | " +
                     " | ".join(f"{v:.1f}%" for _k, v in sorted(r["per_ds"].items())) +
                     " |")
    lines.append("")
    lines.append(f"**Best by pooled accuracy: `{best['model']}` "
                 f"({best['pooled']:.1f}%).**")
    base_row = rows[0]
    if len(rows) > 1:
        wider = next((r for r in rows if r["model"] == "mlp 256-128"), None)
        if wider is not None:
            lines.append(
                f"\nWidening the MLP moved the pooled number by "
                f"{wider['pooled'] - base_row['pooled']:+.1f} points "
                f"({base_row['pooled']:.1f}% -> {wider['pooled']:.1f}%).")
    lines.append("\nReading: if neither a wider MLP nor gradient boosting buys "
                 "more than about a point, the ceiling is the **features and the "
                 "data**, not the learner - which is what the Stage D table is "
                 "for. The band to compare against is measured in Stage D "
                 "(`reports/ablation_features.md`: same feature set, same data, "
                 "same folds, repeated runs), where it is 1.6-2.5 pooled points, "
                 "and it is widened by the 0.7 points that re-running one "
                 "identical command cost before the seed fix of 2026-11-14. Each "
                 "row here is a single run, so anything inside that band is not a "
                 "win. The per-letter cells are read the same way: they move by "
                 "tens of points between runs and are recorded for orientation, "
                 "not as results. The live requirement is >=20 FPS "
                 "for the whole frame pipeline; MediaPipe landmark detection "
                 "itself costs far more than the classifier, so a classifier "
                 "that needs more than ~10 ms per frame is already too slow to "
                 "be worth it.")
    return "\n".join(l for l in lines if l is not None) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
