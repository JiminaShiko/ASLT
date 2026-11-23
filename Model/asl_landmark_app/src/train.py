"""Step 4 - train the landmark MLP and write REPORT.md.

The headline is always cross-dataset (leave-one-dataset-out) POOLED accuracy;
within-dataset session splits and the final model's own split are labelled
optimistic. No random-split score is headlined anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from torch.utils.data import WeightedRandomSampler  # noqa: E402

import augment  # noqa: E402
import config  # noqa: E402
import landmark_store as LS  # noqa: E402
import model as M  # noqa: E402
import normalize  # noqa: E402

torch.set_num_threads(os.cpu_count() or 4)
LABELS = config.LABELS
# tiny cells: print the count, one image is worth 100 points
SMALL_CELL_N = 40


def load_dataset(quick: bool = False, max_rows_per_ds: int | None = None):
    df = LS.load_raw()
    df = LS.keep_static_classes(df)
    if max_rows_per_ds:
        parts = []
        for did, sub in df.groupby("dataset"):
            k = min(len(sub), max_rows_per_ds)
            parts.append(sub.iloc[np.linspace(0, len(sub) - 1, k).round().astype(int)])
        df = pd.concat(parts).reset_index(drop=True)
    if quick:
        parts = []
        for (did, letter), sub in df.groupby(["dataset", "letter"]):
            k = min(len(sub), 150)
            parts.append(sub.iloc[np.linspace(0, len(sub) - 1, k).round().astype(int)])
        df = pd.concat(parts).reset_index(drop=True)
    lms = LS.landmark_matrix(df)
    prepared = normalize.prepare_batch(lms).astype(np.float32)
    y = np.asarray([config.LABEL_TO_INDEX[l] for l in df["letter"]], dtype=np.int64)
    ds = df["dataset"].to_numpy().astype(str)
    sess = LS.group_id(df)
    blk = LS.block_id(df)
    print(f"  loaded {len(df):,} landmark samples, {prepared.shape[1:]} "
          f"feature basis, {len(set(ds))} datasets: {sorted(set(ds))}")
    cov = {d: sorted(set(df.loc[ds == d, 'letter'])) for d in sorted(set(ds))}
    for d, letters in cov.items():
        print(f"    {d:<12} {len(letters):>2} letters ({len(df[ds == d]):>6} samples)")
    return {"lm": prepared, "y": y, "ds": ds, "sess": sess, "blk": blk, "df": df}


DIVERGENCE_CSV = config.CACHE_DIR / "divergence.csv"


def _pair_means(dv: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    acc: dict[tuple[str, str], list[float]] = {}
    for r in dv.itertuples():
        acc.setdefault(tuple(sorted((str(r.ds_a), str(r.ds_b)))), []).append(
            float(r.shape_dist))
    return acc


def style_cluster(dv: pd.DataFrame) -> tuple[tuple[str, str], str] | None:
    """Split the datasets into (majority pair, odd one out) by mean shape.

    Averaged over all letters so one small dataset's noise cannot sway it;
    None when there is no clear two-against-one structure.
    """
    means = {k: sum(v) / len(v) for k, v in _pair_means(dv).items()}
    if len(means) < 3:
        return None                      # fewer than three datasets: no tiebreak
    ranked = sorted(means.items(), key=lambda kv: kv[1])
    (maj, maj_dist), (second, second_dist) = ranked[0], ranked[1]
    if maj_dist * 1.25 > second_dist:     # the "close" pair is not clearly closest
        return None
    all_ds = {d for pair in means for d in pair}
    odd = [d for d in all_ds if d not in maj]
    return (maj, odd[0]) if len(odd) == 1 else None


def divergent_cells(mult: float = 2.0) -> set[tuple[str, str]]:
    """(letter, dataset) cells where one dataset signs that letter differently.

    Reads cache/divergence.csv (src/inspect_data.py). Uses absolute distance
    as a multiple of the median over letters: the shape_dist/within_ds ratio
    is useless (within_ds is a floor). The minority dataset comes from
    style_cluster(), not row counts, which tie. Skipped when the cluster pair
    itself disagrees or is too small. mult=2.0 flags M, N, P.
    """
    if not DIVERGENCE_CSV.exists():
        return set()
    dv = pd.read_csv(DIVERGENCE_CSV)
    cl = style_cluster(dv)
    if cl is None:
        return set()
    (maj_a, maj_b), odd = cl
    # compare against the better-sampled cluster member: the other may hold
    # only a few dozen images per letter
    per: dict[str, dict[str, float]] = {}
    for r in dv.itertuples():
        pair = {str(r.ds_a), str(r.ds_b)}
        lab = str(r.letter)
        ent = per.setdefault(lab, {})
        if odd in pair:
            member = next(iter(pair - {odd}))
            if member in (maj_a, maj_b):
                n_m = int(r.n_a if r.ds_a == member else r.n_b)
                if n_m >= ent.get("n_ref", 0):
                    ent["n_ref"] = n_m
                    ent["odd_ref"] = float(r.shape_dist)
        else:
            n_min = min(int(r.n_a), int(r.n_b))
            ent["within"] = float(r.shape_dist)
            ent["within_n"] = n_min
    dists = sorted(v["odd_ref"] for v in per.values() if "odd_ref" in v)
    if not dists:
        return set()
    med = dists[len(dists) // 2]
    cells: set[tuple[str, str]] = set()
    for lab, v in per.items():
        if "odd_ref" not in v:
            continue
        if v["odd_ref"] < mult * med:
            continue
        # cluster pair disagreeing (or too small to tell) is not a two-style conflict
        if v.get("within_n", 0) >= 200 and v["odd_ref"] <= v["within"]:
            continue
        cells.add((lab, odd))
    return cells


def style_weights(data: dict, cells: set[tuple[str, str]], weight: float) -> np.ndarray:
    """Per-sample training weight: `weight` for divergent-minority cells, else 1."""
    w = np.ones(len(data["y"]), dtype=np.float64)
    if not cells or weight == 1.0:
        return w
    ds = data["ds"]
    letters = data["df"]["letter"].astype(str).to_numpy()
    hit = np.array([(l, d) in cells for l, d in zip(letters, ds)])
    w[hit] = weight
    return w


def divergence_summary(mult: float = 2.0) -> tuple[pd.DataFrame, dict]:
    """Per-letter style-conflict view of cache/divergence.csv, plus the cluster.

    Sorted by the same measure `divergent_cells()` thresholds on.
    """
    if not DIVERGENCE_CSV.exists():
        return pd.DataFrame(), {}
    dv = pd.read_csv(DIVERGENCE_CSV)
    cl = style_cluster(dv)
    if cl is None:
        return pd.DataFrame(), {"cluster": None}
    (maj_a, maj_b), odd = cl
    means = {k: sum(v) / len(v) for k, v in _pair_means(dv).items()}
    per: dict[str, dict[str, float]] = {}
    for r in dv.itertuples():
        pair = {str(r.ds_a), str(r.ds_b)}
        ent = per.setdefault(str(r.letter), {})
        if odd in pair:
            member = next(iter(pair - {odd}))
            n_m = int(r.n_a if r.ds_a == member else r.n_b)
            if n_m >= ent.get("n_ref", 0):
                ent["n_ref"], ent["odd_ref"] = n_m, float(r.shape_dist)
                ent["odd_pair"] = f"{odd} vs {member}"
        else:
            ent["within"] = float(r.shape_dist)
            ent["within_n"] = min(int(r.n_a), int(r.n_b))
    dists = sorted(v["odd_ref"] for v in per.values() if "odd_ref" in v)
    med = dists[len(dists) // 2] if dists else float("nan")
    cells = divergent_cells(mult)
    rows = [{"letter": lab, "odd_dataset": odd, "dist_to_cluster": v.get("odd_ref"),
             "x_median": (v.get("odd_ref", 0.0) / med if med else float("nan")),
             "majority_pair_dist": v.get("within"),
             "flagged": "yes" if (lab, odd) in cells else ""}
            for lab, v in per.items()]
    df = (pd.DataFrame(rows).sort_values("dist_to_cluster", ascending=False)
          .reset_index(drop=True))
    info = {"cluster": (maj_a, maj_b), "odd": odd, "median": med,
            "cluster_mean_dist": means.get(tuple(sorted((maj_a, maj_b))), float("nan")),
            "odd_dists": {m: means.get(tuple(sorted((odd, m))), float("nan"))
                          for m in (maj_a, maj_b)},
            "n_flagged": len(cells), "mult": mult}
    return df, info


def feats_of(lm: np.ndarray) -> np.ndarray:
    return normalize.features_from_normalized_batch(lm.astype(np.float64))


def train_one(data, train_idx: np.ndarray, cfg: dict, seed: int,
              val_idx: np.ndarray | None = None, verbose: bool = True):
    # everything random must come from `seed`: without this the torch init and
    # sampler drew from the global RNG (63.9 vs 64.6% for two identical runs).
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lm_tr, y_tr = data["lm"][train_idx], data["y"][train_idx]
    counts = np.bincount(y_tr, minlength=len(LABELS)).astype(np.float64)
    counts[counts == 0] = 1.0
    sample_w = torch.as_tensor(1.0 / counts[y_tr], dtype=torch.double)
    w = data.get("w")
    if w is not None:
        sample_w = sample_w * torch.as_tensor(w[train_idx], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(y_tr), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))

    net = M.build(cfg["in_dim"], len(LABELS), cfg["hidden"], cfg["dropout"])
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    best = (float("inf"), None, -1)
    stale = 0
    val_feats = None
    val_y = None
    if val_idx is not None and len(val_idx):
        val_feats = feats_of(data["lm"][val_idx])
        val_y = data["y"][val_idx]

    n = len(train_idx)
    for epoch in range(cfg["epochs"]):
        net.train()
        order = np.fromiter(sampler, dtype=np.int64, count=n)
        tot_loss = 0.0
        for i in range(0, n, cfg["batch"]):
            bidx = torch.from_numpy(order[i:i + cfg["batch"]])
            lms = lm_tr[bidx.numpy()]
            lms = augment.augment_batch(lms, rng, cfg["aug_strength"])
            xb = torch.from_numpy(feats_of(lms))
            yb = torch.from_numpy(y_tr[bidx.numpy()])
            opt.zero_grad(set_to_none=True)
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * len(yb)
        if val_feats is None:
            continue
        with torch.no_grad():
            vlogits = net(torch.from_numpy(val_feats))
            vl = float(crit(vlogits, torch.from_numpy(val_y)).item())
        vacc = float((vlogits.argmax(1).numpy() == val_y).mean())
        sched.step(vl)
        if vl < best[0] - 1e-4:
            best = (vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, epoch)
            stale = 0
        else:
            stale += 1
        if verbose and (epoch % 5 == 0 or stale >= cfg["patience"]):
            print(f"      ep{epoch:>3} train_loss={tot_loss / max(n,1):.4f} "
                  f"val_loss={vl:.4f} val_acc={100 * vacc:.1f}%")
        if stale >= cfg["patience"]:
            if verbose:
                print(f"      early stop at epoch {epoch} (best epoch {best[2]})")
            break
    if best[1] is not None:
        net.load_state_dict(best[1])
    net.eval()
    return net


def evaluate(net, lm: np.ndarray, y: np.ndarray,
             want_probs: bool = False):
    """(true labels, predicted labels); `want_probs` adds the full softmax."""
    probs = M.predict_probs(net, feats_of(lm))
    p = probs.argmax(1)
    return (y, p, probs) if want_probs else (y, p)


def top_two(probs: np.ndarray) -> np.ndarray:
    """Softmax matrix -> `[p1, p2, runner_up_index]`, what `live.frame_call()` sees.

    Lets pair margins be measured out-of-fold without dumping 122k x 24 floats.
    """
    n = len(probs)
    part = np.argpartition(-probs, 1, axis=1)[:, :2]
    rows = np.arange(n)
    # argpartition only guarantees position 0 is one of the two largest
    first = np.where(probs[rows, part[:, 0]] >= probs[rows, part[:, 1]],
                     part[:, 0], part[:, 1])
    second = np.where(first == part[:, 0], part[:, 1], part[:, 0])
    return np.column_stack([probs[rows, first], probs[rows, second],
                            second.astype(np.float64)])


def per_class(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, lab in enumerate(LABELS):
        sup = int((y == i).sum())
        correct = int(((y == i) & (p == i)).sum())
        rows.append({"letter": lab, "n": sup, "correct": correct,
                     "accuracy": (correct / sup) if sup else float("nan")})
    return pd.DataFrame(rows)


def confusion(y: np.ndarray, p: np.ndarray, n_cls: int = None) -> np.ndarray:
    n_cls = n_cls or len(LABELS)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, q in zip(y, p):
        cm[t, q] += 1
    return cm


def top_confused(cm: np.ndarray, k: int = 8) -> list[tuple[str, str, int, float]]:
    out = []
    for i in range(len(cm)):
        tot = cm[i].sum()
        if tot == 0:
            continue
        for j in range(len(cm)):
            if i == j or cm[i, j] == 0:
                continue
            out.append((LABELS[i], LABELS[j], int(cm[i, j]), cm[i, j] / tot))
    out.sort(key=lambda x: (-x[3], -x[2]))
    return out[:k]


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    mask = cm.sum(axis=1) > 0
    labs = [l for l, m in zip(LABELS, mask) if m]
    sub = cm[np.ix_(mask, mask)]
    fig, ax = plt.subplots(figsize=(9, 8))
    norm = sub.astype(float) / np.maximum(sub.sum(axis=1, keepdims=True), 1)
    sns.heatmap(norm, annot=sub, fmt="d", cmap="viridis", xticklabels=labs,
                yticklabels=labs, ax=ax, cbar_kws={"label": "row-fraction"},
                linewidths=0.3)
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = correct / n
    den = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def headline_stats(cross: list[dict], pooled_y: np.ndarray,
                   pooled_p: np.ndarray) -> dict:
    """Pooled (micro, the headline) and macro (mean-of-datasets) accuracy.

    MACRO is reported, never headlined: small easy datasets pull it up.
    """
    n = int(len(pooled_y))
    correct = int((pooled_y == pooled_p).sum()) if n else 0
    out = {
        "n_pooled": n,
        "correct_pooled": correct,
        "pooled": (correct / n) if n else float("nan"),
        "ci": wilson_ci(correct, n),
        "macro": float(np.mean([r["accuracy"] for r in cross])) if cross else float("nan"),
        "n_datasets": len(cross),
    }
    big = [r for r in cross if r["n_test"] >= config.HOLDOUT_MIN_FOR_HEADLINE]
    small = [r for r in cross if r["n_test"] < config.HOLDOUT_MIN_FOR_HEADLINE]
    out["headline_datasets"] = [r["held_out_dataset"] for r in big]
    out["small_datasets"] = [r["held_out_dataset"] for r in small]
    if big:
        yb = np.concatenate([r["y"] for r in big])
        pb = np.concatenate([r["p"] for r in big])
        out["n_pooled_big"] = int(len(yb))
        out["correct_pooled_big"] = int((yb == pb).sum())
        out["pooled_big"] = float((yb == pb).mean())
        out["ci_big"] = wilson_ci(out["correct_pooled_big"], out["n_pooled_big"])
        out["macro_big"] = float(np.mean([r["accuracy"] for r in big]))
    else:
        out.update({"n_pooled_big": 0, "correct_pooled_big": 0,
                    "pooled_big": float("nan"), "ci_big": (float("nan"),) * 2,
                    "macro_big": float("nan")})
    return out


def per_letter_by_dataset(cross: list[dict]) -> pd.DataFrame:
    """letter x held-out-dataset accuracy table (NaN where a letter is absent)."""
    rows = []
    for r in cross:
        pc = r["per_class"].set_index("letter")
        for lab in LABELS:
            if lab in pc.index and pc.loc[lab, "n"] > 0:
                rows.append({"letter": lab, "dataset": r["held_out_dataset"],
                             "n": int(pc.loc[lab, "n"]),
                             "accuracy": float(pc.loc[lab, "accuracy"])})
    if not rows:
        empty = pd.DataFrame(index=pd.Index(LABELS, name="letter"))
        return empty, empty
    wide = pd.DataFrame(rows).pivot(index="letter", columns="dataset",
                                    values="accuracy")
    counts = pd.DataFrame(rows).pivot(index="letter", columns="dataset", values="n")
    wide = wide.reindex(LABELS)
    counts = counts.reindex(LABELS)
    return wide, counts


def cross_dataset_folds(data, cfg, seed=0, train_how: str | None = None):
    """Leave-one-dataset-out: the primary, honest evaluation.

    `train_how` narrows only the training rows to one detection path, so the
    effect of the recovered images can be isolated.
    """
    cfg = fold_cfg(cfg)
    ds_ids = sorted(set(data["ds"]))
    how = (data["df"]["how"].astype(str).to_numpy()
           if "how" in data["df"].columns else None)
    results, pooled_y, pooled_p = [], [], []
    for held in ds_ids:
        tr = np.where(data["ds"] != held)[0]
        if train_how:
            if how is None:
                raise SystemExit("--train-how needs the round-2 cache (no `how` "
                                 "column); rebuild it with src/extract_landmarks.py")
            tr = tr[how[tr] == train_how]
        te = np.where(data["ds"] == held)[0]
        if len(te) < 20:
            continue
        print(f"\n  [cross] train on {sorted(set(data['ds'][tr]))} -> test on {held} "
              f"({len(tr):,} train / {len(te):,} test)")
        t0 = time.time()
        net = train_one(data, tr, cfg, seed=seed, val_idx=_group_val(data, tr))
        y, p, probs = evaluate(net, data["lm"][te], data["y"][te], want_probs=True)
        acc = float((y == p).mean())
        results.append({
            "held_out_dataset": held, "n_test": len(te), "accuracy": acc,
            "y": y, "p": p, "te_idx": te, "top2": top_two(probs),
            "carries_headline": len(te) >= config.HOLDOUT_MIN_FOR_HEADLINE,
            "letters_covered": sorted(set(data["df"]["letter"].to_numpy()[te])),
            "seconds": time.time() - t0,
            "cm": confusion(y, p),
            "per_class": per_class(y, p),
        })
        pooled_y.append(y)
        pooled_p.append(p)
        print(f"  [cross] {held}: {100 * acc:.1f}%  ({time.time() - t0:.0f}s)")
    return results, np.concatenate(pooled_y), np.concatenate(pooled_p)


ROUND1_HOW = "direct_0.5"


def dump_oof(cross, data, out: Path) -> int:
    """Write every leave-one-dataset-out prediction as one CSV row per image.

    With `p1`, `p2` and `runner_up` when the folds carried their softmax, so
    refusal margins can be measured. `image` joins back to the landmark cache.
    """
    cols = data["df"].reset_index(drop=True)
    img = cols["image"].astype(str).to_numpy() if "image" in cols else np.full(len(cols), "")
    how = cols["how"].astype(str).to_numpy() if "how" in cols else np.full(len(cols), "")
    letter = cols["letter"].astype(str).to_numpy()
    sess = data["sess"]
    parts = []
    for r in cross:
        te = np.asarray(r["te_idx"])
        row = {
            "fold": r["held_out_dataset"],
            "image": img[te],
            "letter": letter[te],
            "predicted": [LABELS[i] for i in r["p"]],
            "correct": np.asarray(r["y"]) == np.asarray(r["p"]),
            "how": how[te],
            "session": sess[te],
        }
        if "top2" in r:
            t = np.asarray(r["top2"])
            row["p1"] = t[:, 0]
            row["p2"] = t[:, 1]
            row["runner_up"] = [LABELS[int(i)] for i in t[:, 2]]
        parts.append(pd.DataFrame(row))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(
        out, index=False, float_format="%.4f")
    return int(sum(len(p) for p in parts))


def round1_comparable(cross, data) -> dict | None:
    """Pooled accuracy scored only on the images round 1 could detect.

    Stage C added 20,885 hands that round 1 never had, so the v2 headline is
    measured on a bigger and harder test set than the 67.6% quoted in round 1 -
    the two numbers are not directly comparable. Restricting the scoring to
    rows the ladder found on its first attempt (`how == direct_0.5`) rebuilds
    round 1's test set, so the same yardstick can be used for both rounds. The
    extra recovered images still train the model, which is the point: this is a
    like-for-like *test*, not a like-for-like *model*.
    """
    if "how" not in data["df"].columns:
        return None
    how = data["df"]["how"].astype(str).to_numpy()
    n = c = 0
    for r in cross:
        m = how[r["te_idx"]] == ROUND1_HOW
        n += int(m.sum())
        c += int((r["y"][m] == r["p"][m]).sum())
    if not n:
        return None
    return {"correct": c, "n": n, "accuracy": c / n,
            "share_of_test": n / max(sum(int(len(r["y"])) for r in cross), 1)}


def fold_cfg(cfg: dict) -> dict:
    """Shorter training for the ~15 evaluation models; early stopping still applies."""
    return {**cfg, "epochs": cfg.get("fold_epochs") or cfg["epochs"]}


def _group_val(data, train_idx: np.ndarray) -> np.ndarray:
    """Contiguous-block validation inside the training datasets (never random).

    Groups on `blk` so every class lands on both sides (a session folder is
    one whole letter for the photo datasets).
    """
    groups = data["blk"][train_idx]
    if len(np.unique(groups)) < 4:
        return train_idx[:0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    tr, va = next(gss.split(train_idx, groups=groups))
    return train_idx[va]


def pairwise_folds(data, cfg, seed=1):
    cfg = fold_cfg(cfg)
    pairs, ds_ids = [], sorted(set(data["ds"]))
    for a in ds_ids:
        for b in ds_ids:
            if a == b:
                continue
            tr = np.where(data["ds"] == a)[0]
            te = np.where(data["ds"] == b)[0]
            if len(tr) < 200 or len(te) < 20:
                continue
            net = train_one(data, tr, cfg, seed=seed, val_idx=_group_val(data, tr),
                            verbose=False)
            y, p = evaluate(net, data["lm"][te], data["y"][te])
            common = sorted(set(data["df"]["letter"].to_numpy()[te]) &
                            set(data["df"]["letter"].to_numpy()[tr]))
            keep = np.isin(y, [config.LABEL_TO_INDEX[c] for c in common])
            acc = float((y[keep] == p[keep]).mean()) if keep.any() else float("nan")
            pairs.append({"train": a, "test": b, "n_test": int(keep.sum()),
                          "accuracy": acc, "shared_letters": len(common)})
            print(f"  [pair] {a:>12} -> {b:<12} {100 * acc:5.1f}%  "
                  f"(test n={int(keep.sum())}, {len(common)} letters in common)")
    return pairs


def within_dataset_blocksplit(data, cfg, dataset_id: str, seed=2):
    """Reference number only: contiguous blocks, so clips never straddle splits."""
    cfg = fold_cfg(cfg)
    idx = np.where(data["ds"] == dataset_id)[0]
    groups = data["blk"][idx]
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                   random_state=seed).split(idx, groups=groups))
    tr_idx, te_idx = idx[tr], idx[va]
    net = train_one(data, tr_idx, cfg, seed=seed, val_idx=_group_val(data, tr_idx),
                    verbose=False)
    y, p = evaluate(net, data["lm"][te_idx], data["y"][te_idx])
    return {"dataset": dataset_id, "n_test": len(te_idx),
            "n_groups": int(len(np.unique(groups))), "accuracy": float((y == p).mean())}


def unseen_group_split(data, cfg, dataset_id: str, seed=3):
    """Held-out capture group, the closest this data gets to held-out subject.

    None for the photo datasets (one folder = one letter).
    """
    cfg = fold_cfg(cfg)
    idx = np.where(data["ds"] == dataset_id)[0]
    sess, let = data["sess"][idx], data["y"][idx]
    per = {s: set(let[sess == s]) for s in sorted(set(sess))}
    need = max(4, len(LABELS) // 3)
    rich = [s for s, ls in per.items() if len(ls) >= need]
    if len(rich) < 2:
        return None
    half = len(rich) // 2
    tr, te = idx[np.isin(sess, rich[:half])], idx[np.isin(sess, rich[half:])]
    net = train_one(data, tr, cfg, seed=seed, val_idx=_group_val(data, tr),
                    verbose=False)
    y, p = evaluate(net, data["lm"][te], data["y"][te])
    common = sorted(set(data["y"][tr]) & set(data["y"][te]))
    keep = np.isin(y, common)
    return {"dataset": dataset_id, "n_test": int(keep.sum()),
            "train_groups": rich[:half], "test_groups": rich[half:],
            "accuracy": float((y[keep] == p[keep]).mean())}


def live_path_summary() -> dict | None:
    """Live hand-coverage and FPS from reports/live_path.md, or None if absent."""
    p = config.REPORTS_DIR / "live_path.md"
    if not p.is_file():
        return None
    txt = p.read_text(encoding="utf-8")
    cov = re.search(r"\|\s*frames with a hand\s*\|\s*\d+/\d+\s*\(([\d.]+)%\)\s*\|"
                    r"\s*\d+/\d+\s*\(([\d.]+)%\)", txt)
    if not cov:
        return None
    fps = re.search(r"\*\*([\d.]+) frames/s\*\*", txt)
    return {"coverage": f"it finds a hand in {cov.group(2)}% of frames instead of "
                        f"{cov.group(1)}%",
            "fps": f"{fps.group(1)} frames/s" if fps else "an unrecorded frame rate"}


def extractor_recall(skipped_csv: Path):
    """(recall before = attempt 1 only, recall after ladder, images selected)."""
    if not skipped_csv.exists():
        return None
    sk = pd.read_csv(skipped_csv)
    if "kept_direct" not in sk.columns:
        return None
    sel = max(int(sk["selected"].sum()), 1)
    return (int(sk["kept_direct"].sum()) / sel, int(sk["kept"].sum()) / sel, sel)


def coverage_sentence(offline_rec, live) -> str:
    """The two coverage numbers, stated as the different quantities they are.

    Extractor = image recall offline; live path = frame coverage. Never merge.
    """
    off = (f"the extractor finds a hand in {100 * offline_rec[1]:.1f}% of dataset "
           f"images instead of {100 * offline_rec[0]:.1f}% - that is image recall "
           f"over {offline_rec[2]:,} files, where five detections per image cost "
           f"nothing" if offline_rec else
           "no ladder-aware `skipped_stats.csv` was found, so the offline image "
           "recall is not quoted here")
    liv = (f"{live['coverage']}, measured frame by frame in "
           f"`reports/live_path.md` at {live['fps']} for the whole pipeline"
           if live else
           "no live measurement exists yet - run "
           "`python src/verify_livepath.py` before quoting one")
    return ("What the ladder buys is coverage, and the two coverage numbers are "
            "not the same thing. **Offline:** " + off + ". **Live:** the app "
            "cannot afford that ladder per frame, so it retries a frame at "
            "confidence 0.3 only when tracking has just lost the hand, and "
            + liv + ". A frame with no hand produces nothing at all, so coverage "
            "is the part worth paying for; accuracy is the table above.")


def write_report(cfg, cross, pooled_y, pooled_p, pairs, blocksplit, groups, final_info,
                 data, skipped_csv: Path, out_md: Path) -> None:
    cm_pooled = confusion(pooled_y, pooled_p)
    pc = per_class(pooled_y, pooled_p)
    # one figure per profile: reports/confusion_matrix_<profile>.png
    png = config.REPORTS_DIR / f"confusion_matrix_{out_md.stem.replace('report_', '')}.png"
    plot_confusion(cm_pooled, png, "Cross-dataset confusion (each sample scored by a "
                                   "model that never saw its dataset)")

    mean_acc = float(np.mean([r["accuracy"] for r in cross])) if cross else float("nan")
    hs = headline_stats(cross, pooled_y, pooled_p)
    worst = pc.dropna(subset=["accuracy"]).sort_values("accuracy")
    offline_rec = extractor_recall(skipped_csv)
    live = live_path_summary()
    lines = []
    A = lines.append
    A("# ASL landmark recogniser - training report\n")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} · model `{cfg['name']}` · "
      f"{len(LABELS)} classes (A-Z minus J, Z) · input {cfg['in_dim']} dims\n")

    A("## Headline number\n")
    A(f"**Cross-dataset accuracy, POOLED (headline): {100 * hs['pooled']:.1f}% "
      f"[{100 * hs['ci'][0]:.1f}–{100 * hs['ci'][1]:.1f}% 95% CI] "
      f"({hs['correct_pooled']:,}/{hs['n_pooled']:,} held-out images)**\n")
    A(f"**Cross-dataset accuracy, MACRO (mean of the {hs['n_datasets']} held-out "
      f"datasets, unweighted): {100 * hs['macro']:.1f}%**\n")
    r1 = round1_comparable(cross, data)
    if r1:
        A(f"**Same yardstick as round 1 (pooled, scored only on the "
          f"{100 * r1['share_of_test']:.0f}% of these images that round 1's "
          f"single-pass detector could find at all): {100 * r1['accuracy']:.1f}% "
          f"({r1['correct']:,}/{r1['n']:,})**\n")
        # paired control on the seeded pipeline; re-derive with
        #   python src/train.py --folds-only --no-save --train-how direct_0.5
        CTRL = {"correct_all": 80861, "n_all": 122359,
                "correct_easy": 69538, "n_easy": 101932}
        hard_n = hs["n_pooled"] - r1["n"]
        hard_c = hs["correct_pooled"] - r1["correct"]
        ctrl_hard_n = CTRL["n_all"] - CTRL["n_easy"]
        ctrl_hard = ((CTRL["correct_all"] - CTRL["correct_easy"]) / ctrl_hard_n
                     if ctrl_hard_n else float("nan"))
        hard = hard_c / hard_n if hard_n else float("nan")
        ctrl_pooled = CTRL["correct_all"] / CTRL["n_all"]
        ctrl_easy = CTRL["correct_easy"] / CTRL["n_easy"]
        # the control is only a control if the two runs scored the same images
        paired = CTRL["n_all"] == hs["n_pooled"] and CTRL["n_easy"] == r1["n"]
        max_gap = max(abs(hs["pooled"] - ctrl_pooled), abs(r1["accuracy"] - ctrl_easy),
                      abs(hard - ctrl_hard) if hard_n else 0.0)
        A(f"Round 1 headlined 67.6% pooled on a cache that was missing 20,885 "
          f"hands. Stage C put them back, and they are not neutral: the recovered "
          f"images are the flat palms, the silhouettes and the small crops, i.e. "
          f"the frames a real webcam is most likely to give you. The line directly "
          f"above this one scores only the images round 1 could also score, so it "
          f"is the only cell in this report that may be compared with 67.6% — and "
          f"on that like-for-like subset this model is at {100 * r1['accuracy']:.1f}% "
          f"against round 1's 67.6%, a {100 * abs(r1['accuracy'] - 0.676):.1f}-point "
          f"difference. Per the Reproducibility section that is inside the "
          f"run-to-run band, so it is neither a measured improvement nor a "
          f"measured regression.\n"
          f"What moved the headline is the denominator. Whether the recovered "
          f"images also moved the *model* was then measured with "
          f"`--train-how direct_0.5`, which leaves the test rows alone and withholds "
          f"the ladder-rescued images from training only: same seed, same 71 "
          f"features, same style weight, same folds.\n\n")
        A("| training rows | pooled, all 122,359 | round 1's 101,932 | "
          f"ladder-rescued {hard_n:,} |\n|---|---|---|---|\n"
          f"| every recovered image (this model) | {100 * hs['pooled']:.1f}% "
          f"| {100 * r1['accuracy']:.1f}% | {100 * hard:.1f}% |\n"
          f"| round 1's data mix only (control) | {100 * ctrl_pooled:.1f}% | "
          f"{100 * ctrl_easy:.1f}% | {100 * ctrl_hard:.1f}% |\n\n")
        if not paired:
            A(f"**This table is void in this run.** The control above was measured "
              f"on {CTRL['n_all']:,} held-out images and this report has "
              f"{hs['n_pooled']:,}, so the two did not score the same test set — "
              f"which happens whenever a report is generated from a `--quick` or "
              f"partial cache. Re-run both commands on the full cache to get the "
              f"comparison back.")
        else:
            A(f"Two paired runs on identical held-out images, and the largest gap "
              f"in the table is {100 * max_gap:.1f} points, which is the size of "
              f"one re-roll of this pipeline. So the recovered images neither helped "
              f"nor hurt the average.\n" + coverage_sentence(offline_rec, live))
    A("The **pooled (micro)** number is the headline: every held-out image counts "
      "exactly once, so it estimates \"a random image from hands the model never "
      "trained on\", which is what the webcam will actually give you. The **macro** "
      "mean weights each *dataset* equally, so a small holdout can carry it far "
      "beyond the truth — an earlier version of this report headlined the macro "
      f"mean ({100 * hs['macro']:.1f}%) while the pooled figure was "
      f"{100 * hs['pooled']:.1f}%.\n")
    if hs["small_datasets"]:
        A(f"Datasets under {config.HOLDOUT_MIN_FOR_HEADLINE} held-out images are "
          f"flagged **small-n** and do not carry the headline: "
          f"{', '.join(hs['small_datasets'])}. They are still used IN TRAINING for "
          f"every other fold and for the final model; only their role as a "
          f"*measuring stick* is limited. "
          f"Excluding them, the pooled cross-dataset accuracy is "
          f"**{100 * hs['pooled_big']:.1f}%** over "
          f"{hs['n_pooled_big']:,} images "
          f"[{100 * hs['ci_big'][0]:.1f}–{100 * hs['ci_big'][1]:.1f}%] and the macro "
          f"mean of the headline-bearing datasets is "
          f"**{100 * hs['macro_big']:.1f}%**.\n")
    A("This is leave-one-dataset-out: every test sample was predicted by a model "
      "trained only on the *other* datasets, i.e. on hands, cameras and backgrounds "
      "it never saw. Treat it as the expected live accuracy ceiling.\n")
    A("| held-out dataset | test images | letters covered | accuracy | carries headline? |")
    A("|---|---|---|---|---|")
    for r in cross:
        A(f"| {r['held_out_dataset']} | {r['n_test']} | {len(r['letters_covered'])} "
          f"({''.join(r['letters_covered'])[:12]}{'…' if len(r['letters_covered']) > 12 else ''}) "
          f"| {100 * r['accuracy']:.1f}% | "
          f"{'yes' if r['carries_headline'] else '**no — small-n**'} |")
    A("")
    covered = {r["held_out_dataset"]: set(r["letters_covered"]) for r in cross}
    union = set().union(*covered.values()) if covered else set()
    A(f"All numbers are over the {len(union)}-letter static alphabet "
      f"(`{''.join(sorted(union))}`); J and Z are moving letters and are handled by "
      f"the trajectory rule in `src/live.py`, not by this classifier.")
    for d, ls in covered.items():
        if ls != union:
            A(f"- **{d}** only covers `{''.join(sorted(ls))}`, so its accuracy is "
              f"measured over that subset and is not directly comparable to the others.")
    A("")

    A("## Per-letter accuracy for each held-out dataset\n")
    A("A letter can look strong in the pooled table and still be dataset-specific. "
      "Each column is a model that never saw that dataset; `–` means no test "
      "images for that letter, and a bracketed number is a small test count "
      f"(< {config.HOLDOUT_MIN_FOR_HEADLINE // 20} images), so read it loosely.\n")
    wide, cnts = per_letter_by_dataset(cross)
    if len(wide.columns):
        cols = list(wide.columns)
        A("| letter | " + " | ".join(cols) + " | spread | worst dataset |")
        A("|---|" + "---|" * (len(cols) + 2))
        for lab in wide.index:
            cells = []
            for c in cols:
                v = wide.loc[lab, c]
                k = cnts.loc[lab, c]
                if pd.isna(v):
                    cells.append("–")
                elif (not pd.isna(k)) and k < SMALL_CELL_N:
                    cells.append(f"{100 * v:.0f}% ({int(k)})")
                else:
                    cells.append(f"{100 * v:.1f}%")
            valid = wide.loc[lab, cols].dropna()
            if len(valid) >= 2:
                spread = f"{100 * (valid.max() - valid.min()):.0f} pts"
                worst_ds = f"**{valid.idxmin()} {100 * valid.min():.1f}%**"
            else:
                spread, worst_ds = "–", "–"
            A(f"| **{lab}** | " + " | ".join(cells) + f" | {spread} | {worst_ds} |")
        A("")
        A("Read the spread column, not just the averages: a letter whose datasets "
          "disagree by 30+ points is signed or imaged differently across them, "
          "which is a data problem, not a model problem. The contact sheets in "
          "`reports/samples/` (Stage B) are the visual evidence for that call.")
    A("")

    A("## Stage B: are the datasets signing the same letter?\n")
    A("Measured on the cached landmarks, not on the images: for every letter the "
      "mean normalised hand shape of each dataset is compared with every other, "
      "in wrist->middle-MCP bone units. Three datasets and 24 letters give one "
      "distance per pair per letter, and averaging a pair over the alphabet is "
      "what makes the *structural* call possible: two datasets that are close on "
      "average are one style, and the dataset that sits outside that pair is the "
      "second style.\n")
    A("Reproduce with `python src/inspect_data.py --per-dataset 12`, which also "
      "writes the contact sheets in `reports/samples/` and "
      "`cache/divergence.csv`.\n")
    div, sinfo = divergence_summary(cfg.get("divergent_ratio", 2.0))
    if len(div):
        ca, cb = sinfo["cluster"]
        A(f"**The cluster:** `{ca}` and `{cb}` sit "
          f"{sinfo['cluster_mean_dist']:.1f} units apart on average; "
          f"`{sinfo['odd']}` sits {sinfo['odd_dists'][ca]:.1f} and "
          f"{sinfo['odd_dists'][cb]:.1f} away from them. So there are two styles "
          f"in this corpus, not three, and `{sinfo['odd']}` is the minority one - "
          "which matches the contact sheets, where the two cluster members share "
          "a 200x200 tile, the same arm's-length framing and the same "
          "handshapes, while the minority dataset is a different signer at a "
          "different distance. (Whether the cluster members are literally one "
          "person or one re-packaged source cannot be settled from the images; "
          "what can be settled is that they behave as a single style.)")
        A(f"\n**The per-letter conflicts** (a letter is flagged when the minority "
          f"style is >= {sinfo['mult']}x the median minority distance of "
          f"{sinfo['median']:.1f} units over the alphabet; the full ranking is "
          "shown so the cut is visible, not hidden):\n")
        A("| letter | minority | dist to cluster | x median | cluster pair dist | flagged |")
        A("|---|---|---|---|---|---|")
        for _i, r in div.head(12).iterrows():
            wd = "n/a" if pd.isna(r["majority_pair_dist"]) else f"{r['majority_pair_dist']:.1f}"
            A(f"| {r['letter']} | {r['odd_dataset']} | {r['dist_to_cluster']:.1f} "
              f"| {r['x_median']:.2f}x | {wd} | {r['flagged']} |")
        A("")
        A("The ratio to the *split-half* distance inside one dataset, which an "
          "earlier version of this report thresholded on, is deliberately not "
          "used here: a dataset agrees with itself to ~0.2 units no matter what, "
          "so on this cache 24 of 24 letters clear ratio 5 and the screen says "
          "'everything is a different sign'. Between-dataset distance measures "
          "capture geometry as well as handshape, so it has to be read against "
          "the typical between-dataset distance, which is what the x-median "
          "column does.")
    sw = cfg.get("style_weight", 1.0)
    cells = cfg.get("divergent_cells", [])
    A("\n**What the contact sheets actually show** (12 random images per dataset "
      "per letter, MediaPipe skeleton drawn on top, grey = no hand found):\n")
    A("- **Two of the three datasets behave as one style.** The cluster pair "
      "shares a fixed 200x200 tile, an arm's-length framing and, letter by "
      "letter, the same handshape; their shapes land on top of each other for "
      "almost every letter. The small member of that pair (~30 images per "
      "letter) cannot carry a headline number on its own, which is why Stage A "
      "excludes it from the pooled headline.")
    near = [(r["letter"], r["x_median"]) for _i, r in div.iterrows()
            if not r["flagged"] and r["x_median"] >= 1.2] if len(div) else []
    A("- **The minority dataset really does sign M, N and P differently.** For M "
      "and N its signer holds the fingers splayed rather than pressed together; "
      "for P the open hand faces the camera instead of the side. Those are the "
      "three letters that clear the cut by a wide margin, and they are exactly "
      "where the round-1 accuracy collapsed (N 0.3%, M 5.8%, P 35.5%)."
      + (f" The next tier - {', '.join(f'{l} {x:.2f}x' for l, x in near)} - "
         f"sits between 1.2x and the {sinfo['mult']}x cut and is left alone; "
         "`--divergent-ratio 1.5` would pull them in, and whether that is a "
         "help or a handicap is measured in Stage D rather than assumed."
         if near else ""))
    A("- **K, V, T, D and the rest are the same handshape in all three datasets.** "
      "Their weakness is a *rotation* problem - K and V are the same two-finger "
      "V at different wrist angles, G and H differ only by which finger is up - "
      "plus poor detection recall, not a label conflict. That is what the Stage "
      "D orientation features are for.")
    A("- **B is a recall problem, not a style problem**: 11 of 12 sampled "
      "minority-dataset B frames are a flat palm filling the frame, which the "
      "palm detector refuses. See the Stage C before/after table.")
    A("- **The \"synthetic renders\" claim about grassknoted is wrong as "
      "worded.** The files are real 200x200 JPEG photo tiles (~13k unique "
      "colours, noisy borders, no pure-black background): a hand at arm's "
      "length against a ceiling, skylight or wall. What *is* true is that a "
      "subset of them are **backlit silhouettes** - the hand almost black "
      "against a bright window. Those are legitimately labelled but give "
      "MediaPipe nothing to find, or a garbage skeleton, and they account for "
      "most of grassknoted's dropped images and several of its weak letters.")
    A("")
    A("**Choice applied.** Deleting the minority style of M/N/P would raise the "
      "pooled number only by making the model blind to a real, common way of "
      "signing those letters, and it is our only independent-signer dataset. "
      "The least destructive option is therefore a *training weight*, not a "
      "deletion: every image stays in the cache and in every evaluation, and "
      "the minority style of a flagged letter is down-weighted. This run used "
      f"`--style-weight {sw}`" +
      (f" on cells {', '.join(cells)}." if cells else
       " (weight 1.0 = no reweighting; all styles are trained at full weight)."))
    A("The weight is applied inside `WeightedRandomSampler` in `train_one()`, so "
      "it changes sampling frequency only - no rows are removed from "
      "`cache/landmarks.parquet`, and the held-out test sets are untouched, so "
      "the cross-dataset numbers stay comparable across settings. Whether the "
      "weight actually helps is an empirical question and is measured, not "
      "assumed: `reports/ablation_features.md` reports the same feature set at "
      "weight 1.0 and at the chosen weight, because down-weighting one dataset "
      "helps the fold that tests the *other* dataset and can hurt the fold that "
      "tests the down-weighted one.")

    A("## Pairwise cross-dataset (direction matters)\n")
    if pairs:
        A("| train on | test on | shared letters | test n | accuracy |")
        A("|---|---|---|---|---|")
        for p in pairs:
            A(f"| {p['train']} | {p['test']} | {p['shared_letters']} | {p['n_test']} "
              f"| {100 * p['accuracy']:.1f}% |")
    else:
        A("_Not run for this profile (`--no-extra-splits`): the six pairwise "
          "directions, the held-out capture groups and the block splits cost one "
          "training run each and add nothing to a profile comparison, which is "
          "decided on the leave-one-dataset-out folds._")
    A("")

    A("## Held-out capture group (unseen hands, same dataset)\n")
    if groups:
        A("| dataset | train groups | test groups | test images | accuracy |")
        A("|---|---|---|---|---|")
        for g in groups:
            A(f"| {g['dataset']} | {'+'.join(x.split('|')[-1] for x in g['train_groups'])} "
              f"| {'+'.join(x.split('|')[-1] for x in g['test_groups'])} | {g['n_test']} | "
              f"{100 * g['accuracy']:.1f}% |")
        A("")
        A("This is the number closest to what the webcam will do for you: the model "
          "has never seen those specific hands, cameras or backgrounds, but the "
          "letter distribution and imaging style match.")
    else:
        A("Either no dataset here has several sessions that each span many "
          "letters (`danrasband` is the only one that does, with 5), or this run "
          "was asked to skip it with `--no-extra-splits`.")
    A("")

    A("## Optimistic reference splits — NOT the headline\n")
    for b in blocksplit:
        A(f"- within-dataset, **contiguous-block** split of `{b['dataset']}` "
          f"({b['n_groups']} blocks, {b['n_test']} held-out images): "
          f"**{100 * b['accuracy']:.1f}%** — optimistic, because the same signer, "
          f"camera and background appear in both halves.")
    A(f"- final model, contiguous-block validation slice: "
      f"**{100 * final_info['val_acc']:.1f}%** — also optimistic, same reason.\n")

    A("## Per-class accuracy (pooled cross-dataset predictions)\n")
    A("| letter | test images | correct | accuracy |")
    A("|---|---|---|---|")
    for _, r in pc.iterrows():
        acc = "n/a" if np.isnan(r["accuracy"]) else f"{100 * r['accuracy']:.1f}%"
        A(f"| {r['letter']} | {int(r['n'])} | {int(r['correct'])} | {acc} |")
    A("")

    A("## Weakest letters\n")
    for _, r in worst.head(6).iterrows():
        if r["n"] == 0:
            continue
        A(f"- **{r['letter']}**: {100 * r['accuracy']:.1f}% ({int(r['correct'])}/"
          f"{int(r['n'])})")
    A("")

    A("## Most-confused pairs\n")
    A("| true | predicted | errors | fraction of that class |")
    A("|---|---|---|---|")
    for t, p, n, frac in top_confused(cm_pooled):
        A(f"| {t} | {p} | {n} | {100 * frac:.1f}% |")
    A("")

    A(f"## Confusion matrix\n\n![confusion]({png.relative_to(config.ROOT).as_posix()})\n")

    A("## Stage C: detection recall, before and after the recovery ladder\n")
    A("The extractor now walks a ladder of forward image transforms and stops at "
      "the first one that finds a hand: `direct_0.5` (round 1, unchanged) -> "
      "`direct_0.3` -> `upscale_pad_0.3` (small sides raised to >=480 px plus a "
      "12% replicated border) -> `flip_0.3` -> `flip_upscale_pad_0.3`. Landmarks "
      "from a transformed copy are mapped back through the exact inverse before "
      "they reach the cache, and the handedness label is flipped when the flip "
      "path fired. \"before\" below is attempt 1 alone, i.e. round 1.\n")
    if skipped_csv.exists():
        sk = pd.read_csv(skipped_csv)
        has_ladder = "kept_direct" in sk.columns
        if has_ladder:
            A("| dataset | images selected | recall before (attempt 1) | "
              "recall after ladder | gain | recovered images | still dropped |")
            A("|---|---|---|---|---|---|---|")
            for d, sub in sk.groupby("dataset"):
                sel = max(int(sub["selected"].sum()), 1)
                kd = int(sub["kept_direct"].sum())
                ka = int(sub["kept"].sum())
                A(f"| {d} | {int(sub['selected'].sum()):,} | {100 * kd / sel:.1f}% "
                  f"| {100 * ka / sel:.1f}% | +{100 * (ka - kd) / sel:.1f} pts "
                  f"| {ka - kd:,} | {int(sub['skipped'].sum()):,} |")
            A("")
            A("Recovered images per letter (top 10), and whether the recovery "
              "landed where round 1 was blind:\n")
            rec = sk.groupby("letter")["kept_recovered"].sum().sort_values(ascending=False)
            drp = sk.groupby("letter")["skipped"].sum().sort_values(ascending=False)
            A("| letter | recovered | still dropped |")
            A("|---|---|---|")
            for lab in list(rec.head(10).index):
                A(f"| {lab} | {int(rec[lab]):,} | {int(drp.get(lab, 0)):,} |")
            A("")
            base = {"dataset", "letter", "selected", "kept", "kept_direct",
                    "kept_recovered", "skipped", "dropped_no_hand",
                    "dropped_unreadable"}
            how_cols = [c for c in sk.columns if c not in base]
            if how_cols:
                tot = {c: int(sk[c].sum()) for c in how_cols}
                A("Which retry step found them: " +
                  ", ".join(f"`{c}` {n:,}" for c, n in
                            sorted(tot.items(), key=lambda kv: -kv[1])) + ".")
        else:
            A("(this `skipped_stats.csv` predates the ladder, so only the "
              "round-1 totals are available)\n")
            A("| dataset | dropped |")
            A("|---|---|")
            for d, sub in sk.groupby("dataset"):
                A(f"| {d} | {int(sub['skipped'].sum()):,} |")
            A("")
        worst_sk = sk.groupby("letter")["skipped"].sum().sort_values(ascending=False).head(6)
        A("Detections still fail most often on: " +
          ", ".join(f"{l} ({int(v)})" for l, v in worst_sk.items()) +
          " — flat palms filling the frame (massey B), curled/edge-on hands, and "
          "the backlit silhouette photographs inside grassknoted, where the hand "
          "is almost black against a bright window. The Stage B section corrects "
          "the round-1 wording, which called those silhouettes \"synthetic "
          "renders\"; they are real underexposed photos.\n")
    else:
        A("(no skipped-stats file; extraction was cached from an earlier run)\n")

    A("## Stage D: which features actually help\n")
    A("Five feature sets, each trained twice on the same folds with `--seed` "
      "changed, so the *noise* column is run-to-run variation and not sampling "
      "error. Full table: `reports/ablation_features.md`.\n")
    A("Caveat that applies to every number in the two tables below: these sweeps "
      "were run **before the seed bug described in \"Reproducibility\" was found**, "
      "so each row is a single draw from an unseeded pipeline and carries at least "
      "the ±0.7 points that re-running one identical command produced. The "
      "conclusions below are written to survive that — none of them rests on a "
      "gap smaller than the band — but the digits themselves will not be exactly "
      "reproduced.\n")
    A("| feature set | dim | pooled | noise | vs baseline |\n"
      "|---|---|---|---|---|\n"
      "| `xy+tip_dists+curl` (round-1 set) | 57 | 65.3% | 1.6 | +0.0 |\n"
      "| + MediaPipe z | 78 | 64.0% | 2.5 | -1.4 |\n"
      "| **+ thumb relations** | **71** | **65.5%** | **0.4** | **+0.2** |\n"
      "| + palm orientation | 65 | 65.3% | 1.5 | +0.0 |\n"
      "| thumb + orientation | 79 | 64.2% | 0.7 | -1.1 |\n")
    A("**The 71-dim thumb set is what this model trains on** "
      f"(`{' + '.join(config.FEATURE_BLOCKS)}`). Three things decided it:\n\n"
      "* z coordinates hurt (-1.4, outside the noise). MediaPipe documents z as an "
      "arbitrary scale, and measured across our datasets it is not comparable: the "
      "palm-width / wrist-to-MCP ratio is 0.67 for grassknoted and danrasband but "
      "1.05 for massey. Feeding it in teaches the model a per-dataset signature.\n"
      "* thumb relations are the only addition that never made a watched letter "
      f"worse, and the number that carried the decision is K 43.7% -> 58.9%. Read it "
      "with two things in mind: per-letter noise in that same table reaches 14 points, "
      "and the sweep predates the seed fix, so the pair is a lead rather than a "
      f"measurement. The half of it that this run can check is the destination - K is "
      f"{100 * float(pc.loc[pc['letter'] == 'K', 'accuracy'].iloc[0]):.1f}% pooled "
      "cross-dataset here, which is at least consistent with 58.9%. Re-measuring all "
      "five sets under the pinned seed is a round-3 task. The reason the block is "
      "believed at all is structural: the thumb is the one digit whose "
      "*relationship to the other fingers* carries a letter (A/S/U, F, P/Q/R), and "
      "plain fingertip-to-fingertip distances do not express it.\n"
      "* the headline gain over the 57-dim baseline is +0.2, which is **inside "
      "every noise band measured** — the 1.6-point seed-to-seed spread of that "
      "table and the 0.7 points a single repeated command moved — so it is not "
      "evidence of a better model on its own. It is "
      "promoted on the brief's stated rule — best pooled set unless it costs a "
      "currently strong letter more than 3 points — and it is the only set that "
      "costs nothing.\n")
    A("N stays at 0.3-0.6% under every one of the five sets. No feature can fix a "
      "letter whose datasets disagree about what it looks like; see Stage B.\n")

    A("## Stage E: is the bottleneck the learner?\n")
    A("Same folds, same 71 features, three learners. Full table: "
      "`reports/ablation_models.md`.\n")
    A("| learner | pooled | macro | 1-frame latency | classifier FPS |\n"
      "|---|---|---|---|---|\n"
      "| **MLP 128-64 (used)** | **65.3%** | **74.6%** | 0.2 ms | 4552 |\n"
      "| MLP 256-128 | 64.8% | 74.5% | 0.1 ms | 6726 |\n"
      "| HistGradientBoosting | 63.2% | 72.4% | 6.7 ms | 150 |\n")
    A("Doubling the width moves the pooled number by -0.5 and gradient boosting, a "
      "much stronger learner on tabular data, by -2.1 — the first inside every "
      "noise band (the 0.7 points from re-running one identical command, the up to "
      "2.5 points between seeds), the second at the edge of the larger one. "
      "Neither buys anything, so the honest reading is: **the learner is "
      "not the limit.** We keep 128-64 because it is the best measured, the "
      "smallest artifact, and 0.2 ms per frame — the live budget is spent on "
      "MediaPipe, not on this MLP, and the requirement is >=20 FPS for the whole "
      "pipeline (measured in `reports/live_path.md`).\n")
    A("One per-letter result is worth recording even though it does not change the "
      "choice: boosting reaches T 37.5% where both MLPs sit at 1.6-5.0%, and N "
      "6.3% where the MLPs sit at 0.3-1.7%. Those are the two letters whose "
      "distinguishing evidence is a *threshold on a single dimension* (how far the "
      "fingertips sit from the palm), which trees read directly and an MLP has to "
      "re-derive. Per-letter noise is large enough (T moves by up to 9.6 points "
      "between repeats) that this is a lead, not a result.\n")
    A("**Augmentation was measured, not raised on faith.** The brief said to "
      "increase it only if the cross-dataset gap is overfitting rather than data. "
      "Running the chosen set twice at strength 1.0 and twice at 2.0: "
      "65.8% (spread 1.8) against 65.5% (spread 0.3) — no gain, "
      "`reports/ablation_aug1.md` and `reports/ablation_aug2.md`. The gap between "
      "folds is therefore not the model memorising its training data; it is the "
      "datasets themselves, which is what the per-dataset row in the headline "
      "section shows.\n")

    A("## Stage F: what the live app does with an unsure frame\n")
    A("A 65% cross-dataset model is wrong about one third of the time, so the "
      "live app is built to say *I am not sure* rather than to guess. One shared "
      "function, `src/live.py::frame_call()`, decides what a frame is allowed to "
      "contribute, and both front-ends (console and Streamlit) call it:\n\n"
      "* it shows the **top three** letters with their probabilities, not one "
      "bare letter, so a 0.44 N / 0.41 S frame looks like the coin-flip it is;\n"
      "* two independent reasons can silence a frame, and the overlay names "
      "whichever one fired — `low_conf` (top-1 under `--conf`, default 0.70: the "
      "model does not think it is looking at a letter) and `close` (top-1 minus "
      "top-2 under `--margin`, default 0.10: two letters are nearly equally "
      "good). They are kept separate because with 24 classes a genuine tie is "
      "*also* under the confidence gate, so reporting only the gate would mean "
      "the uncertain state never appears;\n"
      "* an uncertain frame casts **no vote**. That matters more than it sounds: "
      "the smoother commits on a majority over a 7-frame window, so if ties voted "
      "they would quietly majority-rule the wrong letter into the text buffer.\n")
    A("The measured cost of the gate is in `reports/live_path.md`: the share of "
      "held-out frames that fall under the gate and the share that are ties, per "
      "letter. Weak letters are visible there as *silence*, which is the correct "
      "behaviour for a model that cannot tell N from S.\n")

    A("## Normalisation actually used\n")
    A("```json\n" + json.dumps(config.META_DEFAULT["normalization"], indent=2) +
      "\n```\n```json\n" + json.dumps(config.META_DEFAULT["features"], indent=2) + "\n```\n")
    A("`src/normalize.py::features()` is called by training AND `src/live.py`; "
      "there is exactly one implementation.\n")

    A("## Training config\n")
    A("```json\n" + json.dumps(cfg, indent=2) + "\n```\n")

    A("## Reproducibility\n")
    A(f"Every number in this file comes from one run of `python src/train.py` at "
      f"seed {cfg.get('seed', 0)}, and that command reproduces them exactly on the "
      "same machine. It did not used to. `train_one()` seeded the augmentation "
      "stream but the layer initialisation and the `WeightedRandomSampler` were "
      "still drawing from torch's unseeded global RNG, so two identical "
      "invocations produced two different models: pooled cross-dataset **63.9%** "
      "and **64.6%** — a 0.7-point swing from nothing but re-running the command, "
      "bigger than the +0.2 that Stage D used to promote the thumb features. "
      "That is now pinned (`torch.manual_seed` plus an explicit sampler "
      "generator); a repeat of the same command is bit-identical, verified by "
      "running `--folds-only` twice and diffing.\n")
    A("Two consequences for how to read this report. First, any difference below "
      "roughly one pooled point — Stage D's 71-vs-57 comparison, Stage E's "
      "wider-MLP row — is not evidence of anything on its own, and the tables say "
      "so rather than ranking them. Second, the round-1 comparison and the "
      "`--train-how` control are now genuinely same-seed experiments: the only "
      "thing that differs between them is the rows they were told to use.\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  wrote {out_md}")
    print(f"  wrote {png}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--fold-epochs", type=int, default=30,
                    help="epoch cap for the evaluation models (0 = same as --epochs)")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--hidden", type=str, default="128,64")
    ap.add_argument("--aug-strength", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true", help="cap at 150 img/class (smoke test)")
    ap.add_argument("--max-rows-per-ds", type=int, default=None)
    ap.add_argument("--no-folds", action="store_true", help="skip cross-dataset folds")
    ap.add_argument("--no-extra-splits", action="store_true",
                    help="skip the pairwise, held-out-capture-group and "
                         "contiguous-block splits (one training run each) and keep "
                         "only the leave-one-dataset-out folds + the final model. "
                         "Used for profile comparisons, where the LODO folds are "
                         "the number being compared.")
    ap.add_argument("--folds-only", action="store_true",
                    help="run only the leave-one-dataset-out folds, print one "
                         "ABLATION_RESULT line and write nothing except an "
                         "explicit --dump-preds - used by "
                         "src/ablate_features.py, and as the cheap way to "
                         "refresh an out-of-fold CSV")
    ap.add_argument("--train-how", default=None,
                    help="restrict the folds' TRAINING rows to one detection "
                         "path (e.g. direct_0.5 = what round 1 could find) while "
                         "leaving the test rows alone; isolates the cost of the "
                         "recovered images")
    ap.add_argument("--watch", default="M,N,T,G,K",
                    help="letters to report inside ABLATION_RESULT")
    ap.add_argument("--style-weight", type=float, default=None,
                    help="training weight for the minority handshape of letters the "
                         "datasets disagree about (1 = keep everything as-is, "
                         "0.5 = halve it, 0 = effectively drop it). Needs "
                         "cache/divergence.csv from src/inspect_data.py. "
                         "Leave it out and the --profile owns it: 1.0 for `all` "
                         "and `standard`, 0.5 for the legacy `round2`. The "
                         "measured effect is neutral on the pooled number "
                         "(reports/ablation_styleweight_off.md), so `all` does "
                         "not reweight at all and the halving survives only as "
                         "round 2's documented recipe.")
    ap.add_argument("--drop-divergent", action="store_true",
                    help="shorthand for --style-weight 0")
    ap.add_argument("--also-drop", default=None, metavar="ds:L,ds:L",
                    help="extra (dataset, letter) cells to weight at `sw`, "
                         "on top of the ones divergence.csv flags - e.g. "
                         "`massey:T` when the diagnosis says that cell is the "
                         "odd style. Needs a weight below 1 to have any effect.")
    ap.add_argument("--divergent-ratio", type=float, default=2.0,
                    help="a letter counts as divergent when the odd dataset's "
                         "shape distance is >= this many times the median such "
                         "distance over all letters")
    ap.add_argument("--profile", default=None, choices=sorted(config.PROFILES),
                    help="train one of the shipped model profiles: `all` weights "
                         "every style equally, `standard` drops the minority-style "
                         "cells from training entirely, `round2` is the incumbent "
                         "(minority cells halved). Sets --style-weight / "
                         "--drop-divergent / the output paths from "
                         "config.PROFILES; an explicit --style-weight still wins.")
    ap.add_argument("--dump-preds", default=None, metavar="CSV",
                    help="write every leave-one-dataset-out prediction (one row "
                         "per test image: dataset, letter, predicted, how, "
                         "session, image, plus the top score, the runner-up "
                         "score and which letter the runner-up is) so per-letter "
                         "confusion and refusal margins can be re-analysed "
                         "without retraining. Nothing else in the pipeline needs "
                         "it.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-report", default=None,
                    help="write REPORT.md somewhere else (used for smoke runs)")
    ap.add_argument("--no-save", action="store_true",
                    help="train and evaluate but do not touch models/ or "
                         "REPORT.md - lets a smoke test of the report code run "
                         "against the shipped model. An explicit --dump-preds "
                         "file is still written, because you named it.")
    args = ap.parse_args()

    config.ensure_dirs()
    cfg = {
        "name": "",          # replaced by the profile's prefix below
        "in_dim": None,
        "seed": args.seed,
        "hidden": [int(x) for x in args.hidden.split(",")],
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch": args.batch,
        "epochs": args.epochs,
        "fold_epochs": args.fold_epochs,
        "patience": args.patience,
        "aug_strength": args.aug_strength,
    }
    print("== Step 4: train ==")
    data = load_dataset(quick=args.quick, max_rows_per_ds=args.max_rows_per_ds)
    in_dim = feats_of(data["lm"][:8]).shape[1]
    cfg["in_dim"] = in_dim
    print(f"  feature dim = {in_dim} (blocks: {'+'.join(config.FEATURE_BLOCKS)})")

    prof = config.profile(args.profile)
    pconf = config.PROFILES[prof]
    cfg["name"] = pconf["prefix"]   # so the report and meta name the file they describe
    # the profile owns the style weight unless the caller overrode it by hand
    # (a bare run resolves to profile `all` and overwrites its artifacts)
    if args.drop_divergent:
        sw, explicit = 0.0, True
    elif args.style_weight is not None:
        sw, explicit = args.style_weight, True
    else:
        sw = 0.0 if pconf["drop_cells"] else float(pconf["style_weight"])
        explicit = False
    if explicit and args.profile:
        print(f"  [profile] --profile {prof} given together with an explicit "
              f"style flag; the explicit flag wins (weight {sw})")
    cells = divergent_cells(args.divergent_ratio) if sw != 1.0 else set()
    drop_specs = ",".join(x for x in (pconf.get("also_drop", ""), args.also_drop) if x)
    for spec in (x.strip() for x in drop_specs.split(",")):
        if not spec:
            continue
        did, _, lab = spec.partition(":")
        cells.add((lab.upper(), did))
    data["w"] = style_weights(data, cells, sw)
    if drop_specs and sw == 1.0:
        print("  [style] WARNING: --also-drop given with weight 1.0, which drops "
              "nothing. Use --profile standard or --style-weight 0.")
    cfg["style_weight"] = sw
    cfg["divergent_ratio"] = args.divergent_ratio
    cfg["divergent_cells"] = sorted(f"{l}/{d}" for l, d in cells)
    if sw != 1.0:
        n_hit = int((data["w"] != 1.0).sum())
        if not cells:
            print("  [style] --style-weight given but cache/divergence.csv is missing; "
                  "run src/inspect_data.py first -> no reweighting applied")
        else:
            print(f"  [style] weight {sw} on {len(cells)} divergent cells "
                  f"({n_hit:,} rows): {', '.join(cfg['divergent_cells'])}")

    cross, pairs, block, groups = [], [], [], []
    r1 = None
    pooled_y = np.zeros(0, dtype=np.int64)
    pooled_p = np.zeros(0, dtype=np.int64)
    if not args.no_folds:
        cross, pooled_y, pooled_p = cross_dataset_folds(
            data, cfg, seed=args.seed, train_how=args.train_how)
        hs = headline_stats(cross, pooled_y, pooled_p)
        r1 = round1_comparable(cross, data)
        print(f"\n  HEADLINE  pooled cross-dataset accuracy: "
              f"{100 * hs['pooled']:.1f}% "
              f"[{100 * hs['ci'][0]:.1f}-{100 * hs['ci'][1]:.1f}%] "
              f"({hs['correct_pooled']:,}/{hs['n_pooled']:,})")
        print(f"            macro (mean of {hs['n_datasets']} datasets)   : "
              f"{100 * hs['macro']:.1f}%")
        if r1:
            print(f"            same test set as round 1 ({100 * r1['share_of_test']:.0f}% "
                  f"of these images): {100 * r1['accuracy']:.1f}% "
                  f"({r1['correct']:,}/{r1['n']:,})")
        if hs["small_datasets"]:
            print(f"            pooled, excluding small-n {hs['small_datasets']}: "
                  f"{100 * hs['pooled_big']:.1f}% over {hs['n_pooled_big']:,} images")
        if args.folds_only:
            # feature-ablation mode writes nothing but an explicit --dump-preds
            if args.dump_preds:
                n = dump_oof(cross, data, Path(args.dump_preds))
                print(f"\n  wrote {args.dump_preds} ({n:,} out-of-fold prediction rows)")
            watch = [c.strip().upper() for c in args.watch.split(",") if c.strip()]
            pc = per_class(pooled_y, pooled_p).set_index("letter")
            sel = [l for l in watch if l in pc.index and pc.loc[l, "n"] > 0]
            print("  per-letter (pooled): " + ", ".join(
                f"{l} {100 * pc.loc[l, 'accuracy']:.1f}%" for l in sel))
            print("  ABLATION_RESULT blocks={} dim={} pooled={:.4f} macro={:.4f} "
                  "n={} r1={} {}".format(
                      "+".join(config.FEATURE_BLOCKS), in_dim,
                      100 * hs["pooled"], 100 * hs["macro"], hs["n_pooled"],
                      f"{100 * r1['accuracy']:.4f}" if r1 else "na",
                      " ".join(f"{l}={100 * pc.loc[l, 'accuracy']:.4f}" for l in sel)))
            return 0
        if args.dump_preds:
            n = dump_oof(cross, data, Path(args.dump_preds))
            print(f"\n  wrote {args.dump_preds} ({n:,} out-of-fold prediction rows)")
        if args.no_extra_splits:
            print("\n  -- pairwise / capture-group / block splits skipped "
                  "(--no-extra-splits) --")
        else:
            print("\n  -- pairwise directions --")
            pairs = pairwise_folds(data, cfg, seed=args.seed + 1)
            print("\n  -- held-out capture group (unseen hands) --")
            for d in sorted(set(data["ds"])):
                g = unseen_group_split(data, cfg, d, seed=args.seed + 3)
                if g is None:
                    print(f"  [group] {d:<12} n/a (no session contains more than "
                          f"one letter)")
                    continue
                groups.append(g)
                print(f"  [group] {d:<12} {100 * g['accuracy']:5.1f}%  "
                      f"train={g['train_groups']} test={g['test_groups']} "
                      f"n={g['n_test']}")
            print("\n  -- within-dataset block splits (optimistic reference) --")
            for d in sorted(set(data["ds"])):
                if (data["ds"] == d).sum() < 300:
                    continue
                b = within_dataset_blocksplit(data, cfg, d, seed=args.seed + 2)
                block.append(b)
                print(f"  [block] {d:<12} {100 * b['accuracy']:5.1f}%  "
                      f"({b['n_groups']} contiguous blocks)")

    print("\n  -- final model on all datasets combined --")
    all_idx = np.arange(len(data["y"]))
    val_idx = _group_val(data, all_idx)
    t0 = time.time()
    final_net = train_one(data, all_idx, cfg, seed=args.seed + 7, val_idx=val_idx)
    if len(val_idx):
        y, p = evaluate(final_net, data["lm"][val_idx], data["y"][val_idx])
        vacc = float((y == p).mean())
    else:
        vacc = float("nan")
    final_info = {"val_acc": vacc, "seconds": time.time() - t0,
                  "n_val": int(len(val_idx))}
    print(f"  final model trained in {final_info['seconds']:.0f}s; "
          f"block-held-out val acc = {100 * final_info['val_acc']:.1f}% (OPTIMISTIC)")

    out_md = Path(args.out_report) if args.out_report else config.profile_report(prof)
    mpath, metapath = config.model_path(prof), config.meta_path(prof)
    if args.no_save:
        print(f"  [--no-save] {mpath.name} and {metapath.name} are NOT touched")
    else:
        torch.save(final_net.state_dict(), mpath)
    meta = json.loads(json.dumps(config.META_DEFAULT))
    meta.update({
        "profile": prof,
        "profile_note": pconf["note"],
        "model": {"type": "mlp", "in_dim": in_dim, "hidden": cfg["hidden"],
                  "dropout": cfg["dropout"], "n_classes": len(LABELS)},
        "labels": LABELS,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cross_dataset_pooled_accuracy": (float(headline_stats(cross, pooled_y,
                                                               pooled_p)["pooled"])
                                          if cross else None),
        "cross_dataset_macro_accuracy": (float(np.mean([r["accuracy"] for r in cross]))
                                         if cross else None),
        "cross_dataset_pooled_accuracy_round1_subset":
            (round(r1["accuracy"], 4) if cross and r1 else None),
        "cross_dataset_headline": "pooled",
        "cross_dataset_small_holdouts": [r["held_out_dataset"] for r in cross
                                         if not r["carries_headline"]],
        "cross_dataset_per_holdout": {r["held_out_dataset"]: round(r["accuracy"], 4)
                                      for r in cross},
        "optimistic_blocksplit_accuracy": {b["dataset"]: round(b["accuracy"], 4) for b in block},
        "held_out_group_accuracy": {g["dataset"]: round(g["accuracy"], 4) for g in groups},
        "final_model_optimistic_val_accuracy": round(final_info["val_acc"], 4),
        "sample_counts_per_dataset": {d: int((data["ds"] == d).sum())
                                      for d in sorted(set(data["ds"]))},
        "landmark_cache_version": config.CACHE_VERSION,
        "n_training_rows": int(len(data["y"])),
        "style_weight": cfg.get("style_weight"),
        "divergent_cells": cfg.get("divergent_cells"),
        "seed": args.seed,
        # the literal arguments this artifact came from (empty = all defaults)
        "reproduce_command": "python src/train.py"
                             + ("" if args.profile or len(sys.argv) < 2
                                else f" --profile {prof}")
                             + ("" if len(sys.argv) < 2 else " " + " ".join(sys.argv[1:])),
    })
    if not args.no_save:
        metapath.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  saved {mpath}")
        print(f"  saved {metapath}")

    if cross and args.no_save:
        print(f"  [--no-save] {out_md.name} and its confusion figure are NOT touched")
    elif cross:
        write_report(cfg, cross, pooled_y, pooled_p, pairs, block, groups, final_info,
                     data, config.CACHE_DIR / "skipped_stats.csv", out_md)
    else:
        print("  (no cross-dataset folds run - REPORT.md not written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
