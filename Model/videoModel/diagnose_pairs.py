"""Stage-A diagnostics on cache/landmarks, manifest and predictions; writes nothing.

The AUC columns are a linear probe - directional evidence on a few dozen
clips, not a verdict. See README for flags.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, extract, features, landmarks  # noqa: E402

BG = (config.REST_LABEL, config.OTHER_LABEL)


def segment(video: str, s: float, e: float) -> np.ndarray | None:
    """The trimmed signing segment, exactly as the REST/word split sees it."""
    p = extract.landmark_file(video)
    if not p.is_file():
        return None
    arr = np.load(p).astype(np.float32)
    if not len(arr):
        return None
    a, b = int(max(0, s)), int(min(len(arr), e if e > s else len(arr)))
    return arr[a:b] if b > a else arr


def hand_rate(seg: np.ndarray) -> float:
    """Fraction of frames with at least one wrist - the segment, not the clip."""
    if not len(seg):
        return float("nan")
    l = seg[:, config.HAND_WRIST_COLS[0], :2]
    r = seg[:, config.HAND_WRIST_COLS[1], :2]
    return float((~(np.isnan(l).any(1) & np.isnan(r).any(1))).mean())


def measures(seg: np.ndarray) -> dict:
    """Wrist geometry in shoulder widths, y up, origin at the shoulder line.

    's/m/e' = first/middle/last frame the hand is seen; 'a' = closest to the mouth.
    """
    out: dict[str, float] = {}
    if len(seg) < 3:
        return out
    sh = np.array([landmarks.shoulder_mid(f) for f in seg])
    w = np.linalg.norm(seg[:, config.POSE_L_SHOULDER, :2]
                       - seg[:, config.POSE_R_SHOULDER, :2], axis=1)
    scale = np.where(w > 1e-6, w, np.nan)
    mo = np.array([(f[config.POSE_MOUTH_L, :2] + f[config.POSE_MOUTH_R, :2]) / 2
                   for f in seg]) - sh
    if not np.isfinite(scale).all():
        return out
    for name, col in (("L", config.HAND_WRIST_COLS[0]),
                      ("R", config.HAND_WRIST_COLS[1])):
        xy = (seg[:, col, :2] - sh) / scale[:, None]
        xy = np.column_stack([xy[:, 0], -xy[:, 1]])
        pres = float(np.mean(~np.isnan(xy[:, 0])))
        out[f"{name}_pres"] = pres
        if pres == 0:
            continue
        mouth = np.hypot(*(xy - mo).T)
        idx = np.where(~np.isnan(xy[:, 0]))[0]
        near = idx[int(np.argmin(np.where(np.isnan(mouth), np.inf,
                                          mouth)[idx]))]
        out[f"{name}_mouth_a"] = float(mouth[near])
        out[f"{name}_y_a"] = float(xy[near, 1])
        out[f"{name}_mouth_min"] = float(np.nanmin(mouth))
        for tag, k in (("s", idx[0]), ("m", idx[len(idx) // 2]), ("e", idx[-1])):
            out[f"{name}_y_{tag}"] = float(xy[k, 1])
            out[f"{name}_mouth_{tag}"] = float(mouth[k])
        if len(idx) >= 4:
            out[f"{name}_dx"] = float(xy[idx[-1], 0] - xy[idx[0], 0])
            out[f"{name}_dy"] = float(xy[idx[-1], 1] - xy[idx[0], 1])
            out[f"{name}_travel"] = float(np.hypot(out[f"{name}_dx"],
                                                   out[f"{name}_dy"]))
            # travel after the mouth-closest frame: where GOOD and THANKS should differ
            tail = idx[idx >= near]
            if len(tail) >= 2:
                out[f"{name}_away"] = float(mouth[tail[-1]] - mouth[near])
                out[f"{name}_dy_after"] = float(xy[tail[-1], 1] - xy[near, 1])
                out[f"{name}_travel_after"] = float(np.hypot(
                    xy[tail[-1], 0] - xy[near, 0],
                    xy[tail[-1], 1] - xy[near, 1]))
            out[f"{name}_range_y"] = float(np.nanmax(xy[:, 1])
                                           - np.nanmin(xy[:, 1]))
    lw = seg[:, config.HAND_WRIST_COLS[0], :2]
    rw = seg[:, config.HAND_WRIST_COLS[1], :2]
    both = ~(np.isnan(lw[:, 0]) | np.isnan(rw[:, 0]))
    out["ww_both"] = float(both.mean())
    if both.any():
        j = np.where(both)[0][-1]
        out["ww_end"] = float(np.linalg.norm(lw[j] - rw[j]) / scale[j])
    return out


def dominant(row: dict) -> str:
    """Which hand the signer actually used for this clip."""
    return "R" if (row.get("R_pres") or 0) >= (row.get("L_pres") or 0) else "L"


def _runs(mask: np.ndarray) -> list[int]:
    """Lengths of the True runs in a boolean vector."""
    out, t, T = [], 0, len(mask)
    while t < T:
        if not mask[t]:
            t += 1
            continue
        b = t
        while b + 1 < T and mask[b + 1]:
            b += 1
        out.append(b - t + 1)
        t = b + 1
    return out


def gap_report(man: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """What fill_gaps() recovers per word, plus a gap-length histogram.

    Many 1-3 frame holes are noise; a tail past GAP_FILL_FRAMES is a hand
    genuinely out of shot.
    """
    from collections import Counter
    hist: Counter = Counter()
    per: list[tuple] = []
    for r in man.itertuples(index=False):
        seg = cache.get(r.video)
        if seg is None or len(seg) < 5:
            continue
        for col in (config.COLS_LHAND, config.COLS_RHAND):
            raw = np.asarray(seg, np.float32)[:, col, :2]
            miss = np.isnan(raw).any(2).any(1)
            filled = features.fill_gaps(raw)
            seen_now = ~np.isnan(filled).any(2).any(1)
            for L in _runs(miss):
                hist[L if L <= 10 else "11+"] += 1
            per.append((r.word, float(miss.mean() - (~seen_now).mean()),
                        float(1.0 - miss.mean()), float(seen_now.mean())))
    d = pd.DataFrame(per, columns=["word", "recovered", "seen_raw",
                                   "seen_filled"])
    g = d.groupby("word").agg(clips=("recovered", "size"),
                              seen_raw=("seen_raw", "mean"),
                              seen_filled=("seen_filled", "mean"),
                              frames_back=("recovered", "mean"))
    g["any_bridge"] = d.groupby("word")["recovered"].apply(
        lambda x: float((x > 0).mean()))
    print(g.sort_values("frames_back", ascending=False).round(3).to_string())
    tot = sum(hist.values())
    print(f"\n  hand-absence runs, all clips ({tot} runs): "
          + "  ".join(f"{k}:{hist[k] * 100 / tot:.0f}%"
                      for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "11+"]
                      if hist.get(k)))
    filled_frac = sum(v for k, v in hist.items()
                      if isinstance(k, int) and k <= config.GAP_FILL_FRAMES)
    print(f"  filled by the {config.GAP_FILL_FRAMES}-frame rule: "
          f"{100 * filled_frac / max(tot, 1):.1f}% of all absences")
    return g


def rates(man: pd.DataFrame, cache: dict) -> pd.DataFrame:
    per = []
    for r in man.itertuples(index=False):
        seg = cache.get(r.video)
        if seg is None:
            continue
        per.append((r.word, float(r.hand_rate), hand_rate(seg), len(seg)))
    d = pd.DataFrame(per, columns=["word", "rate_full", "rate_seg", "n"])
    g = d.groupby("word").agg(clips=("rate_seg", "size"),
                              rate_full=("rate_full", "mean"),
                              rate_seg=("rate_seg", "mean"),
                              seg_below_50=("rate_seg", lambda x: (x < .5).mean()))
    g["delta"] = g["rate_seg"] - g["rate_full"]
    return g.sort_values("rate_seg").round(3)


def detection_ceiling(man: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Does the model fail because MediaPipe failed, or in spite of it?"""
    pred = pd.read_csv(config.REPORTS_DIR / "predictions.csv")
    pred["seg_rate"] = pred["video"].map({v: hand_rate(s)
                                          for v, s in cache.items()})
    pred["correct"] = pred["correct"].astype(bool)
    w = pred[~pred["label"].isin(BG) & pred["seg_rate"].notna()]
    rows = []
    for word, grp in w.groupby("label"):
        hi = grp[grp["seg_rate"] >= 0.8]
        lo = grp[grp["seg_rate"] < 0.8]
        rows.append((word, len(grp), float(grp["correct"].mean()),
                     len(hi), float(hi["correct"].mean()) if len(hi) else np.nan,
                     len(lo), float(lo["correct"].mean()) if len(lo) else np.nan,
                     float(grp["seg_rate"].mean())))
    return (pd.DataFrame(rows, columns=["word", "n", "acc", "n_hi", "acc_hi",
                                        "n_lo", "acc_lo", "seg_rate"])
            .sort_values("acc").round(3))


def probe(M: pd.DataFrame, y: np.ndarray, X: np.ndarray,
          labels: list[str]) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def auc(v):
        v = np.asarray(v, float)
        keep = ~np.isnan(v)
        if keep.sum() < 8 or len(np.unique(y[keep])) < 2:
            return float("nan"), int(keep.sum())
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y[keep], v[keep])), int(keep.sum())

    print("  one number at a time (AUC>0.5 = bigger for the SECOND word):")
    res = []
    for i, c in enumerate(labels):
        a, n = auc(M[c])
        if np.isfinite(a):
            res.append((abs(a - 0.5), c, a, n))
    for _s, c, a, n in sorted(res, reverse=True)[:10]:
        print(f"    {c:16s} AUC {a:5.3f}  n={n:3d}")

    def logit(Z):
        Z = np.asarray(Z, float)
        if not np.isfinite(Z).all() or len(np.unique(y)) < 2:
            return float("nan")
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        try:
            return float(cross_val_score(LogisticRegression(max_iter=5000), Z,
                                         y, cv=4, scoring="roc_auc").mean())
        except ValueError:
            return float("nan")

    print(f"    pooled {X.shape[1]}-dim mean  AUC {logit(X):.3f}")
    for block in features.BLOCK_NAMES:
        cols = features.block_columns([block])
        print(f"    block {block:14s} {len(np.atleast_1d(cols)):3d}d  "
              f"AUC {logit(X[:, cols]):.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--words", default="good,thanks,happy,please",
                    help="comma separated; the first two are probed against "
                         "each other")
    ap.add_argument("--splits", default="val,test",
                    help="held-out signers by default; training signers only "
                         "mislead here")
    ap.add_argument("--all-rates", action="store_true",
                    help="just the hand-rate table for every word")
    ap.add_argument("--gaps", action="store_true",
                    help="only the dropout histogram: how much of a hand's "
                         "absence is detector noise (the Stage B1 evidence)")
    ap.add_argument("--n", type=int, default=0, help="cap clips per word")
    a = ap.parse_args(argv)
    splits = tuple(s.strip() for s in a.splits.split(",") if s.strip())
    words = [w.strip().lower() for w in a.words.split(",") if w.strip()]

    man = pd.read_csv(config.CACHE_DIR / "manifest.csv")
    man = man[(man["status"] == "ok") & man["split"].isin(splits)].copy()
    print(f"{len(man)} clips in {splits}, reading the cache (no re-extraction)"
          f" ... ")
    cache = {}
    for r in man.itertuples(index=False):
        seg = segment(r.video, r.sign_start, r.sign_end)
        if seg is not None:
            cache[r.video] = seg

    print("=== 1. hand detection: the manifest column counts the whole clip, "
          "the segment is what matters")
    print(rates(man, cache).to_string())
    if a.all_rates:
        return 0
    if a.gaps:
        print("\n=== 2b. what bridging a short hand dropout buys back")
        gap_report(man, cache)
        return 0

    print("\n=== 2. is detection the ceiling, or is it the model? "
          "(acc = accuracy on the test signers)")
    print(detection_ceiling(man, cache).to_string(index=False))

    focus = words[:2]
    print(f"\n=== 3. what separates {focus[0]} from {focus[1]} "
          "on held-out signers")
    rows = []
    feats = []
    for r in man.itertuples(index=False):
        if r.word not in words or r.video not in cache:
            continue
        seg = cache[r.video]
        m = measures(seg)
        if not m:
            continue
        m.update({"video": r.video, "word": r.word})
        rows.append(m)
        feats.append(np.nanmean(features.sequence(seg), axis=0))
    M = pd.DataFrame(rows)
    if len(M) < 8 or M["word"].nunique() < 2:
        print("not enough clips cached for this pair")
        return 0
    cols = [c for c in M.columns if c not in ("video", "word")]
    print(M.groupby("word")[cols].median().T.round(3).to_string())
    print("\nn clips:", M["word"].value_counts().to_dict())

    keep = M["word"].isin(focus)
    sub = M[keep]
    y = (sub["word"] == focus[1]).to_numpy(float)
    X = np.array(feats)[np.where(keep)[0]]
    print(f"\n=== 4. linear probe: {focus[0]}(0) vs {focus[1]}(1), "
          f"{len(sub)} clips")
    probe(sub.reset_index(drop=True), y, X, cols)
    print("\nAUC on a few dozen clips is directional. Do not ship a change "
          "on it; ship it on --seeds 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
