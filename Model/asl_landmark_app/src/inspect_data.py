"""Stage B - is a letter actually signed the same way in every dataset?

Two independent tools, one script:

  1. CONTACT SHEETS  (`--letters M,N,...`) -> reports/samples/contact_<L>.png
     One image per letter: a row per dataset, N random images per row, with the
     MediaPipe skeleton drawn on each thumb and undetected hands greyed out.
     Sampling comes from the dataset itself (via the same strided worklist as
     extraction), NOT from the landmark cache, so the pictures are not biased
     toward the hands MediaPipe happened to like.

  2. SHAKE DIVERGENCE (numeric, all 24 letters)
     For every letter, compare the MEAN normalised hand shape between datasets,
     and compare that gap with the same letter's own within-dataset split-half
     gap. A between/within ratio >> 1 means the datasets do not agree on what
     that letter looks like - a labelling/annotation-style problem, not a model
     problem. Units are wrist->middle-MCP bone lengths.

  3. IMAGE PROVENANCE stats per dataset (size, unique colours, border std,
     pure-black fraction) so claims like "grassknoted is synthetic renders"
     are checked instead of repeated.

Run:
    python src/inspect_data.py                       # numeric + provenance only
    python src/inspect_data.py --per-dataset 12 --seed 7   # + contact sheets
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import normalize  # noqa: E402
import landmark_store as LS  # noqa: E402

DATASETS = [d["id"] for d in config.DATASETS]


# ---------------------------------------------------------------------------
# image sampling (from the datasets, not the cache)
# ---------------------------------------------------------------------------
def worklist(per_class_cap: int = 400) -> list[tuple[str, str, str, str]]:
    """(path, dataset, letter, session) for up to per_class_cap images per class."""
    import extract_landmarks as EX

    EX.build_worklist(per_class_cap)
    out = list(EX._WL)
    EX._WL.clear()
    return out


def pick_images(letters: list[str], per_dataset: int, seed: int,
                per_class_cap: int = 400) -> dict[str, dict[str, list[str]]]:
    rng = random.Random(seed)
    buckets: dict[str, dict[str, list[str]]] = collections.defaultdict(dict)
    wl = worklist(per_class_cap)
    grouped = collections.defaultdict(list)
    for path, did, letter, _sess in wl:
        grouped[(did, letter)].append(path)
    for letter in letters:
        for did in DATASETS:
            pool = sorted(grouped.get((did, letter), []))
            if not pool:
                continue
            buckets[letter][did] = rng.sample(pool, min(per_dataset, len(pool)))
    return buckets


# ---------------------------------------------------------------------------
# contact sheets
# ---------------------------------------------------------------------------
def thumb(path: str, detector, size: int, cache: dict,
          highlight_thumb: bool = False) -> tuple[np.ndarray, bool]:
    """RGB thumbnail of one dataset image with the skeleton drawn on it."""
    import cv2
    import hand_backend as hb

    key = (path, highlight_thumb)
    if key in cache:
        return cache[key]
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        res = (np.full((size, size, 3), 60, np.uint8), False)
        cache[key] = res
        return res
    rgb = hb.bgr_to_rgb(bgr)
    try:
        hand = hb.detect_image(detector, bgr)
    except Exception:
        hand = None
    th = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    th = cv2.cvtColor(th, cv2.COLOR_RGB2BGR)
    ok = hand is not None and hand.landmarks is not None
    if ok:
        hb.draw_landmarks(th, hand.landmarks[:, :2])   # normalised coords scale
        if highlight_thumb:
            hh, ww = th.shape[:2]
            tp = [(int(hand.landmarks[i][0] * ww), int(hand.landmarks[i][1] * hh))
                  for i in THUMB_CHAIN]
            for a, b in zip(tp, tp[1:]):
                cv2.line(th, a, b, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(th, tp[-1], 5, (0, 255, 255), 1, cv2.LINE_AA)  # TIP ringed
    else:
        th = (0.35 * th + 0.65 * np.zeros_like(th)).astype(np.uint8)
        cv2.putText(th, "no hand", (4, size // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 1, cv2.LINE_AA)
    res = (th, ok)
    cache[key] = res
    return res


def contact_sheet(letter: str, picked: dict[str, list[str]], out_png: Path,
                  detector, per_dataset: int, size: int = 112,
                  highlight_thumb: bool = False) -> None:
    import cv2

    cols = per_dataset
    label_w = 104
    rows = [d for d in DATASETS if picked.get(d)]
    canvas = np.zeros((len(rows) * (size + 4) + 26, label_w + cols * (size + 3) + 3, 3),
                      np.uint8)
    canvas[:] = 24
    cache: dict = {}
    cv2.putText(canvas,
                f"letter {letter}  -  skeleton = MediaPipe, grey = no hand"
                + (", red = thumb chain, yellow ring = thumb tip"
                   if highlight_thumb else ""),
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for r, did in enumerate(rows):
        y0 = 26 + r * (size + 4)
        for i, path in enumerate(picked[did][:cols]):
            th, _ok = thumb(path, detector, size, cache, highlight_thumb)
            x0 = label_w + 3 + i * (size + 3)
            canvas[y0:y0 + size, x0:x0 + size] = th
        cv2.putText(canvas, did[:12], (4, y0 + size // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", canvas)
    buf.tofile(str(out_png))
    print(f"  wrote {out_png.name}  ({len(rows)} datasets x {cols} images)")


# ---------------------------------------------------------------------------
# numeric: does the mean hand shape agree across datasets?
# ---------------------------------------------------------------------------
def shape_stats(sample_cap: int | None = None):
    """(letter, dataset) -> mean normalised xy, plus per-dataset per-letter arrays.

    Also returns the mirror rate: the fraction of RAW detections whose palm
    cross-product says "the other hand", i.e. that canonicalize() flips. A
    dataset that mirrors ~100% of its hands was recorded from the opposite side
    of the signer from one that mirrors ~0%; after canonicalisation both land in
    the same frame, so this is an imaging property, not a shape disagreement.
    """
    df = LS.keep_static_classes(LS.load_raw())
    if sample_cap:
        parts = [sub.iloc[:sample_cap] for _k, sub in df.groupby(["dataset", "letter"])]
        df = pd.concat(parts).reset_index(drop=True)
    lms = LS.landmark_matrix(df)
    prepared = normalize.prepare_batch(lms)
    pts = prepared[:, :, :2]
    mirrored = np.array([normalize.should_mirror(lms[i]) for i in range(len(lms))])
    key = df["letter"].astype(str) + "|" + df["dataset"].astype(str)
    means, store, mirror = {}, {}, {}
    for k, idx in df.groupby(key, sort=True).indices.items():
        lab, did = k.split("|", 1)
        if len(idx) < 10:
            continue
        store[(lab, did)] = pts[idx]
        means[(lab, did)] = pts[idx].mean(axis=0)
        mirror[(lab, did)] = float(mirrored[idx].mean())
    return means, store, df, mirror


def joint_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-joint L2 distance between two (21,2) mean shapes."""
    return np.linalg.norm(a - b, axis=1)


def within_gap(store, lab, did, rng, splits: int = 6) -> float:
    """Mean split-half distance of the SAME letter in the SAME dataset."""
    arr = store.get((lab, did))
    if arr is None or len(arr) < 20:
        return float("nan")
    idx = np.arange(len(arr))
    gaps = []
    for _ in range(splits):
        r = rng.permutation(idx)
        h1, h2 = r[: len(r) // 2], r[len(r) // 2:]
        gaps.append(float(joint_dist(arr[h1].mean(0), arr[h2].mean(0)).sum()))
    return float(np.mean(gaps))


def divergence_table(means, store, mirror=None) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    letters = sorted({lab for lab, _d in means})
    for lab in letters:
        ds_here = [d for d in DATASETS if (lab, d) in means]
        within = {d: within_gap(store, lab, d, rng) for d in ds_here}
        pairs = []
        for i, a in enumerate(ds_here):
            for b in ds_here[i + 1:]:
                dsum = float(joint_dist(means[(lab, a)], means[(lab, b)]).sum())
                base = np.nanmean([within[a], within[b]])
                pairs.append((a, b, dsum, dsum / base if base and base > 0 else np.nan))
        if not pairs:
            continue
        worst = max(pairs, key=lambda p: (p[3] if p[3] == p[3] else -1))
        diff = joint_dist(means[(lab, worst[0])], means[(lab, worst[1])])
        top = np.argsort(-diff)[:3]
        base = np.nanmean([within[worst[0]], within[worst[1]]])
        row = {
            "letter": lab,
            "n_per_ds": "/".join(str(len(store[(lab, d)])) for d in ds_here),
            "worst_pair": f"{worst[0][:5]} vs {worst[1][:5]}",
            # absolute distance between the two mean hand shapes, in bone units
            "shape_dist": round(worst[2], 2),
            # typical distance between two random halves of the SAME dataset
            "within_ds": round(float(base), 3) if base == base else float("nan"),
            # how many times bigger the between-dataset gap is than that
            "ratio": round(worst[3], 1) if worst[3] == worst[3] else float("nan"),
            "divergent_joints": " ".join(
                f"{int(j)}:{diff[j]:.2f}" for j in top),
        }
        if mirror is not None:
            row["mirror%"] = "/".join(
                f"{100 * mirror.get((lab, d), float('nan')):.0f}" for d in ds_here)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ratio", ascending=False)


def divergence_long(means, store) -> pd.DataFrame:
    """Every (letter, dataset_a, dataset_b) shape distance and its within-ratio.

    Written to cache/divergence.csv so that src/train.py can flag or down-weight
    a specific (letter, dataset) cell without re-running this analysis.
    """
    rng = np.random.default_rng(0)
    rows = []
    for lab in sorted({lab for lab, _d in means}):
        ds_here = [d for d in DATASETS if (lab, d) in means]
        within = {d: within_gap(store, lab, d, rng) for d in ds_here}
        for i, a in enumerate(ds_here):
            for b in ds_here[i + 1:]:
                dsum = float(joint_dist(means[(lab, a)], means[(lab, b)]).sum())
                base = float(np.nanmean([within[a], within[b]]))
                rows.append({"letter": lab, "ds_a": a, "ds_b": b,
                             "shape_dist": round(dsum, 3),
                             "within_ds": round(base, 4),
                             "ratio": round(dsum / base, 2) if base > 0 else np.nan,
                             "n_a": len(store[(lab, a)]), "n_b": len(store[(lab, b)])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage B (round 3): style conflict, or a thumb MediaPipe cannot see?
# ---------------------------------------------------------------------------
PALM_QUAD = (0, 5, 9, 17)      # wrist + the four MCP knuckles
THUMB_CHAIN = (1, 2, 3, 4)     # CMC, MCP, IP, TIP


def tip_in_palm(xy: np.ndarray) -> np.ndarray:
    """Per-frame: is the thumb TIP (4) inside the palm quad, in canonical space?

    Convex test: the tip is inside while every edge leaves it on the same side.
    A fist still gives MediaPipe a plausible palm, so a True here usually means
    "the thumb is hidden behind fingers and the landmark is a guess", which is
    the difference between a letter the model can learn and one it cannot.
    """
    tip = xy[:, 4, :]
    pos = np.zeros(len(tip), bool)
    neg = np.zeros(len(tip), bool)
    for i in range(4):
        a = xy[:, PALM_QUAD[i], :]
        b = xy[:, PALM_QUAD[(i + 1) % 4], :]
        c = (b[:, 0] - a[:, 0]) * (tip[:, 1] - a[:, 1]) \
            - (b[:, 1] - a[:, 1]) * (tip[:, 0] - a[:, 0])
        pos |= c > 0
        neg |= c < 0
    return ~(pos & neg)


def thumb_audit(letters: list[str], sample_cap: int = 2000,
                out_csv: Path | None = None) -> pd.DataFrame:
    """Thumb geometry per (letter, dataset), in palm-length units.

    `prepare()` puts the wrist at the origin and divides by the wrist->middle
    knuckle distance, so 1.0 = one palm and the numbers compare across image
    sizes, distances and hand sizes. `jitter` is the mean distance of a thumb tip
    from that cell's median tip: it separates a letter that is signed
    consistently but differently by dataset (style) from one where the landmark
    itself wanders frame to frame (bad detection, unlearnable).
    """
    df = LS.keep_static_classes(LS.load_raw())
    rows = []
    for lab in letters:
        for did in DATASETS:
            sub = df[(df["letter"] == lab) & (df["dataset"] == did)]
            if sample_cap and len(sub) > sample_cap:
                sub = sub.iloc[np.linspace(0, len(sub) - 1, sample_cap)
                               .round().astype(int)]
            if len(sub) < 10:
                continue
            xy = normalize.prepare_batch(LS.landmark_matrix(sub))[:, :, :2]
            tip = xy[:, 4, :]
            med = np.median(tip, axis=0)
            d = lambda a, b: np.linalg.norm(xy[:, a] - xy[:, b], axis=1)  # noqa: E731
            rows.append({
                "letter": lab, "dataset": did, "n": len(sub),
                "in_palm%": 100 * float(tip_in_palm(xy).mean()),
                "tip_knuckle": float(np.mean(d(4, 5))),
                "tip_little": float(np.mean(d(4, 17))),
                "thumb_len": float(np.mean(d(1, 4))),
                "jitter": float(np.mean(np.linalg.norm(tip - med, axis=1))),
            })
    tbl = pd.DataFrame(rows)
    if not len(tbl):
        return tbl
    for c in ("in_palm%", "tip_knuckle", "thumb_len", "jitter"):
        tbl[c] = tbl[c].round(3)

    print(f"\n  thumb geometry per (letter, dataset)  [units = palm length, "
          f"wrist->middle knuckle = 1.0]")
    print("  in_palm% = share of frames whose thumb TIP sits inside the palm quad "
          "(a hidden/tucked thumb)")
    print("  tip_knuckle = thumb tip -> index knuckle; thumb_len = thumb root -> "
          "tip (extension)")
    print("  jitter = mean distance of the tip from that cell's median tip: HIGH = "
          "the landmark itself is unstable\n")
    for lab in tbl["letter"].unique():
        s = tbl[tbl["letter"] == lab]
        span_palm = s["in_palm%"].max() - s["in_palm%"].min()
        span_kn = s["tip_knuckle"].max() - s["tip_knuckle"].min()
        print(s.to_string(index=False))
        print(f"    -> {lab}: across-dataset span  in_palm {span_palm:.1f} pts, "
              f"tip_knuckle {span_kn:.3f}; worst within-cell jitter "
              f"{s['jitter'].max():.3f}\n")
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tbl.to_csv(out_csv, index=False)
        print(f"  wrote {out_csv}")
    return tbl


def nearest_shapes(means: dict, letters: list[str] | None = None,
                   store: dict | None = None) -> pd.DataFrame:
    """For each (dataset, letter): which OTHER letter is its shape closest to?

    The divergence table only compares one letter across datasets, so it can see
    a style conflict but not a collision - two letters of the SAME signer whose
    mean hands sit on top of each other. A letter that is nearly 0 away from
    another letter cannot be separated by any amount of re-weighting, and a
    near-zero cross-dataset score is not evidence of a style conflict either: it
    means the model never had two separable shapes to begin with.

    Distances are the same summed-per-joint units as `divergence.csv` (a hand is
    ~1 palm from wrist to middle knuckle, so 21 joints can move a few units for
    a genuinely different letter and tens for a different handshape).
    """
    rows = []
    for did in DATASETS:
        have = [(lab, means[(lab, did)]) for lab in (letters or config.LABELS)
                if (lab, did) in means]
        for lab, m in have:
            dists = sorted((float(joint_dist(m, o).sum()), other)
                           for other, o in have if other != lab)
            if not dists:
                continue
            d, near = dists[0]
            all_d = [x for x, _ in dists]
            rows.append({"dataset": did, "letter": lab,
                         "n": len(store[(lab, did)]) if store else 21,
                         "nearest": near, "dist": round(d, 2),
                         "median_other": round(float(np.median(all_d)), 2),
                         "crowding": round(d / float(np.median(all_d)), 3)})
    return pd.DataFrame(rows)


def print_nearest(tbl: pd.DataFrame, letters: list[str]) -> None:
    print("\n  shape collisions INSIDE one dataset (units = summed joint distance,"
          "\n  hand scale ~1 palm): `dist` = distance to the closest OTHER letter's"
          "\n  mean hand, `crowding` = that over the median distance to any other"
          "\n  letter. A crowding near 0 means the two letters share one region of"
          "\n  feature space, which no re-weighting can separate.")
    for did in tbl["dataset"].unique():
        s = tbl[tbl["dataset"] == did].sort_values("crowding")
        print(f"\n  {did} - the eight most crowded letters:")
        print(s.head(8).to_string(index=False))
        if letters:
            print(f"\n  {did} - asked about ({', '.join(letters)}):")
            print(s[s["letter"].isin(letters)].to_string(index=False))


# ---------------------------------------------------------------------------
# numeric: what kind of images are these really?
# ---------------------------------------------------------------------------
def provenance(per_dataset: int, seed: int, cap: int) -> None:
    import cv2

    rng = random.Random(seed + 1)
    wl = worklist(cap)
    grouped = collections.defaultdict(list)
    for path, did, letter, _s in wl:
        grouped[did].append(path)
    print("\n  image provenance (what the files actually look like):")
    print(f"  {'dataset':<13} {'n':>5} {'size':>12} {'uniq_col':>9} "
          f"{'border_sd':>9} {'black%':>7} {'jpeg%':>6}")
    for did in DATASETS:
        pool = grouped.get(did, [])
        if not pool:
            continue
        sample = rng.sample(pool, min(240, len(pool)))
        uniq, border_sd, black, sizes, jpeg = [], [], [], [], 0
        for p in sample:
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            sizes.append(img.shape[:2])
            uniq.append(len(np.unique(img.reshape(-1, 3), axis=0)))
            edge = np.concatenate([img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3),
                                   img[:, :4].reshape(-1, 3), img[:, -4:].reshape(-1, 3)])
            border_sd.append(float(edge.std()))
            black.append(float((img.max(axis=2) < 8).mean()))
            if str(p).lower().endswith((".jpg", ".jpeg")):
                jpeg += 1
        if not uniq:
            continue
        h = [s[0] for s in sizes]
        w = [s[1] for s in sizes]
        fmt = (f"{h[0]}x{w[0]}" if len(set(h)) == 1 and len(set(w)) == 1
               else f"{min(h)}-{max(h)}x{min(w)}-{max(w)}")
        print(f"  {did:<13} {len(sample):>5} {fmt:>12} {int(np.median(uniq)):>9} "
              f"{int(np.median(border_sd)):>9} {100 * np.median(black):>6.0f}% "
              f"{100 * jpeg / len(sample):>5.0f}%")
    print("  (a fixed tile size, very few unique colours, a flat border and a high "
          "pure-black fraction mean composited/rendered images, not photographs)")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--letters", default="M,N,T,K,V,G,H,C,B,P,Q,D",
                    help="comma-separated letters for the contact sheets")
    ap.add_argument("--per-dataset", type=int, default=0,
                    help="images per dataset per sheet (0 = skip the sheets)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--thumb", type=int, default=112)
    ap.add_argument("--cap", type=int, default=400,
                    help="max images listed per class when sampling")
    ap.add_argument("--sample-cache", type=int, default=1500,
                    help="cache rows per dataset-letter for the divergence table")
    ap.add_argument("--thumb-audit", default="",
                    help="comma-separated letters for the thumb-geometry table "
                         "(e.g. T,M,N,A,B,V). Runs without the divergence table "
                         "unless --skip-divergence is given as well.")
    ap.add_argument("--skip-divergence", action="store_true",
                    help="do not build the shape-divergence table (fast, and the "
                         "only way to run --thumb-audit on its own)")
    ap.add_argument("--highlight-thumb", action="store_true",
                    help="draw the thumb chain in red on the contact sheets")
    ap.add_argument("--skip-provenance", action="store_true",
                    help="do not decode the 240-image-per-dataset provenance "
                         "sample (saves a couple of minutes on a busy machine)")
    ap.add_argument("--nearest", default="",
                    help="comma-separated letters: report which OTHER letter each "
                         "one's mean hand shape collides with, inside each dataset")
    args = ap.parse_args()

    config.ensure_dirs()
    print("== Stage B diagnosis ==")
    means = None
    store = {}
    if not args.skip_divergence:
        means, store, _df, mirror = shape_stats(sample_cap=args.sample_cache)
        tbl = divergence_table(means, store, mirror)
        print("\n  landmark-shape divergence between datasets (bone-length units)")
        print("  worst_pair = the two datasets that disagree most about this letter;")
        print("  within_ds  = the same letter's split-half gap inside one dataset;")
        print("  ratio      = shape_dist / within_ds; >1 means the datasets disagree")
        print("               MORE than a dataset disagrees with itself, i.e. the")
        print("               disagreement is not just noise;")
        print("  mirror%    = share of raw detections canonicalize() flips, per dataset "
              "(grass/massey/danras order)")
        print(tbl.to_string(index=False))

        long = divergence_long(means, store)
        out_csv = config.CACHE_DIR / "divergence.csv"
        long.to_csv(out_csv, index=False)
        print(f"\n  wrote {out_csv}  ({len(long)} letter x pair rows)")
    else:
        print("  (shape-divergence table skipped: --skip-divergence)")

    if args.nearest:
        ask = [x.strip().upper() for x in args.nearest.split(",") if x.strip()]
        if means is None:
            means, store, _df, _mirror = shape_stats(sample_cap=args.sample_cache)
        tbl = nearest_shapes(means, store=store)
        near_csv = config.REPORTS_DIR / "nearest_shapes.csv"
        tbl.to_csv(near_csv, index=False)
        print(f"\n  wrote {near_csv}  ({len(tbl)} letter x dataset rows)")
        print_nearest(tbl, ask)

    if args.thumb_audit:
        letters = [x.strip().upper() for x in args.thumb_audit.split(",") if x.strip()]
        thumb_audit(letters, out_csv=config.REPORTS_DIR / "thumb_audit.csv")

    if args.skip_provenance:
        print("\n  (provenance table skipped: --skip-provenance)")
    else:
        provenance(args.per_dataset or 12, args.seed, args.cap)

    if args.per_dataset:
        import hand_backend as hb

        det = hb.create_detector(static=True)
        letters = [x.strip().upper() for x in args.letters.split(",") if x.strip()]
        picked = pick_images(letters, args.per_dataset, args.seed, args.cap)
        out_dir = config.REPORTS_DIR / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  contact sheets -> {out_dir}")
        for letter in letters:
            if not picked.get(letter):
                print(f"  {letter}: no images found - skipped")
                continue
            contact_sheet(letter, picked[letter], out_dir / f"contact_{letter}.png",
                          det, args.per_dataset, args.thumb, args.highlight_thumb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
