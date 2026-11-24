"""Step 2 - every dataset image -> 21x3 landmarks, cached in cache/landmarks.parquet.

Attempt 1 is the spec setting (direct, conf 0.5); failures escalate through
conf 0.3, upscale+pad, flip, flip+upscale/pad. Lower confidence is only ever a
retry. Recovered landmarks are mapped back to the original frame and `how`
records the attempt. Selection is strided so one clip cannot fill a class.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# keep every worker single-threaded; otherwise N workers each spawn M threads
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

import config  # noqa: E402
from download_data import iter_letter_images, normalize_label  # noqa: E402

MANIFEST = config.DATA_DIR / "datasets.json"
CHUNK = 64  # images per IPC message

# recovery settings (recorded in the cache sidecar; changing them must bump
# config.CACHE_VERSION)
HI_CONF = config.MIN_HAND_DETECTION_CONFIDENCE      # 0.5, the brief's value
LO_CONF = 0.3                                       # retry threshold
MIN_SIDE = 480                                      # upscale small images to this
PAD_FRAC = 0.12                                     # symmetric border padding
LADDER = ["direct_0.5", "direct_0.3", "upscale_pad_0.3", "flip_0.3",
          "flip_upscale_pad_0.3"]

_WL: list[tuple[str, str, str, str]] = []  # (path, dataset, letter, session)


def build_worklist(max_per_class: int, datasets: list[str] | None = None) -> tuple[dict, dict]:
    """Fill _WL with (path, dataset, letter, session) and return inventory stats."""
    if not MANIFEST.exists():
        raise SystemExit(f"No dataset manifest at {MANIFEST}. "
                         "Run src/download_data.py first.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_id = {d["id"]: d for d in config.DATASETS}
    stats: dict[str, dict] = {}
    coverage: dict[str, dict[str, int]] = {}

    for entry in manifest:
        did = entry["id"]
        if entry["n_images"] == 0 or did not in by_id:
            stats[did] = {"status": entry["status"], "selected": 0, "available": 0}
            coverage[did] = {}
            continue
        if datasets and did not in datasets:
            stats[did] = {"status": "excluded_by_flag", "selected": 0, "available": 0}
            coverage[did] = {}
            continue
        spec = by_id[did]
        per_letter: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
        for root in entry["roots"]:
            root_path = Path(root)
            if not root_path.is_absolute():
                root_path = config.ROOT / root_path
            for img, letter, session in iter_letter_images(root_path, spec):
                if letter in config.IGNORED_LABELS:
                    continue
                per_letter[letter].append((str(img), session))
        available = sum(len(v) for v in per_letter.values())

        chosen: list[tuple[str, str, str]] = []
        for letter, items in sorted(per_letter.items()):
            items.sort()  # keeps each clip/session contiguous on purpose
            k = min(len(items), max_per_class)
            # evenly spaced stride over the sorted list -> spreads across sessions
            idx = sorted(set(np.linspace(0, len(items) - 1, k).round().astype(int).tolist()))
            chosen.extend((items[i][0], letter, items[i][1]) for i in idx)

        _WL.extend((p, did, letter, session) for p, letter, session in chosen)
        stats[did] = {"status": "ok", "selected": len(chosen), "available": available}
        coverage[did] = {k: len(v) for k, v in sorted(per_letter.items())}
    return stats, coverage


_DET: dict[float, object] = {}
_RETRY = True


def _init_worker(retry: bool = True):
    """One detector per confidence level; retries off builds only the spec one."""
    global _DET, _RETRY
    import hand_backend

    _RETRY = bool(retry)
    confs = [HI_CONF, LO_CONF] if _RETRY else [HI_CONF]
    _DET = {c: hand_backend.create_detector(static=True, confidence=c) for c in confs}


def _variants(bgr):
    """Yield (label, image, new_w, new_h, pad, flipped, confidence) per attempt."""
    import cv2

    h, w = bgr.shape[:2]
    scale = max(1.0, MIN_SIDE / max(1, min(h, w)))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    big = (cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
           if scale > 1.0 else bgr)
    pad = int(round(PAD_FRAC * max(new_w, new_h)))
    padded = (cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
              if pad else big)
    attempts = [("direct_0.5", bgr, w, h, 0, False, HI_CONF)]
    if not _RETRY:
        return attempts
    attempts += [
        ("direct_0.3", bgr, w, h, 0, False, LO_CONF),
        ("upscale_pad_0.3", padded, new_w, new_h, pad, False, LO_CONF),
        ("flip_0.3", cv2.flip(bgr, 1), w, h, 0, True, LO_CONF),
        ("flip_upscale_pad_0.3", cv2.flip(padded, 1), new_w, new_h, pad, True, LO_CONF),
    ]
    return attempts


def _unmap(lm, new_w, new_h, pad, flipped) -> np.ndarray:
    """Landmarks from a scaled/padded/flipped copy -> original normalised coords.

    z is left alone: it is wrist-relative depth, unaffected by mirroring.
    """
    out = np.array(lm, dtype=np.float32, copy=True)
    w2, h2 = new_w + 2 * pad, new_h + 2 * pad
    x = out[:, 0] * w2 - pad
    y = out[:, 1] * h2 - pad
    if flipped:
        x = new_w - x              # undo the flip in the un-padded processed frame
    out[:, 0] = x / new_w
    out[:, 1] = y / new_h
    return out


def _process(paths: list[str]) -> list[tuple]:
    """Return rows: (path, ok, handedness, cross_sign, 63 floats, err, how)."""
    global _DET
    import cv2

    import hand_backend
    import normalize

    if not _DET:
        _init_worker(_RETRY)
    rows = []
    for p in paths:
        bgr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            rows.append((p, False, "", 0.0, None, "unreadable", "none"))
            continue
        found = None
        for label, img, nw, nh, pad, flip, conf in _variants(bgr):
            try:
                res = hand_backend.detect_image(_DET[conf], img)
            except Exception:      # never let one bad frame kill a worker
                continue
            if res.landmarks is None:
                continue
            lm = _unmap(res.landmarks, nw, nh, pad, flip)
            hand = hand_backend.swap_label(res.handedness) if flip \
                else res.handedness
            found = (lm, hand or "", float(normalize.palm_cross_sign(lm)), label)
            break
        if found is None:
            rows.append((p, False, "", 0.0, None, "no_hand", "none"))
            continue
        lm, hand, cross, how = found
        flat = lm.astype(np.float32).reshape(-1).tolist()
        rows.append((p, True, hand, cross, flat, "", how))
    return rows


# ---------------------------------------------------------------------------
def settings(no_retry: bool) -> dict:
    return {
        "cache_version": config.CACHE_VERSION,
        "hi_conf": HI_CONF, "lo_conf": LO_CONF,
        "min_side": MIN_SIDE, "pad_frac": PAD_FRAC,
        "retry": not no_retry,
        "num_hands": config.NUM_HANDS,
        "min_hand_presence_confidence": config.MIN_HAND_PRESENCE_CONFIDENCE,
        "hand_model": config.HAND_MODEL_PATH.name,
    }


def manifest_ids(datasets: list[str] | None = None) -> list[str]:
    """Dataset ids that are downloaded and would contribute rows.

    Recorded in the sidecar so that adding a dataset invalidates the cache
    instead of printing "cache already present - skipping".
    """
    if not MANIFEST.exists():
        return []
    known = {d["id"] for d in config.DATASETS}
    out = []
    for entry in json.loads(MANIFEST.read_text(encoding="utf-8")):
        did = entry["id"]
        if entry.get("n_images", 0) == 0 or did not in known:
            continue
        if datasets and did not in datasets:
            continue
        out.append(did)
    return sorted(out)


def _write_sidecar(st: dict, n_rows: int, elapsed: float,
                   used: list[str] | None = None) -> None:
    config.EXTRACT_META_JSON.write_text(json.dumps({
        **st, "n_rows": n_rows, "elapsed_s": round(elapsed, 1),
        "datasets": sorted(used or []),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")


def _recall_table(n_how: collections.Counter) -> pd.DataFrame:
    how_tbl = pd.DataFrame([(d, l, how, c) for (d, l, how), c in n_how.items()],
                           columns=["dataset", "letter", "how", "n"])
    wide = how_tbl.pivot_table(index=["dataset", "letter"], columns="how",
                               values="n", aggfunc="sum", fill_value=0).reset_index()
    for col in LADDER + ["dropped_no_hand", "dropped_unreadable"]:
        if col not in wide.columns:
            wide[col] = 0
    wide["selected"] = (wide[LADDER].sum(axis=1) + wide["dropped_no_hand"]
                        + wide["dropped_unreadable"])
    wide["skipped"] = wide["dropped_no_hand"] + wide["dropped_unreadable"]
    wide["kept"] = wide["selected"] - wide["skipped"]
    wide["kept_direct"] = wide["direct_0.5"]
    wide["kept_recovered"] = wide["kept"] - wide["direct_0.5"]
    return wide


def _atomic_write(df: pd.DataFrame, path: Path, retries: int = 4) -> None:
    """Write the cache through a sibling temp file and rename it into place.

    A half-written parquet would train on truncated data, and on Windows
    truncating an in-use file fails with EINVAL/EPERM; the old cache survives.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            if tmp.exists():
                tmp.unlink()
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            return
        except OSError as exc:
            last = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            print(f"  [write] {path.name} attempt {attempt + 1}/{retries} "
                  f"failed: {exc}; retrying", flush=True)
            time.sleep(3.0 * (attempt + 1))
    raise SystemExit(
        f"Could not write {path} after {retries} attempts: {last}\n"
        f"  The previous cache at {path} is untouched, so nothing is trained "
        f"on a truncated file.\n"
        f"  Close whatever is reading it (the Streamlit app, an editor, a "
        f"shell sitting in cache/) and re-run: "
        f"python src/extract_landmarks.py --force")


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"  [write] {path.name}: {exc} - kept the previous copy")


def run(max_per_class: int, workers: int, force: bool, datasets,
        no_retry: bool = False) -> int:
    config.ensure_dirs()
    st = settings(no_retry)
    want_ids = manifest_ids(datasets)
    if config.LANDMARKS_PARQUET.exists() and not force:
        size = config.LANDMARKS_PARQUET.stat().st_size / 1e6
        cur = {}
        if config.EXTRACT_META_JSON.exists():
            cur = json.loads(config.EXTRACT_META_JSON.read_text(encoding="utf-8"))
        same_settings = all(cur.get(k) == v for k, v in st.items())
        cached_ids = cur.get("datasets")
        new_ids = ([d for d in want_ids if d not in cached_ids]
                   if cached_ids is not None else [])
        if same_settings and not new_ids:
            print(f"== Step 2: landmarks cache already present "
                  f"({config.LANDMARKS_PARQUET.name}, {size:.1f} MB, "
                  f"cache_version={cur.get('cache_version')}) - skipping MediaPipe. "
                  f"Use --force to rebuild.")
            if cached_ids is None:
                held = ", ".join(want_ids) or "unknown"
                print("  note: this cache was written before the extractor "
                      "recorded which datasets it contains, so it holds "
                      f"{held} as of the time it was built. If you have "
                      "downloaded a dataset since then, re-run with --force so "
                      "its images are included.")
            return 0
        if same_settings and new_ids:
            missing = ", ".join(new_ids)
            print(f"== Step 2: the cache was built without {missing} - "
                  "rebuilding it from scratch so the new data is included; "
                  "no old rows are reused.")
        else:
            print("== Step 2: the existing cache was made with different settings "
                  f"(stored v{cur.get('cache_version', '?')} vs code "
                  f"v{config.CACHE_VERSION}) - rebuilding it from scratch; no old "
                  "rows are reused.")
    if config.LANDMARKS_PARQUET.exists():
        # previous drop numbers, for the report's before/after table
        prev = config.CACHE_DIR / "skipped_stats_v1.csv"
        cur_csv = config.CACHE_DIR / "skipped_stats.csv"
        if cur_csv.exists() and not prev.exists():
            try:
                cur_csv.rename(prev)
            except OSError:
                pass

    print("== Step 2: landmark extraction ==")
    t0 = time.time()
    stats, _ = build_worklist(max_per_class, datasets)
    work = list(_WL)
    _WL.clear()
    for did, s in stats.items():
        print(f"  {did:<12} {s['status']:<22} selected={s['selected']:>6} "
              f"available={s['available']:>7}")
    print(f"  total images to process: {len(work):,}")
    if not work:
        raise SystemExit("Nothing to extract - run src/download_data.py first.")
    print("  recovery ladder: "
          f"{'OFF (round-1 behaviour)' if no_retry else ' -> '.join(LADDER)}")

    path_to_key = {(p): (d, l, s) for p, d, l, s in work}
    batches = [list(path_to_key)[i:i + CHUNK] for i in range(0, len(path_to_key), CHUNK)]

    from concurrent.futures import ProcessPoolExecutor

    rows = []
    n_fail = collections.Counter()
    n_how = collections.Counter()
    print(f"  workers={workers} batches={len(batches)}")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(not no_retry,)) as ex:
        for out in tqdm(ex.map(_process, batches, chunksize=1),
                        total=len(batches), desc="landmarks", unit="batch", miniters=5):
            for path, ok, hand, cross, flat, err, how in out:
                d, l, s = path_to_key[path]
                if not ok:
                    n_fail[(d, l, err)] += 1
                    n_how[(d, l, "dropped_" + err)] += 1
                    continue
                rows.append((path, d, l, s, hand, cross, flat, how))
                n_how[(d, l, how)] += 1
    elapsed = time.time() - t0

    df = pd.DataFrame(rows, columns=["image", "dataset", "letter", "session",
                                     "mp_handedness", "cross_sign", "landmarks", "how"])
    _atomic_write(df, config.LANDMARKS_PARQUET)
    _write_sidecar(st, len(df), elapsed, sorted({d for _, d, _, _ in work}))

    total_sel = len(path_to_key)
    direct = int((df["how"] == "direct_0.5").sum()) if len(df) else 0
    print(f"\n  extracted  : {len(df):,} of {total_sel:,} images "
          f"({100 * len(df) / max(total_sel, 1):.1f}%) in {elapsed:.0f}s "
          f"({total_sel / max(elapsed, 1e-6):.1f} img/s)")
    print(f"  recovered  : {len(df) - direct:,} extra hands that attempt 1 alone "
          f"would have dropped (attempt 1 only = {direct:,} = "
          f"{100 * direct / max(total_sel, 1):.1f}%)")
    print(f"  cache      : {config.LANDMARKS_PARQUET} "
          f"({config.LANDMARKS_PARQUET.stat().st_size / 1e6:.1f} MB)")
    print(f"  sidecar    : {config.EXTRACT_META_JSON}")

    wide = _recall_table(n_how)
    cols = (["dataset", "letter", "selected", "kept", "kept_direct", "kept_recovered",
             "skipped"] + [c for c in LADDER[1:] if c in wide.columns]
            + ["dropped_no_hand", "dropped_unreadable"])
    _atomic_csv(wide[cols], config.CACHE_DIR / "skipped_stats.csv")

    sel_all = max(int(wide["selected"].sum()), 1)
    print("\n  detection recall, before (attempt 1 only) vs after the ladder:")
    print(f"  {'dataset':<13} {'selected':>9} {'attempt1':>9} {'after':>8} "
          f"{'gain':>7} {'still dropped':>14}")
    for d in sorted(set(wide["dataset"])):
        sub = wide[wide["dataset"] == d]
        sel = max(int(sub["selected"].sum()), 1)
        kept, kd = int(sub["kept"].sum()), int(sub["kept_direct"].sum())
        print(f"  {d:<13} {sel:>9,} {100 * kd / sel:>8.1f}% "
              f"{100 * kept / sel:>7.1f}% {100 * (kept - kd) / sel:>+6.1f}% "
              f"{sel - kept:>14,}")
    print(f"  {'ALL':<13} {sel_all:>9,} "
          f"{100 * int(wide['kept_direct'].sum()) / sel_all:>8.1f}% "
          f"{100 * int(wide['kept'].sum()) / sel_all:>7.1f}%")
    rec = wide.groupby("letter")["kept_recovered"].sum().sort_values(ascending=False)
    print("\n  recovered hands per letter (top 12): " +
          ", ".join(f"{l}:{int(v)}" for l, v in rec.head(12).items()))
    still = wide.groupby("letter")["skipped"].sum().sort_values(ascending=False)
    print("  still dropped per letter (top 8): " +
          ", ".join(f"{l}:{int(v)}" for l, v in still.head(8).items()))
    print("  which attempt found them: " + ", ".join(
        f"{k}={int(wide[k].sum())}" for k in LADDER + ["dropped_no_hand"]
        if k in wide.columns))
    print(f"  wrote {config.CACHE_DIR / 'skipped_stats.csv'}")
    print("\n  kept images per dataset x letter:")
    print(df.groupby("dataset")["letter"].value_counts().unstack(fill_value=0).to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--force", action="store_true", help="rebuild the cache")
    ap.add_argument("--no-retry", action="store_true",
                    help="disable the recovery ladder (reproduces the round-1 "
                         "drop rate; used to measure the gain honestly)")
    ap.add_argument("--datasets", type=str, default=None,
                    help="comma-separated dataset ids to include")
    args = ap.parse_args()
    ds = [x.strip() for x in args.datasets.split(",")] if args.datasets else None
    return run(args.max_per_class, args.workers, args.force, ds, args.no_retry)


if __name__ == "__main__":
    raise SystemExit(main())
