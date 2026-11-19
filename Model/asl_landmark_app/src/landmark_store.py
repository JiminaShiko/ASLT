"""Loading the landmark cache into arrays for training / evaluation / export."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import normalize  # noqa: E402


def _check_cache_version(path: Path) -> None:
    """Refuse to train on a cache built by a different extraction version.

    The detection ladder (Stage C) changes WHICH images get a landmark set and
    on what geometry, so mixing a v1 cache with a v2 run would silently compare
    two different datasets. extract_landmarks.py records its settings in
    cache/extract_meta.json; we re-read them here so the guard cannot be
    forgotten by any of the callers (train, eval, export, the Streamlit app).
    """
    if path != config.LANDMARKS_PARQUET:
        return                      # an explicitly supplied cache is the caller's call
    side = config.EXTRACT_META_JSON
    if not side.exists():
        print(f"  [cache] WARNING: {side.name} is missing, so this cache's "
              f"extraction settings are unknown (pre-v2 cache).")
        return
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [cache] WARNING: could not read {side.name}: {exc}")
        return
    got = meta.get("cache_version")
    if got != config.CACHE_VERSION:
        raise SystemExit(
            f"{path.name} was extracted with cache_version={got} but the code "
            f"now expects {config.CACHE_VERSION}.\n"
            f"  The landmark cache and the extractor would disagree, so nothing "
            f"is trained on a stale cache.\n"
            f"  Fix: python src/extract_landmarks.py --force")
    if not meta.get("retry", True):
        print("  [cache] note: this cache was built with --no-retry, i.e. the "
              "detection ladder was off; recall numbers will be round-1 level.")


def load_raw(parquet_path: Path | None = None) -> pd.DataFrame:
    path = Path(parquet_path or config.LANDMARKS_PARQUET)
    if not path.exists():
        raise SystemExit(f"{path} not found - run src/extract_landmarks.py first.")
    _check_cache_version(path)
    df = pd.read_parquet(path)
    df["letter"] = df["letter"].astype(str).str.upper()
    return df


def landmark_matrix(df: pd.DataFrame) -> np.ndarray:
    """(N, 21, 3) float32 raw MediaPipe landmarks (image-normalised coords)."""
    arr = np.stack([np.asarray(x, dtype=np.float32).reshape(21, 3) for x in df["landmarks"]])
    return arr


def keep_static_classes(df: pd.DataFrame) -> pd.DataFrame:
    """A-Z minus the motion letters J and Z, and minus classes we have no data for."""
    allowed = set(config.LABELS)
    out = df[df["letter"].isin(allowed)].reset_index(drop=True)
    counts = out["letter"].value_counts()
    thin = counts[counts < 50]
    if len(thin):
        out = out[~out["letter"].isin(thin.index)].reset_index(drop=True)
        print(f"  dropped under-represented classes ({len(thin)}): {''.join(thin.index)}")
    return out


def build_arrays(df: pd.DataFrame, lms: np.ndarray | None = None,
                 augment_fn=None, seed: int = 0):
    """Return (lm (N,21,3) canonical/centred/scaled, y (N,), ds, sess)."""
    lms = landmark_matrix(df) if lms is None else lms
    prepared = normalize.prepare_batch(lms)
    if augment_fn is not None:
        prepared = augment_fn(prepared, np.random.default_rng(seed))
    y = np.asarray([config.LABEL_TO_INDEX[l] for l in df["letter"]], dtype=np.int64)
    return prepared, y, df["dataset"].to_numpy(), df["session"].to_numpy()


def feature_dim() -> int:
    return normalize.features_from_normalized(np.zeros((21, 3))).shape[0]


def group_id(df: pd.DataFrame) -> np.ndarray:
    """dataset + session, i.e. one recording session of one signer.

    Used by the block-wise split so that neighbouring frames of the same clip
    never straddle train and test.
    """
    return (df["dataset"].astype(str) + "|" + df["session"].astype(str)).to_numpy()


def block_id(df: pd.DataFrame, n_blocks: int = 8) -> np.ndarray:
    """Contiguous rank buckets *inside* each (dataset, session, letter) run.

    `group_id` alone is not enough for the photo datasets: there one session
    folder holds one whole letter, so holding out a session deletes entire
    classes and the split scores 0% for the wrong reason. Bucketing by the
    position of the image inside its own class run keeps every class present
    on both sides while still never splitting a contiguous block - which is
    what guards against near-duplicate frames leaking between train and test.

    Images are ordered by their path. Massey frame names are zero padded
    (color_0_0007.png), so lexicographic order is the real temporal order;
    for the photo sets there is no temporal order to preserve and the blocks
    simply form a fixed, non-random subset of each class.
    """
    ds = pd.Series(df["dataset"].astype(str).to_numpy())
    ss = pd.Series(df["session"].astype(str).to_numpy())
    le = pd.Series(df["letter"].astype(str).to_numpy())
    im = pd.Series(df["image"].astype(str).to_numpy())
    d = pd.DataFrame({"dataset": ds, "session": ss, "letter": le, "image": im,
                      "row": np.arange(len(df))})
    key = ["dataset", "session", "letter"]
    d = d.sort_values(key + ["image"], kind="mergesort").reset_index(drop=True)
    pos = d.groupby(key, sort=False).cumcount().to_numpy()
    size = d.groupby(key, sort=False)["image"].transform("size").to_numpy()
    per = np.maximum(np.ceil(size / float(n_blocks)), 1.0)
    bucket = np.floor(pos / per).astype(str)
    lab = d["dataset"] + "|" + d["session"] + "|" + d["letter"] + "|b" + bucket
    out = np.empty(len(d), dtype=object)
    out[d["row"].to_numpy()] = lab.to_numpy()
    return out

