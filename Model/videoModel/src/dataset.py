"""Stage 4a - landmark cache to labelled samples (word windows, REST slices, OTHER); numpy only, no videos."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, extract, vocab

SAMPLE_COLUMNS = ["video", "split", "participant", "gloss", "label", "kind",
                  "start", "end", "frames"]


@dataclass
class Split:
    """Samples plus the landmark arrays they refer to, held in memory."""
    name: str
    samples: pd.DataFrame
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def window(self, i: int) -> tuple[int, int]:
        r = self.samples.iloc[i]
        return int(r["start"]), int(r["end"])

    def packed(self, i: int) -> np.ndarray:
        """The clip's landmark array trimmed to this sample's window."""
        r = self.samples.iloc[i]
        arr = self.arrays[r["video"]]
        return arr[int(r["start"]):int(r["end"])]

    def counts(self) -> dict[str, int]:
        return self.samples["label"].value_counts().to_dict()

    def signers(self) -> int:
        return int(self.samples["participant"].nunique())


def _manifest(splits: tuple[str, ...] = ("train", "val", "test"),
              require_files: bool = True) -> pd.DataFrame:
    df = extract.read_manifest()
    df = df[df["split"].isin(splits)]
    df = df[df["status"].astype(str) == "ok"]
    df = df[df["landmark_version"].astype(int) == config.LANDMARK_VERSION]
    if require_files:
        have = {v for v in df["video"] if extract.landmark_file(v).is_file()}
        df = df[df["video"].isin(have)]
        if len(df) and config.dataset_available():
            # a stale resumable cache must not widen the current work list
            source_splits = tuple(s for s in splits if s != "ps_test")
            if "ps_test" in splits:
                source_splits = tuple(dict.fromkeys((*source_splits, "val", "test")))
            planned = vocab.videos_to_process(source_splits)
            wanted = planned.loc[planned["split"].isin(splits), "video"]
            df = df[df["video"].isin(wanted)]
    return df.drop_duplicates("video", keep="last").reset_index(drop=True)


def sign_samples(split: str, spec: vocab.VocabSpec,
                 man: pd.DataFrame) -> pd.DataFrame:
    """The signing window of every cached clip that resolves to a label."""
    g2w = spec.gloss_to_word
    core = [w for w in spec.labels
            if w not in (config.REST_LABEL, config.OTHER_LABEL)]
    df = man[man["split"] == split].copy()
    # OTHER clips carry a gloss that resolves to nothing
    keep = (df["gloss"].isin(g2w)
            | df["word"].astype(str).isin(core)
            | (df["word"].astype(str) == config.OTHER_LABEL))
    df = df[keep].copy()
    df["label"] = df["gloss"].map(g2w)
    # ps: rows were resolved via vocab.EXTERNAL_GLOSS_WORDS, so trust their word
    via_word = df["word"].astype(str).isin(core) & df["label"].isna()
    df.loc[via_word, "label"] = df["word"]
    # a clip that is now off-vocabulary carries a stale label
    old_other = ((df["word"].astype(str) == config.OTHER_LABEL)
                 & df["label"].isna())
    df.loc[old_other, "label"] = config.OTHER_LABEL
    df["kind"] = "sign"
    df["start"] = df["sign_start"].astype(int)
    df["end"] = df["sign_end"].astype(int)
    df = df[df["end"] - df["start"] >= 2]
    df["frames"] = (df["end"] - df["start"]).astype(int)
    df["split"] = split
    return df[SAMPLE_COLUMNS].reset_index(drop=True)


def _rng_for(kind: str, split: str, seed: int) -> random.Random:
    return random.Random(f"{kind}:{split}:{seed}")


def rest_samples(split: str, spec: vocab.VocabSpec, man: pd.DataFrame,
                 min_frames: int | None = None,
                 max_frames: int | None = None) -> pd.DataFrame:
    """Capped idle slices from the head/tail of the same clips -> REST."""
    spec_rest = spec.rest
    cap = int(spec_rest.get(f"cap_{split}", 0))
    lo = int(min_frames or spec_rest.get("min_frames", 4))
    hi = int(max_frames or spec_rest.get("max_frames", 96))
    if cap <= 0:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    # ASL Citizen only: a PopSign clip is a single isolated sign
    df = man[(man["split"] == split)
             & (~man["video"].astype(str).str.startswith("ps:"))]
    pool: list[tuple[str, int, int]] = []
    for _, r in df.iterrows():
        T, s, e = int(r["frames"]), int(r["sign_start"]), int(r["sign_end"])
        if s >= lo:
            pool.append((r["video"], 0, s))
        if T - e >= lo:
            pool.append((r["video"], e, T))
    if not pool:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    rng = _rng_for("REST", split, int(spec_rest.get("seed", config.SEED)))
    rng.shuffle(pool)
    signer = dict(zip(df["video"], df["participant"]))
    gloss = dict(zip(df["video"], df["gloss"]))
    used_per_video: dict[str, int] = {}
    rows: list[dict] = []
    for video, a, b in pool:
        if len(rows) >= cap:
            break
        if used_per_video.get(video, 0) >= 2:
            continue
        avail = b - a
        n = min(avail, rng.randint(lo, max(lo, min(hi, avail))))
        off = rng.randint(0, avail - n) if avail > n else 0
        rows.append({"video": video, "split": split,
                     "participant": signer.get(video, ""),
                     "gloss": gloss.get(video, ""),
                     "label": config.REST_LABEL, "kind": "rest",
                     "start": a + off, "end": a + off + n, "frames": n})
        used_per_video[video] = used_per_video.get(video, 0) + 1
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def build_samples(splits: tuple[str, ...] = ("train", "val", "test"),
                  vocab_path=None, man: pd.DataFrame | None = None) -> pd.DataFrame:
    spec = vocab.load_vocab(vocab_path)
    man = _manifest(splits) if man is None else man
    out = []
    for s in splits:
        out.append(sign_samples(s, spec, man))
        out.append(rest_samples(s, spec, man))
    df = pd.concat(out, ignore_index=True)
    return df.drop_duplicates(["video", "kind", "start", "end"],
                              keep="first").reset_index(drop=True)


def load_split(split: str, samples: pd.DataFrame | None = None,
               vocab_path=None, man: pd.DataFrame | None = None,
               limit: int | None = None) -> Split:
    """Materialise one split: samples + their landmark arrays in memory."""
    spec = vocab.load_vocab(vocab_path)
    man = _manifest((split,)) if man is None else man[man["split"] == split]
    samples = (build_samples((split,), vocab_path, man)
               if samples is None else samples[samples["split"] == split])
    if limit:
        samples = samples.head(int(limit)).reset_index(drop=True)
    arrays: dict[str, np.ndarray] = {}
    keep: list[int] = []
    for i, r in samples.iterrows():
        f = extract.landmark_file(r["video"])
        if not f.is_file():
            continue
        if r["video"] not in arrays:
            # memory-mapped: three splits must not each hold a copy
            arrays[r["video"]] = np.load(f, mmap_mode="r")
        if int(r["end"]) <= arrays[r["video"]].shape[0]:
            keep.append(i)
    samples = samples.loc[keep].reset_index(drop=True)
    used = set(samples["video"])
    arrays = {k: v for k, v in arrays.items() if k in used}
    return Split(name=split, samples=samples, arrays=arrays,
                 labels=spec.labels)


def label_index(labels: list[str]) -> dict[str, int]:
    return {w: i for i, w in enumerate(labels)}


def class_weights(labels: list[str], counts: dict[str, int]) -> np.ndarray:
    """Inverse-frequency weights, mean 1; empty classes borrow the median count (finite loss)."""
    n = np.array([counts.get(l, 0) for l in labels], dtype=np.float64)
    present = n[n > 0]
    if present.size:
        n[n == 0] = float(np.median(present))
    else:
        n[:] = 1.0
    w = n.sum() / (len(labels) * n)
    return (w / w.mean()).astype(np.float32)


def sampler_weights(labels: list[str], samples: pd.DataFrame) -> np.ndarray:
    """Per-sample weights (equal class mass, times WEAK_WORD_BOOST), renormalised to mean 1."""
    counts = samples["label"].value_counts().to_dict()
    per = np.array([1.0 / max(counts.get(l, 1), 1)
                    for l in samples["label"]], dtype=np.float64)
    if not per.size:
        return per
    boost = getattr(config, "WEAK_WORD_BOOST", None) or {}
    if boost:
        per *= samples["label"].map(
            lambda label: float(boost.get(label, 1.0))).to_numpy(dtype=np.float64)
    return (per / per.mean()).astype(np.float64)


def describe(samples: pd.DataFrame) -> dict:
    per = (samples.groupby("label")
           .agg(samples=("video", "size"), videos=("video", "nunique"),
                signers=("participant", "nunique"),
                mean_frames=("frames", "mean")))
    per["mean_frames"] = per["mean_frames"].round(1)
    src_ps = (samples["video"].astype(str).str.startswith("ps:")
              if len(samples) else pd.Series(dtype=bool))
    return {"n_samples": int(len(samples)),
            "n_videos": int(samples["video"].nunique()),
            "n_signers": int(samples["participant"].nunique()),
            "n_ac_videos": int((~src_ps).sum()),
            "n_ps_videos": int(src_ps.sum()),
            "per_label": per.reset_index().to_dict("records"),
            "kinds": samples["kind"].value_counts().to_dict()}


def check_disjoint(man: pd.DataFrame) -> list[str]:
    """Audit: no (source, signer) pair may be in train and a held-out split; a violation is a bug, not a knob."""
    man = man.copy()
    man["pid"] = (man["video"].astype(str).map(config.source_of) + ":"
                  + man["participant"].astype(str))
    problems: list[str] = []
    held = man[man["split"].isin(("val", "test", "ps_test"))]
    train = man[man["split"] == "train"]
    if len(held) and len(train):
        leak = sorted(set(held["pid"]) & set(train["pid"]))
        if leak:
            problems.append(
                f"{len(leak)} signer ids appear in train AND a held-out "
                f"split: {leak[:5]}")
    for sp, grp in man.groupby("split"):
        if sp not in ("train", "val", "test", "ps_test"):
            problems.append(f"unknown split name {sp!r} in the manifest")
    return problems


def cache_fingerprint(samples: pd.DataFrame) -> str:
    """Stable hash of the exact sample list, so a report can name its data."""
    key = "|".join(f"{r.video}:{r.kind}:{r.start}:{r.end}:"
                   f"{r.label}:{r.split}:{r.participant}"
                   for r in samples.sort_values(["video", "start", "end", "label"])
                   .itertuples(index=False))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


if __name__ == "__main__":
    import json
    man = _manifest()
    smp = build_samples(man=man)
    print(json.dumps({
        "cached_videos": int(len(man)),
        "samples": describe(smp),
        "by_split": {s: int(len(g)) for s, g in smp.groupby("split")},
    }, indent=1, default=str))
