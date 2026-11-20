"""
Vocabulary resolution against the official ASL Citizen split CSVs.

This is the only place that decides which videos exist in a run:

  * a word  -> the exact raw Gloss strings listed in vocab.json
  * REST    -> the idle head/tail frames of those same videos (src/extract.py
               records the trimmed signing window; src/dataset.py slices it)
  * OTHER   -> a seeded, capped sample of non-vocabulary glosses taken from the
               split it is used in (never from another split)

Splits are signer-disjoint by construction in the official CSVs, so no
signer ever appears in train and test. Nothing here re-randomises them.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from . import config


class VocabError(RuntimeError):
    pass


@dataclass(frozen=True)
class Word:
    word: str
    batch: int
    glosses: tuple[str, ...]
    min_videos: int = 15
    note: str = ""
    status: str = "ok"
    external_glosses: tuple[str, ...] = ()


@dataclass
class VocabSpec:
    words: list[Word]
    enabled_batches: tuple[int, ...]
    rest: dict = field(default_factory=dict)
    other: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def active(self) -> list[Word]:
        """Words that are enabled and actually resolvable in the dataset."""
        return [w for w in self.words
                if w.batch in self.enabled_batches
                and (w.glosses or w.external_glosses)]

    @property
    def skipped(self) -> list[Word]:
        return [w for w in self.words if not (w in self.active)]

    @property
    def labels(self) -> list[str]:
        return ([w.word for w in self.active]
                + [config.REST_LABEL, config.OTHER_LABEL])

    @property
    def gloss_to_word(self) -> dict[str, str]:
        return {g: w.word for w in self.active for g in w.glosses}


def load_vocab(path=None) -> VocabSpec:
    path = path or config.VOCAB_JSON
    spec = json.loads(path.read_text(encoding="utf-8"))
    words = [Word(word=w["word"], batch=int(w.get("batch", 1)),
                  glosses=tuple(w.get("glosses", ())),
                  min_videos=int(w.get("min_videos", 15)),
                  note=str(w.get("note", "")),
                  status=str(w.get("status", "ok")),
                  external_glosses=tuple(w.get("external_glosses", ())))
             for w in spec["words"]]
    seen: set[str] = set()
    for w in words:
        if w.word in seen:
            raise VocabError(f"vocab.json lists '{w.word}' twice")
        seen.add(w.word)
    return VocabSpec(
        words=words,
        enabled_batches=tuple(spec.get("enabled_batches", (1,))),
        rest=spec["extra_classes"][config.REST_LABEL],
        other=spec["extra_classes"][config.OTHER_LABEL],
        raw=spec,
    )


# ---------------------------------------------------------------------------
# split table
# ---------------------------------------------------------------------------
SPLIT_COLUMNS = ["split", "participant", "video", "gloss", "word", "asl_lex"]

@lru_cache(maxsize=4)
def _read_split(split: str) -> pd.DataFrame:
    if split not in config.SPLIT_FILES:
        raise VocabError(f"unknown split '{split}'")
    path = config.SPLITS_DIR / config.SPLIT_FILES[split]
    if not path.is_file():
        config.require_dataset()
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns=lambda c: c.strip())
    need = ["Participant ID", "Video file", "Gloss"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise VocabError(f"{path.name} is missing column(s): {missing}")
    out = pd.DataFrame({
        "split": split,
        "participant": df["Participant ID"].astype(str).str.strip(),
        "video": df["Video file"].astype(str).str.strip(),
        "gloss": df["Gloss"].astype(str).str.strip(),
        "asl_lex": (df["ASL-LEX Code"].astype(str).str.strip()
                    if "ASL-LEX Code" in df.columns else ""),
    })
    return out


@lru_cache(maxsize=4)
def split_table(split: str, vocab_path=None) -> pd.DataFrame:
    """Split rows with the resolved vocabulary word attached ( '' = off-vocab )."""
    spec = load_vocab(vocab_path)
    g2w = spec.gloss_to_word
    df = _read_split(split).copy()
    df["word"] = df["gloss"].map(g2w).fillna("")
    return df[SPLIT_COLUMNS]


def signers(split: str, vocab_path=None) -> list[str]:
    return sorted(split_table(split, vocab_path)["participant"].unique())


# ---------------------------------------------------------------------------
# resolution table (the word -> gloss -> videos -> signers audit)
# ---------------------------------------------------------------------------
def resolve_table(vocab_path=None) -> pd.DataFrame:
    """One row per (word, gloss): official counts, plus the flags the brief asks for."""
    config.require_dataset()
    spec = load_vocab(vocab_path)
    rows = []
    frames = {s: split_table(s, vocab_path) for s in ("train", "val", "test")}
    all_glosses = set()
    for f in frames.values():
        all_glosses |= set(f["gloss"].unique())
    for w in spec.words:
        enabled = w.batch in spec.enabled_batches
        if not w.glosses:
            rows.append(dict(word=w.word, batch=w.batch, gloss="(none)",
                             train_videos=0, train_signers=0, val_videos=0,
                             test_videos=0, test_signers=0,
                             status=("external-only" if w.external_glosses
                                     else "MISSING"), note=w.note))
            continue
        for gloss in w.glosses:
            present = gloss in all_glosses
            sub = {s: frames[s][frames[s]["gloss"] == gloss]
                   for s in frames}
            rows.append(dict(
                word=w.word, batch=w.batch,
                gloss=gloss if present else f"{gloss} (NOT IN DATASET)",
                train_videos=int((sub["train"]["video"].nunique())),
                train_signers=int(sub["train"]["participant"].nunique()),
                val_videos=int(sub["val"]["video"].nunique()),
                test_videos=int(sub["test"]["video"].nunique()),
                test_signers=int(sub["test"]["participant"].nunique()),
                status=("ok" if present and enabled else
                        ("disabled" if present else "MISSING")),
                note=w.note,
            ))
    return pd.DataFrame(rows)


def vocab_markdown(vocab_path=None) -> str:
    df = resolve_table(vocab_path)
    spec = load_vocab(vocab_path)
    lines = [
        "# Vocabulary resolution",
        "",
        f"vocab.json batches enabled: {list(spec.enabled_batches)}",
        "Counts come from the official ASL Citizen split CSVs only "
        "(signer-disjoint).",
        "",
    ]
    head = ("word", "batch", "gloss", "train_videos", "train_signers",
            "val_videos", "test_videos", "test_signers", "status")
    widths = [max(len(head[i]),
                  *(len(str(r[head[i]])) for _, r in df.iterrows()))
              for i in range(len(head))]

    def fmt(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    lines.append(fmt(head))
    lines.append("  ".join("-" * w for w in widths))
    for _, r in df.iterrows():
        lines.append(fmt([r[c] for c in head]))
    notes = df[df["note"].astype(bool)].drop_duplicates("word")
    missing = sorted(set(df.loc[df["status"] == "MISSING", "word"]))
    disabled = sorted(set(df.loc[df["status"] == "disabled", "word"]))
    thin = sorted(set(df.loc[(df["train_videos"] < 15)
                             & (df["status"] == "ok"), "word"]))
    merged = sorted(set(df.loc[(df["status"] == "ok")
                               & df["word"].duplicated(keep=False), "word"]))
    lines += [
        "",
        "Flags:",
        f"  MISSING glosses       : {missing}",
        f"  disabled batch        : {disabled}",
        f"  under 15 train videos : {thin}",
        f"  merged variant glosses: {merged}",
        "",
        "Per-word notes:",
    ]
    for _, r in notes.iterrows():
        lines.append(f"  {r['word']:<12s} {r['note']}")
    lines += [
        "",
        "ASL-LEX evidence for the merged variants (different codes = different",
        "signs, so the merge is 'same English word', not 'same sign'):",
    ]
    for w in spec.active:
        if len(w.glosses) > 1:
            sub = frames_train_codes(w.glosses)
            lines.append(f"  {w.word:<10s} " + ", ".join(
                f"{g}={','.join(sorted(c)) or '?'}" for g, c in sub))
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=32)
def frames_train_codes(glosses: tuple[str, ...]) -> tuple:
    df = split_table("train")
    return tuple((g, set(df[df["gloss"] == g]["asl_lex"])) for g in glosses)


# ---------------------------------------------------------------------------
# sample selection
# ---------------------------------------------------------------------------
def word_videos(split: str, vocab_path=None) -> pd.DataFrame:
    df = split_table(split, vocab_path)
    return df[df["word"] != ""][["video", "participant", "gloss", "word"]]


def other_videos(split: str, vocab_path=None) -> pd.DataFrame:
    """Capped, seeded, gloss-diverse sample of that split's non-vocabulary signs."""
    spec = load_vocab(vocab_path)
    df = split_table(split, vocab_path)
    pool = df[df["word"] == ""]
    cap = int(spec.other[f"cap_{split}"])
    per_gloss = int(spec.other.get("per_gloss_max", 2))
    seed = int(spec.other.get("seed", config.SEED))
    if cap <= 0 or pool.empty:
        return pool.assign(word="")[["video", "participant", "gloss", "word"]]
    glosses = sorted(pool["gloss"].unique())
    rng = random.Random(f"{seed}:{split}:{cap}:{per_gloss}")
    rng.shuffle(glosses)
    chosen: list[str] = []
    for gloss in glosses:
        if len(chosen) >= cap:
            break
        vids = sorted(pool[pool["gloss"] == gloss]["video"].unique())
        rng.shuffle(vids)
        chosen.extend(vids[:per_gloss])
    chosen = chosen[:cap]
    out = pool[pool["video"].isin(set(chosen))].copy()
    out["word"] = config.OTHER_LABEL
    return out[["video", "participant", "gloss", "word"]].drop_duplicates("video")


# ---------------------------------------------------------------------------
# external corpus (PopSign v2.1 subset, downloaded by download_popsign_targets.py)
# ---------------------------------------------------------------------------
EXTERNAL_JOBS_COLUMNS = ["video", "split", "participant", "gloss", "word"]

# Exact external mappings share the vocabulary spec with the downloader.
# Chicken is external-only; it must not masquerade as an ASL Citizen gloss.
EXTERNAL_GLOSS_WORDS = {g: w.word for w in load_vocab().active
                        for g in w.external_glosses}


def external_jobs(splits=("train", "val", "test"), per_word_train: int = 60,
                  per_word_held: int = 20,
                  dataset: str = "popsign_v2_1", vocab_path=None) -> pd.DataFrame:
    """The PopSign subset, walked from the folders the downloader wrote.

    One row per clip with a namespaced id (ps:<dataset>/<split>/<gloss>/<file>),
    the 4-character signer code from the filename as participant (namespaced
    'ps:' so it can never collide with an ASL Citizen id), the PopSign gloss
    as-is and our resolved word via EXTERNAL_GLOSS_WORDS. Clips are picked by
    a seeded round-robin over signers, so the subset maximises signer spread
    instead of taking one signer's 60 takes. Splits are the official PopSign
    signer-disjoint splits: train clips join the train split, validation clips
    become 'ps_val' and test clips become 'ps_test'. No validation/test or
    ASL Citizen/PopSign evaluation numbers are pooled. Missing folders contribute nothing (a run
    before the download just sees the ASL Citizen rows).
    """
    mappings = {g: w.word for w in load_vocab(vocab_path).active
                for g in w.external_glosses}
    rows: list[dict] = []
    for split in splits:
        for gloss, word in mappings.items():
            # New downloads use the canonical project. Previously downloaded
            # clips are read in place after Model/ relocation, never copied.
            files: dict[str, Path] = {}
            for root in config.EXTERNAL_READ_DIRS:
                folder = root / dataset / split / gloss
                if folder.is_dir():
                    for f in folder.iterdir():
                        if f.is_file() and f.suffix.lower() == ".mp4":
                            files.setdefault(f.name, f)
            clips = [files[name] for name in sorted(files)]
            by_signer: dict[str, list[Path]] = {}
            for f in clips:
                signer = f.name[:4]                    # <signer>-<gloss>-...
                if len(signer) == 4 and signer.isalnum():
                    by_signer.setdefault(signer, []).append(f)
            want = per_word_train if split == "train" else per_word_held
            order = sorted(by_signer)
            rng = random.Random(
                f"{config.SEED}:external:{dataset}:{split}:{gloss}")
            rng.shuffle(order)
            pools = [by_signer[s] for s in order]
            picked: list[Path] = []
            k = 0
            while len(picked) < want and any(pools):
                pool = pools[k % len(pools)]
                if pool:
                    picked.append(pool.pop(0))
                k += 1
            out_split = {"train": "train", "val": "ps_val", "test": "ps_test"}[split]
            for f in picked:
                rows.append({
                    "video": f"ps:{dataset}/{split}/{gloss}/{f.name}",
                    "split": out_split,
                    "participant": "ps:" + f.name[:4],
                    "gloss": gloss,
                    "word": word,
                })
    return pd.DataFrame(rows, columns=EXTERNAL_JOBS_COLUMNS)


def videos_to_process(splits=("train", "val", "test"), vocab_path=None) -> pd.DataFrame:
    """Everything src/extract.py must run MediaPipe over: words + OTHER."""
    frames = []
    for split in splits:
        w = word_videos(split, vocab_path).assign(split=split)
        o = other_videos(split, vocab_path).assign(split=split)
        frames.append(pd.concat([w, o], ignore_index=True))
    df = pd.concat(frames, ignore_index=True)
    ext = external_jobs(splits, vocab_path=vocab_path)
    if len(ext):
        df = pd.concat([df, ext], ignore_index=True)
    df = df.drop_duplicates("video", keep="first").reset_index(drop=True)
    missing = [v for v in df["video"] if not config.video_path(v).is_file()]
    if missing:
        raise VocabError(
            f"{len(missing)} listed videos are not on disk, e.g. {missing[:3]}")
    return df


def stats(vocab_path=None) -> dict:
    spec = load_vocab(vocab_path)
    out = {"words": spec.labels, "n_classes": len(spec.labels),
           "enabled_batches": list(spec.enabled_batches),
           "skipped_words": [w.word for w in spec.skipped
                             if w.batch in spec.enabled_batches]}
    for split in ("train", "val", "test"):
        df = split_table(split, vocab_path)
        out[f"{split}_word_videos"] = int((df["word"] != "").sum())
        out[f"{split}_signers"] = int(df["participant"].nunique())
        out[f"{split}_other_videos"] = len(other_videos(split, vocab_path))
    return out


if __name__ == "__main__":
    import sys
    if not config.dataset_available():
        sys.exit("ASL Citizen not found - see src/config.py:require_dataset()")
    print(vocab_markdown())
    s = stats()
    for k, v in s.items():
        print(f"  {k:<20s}: {v}")
