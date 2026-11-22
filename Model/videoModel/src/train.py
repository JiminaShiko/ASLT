"""
Stage 4c - training and the official signer-disjoint evaluation.

    python -m src.train --epochs 60

What the numbers depend on is fixed and written to models/meta.json: the
seed, the vocab.json batch, the landmark/feature version and a fingerprint of
the exact sample list. Early stopping watches the VAL split only; TEST is
touched once, at the end, and every test number is reported with the number
of videos and distinct signers behind it.

Outputs
  models/words_core20.pt        weights + arch + standardisation buffers
  models/meta.json              what produced them
  models/checkpoints/           resumable state (epoch, optimiser, patience)
  logs/train.log                human-readable run log
  reports/words_summary.json    the compact file check_results.py reads
  reports/per_word[_split].csv  per-word accuracy, with counts
  reports/confusion_words*.csv  the full confusion matrix
  reports/confused_pairs*.csv   the most confused pairs
  reports/predictions*.csv      per-sample predictions
  reports/seed_runs[.csv|.md]   every seed measured, with its spread (src/seedreport.py)

--fair trains the current vector (v3) and the pre-Stage-B vector (v2, 170
dims, no gap fill, no segment block) with the same seeds and config, decides
on VAL, promotes the winner and keeps the loser loadable via --weights.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config, dataset, features, model, seedreport, vocab

# The sweep table lives in src/seedreport.py (torch-free, so check_results.py
# can read it). These names are re-exported because train.py is what the
# scripts and the tests reach for.
SEED_RUN_COLUMNS = seedreport.SEED_RUN_COLUMNS
SWEEP_METRICS = seedreport.SWEEP_METRICS
append_seed_rows = seedreport.append_seed_rows
compare_table = seedreport.compare_table
mean_spread = seedreport.mean_spread
metric_columns = seedreport.metric_columns
seed_row_path = seedreport.seed_row_path
write_seed_table = seedreport.write_seed_table


# ---------------------------------------------------------------------------
# data plumbing
# ---------------------------------------------------------------------------
class Batcher:
    """(features, label) batches, rebuilt from landmarks on every pass.

    Features are recomputed rather than cached because the augmentation works
    at landmark level - a mirror or a 0.8x sub-window has to be able to change
    the numbers. The landmark arrays are memory-mapped, so the full cache
    never has to fit in RAM at once.
    """

    def __init__(self, sp: dataset.Split, labels: list[str],
                 blocks: list[str] | None = None,
                 version: int | None = None) -> None:
        self.sp = sp
        self.labels = labels
        self.idx = dataset.label_index(labels)
        self.blocks = blocks
        self.version = version
        cols = features.block_columns(blocks, version=version)
        self.dim = int(cols.stop if isinstance(cols, slice) else len(cols))
        self.y = torch.tensor([self.idx[l] for l in sp.samples["label"]],
                              dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sp)

    def row(self, i: int, augment: bool,
            rng: np.random.Generator | None) -> np.ndarray:
        arr = self.sp.packed(i).astype(np.float32)
        return features.sequence(arr, augment=augment, rng=rng,
                                 blocks=self.blocks, version=self.version)

    def batch(self, order, augment: bool, rng):
        x = np.stack([self.row(int(i), augment, rng) for i in order])
        return torch.from_numpy(x), self.y[torch.as_tensor(
            [int(i) for i in order])]

    def stacked(self, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Un-augmented [N,T,D] features and [N] labels for the whole split."""
        xs, ys = [], []
        for a in range(0, len(self), chunk):
            x, y = self.batch(range(a, min(a + chunk, len(self))), False, None)
            xs.append(x.numpy())
            ys.append(y.numpy())
        if not xs:
            return (np.zeros((0, config.NUM_FRAMES, self.dim), np.float32),
                    np.zeros(0, np.int64))
        return np.concatenate(xs), np.concatenate(ys)


def make_batcher(split: str, labels: list[str], limit: int | None,
                 blocks: list[str] | None,
                 version: int | None = None) -> Batcher | None:
    man = dataset._manifest((split,))
    if man.empty:
        return None
    smp = dataset.build_samples((split,), None, man)
    if limit:
        # keep a spread of labels instead of the first N rows
        keep, seen = [], Counter()
        for i, r in smp.iterrows():
            if seen[r["label"]] < 2:
                keep.append(i)
                seen[r["label"]] += 1
            if len(keep) >= limit:
                break
        smp = smp.loc[keep].reset_index(drop=True)
    if smp.empty:
        return None
    return Batcher(dataset.load_split(split, smp, None, man), labels, blocks,
                   version)


def validate_training_samples(train_samples: pd.DataFrame | None,
                              val_samples: pd.DataFrame | None,
                              spec: vocab.VocabSpec,
                              official_val: pd.DataFrame | None = None,
                              *, smoke: bool = False) -> None:
    """Fail before feature building/fitting when the active cache is incomplete.

    Validation must contain genuine official ASL Citizen rows, not external
    held-out clips or a fallback to the training set. Caps of zero explicitly
    disable the corresponding background requirement for that split.
    """
    if smoke:
        return
    words = [w.word for w in spec.active]
    # A word with no ASL Citizen gloss can never appear in the Citizen val
    # split: its held-out clips are the PopSign ps_val group instead.
    ps_only = {w.word for w in spec.active if not w.glosses}
    problems = []
    for split, samples in (("train", train_samples), ("val", val_samples)):
        need = words if split != "val" else [w for w in words if w not in ps_only]
        required = need + [label for label, cfg in
                           ((config.REST_LABEL, spec.rest),
                            (config.OTHER_LABEL, spec.other))
                           if label in spec.labels
                           and int(cfg.get(f"cap_{split}", 0)) > 0]
        if samples is None or samples.empty:
            problems.append(f"{split}: no samples (missing {', '.join(required)})")
            continue
        usable = samples[samples["split"].eq(split)]
        if split == "val":
            if official_val is None:
                problems.append("val: official ASL Citizen validation table is required")
                continue
            keys = ["video", "participant", "gloss"]
            official = official_val[keys + ["word"]].drop_duplicates(keys)
            usable = usable.merge(official, on=keys, how="inner")
            usable = usable[~usable["video"].astype(str).str.startswith("ps:")]
            valid_label = usable["label"].isin(BG) | usable["label"].eq(usable["word"])
            usable = usable[valid_label]
            if len(usable) != len(samples):
                problems.append("val: contains rows outside the official ASL Citizen validation split")
        present = set(usable["label"])
        missing = [w for w in required if w not in present]
        if missing:
            problems.append(f"{split}: missing samples for {', '.join(missing)}")
    if ps_only:
        man = dataset.extract.read_manifest()
        held = man[man["split"].astype(str).isin(("ps_val", "ps_test"))]
        held = held[held["status"].astype(str) == "ok"]
        have = set(held["word"].astype(str))
        absent = sorted(ps_only - have)
        if absent:
            problems.append("val: no cached PopSign held-out clips for "
                            f"{', '.join(absent)}")
    if problems:
        raise ValueError("Training blocked: " + "; ".join(problems)
                         + ". Extract the active vocabulary's train,val cache first; "
                         "test/ps_test cannot replace validation.")


def evaluation_splits(splits, *, smoke: bool = False) -> tuple[str, ...]:
    """Validate requested reporting splits before any run or output writes."""
    values = splits.split(",") if isinstance(splits, str) else splits
    out = tuple(dict.fromkeys(str(s).strip() for s in values if str(s).strip()))
    unknown = set(out) - {"train", "val", "test", "ps_test"}
    if unknown:
        raise ValueError(f"unknown evaluation splits: {', '.join(sorted(unknown))}")
    if not smoke and "val" not in out:
        raise ValueError("--eval-splits must include val: validation alone selects runs")
    return out


# ---------------------------------------------------------------------------
# one epoch / one evaluation
# ---------------------------------------------------------------------------
def run_epoch(bt: Batcher, net, crit, opt, sched, bs: int, seed: int):
    net.train()
    n = len(bt)
    rng = np.random.default_rng(seed)
    p = dataset.sampler_weights(bt.labels, bt.sp.samples)
    order = rng.choice(n, size=n, replace=True, p=p / p.sum())
    loss_sum, correct, total = 0.0, 0, 0
    for a in range(0, n, bs):
        idx = [int(i) for i in order[a:a + bs]]
        x, y = bt.batch(idx, augment=True, rng=rng)
        opt.zero_grad(set_to_none=True)
        logits = net(x)
        loss = crit(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), config.GRAD_CLIP)
        opt.step()
        sched.step()
        loss_sum += float(loss.detach()) * len(idx)
        correct += int((logits.argmax(1) == y).sum())
        total += len(idx)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def score(bt: Batcher, net, word_only: bool = False) -> dict:
    """Probabilities for a whole split, optionally restricted to the words."""
    net.eval()
    P, Y = bt.stacked()
    if len(Y) == 0:
        return {"n": 0, "top1": 0.0, "top5": 0.0, "pred": np.zeros(0, int),
                "y": np.zeros(0, int), "probs": P, "top5_idx": P}
    probs = np.concatenate([torch.softmax(net(torch.from_numpy(P[a:a + 256])),
                                          dim=1).numpy()
                            for a in range(0, len(P), 256)])
    if word_only:
        cols = word_columns(bt.labels)
        keep = np.isin(Y, cols)
        probs, Y = probs[np.ix_(keep, cols)], Y[keep]
    order = np.argsort(-probs, axis=1)
    pred = order[:, 0]
    k = min(5, probs.shape[1])
    return {"n": int(len(Y)),
            "top1": float((pred == Y).mean()) if len(Y) else 0.0,
            "top5": float(sum(int(t) in row for t, row in zip(Y, order[:, :k]))
                          / max(len(Y), 1)),
            "pred": pred, "y": Y, "probs": probs, "top5_idx": order[:, :k]}


def _num(x, nd: int = 4):
    """Round for JSON, and never write NaN: json.dump's NaN is not valid JSON
    and check_results.py has to be able to read this file."""
    v = float(x)
    return None if not np.isfinite(v) else round(v, nd)


def _clean(obj):
    """Recursive NaN/inf -> None, so every summary file is strict JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# the numbers a seed sweep reports (Stage B4)
# ---------------------------------------------------------------------------
BG = (config.REST_LABEL, config.OTHER_LABEL)


def word_columns(labels: list[str]) -> list[int]:
    return [i for i, l in enumerate(labels)
            if l not in (config.REST_LABEL, config.OTHER_LABEL)]


def gate_metrics(pred: pd.DataFrame, gate: float,
                 min_margin: float = 0.0) -> dict:
    """What the live confidence gate would have done with these readings.

    committed  = p1 >= gate AND margin >= min_margin AND the top-1 is a real
                 word. REST and OTHER on top are refusals, not commits, which
                 is what makes the precision column honest: typing a word over
                 a clip that is really not in the vocabulary is a miss.

    typed_share      - of the clips that ARE vocabulary words, how many the
                       gate lets through (the demo's recall for real signs)
    typed_precision  - of everything it let through, how many were the right
                       word
    other_slip / rest_slip - of the clips that are NOT words, how many it
                       would type anyway. This is the false-alarm rate the
                       README has to promise.
    """
    if pred is None or not len(pred):
        return {}
    ok = (pred["p1"] >= gate) & (pred["margin"] >= min_margin)
    word_top = ~pred["pred"].isin(BG)
    truth_word = ~pred["label"].isin(BG)
    committed = ok & word_top
    correct = pred["correct"].astype(bool)
    tw = int(truth_word.sum())
    twc = committed & truth_word
    out = {"gate": _num(gate, 2), "margin_min": _num(min_margin, 2),
           "n_words": tw, "n_rows": int(len(pred)),
           "typed_share": _num(float((committed & truth_word).sum()) / max(tw, 1)),
           # typed_precision counts every commit, including the ones the gate
           # makes over REST/OTHER clips - all of which are wrong - so it is
           # the number the live demo delivers. typed_precision_words drops
           # those rows and answers "when it types on a real word clip, how
           # often is it right".
           "typed_precision": _num(float((committed & correct).sum())
                                    / max(int(committed.sum()), 1)),
           "typed_precision_words": _num(float((twc & correct).sum())
                                          / max(int(twc.sum()), 1)),
           "n_committed": int(committed.sum())}
    for name in BG:
        m = pred["label"] == name
        out[f"{name.lower()}_slip"] = (_num(float((committed & m).sum())
                                             / max(int(m.sum()), 1))
                                       if m.any() else None)
        out[f"{name.lower()}_n"] = int(m.sum())
    return out


def commit_policy(gate: float | None = None) -> dict:
    """Freeze the configured policy in the report; never tune it on test."""
    from .predictor import _pair_key
    def pairs(mapping):
        return {"|".join(sorted(_pair_key(k))): v for k, v in mapping.items()}
    return {"version": "predictor.decide/commitable-v1",
            "confidence_min": float(config.CONFIDENCE_MIN if gate is None else gate),
            "margin_min": float(config.MARGIN_MIN),
            "pair_margins": pairs(config.PAIR_MARGINS),
            "hedge_pairs": pairs(config.HEDGE_PAIRS),
            "scope": "offline clip decisions; not measured live accuracy"}


def gate_stats(res: dict, labels: list[str], policy: dict | None = None) -> dict:
    """Score full-precision probabilities with the actual predictor commit rule."""
    from .predictor import decide, commitable
    policy = policy or commit_policy()
    kw = dict(min_conf=policy["confidence_min"],
              min_margin=policy["margin_min"],
              pair_margins=policy["pair_margins"],
              hedge_pairs=policy["hedge_pairs"])
    readings = [decide(p, labels, **kw) for p in res["probs"]]
    committed = np.array([commitable(r) for r in readings], dtype=bool)
    # Count refusals caused by the hedge, not already-refused low-confidence rows.
    unhedged = {**kw, "hedge_pairs": {}}
    hedge_refused = np.array([
        not commitable(r) and commitable(decide(p, labels, **unhedged))
        for p, r in zip(res["probs"], readings)], dtype=bool)
    truth = np.asarray([labels[int(i)] for i in res["y"]])
    words = ~np.isin(truth, BG)
    correct = np.asarray([r.word for r in readings]) == truth
    nw, nc = int(words.sum()), int(committed.sum())
    ncorrect = int((committed & words & correct).sum())
    out = {"policy": policy, "n_rows": len(readings), "n_words": nw,
           "n_committed": nc, "n_correct_committed": ncorrect,
           "n_hedge_refused": int(hedge_refused.sum()),
           "typed_share": _num((committed & words).sum() / nw) if nw else None,
           "correct_commit_recall": _num(ncorrect / nw) if nw else None,
           "typed_precision": _num(ncorrect / nc) if nc else None,
           "typed_precision_words": (_num(ncorrect / int((committed & words).sum()))
                                     if (committed & words).any() else None),
           "committed": committed, "hedge_refused": hedge_refused,
           "refusal_reason": [r.reason if not commitable(r) else "" for r in readings]}
    for name in BG:
        mask = truth == name
        n = int(mask.sum())
        out[f"{name.lower()}_n"] = n
        out[f"{name.lower()}_slip"] = _num((committed & mask).sum() / n) if n else None
    return out


def word_groups(spec: vocab.VocabSpec, weak_words=()) -> dict[str, list[str]]:
    """Batch-one retention and expansion are separate, never pooled gains."""
    active = {w.word for w in spec.active}
    core = [w.word for w in spec.words if w.batch == 1 and w.word in active]
    return {"core20": core, "weak6": [w for w in weak_words if w in core],
            "new_words": [w.word for w in spec.active if w.batch != 1]}


def group_stats(res: dict, per: pd.DataFrame, labels: list[str],
                groups: dict[str, list[str]], committed: np.ndarray) -> dict:
    """Group accuracy keeps the full classifier argmax, including new labels."""
    out = {}
    for name, members in groups.items():
        mask = np.isin(res["y"], [labels.index(w) for w in members if w in labels])
        n = int(mask.sum())
        vals = per[per["label"].isin(members) & per["n_samples"].gt(0)]["top1"]
        correct = res["pred"] == res["y"]
        out[name] = {"labels": members, "n_samples": n, "n_words_scored": len(vals),
                     "top1": _num(correct[mask].mean()) if n else None,
                     "macro_top1": _num(vals.mean()) if len(vals) else None,
                     "correct_commit_recall": (_num((committed & correct & mask).sum() / n)
                                               if n else None)}
    return out


def pair_swaps(pred: pd.DataFrame, a: str, b: str) -> int:
    """Clips of a read as b plus clips of b read as a - the swap count."""
    if pred is None or not len(pred):
        return 0
    return int(((pred["label"] == a) & (pred["pred"] == b)).sum()
               + ((pred["label"] == b) & (pred["pred"] == a)).sum())


def collect_metrics(rep: Path, pairs: list[tuple[str, str]], gate: float,
                    splits=("val", "test")) -> dict:
    """The compact row for one finished run, read from its report folder."""
    out: dict = {}
    js = rep / "words_summary.json"
    if js.is_file():
        s = json.loads(js.read_text(encoding="utf-8"))
        out["epochs_ran"] = s.get("epochs_ran")
        out["best_val_top1"] = (s.get("best_val_top1"))
        out["sec_per_epoch"] = s.get("sec_per_epoch")
        for key in ("candidate", "label_set_id", "evaluation_id"):
            out[key] = s.get(key)
        out["labels"] = json.dumps(s.get("labels"))
        for sp in splits:
            ev = (s.get("eval") or {}).get(sp) or {}
            out[f"{sp}_top1"] = ev.get("top1_all")
            out[f"{sp}_words_top1"] = ev.get("top1_words_only")
            out[f"{sp}_words_top5"] = ev.get("top5_words_only")
            out[f"{sp}_macro_words"] = ev.get("macro_top1_words")
            out[f"{sp}_other_as_word"] = ((ev.get("background") or {})
                                          .get("other_as_word_rate"))
            # Saved predictor-policy metrics, not a re-tuned gate on test CSVs.
            for k, v in (ev.get("gate") or {}).items():
                if k != "policy":
                    out[f"{sp}_{k}"] = v
            for group, stats in (ev.get("groups") or {}).items():
                for k in ("top1", "macro_top1", "n_samples", "correct_commit_recall"):
                    out[f"{sp}_{group}_{k}"] = stats.get(k)
    for sp in splits:
        p = rep / (f"predictions.csv" if sp == "test" else f"predictions_{sp}.csv")
        if not p.is_file():
            continue
        pred = pd.read_csv(p)
        for a, b in pairs:
            out[f"{sp}_swap_{a}_{b}"] = pair_swaps(pred, a, b)
            sub = pred[pred["label"].isin([a, b])]
            out[f"{sp}_n_{a}_{b}"] = int(len(sub))
    # best epoch, straight from the history the run wrote
    if js.is_file() and out.get("epochs_ran"):
        s = json.loads(js.read_text(encoding="utf-8"))
        hist = s.get("history") or []
        if hist:
            out["best_ep"] = int(max(range(len(hist)),
                                     key=lambda i: hist[i].get("val_top1")
                                     or 0.0)) + 1
    return out


# ---------------------------------------------------------------------------
# regularization: presets plus per-flag overrides (Stage B3)
# ---------------------------------------------------------------------------
@dataclass
class Reg:
    """What this run trains with. Defaults are exactly the 2026-11-16 run."""
    name: str = "none"
    dropout: float = config.DROPOUT
    weight_decay: float = config.WEIGHT_DECAY
    label_smoothing: float = config.LABEL_SMOOTHING
    hidden: int = 0                      # 0 = take model.Arch's default
    gru_layers: int = 0
    conv_channels: tuple[int, ...] = ()
    aug: dict = field(default_factory=dict)      # overrides of config.AUG

    def as_dict(self) -> dict:
        return {"preset": self.name, "dropout": self.dropout,
                "weight_decay": self.weight_decay,
                "label_smoothing": self.label_smoothing,
                "hidden": self.hidden or None,
                "gru_layers": self.gru_layers or None,
                "conv_channels": list(self.conv_channels) or None,
                "aug": self.aug,
                "aug_effective": {k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in config.AUG.items()}}


# Each preset is a hypothesis about why 400 clips are not enough, and only
# one of them can be decided by the validation split:
REG_PRESETS: dict[str, dict] = {
    "none": {},                                     # the measured baseline
    "small": {"hidden": 64, "gru_layers": 1, "conv_channels": (48, 64)},
    "reg": {"dropout": 0.45, "weight_decay": 1e-3},
    "smooth": {"label_smoothing": 0.15},
    "aug+": {"aug": {"jitter": 0.030, "rotate": 15.0, "scale": 0.15,
                      "translate": 0.10, "speed": (0.70, 1.30),
                      "frame_dropout": 0.10}},
    "all": {"hidden": 64, "gru_layers": 1, "conv_channels": (48, 64),
            "dropout": 0.45, "weight_decay": 1e-3, "label_smoothing": 0.10,
            "aug": {"jitter": 0.030, "rotate": 15.0, "scale": 0.15,
                     "translate": 0.10, "speed": (0.70, 1.30),
                     "frame_dropout": 0.10}},
}


def make_reg(preset: str | None, overrides: dict | None) -> Reg:
    """Preset first, then the individual flags on top, then apply the AUG part.

    config.AUG is a module-level dict that features.py reads at build time, so
    overriding it here changes training augmentation for this process only.
    Live inference does not read it (augment is never True in the demo), so
    nothing a run does to AUG can change what the camera path computes.
    """
    name = (preset or "none").strip().lower()
    if name not in REG_PRESETS:
        raise SystemExit(f"unknown --reg '{name}'; presets: "
                         f"{', '.join(REG_PRESETS)}")
    spec = {**REG_PRESETS[name], **(overrides or {})}
    aug = dict(spec.get("aug") or {})
    r = Reg(name=name,
            dropout=float(spec.get("dropout", config.DROPOUT)),
            weight_decay=float(spec.get("weight_decay", config.WEIGHT_DECAY)),
            label_smoothing=float(spec.get("label_smoothing",
                                           config.LABEL_SMOOTHING)),
            hidden=int(spec.get("hidden", 0) or 0),
            gru_layers=int(spec.get("gru_layers", 0) or 0),
            conv_channels=tuple(spec.get("conv_channels") or ()),
            aug=aug)
    for k, v in aug.items():
        if k not in config.AUG:
            raise SystemExit(f"--aug {k}=... is not a knob; known: "
                             f"{', '.join(config.AUG)}")
        config.AUG[k] = tuple(v) if isinstance(v, (list, tuple)) else v
    return r


def parse_aug_flags(items: list[str]) -> dict:
    """--aug jitter=0.03 --aug speed=0.7,1.3  ->  {'jitter': 0.03, ...}

    A key whose old value is a range takes a comma pair; anything else is a
    float. Unknown keys are checked by make_reg, not here.
    """
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--aug needs KEY=VALUE, got '{it}'")
        k, v = it.split("=", 1)
        k, v = k.strip(), v.strip()
        parts = [p for p in v.replace(" ", "").split(",") if p]
        out[k] = tuple(float(p) for p in parts) if len(parts) > 1 else (
            float(v) if len(parts) == 1 else None)
        if out[k] is None:
            raise SystemExit(f"--aug {k}= is empty")
    return out



def per_label(res: dict, labels: list[str], bt: Batcher) -> pd.DataFrame:
    """Per-word accuracy with the counts behind every single number."""
    rows = []
    for i, lab in enumerate(labels):
        m = res["y"] == i
        sel = bt.sp.samples[m]
        if not m.any():
            rows.append({"label": lab, "n_samples": 0, "videos": 0, "signers": 0,
                         "top1": np.nan, "top5": np.nan, "mean_p1": np.nan,
                         "as_word": np.nan})
            continue
        top5 = np.array([i in r for r in res["top5_idx"][m]])
        rows.append({
            "label": lab,
            "n_samples": int(m.sum()),
            "videos": int(sel["video"].nunique()),
            "signers": int(sel["participant"].nunique()),
            "top1": round(float((res["pred"][m] == i).mean()), 4),
            "top5": round(float(top5.mean()), 4),
            "mean_p1": round(float(res["probs"][m].max(1).mean()), 4),
            "wrong": int((res["pred"][m] != i).sum()),
        })
    return pd.DataFrame(rows)


def confusion(res: dict, labels: list[str]) -> pd.DataFrame:
    wide = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for t, p in zip(res["y"], res["pred"]):
        wide.iloc[int(t), int(p)] += 1
    wide.insert(0, "true", labels)
    wide["n_true"] = wide[labels].sum(axis=1).values
    return wide


def confused_pairs(conf: pd.DataFrame, labels: list[str],
                   top: int = 15) -> pd.DataFrame:
    """The pairs the model actually swaps, word classes only."""
    rows = []
    wl = [labels[i] for i in word_columns(labels)]
    for i, a in enumerate(wl):
        for b in wl[i + 1:]:
            ab = int(conf.loc[conf["true"] == a, b].iloc[0])
            ba = int(conf.loc[conf["true"] == b, a].iloc[0])
            na = int(conf.loc[conf["true"] == a, "n_true"].iloc[0])
            nb = int(conf.loc[conf["true"] == b, "n_true"].iloc[0])
            if ab + ba:
                rows.append({"pair": f"{a} / {b}", "swapped": ab + ba,
                             "a_as_b": ab, "b_as_a": ba, "n_a": na, "n_b": nb,
                             "swap_rate": round((ab + ba) / max(na + nb, 1), 4)})
    return (pd.DataFrame(rows, columns=["pair", "swapped", "a_as_b", "b_as_a",
                                        "n_a", "n_b", "swap_rate"])
            .sort_values(["swapped", "swap_rate"], ascending=False)
            .reset_index(drop=True).head(top))


def background_stats(res: dict, labels: list[str]) -> dict:
    """REST / OTHER typed as a word, and words typed as REST / OTHER."""
    words = set(word_columns(labels))
    extra = {n: labels.index(n) for n in (config.REST_LABEL, config.OTHER_LABEL)
             if n in labels}
    out: dict = {}
    for name, col in extra.items():
        m = res["y"] == col
        as_word = int(sum(int(p) in words for p in res["pred"][m]))
        out[f"{name.lower()}_n"] = int(m.sum())
        out[f"{name.lower()}_acc"] = (_num(float((res["pred"][m] == col).mean()))
                                      if m.any() else None)
        out[f"{name.lower()}_as_word"] = as_word
        out[f"{name.lower()}_as_word_rate"] = _num(
            as_word / max(int(m.sum()), 1))
    m_word = np.isin(res["y"], list(words))
    as_extra = int(sum(int(p) in extra.values() for p in res["pred"][m_word]))
    out["word_as_extra"] = as_extra
    out["word_as_extra_rate"] = _num(as_extra / max(int(m_word.sum()), 1))
    return out


# ---------------------------------------------------------------------------
# seed sweep: mean +- spread, so a 2-point gain can stop looking like a win
# ---------------------------------------------------------------------------
def promote_from(rep: Path, weights: Path, log) -> None:
    """Copy one sweep run over the release artefacts, unchanged in behaviour.

    The sweep trains into its own folders; only the run that wins on VAL is
    allowed to touch models/words_core20.pt and reports/*.csv, so status.py
    and check_results.py keep describing exactly one model.
    """
    import shutil
    for p in sorted(rep.glob("*")):
        if p.suffix in (".csv", ".json") and p.name != "meta.json":
            shutil.copyfile(p, config.REPORTS_DIR / p.name)
    if (rep / "meta.json").is_file():
        shutil.copyfile(rep / "meta.json", config.META_JSON)
    shutil.copyfile(weights, config.FOR_RELEASE)
    log.info("promoted this run: %s + %s + reports/* (its own folder stays as "
             "the evidence)", config.FOR_RELEASE, config.META_JSON)


def seed_sweep(seeds: list[int], *, epochs=None, batch=None, limit=None,
               blocks=None, reg_name=None, aug_over=None,
               pairs: list[tuple[str, str]] | None = None,
               gate: float | None = None, tag: str = "",
               promote_best: bool = True, log=None,
               feature_version: int | None = None,
               eval_splits=("val", "test", "ps_test"),
               smoke: bool = False) -> pd.DataFrame:
    """Train N seeds of one config and write the mean +- spread table.

    The point is the spread: with ~430 training clips a seed can move the
    test number by a couple of points on its own, so a change is only real
    when its gain is larger than that. The deciding column is always VAL;
    TEST is reported alongside it and used for nothing.
    """
    eval_splits = evaluation_splits(eval_splits, smoke=smoke)
    if not seeds:
        raise ValueError("seed sweep requires at least one seed")
    if tag and (Path(tag).name != tag or "/" in tag or "\\" in tag or tag in (".", "..")):
        raise ValueError("sweep tag must be a single directory name")
    promote_best = promote_best and not smoke
    log = log or config.logger_for("sweep")
    pairs = pairs or [("good", "thanks"), ("happy", "please")]
    gate = float(config.CONFIDENCE_MIN if gate is None else gate)
    fver = (config.FEATURE_VERSION if feature_version is None
            else int(feature_version))
    r = make_reg(reg_name, aug_over or {})
    tag = tag or f"{r.name}-{time.strftime('%Y%m%d-%H%M')}"
    log.info("sweep '%s': %d seeds %s, reg=%s, feature v%d (%d dims), gate "
             "p1>=%.2f, pairs %s", tag, len(seeds), seeds, r.name, fver,
             features.feature_dim(fver), gate,
             ", ".join(f"{a}/{b}" for a, b in pairs))
    rows: list[dict] = []
    report_root = config.CKPT_DIR / "smoke" / "reports" if smoke else config.REPORTS_DIR
    checkpoint_root = config.CKPT_DIR / "smoke" if smoke else config.CKPT_DIR
    for sd in seeds:
        run_dir = checkpoint_root / f"seed{sd}"
        rep = report_root / "seeds" / f"{tag}-seed{sd}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rep.mkdir(parents=True, exist_ok=True)
        lg = config.logger_for("train", rep / "train.log")
        lg.info("=== sweep run %s / seed %d ===", tag, sd)
        out = train(epochs=epochs, batch=batch, limit=limit, seed=sd,
                    fresh=True, blocks=blocks, log=lg, reg=r,
                    release=run_dir / "words.pt", ckpt=run_dir / "state.pt",
                    reports_dir=rep, meta_json=rep / "meta.json",
                    feature_version=fver, eval_splits=eval_splits,
                    smoke=smoke, gate=gate)
        row = {"run_tag": tag, "reg": r.name, "feature_version": fver,
               "seed": sd, "p1_gate": gate, "epochs_ran": out["epochs_ran"],
               "seconds": out["seconds"], "promoted": 0}
        row.update(collect_metrics(rep, pairs, gate, splits=eval_splits))
        rows.append(row)
        lg.info("seed %d: val words %.3f  test words %s  (test is reported, "
                "not used)", sd, row.get("val_words_top1") or 0.0,
                row.get("test_words_top1"))
    df = pd.DataFrame(rows)
    # The deciding column is plain val top-1 over all 22 classes. The old
    # words-only key (REST/OTHER dropped from the argmax) is measured on
    # only ~84 val rows and ignores exactly the mistakes the live gate
    # pays for: in the 2026-11-17 sweep it tied two seeds at 0.9167 and
    # promoted the first, whose test top-1 was 4.6 points below the seed
    # it left on disk. Both columns are VAL - test is still used for
    # nothing.
    key = "val_top1"
    if smoke:
        # Scratch runs never select or promote a model and never enter the real table.
        append_seed_rows(rows, reports_dir=report_root)
        return df
    if key not in df or pd.to_numeric(df[key], errors="coerce").isna().any():
        raise ValueError("sweep has no valid VAL score for every seed; nothing promoted")
    win = int(pd.to_numeric(df[key], errors="coerce").idxmax())
    if promote_best and len(df):
        rows[win]["promoted"] = 1
    best_seed = int(rows[win]["seed"])
    table = append_seed_rows(rows)
    mine = table[table["run_tag"] == tag]
    print("\n" + "=" * 68)
    print(f"sweep '{tag}'  ({len(mine)} seeds)  mean +- half-range")
    for line in mean_spread(mine, [c for c in mine.columns
                                   if c.startswith(("val_", "test_"))]):
        print(line)
    print(f"\nwinner by VAL ({key}): seed {best_seed} = "
          f"{float(df[key].iloc[win]):.3f}")
    if promote_best and len(df):
        promote_from(config.REPORTS_DIR / "seeds" / f"{tag}-seed{best_seed}",
                     config.CKPT_DIR / f"seed{best_seed}" / "words.pt", log)
        # the release ONNX must always match the model that was just promoted
        # (a plain --seeds sweep used to leave the previous run's .onnx behind)
        from . import export_onnx
        export_onnx.export(log=log)
    else:
        log.info("--promote none: %s and reports/ are left exactly as they "
                 "were; this run is measurement only", config.FOR_RELEASE.name)
    # the best seed's weights always survive: fair_decide() promotes from them
    # even when promote_best is False, and cleans them up afterwards itself
    drop = [int(s["seed"]) for s in rows if int(s["seed"]) != best_seed]
    for sd in drop:
        for junk in (config.CKPT_DIR / f"seed{sd}" / "words.pt",
                     config.CKPT_DIR / f"seed{sd}" / "state.pt"):
            if junk.is_file():
                junk.unlink()
        log.info("deleted the leftover weights of seed %d (its reports under "
                 "reports/seeds/ are kept as the evidence)", sd)
    return table


# ---------------------------------------------------------------------------
# --fair: the same seeds and config on feature v3 and the old v2 vector
# ---------------------------------------------------------------------------
def _drop_stash(stash: Path | None) -> None:
    """Delete the fair stash file and its folder (nothing temp stays behind)."""
    if stash is None:
        return
    stash.unlink(missing_ok=True)
    try:
        stash.parent.rmdir()
    except OSError:
        pass


def _stash_fair_best(tag: str, tab: pd.DataFrame, log) -> Path | None:
    """Copy the v3 sweep's best-seed weights out of the per-seed folder.

    Both fair sweeps write checkpoints into the same per-seed folders
    (models/checkpoints/seed<seed>/), so the v2 sweep would overwrite exactly
    the weights fair_decide needs when v3 wins or is the loser. Returns the
    stash path, or None (with a loud log) when the sweep left nothing.
    """
    import shutil
    g = tab[tab["run_tag"] == tag] if "run_tag" in tab.columns else tab.iloc[0:0]
    if not len(g):
        log.error("fair: no seed rows for '%s' - cannot stash the v3 best "
                  "seed's weights", tag)
        return None
    idx = int(pd.to_numeric(g["val_top1"], errors="coerce").fillna(-1).idxmax())
    sd = int(g.loc[idx, "seed"])
    src = config.CKPT_DIR / f"seed{sd}" / "words.pt"
    if not src.is_file():
        log.error("fair: %s is missing after the v%d sweep - cannot stash "
                  "the best seed's weights", src, config.FEATURE_VERSION)
        return None
    stash = config.CKPT_DIR / "fair_stash" / f"{tag}-seed{sd}.pt"
    stash.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, stash)
    log.info("stashed feature v%d best seed %d -> %s (the v2 sweep reuses "
             "the same seed folders and would overwrite it)",
             config.FEATURE_VERSION, sd, stash)
    return stash


def fair_decide(tags: dict[int, str], smoke: bool, log,
                stash3: Path | None = None) -> None:
    """Decide the fair comparison on VAL and wire the winner in.

    The winner is the higher mean val top-1. v3 is adopted only when its gap
    over v2 is larger than the pooled seed-to-seed spread of the two runs -
    otherwise the simpler v2 vector stays, per the plan. The winner's best
    seed is promoted to the plain release names; the loser's best seed is
    kept loadable as models/words_v<ver>.pt (--weights), and its checkpoint
    meta carries feature_version so predictor/live build the right layout. A
    smoke run computes and prints everything but copies nothing.
    """
    import shutil

    def weights_for(ver: int, sd: int) -> Path:
        """v3 weights live in the stash (the v2 sweep reuses the seed
        folders); v2 weights are the sweep's own last-written files."""
        src = (stash3 if (ver == config.FEATURE_VERSION
                          and stash3 is not None)
               else config.CKPT_DIR / f"seed{sd}" / "words.pt")
        if not src.is_file():
            log.error("fair: %s is missing - cannot wire the winner/loser "
                      "weights, nothing was changed", src)
            raise SystemExit(2)
        return src

    p = seedreport.seed_row_path()
    df = pd.read_csv(p)
    missing = [t for t in tags.values() if t not in set(df["run_tag"])]
    if missing:
        log.error("fair: sweep rows missing for %s - the sweeps must finish "
                  "before deciding", missing)
        _drop_stash(stash3)
        return
    d = seedreport.fair_decision(df, tags[config.FEATURE_VERSION], tags[2])
    print("\n" + "=" * 68)
    print(seedreport.fair_table(tags[config.FEATURE_VERSION], tags[2]))
    print("=" * 68)
    if not d["comparable"]:
        _drop_stash(stash3)
        raise ValueError("fair comparison requires matching label/evaluation identities; nothing promoted")
    log.info("fair decision on VAL: v3 %.3f vs v2 %.3f, diff %+.3f, pooled "
             "seed spread +-%.3f -> keep v%d", d["mean3"], d["mean2"],
             d["diff"], d["spread"], d["winner"])

    def best_seed_of(tag: str) -> int:
        g = df[df["run_tag"] == tag]
        idx = int(pd.to_numeric(g["val_top1"], errors="coerce")
                  .fillna(-1).idxmax())
        return int(g.loc[idx, "seed"])

    if smoke:
        _drop_stash(stash3)
        log.warning("fair SMOKE: nothing promoted, no loser file written")
        return
    w, l = d["winner"], (2 if d["winner"] == config.FEATURE_VERSION else
                         config.FEATURE_VERSION)
    sd_w, sd_l = best_seed_of(tags[w]), best_seed_of(tags[l])
    src_w, src_l = weights_for(w, sd_w), weights_for(l, sd_l)
    promote_from(config.REPORTS_DIR / "seeds" / f"{tags[w]}-seed{sd_w}",
                 src_w, log)
    loser_pt = config.MODELS_DIR / f"words_v{l}.pt"
    shutil.copyfile(src_l, loser_pt)
    _drop_stash(stash3)
    log.info("loser kept loadable: %s (feature v%d, best seed %d, load with "
             "--weights %s)", loser_pt, l, sd_l, loser_pt)
    # mark the promoted row, refresh the md table, clear the leftovers
    m = (df["run_tag"] == tags[w]) & (df["seed"] == sd_w)
    df.loc[m, "promoted"] = 1
    df.to_csv(p, index=False)
    seedreport.write_seed_table(df)
    for sd in (sd_w, sd_l):
        for junk in (config.CKPT_DIR / f"seed{sd}" / "words.pt",
                     config.CKPT_DIR / f"seed{sd}" / "state.pt"):
            if junk.is_file():
                junk.unlink()
    log.info("fair done: release files are feature v%d; %s is feature v%d",
             w, loser_pt.name, l)


def fair(seeds: list[int], *, epochs=None, batch=None, limit=None,
         blocks=None, reg_name=None, aug_over=None,
         pairs: list[tuple[str, str]] | None = None, gate: float | None = None,
         tags: dict[int, str], smoke: bool = False, log=None,
         eval_splits=("val", "test", "ps_test")) -> None:
    """Both feature versions, same seeds, same config, one decision."""
    eval_splits = evaluation_splits(eval_splits, smoke=smoke)
    log = log or config.logger_for("fair")
    if len(seeds) < 2:
        raise SystemExit("--fair compares seed spreads, so it needs "
                         "--seeds >= 2 (3 is what the plan asks for)")
    kw = dict(epochs=epochs, batch=batch, limit=limit, blocks=blocks,
              reg_name=reg_name, aug_over=aug_over, pairs=pairs, gate=gate,
              promote_best=False, log=log, eval_splits=eval_splits, smoke=smoke)
    if smoke:
        for ver in (config.FEATURE_VERSION, 2):
            seed_sweep(seeds, tag=tags[ver], feature_version=ver, **kw)
        log.warning("fair SMOKE: scratch reports only; no selection or promotion")
        return
    tab3 = seed_sweep(seeds, tag=tags[config.FEATURE_VERSION],
                      feature_version=config.FEATURE_VERSION, **kw)
    # the v2 sweep reuses the same per-seed checkpoint folders, so the v3
    # best seed's weights must be stashed aside before it runs
    stash = _stash_fair_best(tags[config.FEATURE_VERSION], tab3, log)
    seed_sweep(seeds, tag=tags[2], feature_version=2, **kw)
    fair_decide(tags, smoke=smoke, log=log, stash3=stash)


def split_stats(
bt: Batcher, res: dict, per: pd.DataFrame, labels: list[str]) -> dict:
    wl = [labels[i] for i in word_columns(labels)]
    vals = per[per["label"].isin(wl) & (per["n_samples"] > 0)]["top1"]
    return {
        "n_samples": res["n"],
        "n_videos": int(bt.sp.samples["video"].nunique()),
        "n_signers": int(bt.sp.samples["participant"].nunique()),
        "top1_all": _num(res["top1"]),
        "top5_all": _num(res["top5"]),
        "macro_top1_words": (_num(float(vals.mean())) if len(vals) else None),
        "n_words_scored": int(len(vals)),
        "background": background_stats(res, labels),
    }


def write_split_report(bt: Batcher, net, labels: list[str], name: str,
                       summary: dict, log,
                       reports_dir: Path | None = None) -> None:
    reports_dir = reports_dir or config.REPORTS_DIR
    res = score(bt, net)
    wres = score(bt, net, word_only=True)
    per = per_label(res, labels, bt)
    conf = confusion(res, labels)
    pairs = confused_pairs(conf, labels)
    stats = split_stats(bt, res, per, labels)
    stats["top1_words_only"] = _num(wres["top1"])
    stats["top1_words_only_n"] = wres["n"]
    stats["top5_words_only"] = _num(wres["top5"])
    decisions = gate_stats(res, labels, summary.get("commit_policy"))
    committed = decisions.pop("committed")
    hedge_refused = decisions.pop("hedge_refused")
    refusal_reason = decisions.pop("refusal_reason")
    stats["gate"] = decisions
    groups = summary.get("word_groups", {})
    if name == "val" and "core20" in groups:
        # Choose once on VAL; test and ps_test reuse this exact membership.
        weak = per[per["label"].isin(groups["core20"]) & per["n_samples"].gt(0)]
        groups["weak6"] = weak.sort_values(["top1", "label"]).head(6)["label"].tolist()
        summary["weak6_source"] = "six weakest core20 words on this run's official val split"
    stats["groups"] = group_stats(res, per, labels, groups, committed)
    summary.setdefault("eval", {})[name] = stats
    tag = "" if name == "test" else f"_{name}"
    per.to_csv(reports_dir / f"per_word{tag}.csv", index=False)
    conf.to_csv(reports_dir / f"confusion_words{tag}.csv", index=False)
    pairs.to_csv(reports_dir / f"confused_pairs{tag}.csv", index=False)
    smp = bt.sp.samples
    pd.DataFrame({
        "video": smp["video"].values, "gloss": smp["gloss"].values,
        "label": smp["label"].values,
        "participant": smp["participant"].values,
        "kind": smp["kind"].values, "start": smp["start"].values,
        "end": smp["end"].values,
        "pred": [labels[i] for i in res["pred"]],
        "p1": np.round(res["probs"].max(1), 4),
        "margin": np.round(np.sort(res["probs"], axis=1)[:, -1] -
                           np.sort(res["probs"], axis=1)[:, -2], 4),
        "top3": ["|".join(labels[i] for i in r) for r in res["top5_idx"][:, :3]],
        "correct": res["pred"] == res["y"],
        "committed": committed, "hedge_refused": hedge_refused,
        "refusal_reason": refusal_reason,
    }).to_csv(reports_dir / f"predictions{tag}.csv", index=False)

    macro = stats["macro_top1_words"]
    log.info("%s: top-1 %.1f%%  top-5 %.1f%%  macro-words %s over %d "
             "samples / %d videos / %d signers", name, 100 * res["top1"],
             100 * res["top5"],
             "n/a" if macro is None else f"{100 * macro:.1f}%",
             res["n"], stats["n_videos"], stats["n_signers"])
    log.info("%s: words-only (argmax over the %d words, REST/OTHER ignored) "
             "top-1 %.1f%% on %d samples", name, len(word_columns(labels)),
             100 * wres["top1"], wres["n"])
    b = stats["background"]
    log.info("%s: REST typed as a word %d/%d, OTHER typed as a word %d/%d, "
             "words typed as REST/OTHER %d/%d", name,
             b.get("rest_as_word", 0), b.get("rest_n", 0),
             b.get("other_as_word", 0), b.get("other_n", 0),
             b["word_as_extra"], res["n"] - b.get("rest_n", 0) - b.get("other_n", 0))
    if len(pairs):
        log.info("%s: most confused %s", name,
                 ", ".join(f"{r['pair']} {r['swapped']}"
                           for _, r in pairs.head(6).iterrows()))
    weak = per[(per["n_samples"] >= 5) & per["label"].isin(
        [labels[i] for i in word_columns(labels)])].nsmallest(5, "top1")
    if len(weak):
        log.info("%s: weakest words %s", name,
                 ", ".join(f"{r['label']} {100 * float(r['top1']):.0f}%"
                           f" (n={int(r['n_samples'])})"
                           for _, r in weak.iterrows()))


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
# Set by the --allow-vocab-change flag. Read by the release guard below on every
# code path (single run, --seeds sweep, --fair), because all three can write
# models/words_core20.pt.
ALLOW_VOCAB_CHANGE = False


def _release_label_guard(labels, log, release) -> None:
    """Refuse to overwrite the shipped weights with a different label set.

    `vocab.json` enables batches 1-4, i.e. 95 words + REST + OTHER = 97
    outputs, while the shipped checkpoint (`models/words_core20.pt`) knows the
    20 core words + REST + OTHER = 22. Before this guard `python -m src.train`
    with the default vocabulary silently replaced the release model with a
    97-class one that had never been evaluated or promoted - the only thing
    that noticed was tests/test_logic.py, which could merely SKIP its
    feature-path check ("expanded vocabulary is not trained yet").

    A candidate run is exempt automatically: config.use_candidate() points
    FOR_RELEASE at candidates/<name>/models/, so `release` is not the shipped
    path. Overwriting the release on purpose needs --allow-vocab-change.
    """
    if ALLOW_VOCAB_CHANGE or Path(release) != Path(config.FOR_RELEASE):
        return
    if not config.META_JSON.is_file():
        return
    try:
        shipped = list(json.loads(
            config.META_JSON.read_text(encoding="utf-8")).get("labels") or [])
    except (OSError, json.JSONDecodeError):
        return
    if not shipped or sorted(shipped) == sorted(labels):
        return
    added = [x for x in labels if x not in shipped]
    removed = [x for x in shipped if x not in labels]
    raise SystemExit(
        f"refusing to overwrite {config.FOR_RELEASE.name}: it was trained on "
        f"{len(shipped)} labels, this run would write {len(labels)}.\n"
        f"  added  : {added[:12]}{' ...' if len(added) > 12 else ''}\n"
        f"  removed: {removed[:12]}{' ...' if len(removed) > 12 else ''}\n"
        "Train it as a candidate instead and keep the shipped model:\n"
        "  python -m src.train --candidate vocab-word\n"
        "  python check_results.py --candidate vocab-word   # compare, then decide\n"
        "Only pass --allow-vocab-change when you mean to replace the release "
        "weights with this vocabulary.")


def train(epochs=None, batch=None, limit=None, seed=None, fresh=False,
          blocks=None, log=None, eval_splits=("val", "test", "ps_test"),
          release: Path | None = None, ckpt: Path | None = None,
          reports_dir: Path | None = None, meta_json: Path | None = None,
          smoke: bool = False, reg: Reg | None = None,
          feature_version: int | None = None, gate: float | None = None) -> dict:
    eval_splits = evaluation_splits(eval_splits, smoke=smoke)
    log = log or config.logger_for("train", config.LOGS_DIR / "train.log")
    config.ensure_dirs()
    config.require_dataset()
    reports_dir = reports_dir or config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    meta_json = meta_json or config.META_JSON
    r = reg or make_reg(None, None)
    seed = config.SEED if seed is None else int(seed)
    epochs = config.EPOCHS if epochs is None else int(epochs)
    batch = config.BATCH_SIZE if batch is None else int(batch)
    blocks = None if not blocks or blocks == ["all"] else blocks
    fver = (config.FEATURE_VERSION if feature_version is None
            else int(feature_version))
    release = release or config.FOR_RELEASE
    ckpt = ckpt or (config.CKPT_DIR / "train_state.pt")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if fver not in (2, config.FEATURE_VERSION):
        raise SystemExit(f"unknown feature version {fver}; "
                         f"known: 2 and {config.FEATURE_VERSION}")

    spec = vocab.load_vocab()
    labels = spec.labels
    if len(set(labels)) != len(labels):
        raise RuntimeError("duplicate label in vocab.json")
    _release_label_guard(labels, log, release)
    log.info("labels: %d (%s)", len(labels), ", ".join(labels))
    log.info("feature version %d -> %d dims%s", fver,
             features.feature_dim(fver),
             "" if fver == config.FEATURE_VERSION
             else " (the pre-Stage-B vector: no gap fill, no segment block)")

    t0 = time.time()
    # the mixed-corpus honesty check, before anything trains: no (source,
    # signer) pair may appear in train and in a held-out split. A violation
    # means a pipeline bug, so it fails loud instead of training on it.
    man_all = dataset._manifest(("train", "val", "test", "ps_test"))
    split_problems = dataset.check_disjoint(man_all)
    if split_problems:
        for p in split_problems:
            log.error("signer-disjointness check: %s", p)
        raise SystemExit(2)
    train_bt = make_batcher("train", labels, limit, blocks, fver)
    val_bt = make_batcher("val", labels, limit, blocks, fver)
    validate_training_samples(
        train_bt.sp.samples if train_bt is not None else None,
        val_bt.sp.samples if val_bt is not None else None,
        spec, official_val=vocab.split_table("val") if not smoke else None,
        smoke=smoke)
    if train_bt is None or not len(train_bt):
        raise ValueError("no cached train samples; extract the train split first")
    # VAL is evaluated first so its weak-six membership is frozen for test.
    ordered_splits = sorted(eval_splits, key=lambda s: s != "val")
    testers = {s: (val_bt if s == "val" else
                   make_batcher(s, labels, limit, blocks, fver))
               for s in ordered_splits}
    testers = {k: v for k, v in testers.items() if v is not None and len(v)}
    log.info("samples: train %d (video-level cap %s) | val %s | %s",
             len(train_bt), limit, len(val_bt) if val_bt else "none",
             ", ".join(f"{k} {len(v)}" for k, v in testers.items()) or "no eval")
    if not testers and not smoke:
        log.error("no eval split is cached; run python -m src.extract "
                  "--splits val,test before training")
        raise SystemExit(2)

    log.info("building features once to measure standardisation ...")
    P, Y = train_bt.stacked()
    mu, sd = model.standardise(P)
    log.info("standardisation from %d train samples, %d frames each (%.1f s, "
             "%.1f ms per feature build)", len(P), config.NUM_FRAMES,
             time.time() - t0, 1000 * (time.time() - t0) / max(len(P), 1))
    # P and Y are kept until the end: the released weights get one pass over
    # them for the words-only TRAIN accuracy - the memorisation number the
    # report shows next to val. Nothing else is written from them.

    defaults = model.Arch()
    arch = model.Arch(
        in_dim=train_bt.dim, num_classes=len(labels), dropout=r.dropout,
        hidden=r.hidden or defaults.hidden,
        gru_layers=r.gru_layers or defaults.gru_layers,
        conv_channels=r.conv_channels or defaults.conv_channels)
    net = model.build(arch, mu, sd)
    log.info("model: {:,} params, [{}, {}] -> {} classes".format(
        net.n_params(), config.NUM_FRAMES, train_bt.dim, len(labels)))
    log.info("regularization [%s]: dropout %.2f  weight decay %.1e  label "
             "smoothing %.2f  hidden %d  gru layers %d  conv %s  aug %s",
             r.name, r.dropout, r.weight_decay, r.label_smoothing,
             arch.hidden, arch.gru_layers, arch.conv_channels,
             json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in r.aug.items()}) if r.aug else "default")
    counts = train_bt.sp.samples["label"].value_counts().to_dict()
    crit = nn.CrossEntropyLoss(
        label_smoothing=r.label_smoothing,
        weight=torch.tensor(dataset.class_weights(labels, counts)))
    opt = torch.optim.AdamW(net.parameters(), lr=config.LEARNING_RATE,
                            weight_decay=r.weight_decay)
    steps_per_epoch = max(1, int(np.ceil(len(train_bt) / batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.LEARNING_RATE,
        total_steps=steps_per_epoch * epochs, pct_start=0.15)

    fp = dataset.cache_fingerprint(train_bt.sp.samples)
    start_epoch, best, bad = 0, float("-inf"), 0
    if ckpt.is_file() and not fresh:
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        same = (st.get("seed") == seed and st.get("epochs") == epochs
                and st.get("fingerprint") == fp
                and st.get("arch") == arch.as_dict())
        if same:
            net.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            start_epoch, best, bad = (int(st["epoch"]), float(st["best_val"]),
                                      int(st["bad"]))
            # OneCycleLR counts batches, not epochs: put the clock back where
            # the interrupted run left it, or the learning rate restarts high.
            sched.last_epoch = start_epoch * steps_per_epoch - 1
            for g in opt.param_groups:
                g["lr"] = sched.get_lr()[0]
            log.info("resumed at epoch %d (best val %.4f, %d without "
                     "improvement, lr %.2e) from %s",
                     start_epoch, best, bad, opt.param_groups[0]["lr"], ckpt)
        else:
            log.warning("checkpoint %s is from a different run (sample "
                        "fingerprint %s); starting fresh", ckpt, fp)

    history: list[dict] = []
    for ep in range(start_epoch, epochs):
        e0 = time.time()
        loss, tr_acc = run_epoch(train_bt, net, crit, opt, sched, batch,
                                  seed * 1_000_003 + ep)
        if val_bt is not None and len(val_bt):
            val_acc = score(val_bt, net)["top1"]
        elif smoke:
            val_acc = tr_acc  # scratch plumbing only, never a validation claim
        else:
            raise RuntimeError("validation disappeared; refusing a training-score fallback")
        improved = val_acc > best + 1e-4
        best = max(best, val_acc)
        bad = 0 if improved else bad + 1
        history.append({"epoch": ep + 1, "train_loss": round(loss, 4),
                        "train_top1": round(tr_acc, 4),
                        "val_top1": round(val_acc, 4),
                        "saved": bool(improved),
                        "lr": round(float(opt.param_groups[0]["lr"]), 6),
                        "seconds": round(time.time() - e0, 1)})
        log.info("epoch %2d/%d  loss %.4f  train %.3f  val %.3f  lr %.2e  "
                 "(%.1f s)%s", ep + 1, epochs, loss, tr_acc, val_acc,
                 opt.param_groups[0]["lr"], time.time() - e0,
                 "  <- best" if improved else "")
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "best_val": best, "bad": bad,
                    "seed": seed, "epochs": epochs, "arch": arch.as_dict(),
                    "fingerprint": fp}, ckpt)
        if improved:
            model.save(net, release, {"best_val_top1": best, "epoch": ep + 1,
                                      "labels": labels,
                                      "feature_dim": train_bt.dim,
                                      "feature_version": fver})
        if bad >= config.PATIENCE:
            log.info("early stop: val did not improve for %d epochs", bad)
            break

    if not release.is_file():
        # no epoch ever improved (possible in a 1-epoch smoke run): keep the
        # weights from the last epoch rather than loading a stale file
        log.warning("%s was not written by an improving epoch; saving the "
                    "final weights", release)
        model.save(net, release, {"best_val_top1": best,
                                  "epoch": len(history), "labels": labels,
                                  "feature_dim": train_bt.dim,
                                  "feature_version": fver})
    best_net, _ = model.load(release)
    policy = commit_policy(gate)
    eval_identity = {
        "protocol": "official-ac-val-v1", "smoke": smoke,
        "landmark_version": config.LANDMARK_VERSION,
        "labels": sorted(labels),
        "glosses": {w.word: sorted(w.glosses) for w in spec.active},
        "policy": policy, "requested_splits": sorted(eval_splits),
        "samples": {name: seedreport.identity(bt.sp.samples.sort_values(
            ["video", "start", "end", "label"]).to_dict("records"))
                    for name, bt in testers.items()},
    }
    summary = {
        "candidate": seedreport.candidate_name(),
        "label_set_id": seedreport.identity(sorted(labels)),
        "evaluation_id": seedreport.identity(eval_identity),
        "evaluation_identity": eval_identity,
        "commit_policy": policy, "word_groups": word_groups(spec),
        "eval_splits": list(eval_splits),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "smoke": bool(smoke), "seed": seed,
        "epochs_requested": epochs, "epochs_ran": len(history),
        "best_val_top1": round(best, 4),
        "sample_fingerprint": fp,
        "feature_dim": train_bt.dim, "feature_version": fver,
        "feature_blocks": blocks or "all",
        "regularization": r.as_dict(),
        "num_classes": len(labels), "labels": labels, "history": history,
        "sec_per_epoch": (round(float(np.mean([h["seconds"] for h in history])), 2)
                          if history else 0.0),
        "seconds": round(time.time() - t0, 1),
    }
    if not smoke and len(P):
        # words-only accuracy of the RELEASED weights on their own training
        # split: the memorisation number the report prints next to val
        tr_probs = np.concatenate([best_net.predict(P[a:a + 256])
                                   for a in range(0, len(P), 256)])
        cols = word_columns(labels)
        wkeep = np.isin(Y, cols)
        tr_ev = {"n": int(len(Y)), "top1_all": _num(float(
            (tr_probs.argmax(1) == Y).mean())),
            "top1_words_only": _num(float(
                (tr_probs[np.ix_(wkeep, cols)].argmax(1) == Y[wkeep]).mean())),
            "top1_words_only_n": int(wkeep.sum()),
            "note": "train split with the released weights: the "
                    "memorisation check, not a held-out number"}
        summary.setdefault("eval", {})["train"] = tr_ev
    del P, Y
    for name, bt in testers.items():
        kind = ("official signer-disjoint" if name in ("val", "test")
                else "PopSign held-out, separate corpus" if name == "ps_test"
                else "smoke hold-out, not a real number")
        if name == "train" and not smoke:
            kind = "train split (memorisation check, not a held-out number)"
        log.info("=== %s split (%s) ===", name, kind)
        write_split_report(bt, best_net, labels, name, summary, log,
                           reports_dir)

    data_splits = ["train", "val", "test"]
    if len(man_all[man_all["split"] == "ps_test"]):
        data_splits.append("ps_test")
    summary["data"] = {s: dataset.describe(
        dataset.build_samples((s,), None, dataset._manifest((s,))))
        for s in data_splits}
    summary["split_check"] = split_problems
    summary["env"] = config.env_summary()
    summary["vocab_batches"] = list(spec.enabled_batches)
    summary["augment"] = {k: list(v) if isinstance(v, tuple) else v
                          for k, v in config.AUG.items()}
    (reports_dir / "words_summary.json").write_text(
        json.dumps(_clean(summary), indent=1, default=str), encoding="utf-8")

    ev = summary.get("eval", {})
    test_stats = ev.get("test") or {}
    meta = {
        "candidate": summary["candidate"],
        "label_set_id": summary["label_set_id"],
        "evaluation_id": summary["evaluation_id"],
        "commit_policy": policy,
        "model": "isolated ASL word recognizer (landmarks, not pixels)",
        "created": summary["trained_at"],
        "labels": labels, "num_classes": len(labels),
        "vocab_batches": summary["vocab_batches"],
        "feature_dim": train_bt.dim, "num_frames": config.NUM_FRAMES,
        "landmark_version": config.LANDMARK_VERSION,
        "feature_version": fver,
        "train_words_top1": (summary.get("eval", {}).get("train") or {})
        .get("top1_words_only"),
        "arch": arch.as_dict(), "params": net.n_params(),
        "regularization": r.as_dict(),
        "train_samples": len(train_bt),
        "train_signers": train_bt.sp.signers(),
        "train_videos": int(train_bt.sp.samples["video"].nunique()),
        "best_val_top1": summary["best_val_top1"],
        "epochs_ran": len(history), "seed": seed,
        "smoke": smoke,
        "sample_fingerprint": fp,
        "eval": ev,
        "test_top1": test_stats.get("top1_all"),
        "test_n_signers": test_stats.get("n_signers"),
        "licence": f"trained on ASL Citizen under the terms in "
                   f"{config.LICENSE_FILE}; non-commercial research only",
        "note": "scored on the official signer-disjoint split; no webcam "
                "data was used for training or calibration, and nothing from "
                "a camera is ever written to disk",
    }
    meta_json.parent.mkdir(parents=True, exist_ok=True)
    meta_json.write_text(json.dumps(_clean(meta), indent=1, default=str),
                         encoding="utf-8")
    log.info("saved %s (%.1f MB) and %s, %.1f s total",
             release, release.stat().st_size / 1e6 if release.is_file() else 0,
             meta_json, time.time() - t0)
    # A finished run must leave no resume state behind: that leftover file is
    # what made run_all.py want to train again after everything had succeeded.
    if ckpt.is_file():
        ckpt.unlink()
        log.info("cleared the resume checkpoint %s - nothing to resume", ckpt)
    return summary


# ---------------------------------------------------------------------------
# --parity: the live feature path must be the training feature path
# ---------------------------------------------------------------------------
def parity(n: int = 5, log=None) -> int:
    log = log or config.logger_for("parity")
    labels = vocab.load_vocab().labels
    bt = None
    for s in ("test", "val", "train"):
        bt = make_batcher(s, labels, n, None)
        if bt is not None:
            log.info("parity checked on the '%s' split cache", s)
            break
    if bt is None:
        log.error("nothing cached; run python -m src.extract --splits train")
        return 2
    # the version that matters is the one the deployed weights expect
    fver = config.FEATURE_VERSION
    net = None
    if config.FOR_RELEASE.is_file():
        net, meta0 = model.load(config.FOR_RELEASE)
        fver = (meta0.get("feature_version")
                or features.version_for_dim(net.arch.in_dim)
                or config.FEATURE_VERSION)
    worst = 0.0
    for i in range(min(n, len(bt))):
        arr = np.ascontiguousarray(bt.sp.packed(i).astype(np.float32))
        a = features.sequence(arr, version=fver)
        buf = features.RingBuffer(maxlen=len(arr) + 1)
        for row in arr:
            buf.append(row)
        worst = max(worst, float(np.abs(a - features.from_live_buffer(
            buf, version=fver)).max()))
    log.info("training path vs live buffer path (feature v%d): max abs "
             "difference %.3e over %d clips", fver, worst, min(n, len(bt)))
    if net is None:
        log.warning("no %s yet, model parity not checked", config.FOR_RELEASE)
        return 0 if worst == 0 else 1
    x = np.stack([features.sequence(np.ascontiguousarray(
        bt.sp.packed(i).astype(np.float32)), version=fver)
        for i in range(min(n, len(bt)))])
    y = bt.y.numpy()[: len(x)]
    probs = net.predict(x)
    log.info("model on those features: top-1 %d/%d, mean p1 %.3f",
             int((probs.argmax(1) == y).sum()), len(x),
             float(probs.max(1).mean()))
    log.info("checkpoint labels match vocab.json: %s",
             meta0.get("labels") == labels)
    return 0 if worst == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the word recognizer.")
    ap.add_argument("--candidate", help="isolate all outputs under candidates/NAME")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap samples per split (smoke test / quick look)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the checkpoint and start at epoch 0")
    ap.add_argument("--blocks", default="all",
                    help="feature block ablation, e.g. pose,hands,mask")
    ap.add_argument("--eval-splits", default="val,test,ps_test")
    ap.add_argument("--smoke", action="store_true",
                    help="3 samples, 1 epoch, scratch outputs only")
    ap.add_argument("--parity", action="store_true",
                    help="prove the live feature path equals the training one")
    ap.add_argument("--reg", default=None, choices=sorted(REG_PRESETS),
                    help="regularization preset (default: none = what ran "
                         "on 2026-11-16)")
    for spec in (("--dropout", float, "override the preset's dropout"),
                 ("--weight-decay", float, "AdamW weight decay"),
                 ("--label-smoothing", float, "cross-entropy smoothing"),
                 ("--hidden", int, "GRU hidden size (smaller model)"),
                 ("--gru-layers", int, "GRU layers (1 = smaller model)")):
        ap.add_argument(spec[0], type=spec[1], default=None, help=spec[2])
    ap.add_argument("--conv-channels", default=None,
                    help="e.g. 48,64 - the two conv widths")
    ap.add_argument("--aug", action="append", default=[], metavar="KEY=VALUE",
                    help="override one augmentation knob, e.g. "
                         "--aug jitter=0.03 --aug speed=0.7,1.3")
    ap.add_argument("--seeds", type=int, default=1,
                    help="train N seeds of this config and report mean +- "
                         "spread (writes reports/seed_runs.csv)")
    ap.add_argument("--tag", default="",
                    help="name for this sweep row, e.g. baseline / v3features")
    ap.add_argument("--pairs", default="good/thanks,happy/please",
                    help="swap counts to report, slash separated pairs")
    ap.add_argument("--gate", type=float, default=None,
                    help="confidence gate for the gated metrics "
                         f"(default config.CONFIDENCE_MIN = "
                         f"{config.CONFIDENCE_MIN})")
    ap.add_argument("--promote", default="best", choices=("best", "none"),
                    help="after a sweep: copy the VAL-best run over "
                         "models/words_core20.pt, or measure only")
    ap.add_argument("--fair", action="store_true",
                    help="train feature v3 AND the old v2 vector with the "
                         "same seeds and config, decide on VAL (adopt v3 "
                         "only above the pooled seed spread, else keep the "
                         "simpler v2), promote the winner and keep the "
                         "loser loadable as models/words_v<ver>.pt")
    ap.add_argument("--compare", action="store_true",
                    help="print the baseline-vs-new table and exit")
    ap.add_argument("--allow-vocab-change", action="store_true",
                    help="permit writing models/words_core20.pt when the label "
                         "set differs from the shipped checkpoint's")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    # One switch, read by every code path that can write the release weights.
    # `global` is not needed here because the guard lives in this module and the
    # flag is set before any training starts.
    global ALLOW_VOCAB_CHANGE
    ALLOW_VOCAB_CHANGE = bool(args.allow_vocab_change)
    # Reject unsupported combinations before even opening a log file.
    if args.fair and args.promote == "none":
        ap.error("--fair does not support --promote none; use separate --seeds sweeps")
    if args.seeds < 1 or (args.fair and args.seeds < 2):
        ap.error("--seeds must be >= 1 (>= 2 for --fair)")
    try:
        eval_splits = evaluation_splits(args.eval_splits, smoke=args.smoke)
        candidate_root = config.use_candidate(args.candidate) if args.candidate else None
    except ValueError as exc:
        ap.error(str(exc))
    if candidate_root is not None and args.log:
        if not Path(args.log).resolve().is_relative_to(candidate_root.resolve()):
            ap.error("candidate --log must stay inside its candidate directory")

    if args.compare:
        print(compare_table())
        return 0

    if args.parity:
        return parity(log=config.logger_for(
            "parity", Path(args.log) if args.log else None))

    log = config.logger_for("train", Path(args.log) if args.log
                            else config.LOGS_DIR / "train.log")
    log.info("env: %s", json.dumps(config.env_summary()))
    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]

    over: dict = {}
    for key, val in (("dropout", args.dropout),
                     ("weight_decay", args.weight_decay),
                     ("label_smoothing", args.label_smoothing),
                     ("hidden", args.hidden), ("gru_layers", args.gru_layers)):
        if val is not None:
            over[key] = val
    if args.conv_channels:
        over["conv_channels"] = tuple(int(x) for x in
                                      args.conv_channels.split(",") if x.strip())
    if args.aug:
        over["aug"] = parse_aug_flags(args.aug)
    pairs = []
    for p in (x.strip() for x in args.pairs.split(",") if x.strip()):
        parts = [q.strip().lower() for q in p.split("/") if q.strip()]
        if len(parts) != 2:
            raise SystemExit(f"--pairs wants word/word, got '{p}'")
        pairs.append((parts[0], parts[1]))
    base_seed = config.SEED if args.seed is None else int(args.seed)

    if args.fair:
        if args.smoke:
            log.warning("SMOKE fair: 1 epoch, a few samples, decides and "
                        "prints but promotes nothing")
        base_tag = args.tag or f"fair-{time.strftime('%Y%m%d-%H%M')}"
        tags = {config.FEATURE_VERSION: f"{base_tag}-v{config.FEATURE_VERSION}",
                2: f"{base_tag}-v2"}
        fair([base_seed + i for i in range(args.seeds)],
             epochs=1 if args.smoke else args.epochs, batch=args.batch,
             limit=6 if args.smoke else args.limit, blocks=blocks,
             reg_name=args.reg, aug_over=over, pairs=pairs, gate=args.gate,
             tags=tags, smoke=args.smoke, log=log, eval_splits=eval_splits)
        return 0

    if args.smoke:
        scratch = config.CKPT_DIR / "smoke"
        scratch.mkdir(parents=True, exist_ok=True)
        log = config.logger_for("train", scratch / "smoke_train.log")
        log.warning("SMOKE TEST: 1 epoch, a handful of samples, everything in "
                    "%s - this produces no real number", scratch)
        if args.seeds > 1:
            log.warning("--seeds %d with --smoke: this only proves the sweep "
                        "runs, the numbers mean nothing", args.seeds)
            seed_sweep([base_seed + i for i in range(args.seeds)], epochs=1,
                       limit=6, blocks=blocks or None, reg_name=args.reg,
                       aug_over=over, pairs=pairs, gate=args.gate,
                       tag=args.tag or "smoke", log=log,
                       promote_best=False, smoke=True, eval_splits=eval_splits)
            return 0
        out = train(epochs=1, limit=6, seed=args.seed, smoke=True, log=log,
                    eval_splits=eval_splits, gate=args.gate,
                    ckpt=scratch / "state.pt",
                    release=scratch / "words_smoke.pt",
                    reports_dir=scratch, meta_json=scratch / "smoke_meta.json",
                    blocks=blocks or None, reg=make_reg(args.reg, over))
        print(json.dumps({"epochs_ran": out["epochs_ran"],
                          "feature_dim": out["feature_dim"],
                          "num_classes": out["num_classes"],
                          "sec_per_epoch": out["sec_per_epoch"],
                          "regularization": out["regularization"],
                          "eval": out.get("eval")}, indent=1, default=str))
        return 0
    if args.seeds > 1 or args.promote == "none":
        seed_sweep([base_seed + i for i in range(args.seeds)],
                   epochs=args.epochs, batch=args.batch, limit=args.limit,
                   blocks=blocks, reg_name=args.reg, aug_over=over,
                   pairs=pairs, gate=args.gate, tag=args.tag,
                   promote_best=args.promote == "best", log=log,
                   eval_splits=eval_splits)
        return 0
    train(epochs=args.epochs, batch=args.batch, limit=args.limit,
          seed=args.seed, fresh=args.fresh, blocks=blocks, log=log,
          reg=make_reg(args.reg, over), gate=args.gate,
          eval_splits=eval_splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
