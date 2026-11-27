"""Train the J/Z stroke head: the model, not a rule, names J and Z.

Data: the MSL dynamic alphabet shards already on disk (`cache/msl_landmarks`,
622 clips, letters J K N-tilde Q X Z, 20 signers, real strokes at 30 fps). The
classes are J / Z / other - "other" is every other moving letter in the corpus
plus motionless windows, so the head has to tell a hook from a zigzag from a
hand that is merely moving.

Honest framing, because this corpus is Mexican SL: the *strokes* transfer (J is
a hook in front of the palm, Z is a zigzag in the air in both languages), the
letters' handshapes are close but not identical. The measurement that decides
whether this ships is on signers the model never saw, and those numbers are
written into `reports/jz_model.md` rather than assumed.

    python src/train_jz.py                 # cross-validate, then fit the shipped head
    python src/train_jz.py --seeds 3       # ... and report seed spread
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

if __package__:
    from . import config, jz_features as JF, normalize
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config  # noqa: E402
    import jz_features as JF  # noqa: E402
    import normalize  # noqa: E402

LETTERS = ("J", "Z")           # the two classes the head exists for
CLASSES = ("J", "Z", "other")  # "other" = any other stroke, or no stroke
NEGATIVE_LETTERS = ("K", "Q", "X", "\u00d1")   # a live moving letter looks like these

WIN_S = JF.WINDOW_S
STRIDE_S = 0.30
MIN_WIN_S = 0.60
STILL_FRACTION = 0.25      # share of "other" made of motionless windows


def _clip_windows(lm: np.ndarray, times: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slide WIN_S windows over one clip, in seconds (not frames)."""
    out = []
    if len(times) < 6:
        return out
    t0, t1 = float(times[0]), float(times[-1])
    if t1 - t0 < MIN_WIN_S:
        return out
    start = t0
    while start + MIN_WIN_S <= t1 + 1e-9:
        stop = min(start + WIN_S, t1)
        sel = (times >= start - 1e-9) & (times <= stop + 1e-9)
        if sel.sum() >= 6:
            out.append((lm[sel], times[sel]))
        if stop >= t1:
            break
        start += STRIDE_S
    return out


def _still_windows(lm: np.ndarray, times: np.ndarray, rng, want: int):
    """Windows from a single held frame: the 'a held letter is not a stroke' prior.

    A real camera holds a letter still for a second before moving to the next
    one; without these, the head only ever sees moving hands and fires on them.
    """
    out = []
    if len(lm) < 2:
        return out
    step = max(1.0 / 30.0, float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30)
    for _ in range(want):
        i = int(rng.integers(0, len(lm)))
        n = max(8, int(round(WIN_S / step)))
        base = np.asarray(lm[i], dtype=np.float64)
        jitter = rng.normal(0.0, 0.0015, size=base.shape)   # sensor shake only
        frames = np.repeat(base[None, :, :], n, axis=0) + jitter
        t = np.arange(n) * step
        out.append((frames, t))
    return out


def _gate_share(lms: np.ndarray) -> float:
    """Share of frames with 'index extended, every other finger curled'.

    The live decider only asks the stroke head once a shape gate holds, so this
    is what decides which offline windows the head is ever really shown.
    """
    up = 0
    for lm in lms:
        c = normalize.canonicalize(np.asarray(lm, dtype=np.float64))
        ratios = []
        for mcp, pip, _dip, tip in config.FINGER_CHAINS:
            a = float(np.linalg.norm(c[pip, :2] - c[mcp, :2]))
            b = float(np.linalg.norm(c[tip, :2] - c[mcp, :2]))
            ratios.append(b / max(a, 1e-6))
        if ratios[1] >= 1.6 and all(r <= 1.20 for r in ratios[2:]):
            up += 1
    return up / max(len(lms), 1)


def build_dataset(seed: int = 0, verbose: bool = True):
    """All windows on disk -> (X, y, signer, clip, split, aux).

    `aux` carries, per window: the shape-gate share and the 71-dim static
    features of its last frame. The live decider opens on the shape gate *or* on
    the letter classifier reading I/X, and only those windows are ones the head
    is ever asked about - `main` filters with exactly that condition.
    """
    shards = sorted(glob.glob(str(config.CACHE_DIR / "msl_landmarks" / "*.npz")))
    if not shards:
        raise SystemExit(
            "no J/Z shards in cache/msl_landmarks - nothing to train on.\n"
            "  python src/download_jz_data.py && python src/extract_jz_landmarks.py")
    rng = np.random.default_rng(seed)
    X, y, signer, clip, split = [], [], [], [], []
    gates, lasts = [], []
    per_letter = Counter()
    for path in shards:
        z = np.load(path, allow_pickle=True)
        lm = np.asarray(z["lm"], dtype=np.float64)
        fps = float(z["fps"]) if float(z["fps"]) > 1 else 30.0
        idx = np.asarray(z["idx"], dtype=np.float64) if "idx" in z else None
        times = (idx if idx is not None and len(idx) == len(lm)
                 else np.arange(len(lm), dtype=np.float64)) / fps
        letter = str(z["letter"])
        label = letter if letter in LETTERS else "other"
        wins = _clip_windows(lm, times)
        for frames, t in wins:
            X.append(JF.features_from_frames(frames, t))
            y.append(CLASSES.index(label))
            signer.append(str(z["signer"]))
            clip.append(Path(path).name)
            split.append(str(z["split"]))
            gates.append(_gate_share(frames))
            lasts.append(normalize.features(frames[-1]))
            per_letter[letter] += 1
        n_still = max(1, int(round(len(wins) * STILL_FRACTION)))
        for frames, t in _still_windows(lm, times, rng, n_still):
            X.append(JF.features_from_frames(frames, t))
            y.append(CLASSES.index("other"))
            signer.append(str(z["signer"]))
            clip.append(Path(path).name + "#still")
            split.append(str(z["split"]))
            gates.append(_gate_share(frames))
            lasts.append(normalize.features(frames[-1]))
            per_letter[letter + "/still"] += 1
    if verbose:
        print(f"  windows: {len(X)}  ({dict(per_letter)})")
        print(f"  classes: {dict(Counter(np.asarray(y)))}  signers: "
              f"{len(set(signer))}")
    aux = {"gate": np.asarray(gates, dtype=np.float64),
           "last": np.stack(lasts).astype(np.float32)}
    return (np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64),
            np.asarray(signer), np.asarray(clip), np.asarray(split), aux)


def _torch():
    import torch
    # A 190->128->64->3 net is far below the crossover where intra-op threads
    # help: with torch's default (one thread per core) the CV loop spent 20x
    # the wall time and burned every core. One thread is both faster and
    # reproducible. Override with ASL_TORCH_THREADS if you ever widen the net.
    threads = int(os.environ.get("ASL_TORCH_THREADS", "1") or 1)
    torch.set_num_threads(max(1, threads))
    torch.manual_seed(0)
    return torch


def _net(dim: int, hidden=(128, 64), dropout=0.2):
    torch = _torch()
    layers: list = []
    prev = dim
    for h in hidden:
        layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        prev = h
    layers.append(torch.nn.Linear(prev, len(CLASSES)))
    return torch.nn.Sequential(*layers)


def fit(X: np.ndarray, y: np.ndarray, seed: int = 0, epochs: int = 320,
        hidden=(128, 64), dropout=0.2, lr: float = 2e-3, wd: float = 1e-4):
    """Fit one head on standardised inputs; returns (net, mean, std)."""
    torch = _torch()
    torch.manual_seed(seed)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xs = (X - mean) / std
    net = _net(X.shape[1], hidden, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = torch.tensor((counts.sum() / (len(CLASSES) * counts)), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    xt = torch.from_numpy(Xs.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.int64))
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    return net, mean, std


def predict(net, mean, std, X: np.ndarray) -> np.ndarray:
    torch = _torch()
    with torch.no_grad():
        logits = net(torch.from_numpy(((X - mean) / std).astype(np.float32)))
        return torch.softmax(logits, 1).numpy()


def _metrics(probs: np.ndarray, y: np.ndarray, thr: dict | None = None) -> dict:
    """Per-class precision/recall; with `thr`, what the live decider would fire."""
    out: dict = {}
    if thr is None:
        pred = probs.argmax(1)
        for i, name in enumerate(CLASSES):
            tp = int(((pred == i) & (y == i)).sum())
            fp = int(((pred == i) & (y != i)).sum())
            fn = int(((pred != i) & (y == i)).sum())
            out[name] = {"n": int((y == i).sum()), "tp": tp, "fp": fp, "fn": fn,
                         "precision": tp / max(tp + fp, 1),
                         "recall": tp / max(tp + fn, 1)}
        return out
    # live decider: fire the allowed letter whose probability clears its threshold
    fired = np.full(len(y), -1)
    for i, name in enumerate(LETTERS):
        ci = CLASSES.index(name)
        ok = (probs[:, ci] >= thr[name]) & (probs[:, ci] > probs[:, CLASSES.index("other")])
        fired[ok & (fired < 0)] = ci
    for i, name in enumerate(LETTERS):
        ci = CLASSES.index(name)
        tp = int(((fired == ci) & (y == ci)).sum())
        fp = int(((fired == ci) & (y != ci)).sum())
        fn = int(((fired != ci) & (y == ci)).sum())
        out[name] = {"n": int((y == ci).sum()), "fired": tp + fp, "tp": tp, "fp": fp,
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
    return out


def _stroke_metrics(probs: np.ndarray, y: np.ndarray, strokes: np.ndarray,
                    thr: dict | None = None) -> dict:
    """Per-class numbers counted per *stroke*, which is what live decides on.

    A stroke is one clip: the decider fires when the probability peaks over its
    threshold on any of the clip's gated windows, then the cooldown swallows the
    rest. Scoring windows instead understates the model - a stroke counts as
    recognised if *any* window clears the bar, not if all of them do.
    """
    out: dict = {}
    ids = np.unique(strokes)
    for name in LETTERS:
        ci = CLASSES.index(name)
        is_cls = {q: bool((y[strokes == q] == ci).any()) for q in ids}
        if thr is None:
            fired = {q for q in ids
                     if int(probs[strokes == q].argmax(1)[0]) == ci and
                     (probs[strokes == q].argmax(1) == ci).any()}
        else:
            fire = (probs[:, ci] >= thr[name]) & (probs[:, ci] > probs[:, CLASSES.index("other")])
            fired = set(np.unique(strokes[fire]).tolist())
        tp = sum(1 for q in fired if is_cls[q])
        n_pos = sum(1 for q in ids if is_cls[q])
        out[name] = {"strokes": n_pos, "fired": len(fired), "tp": tp,
                     "precision": tp / max(len(fired), 1),
                     "recall": tp / max(n_pos, 1)}
    return out


def _curve(probs: np.ndarray, y: np.ndarray, name: str,
           groups: np.ndarray | None = None,
           strokes: np.ndarray | None = None) -> list[dict]:
    """Precision/recall at each threshold, per window *and* per stroke.

    `worst` is the worst signer's precision, because pooled precision hides one
    signer the head fails on - and that signer is the new user. Selection uses
    the stroke numbers when clip ids are supplied: that is the unit live fires
    on, and one bad window inside a good stroke is not an error.
    """
    ci, other = CLASSES.index(name), CLASSES.index("other")
    clip_group: dict = {}
    if strokes is not None and groups is not None:
        clip_group = {q: g for q, g in zip(strokes, groups)}
    clip_ids = np.unique(strokes) if strokes is not None else None
    clip_is_cls = ({q: bool((y[strokes == q] == ci).any()) for q in clip_ids}
                   if clip_ids is not None else None)
    rows = []
    for t in np.arange(0.30, 0.951, 0.05):
        fire = (probs[:, ci] >= t) & (probs[:, ci] > probs[:, other])
        n = int(fire.sum())
        prec = float((y[fire] == ci).mean()) if n else 0.0
        recall = float(((y == ci) & fire).sum() / max((y == ci).sum(), 1))
        # A signer with 3 fires is a coin flip, not evidence: only groups with
        # enough fires to mean something are allowed to set the floor.
        worst = None
        if groups is not None and n:
            per = [float((y[fire & (groups == g)] == ci).mean())
                   for g in sorted(set(groups))
                   if (fire & (groups == g)).sum() >= 8]
            worst = min(per) if per else None
        row = {"threshold": round(float(t), 2), "fired": n, "precision": prec,
               "recall": recall, "worst_signer": worst}
        if clip_ids is not None:
            fired_ids = np.unique(strokes[fire])
            s_tp = sum(1 for q in fired_ids if clip_is_cls[q])
            s_pos = sum(1 for q in clip_ids if clip_is_cls[q])
            s_worst = None
            if clip_group:
                per = [float(sum(1 for q in fired_ids if clip_is_cls[q] and clip_group[q] == g)
                             / max(sum(1 for q in fired_ids if clip_group[q] == g), 1))
                       for g in sorted(set(clip_group.values()))
                       if sum(1 for q in fired_ids if clip_group[q] == g) >= 3]
                s_worst = min(per) if per else None
            row.update({"s_fired": int(len(fired_ids)),
                        "s_precision": s_tp / max(len(fired_ids), 1),
                        "s_recall": s_tp / max(s_pos, 1), "s_worst_signer": s_worst})
        rows.append(row)
    return rows


def choose_thresholds(probs: np.ndarray, y: np.ndarray, target: float = 0.90,
                      min_fires: int = 8, min_thr: float = 0.50,
                      worst_floor: float = 0.60, groups: np.ndarray | None = None,
                      curve: list | None = None,
                      strokes: np.ndarray | None = None,
                      prefer_strokes: bool = False) -> dict:
    """Per-letter threshold at >= `target` precision, highest recall.

    Selection is window-level by default. Stroke-level precision is reported
    next to it but is a coarser statistic (53 Z strokes against 184 windows), so
    a single confused stroke moves it several points; measuring with it turns a
    measurement choice into a model change. `prefer_strokes=True` selects on the
    stroke numbers instead, for experimenting only. Two guards stop the search
    from picking a point that only looks good pooled: a floor of `min_thr` and
    `worst_floor` on the worst signer's precision. A letter with no qualifying
    point gets 1.01 - refusing is honest, a coin flip is not.
    """
    thr = {}
    for name in LETTERS:
        rows = curve if curve is not None else _curve(probs, y, name, groups, strokes)
        best, best_recall = 1.01, -1.0
        for row in rows:
            t = row["threshold"]
            if t < min_thr:
                continue
            if prefer_strokes and "s_fired" in row:
                s_prec, s_rec, s_worst, fired = (row["s_precision"], row["s_recall"],
                                                 row["s_worst_signer"], row["s_fired"])
            else:
                s_prec, s_rec, s_worst, fired = (row["precision"], row["recall"],
                                                 row["worst_signer"], row["fired"])
            if fired < min_fires or s_prec < target:
                continue
            if s_worst is not None and s_worst < worst_floor:
                continue
            if s_rec > best_recall:
                best, best_recall = float(t), s_rec
        thr[name] = round(best, 2) if best <= 1.0 else 1.01
    return thr


GATE_SHAPE = 0.50        # share of a window that must pass the index-up shape gate
GATE_X_P1 = 0.90         # the display gate: a confident X is an X, not a Z


def live_eligible(aux: dict, profile: str = "all") -> np.ndarray:
    """The windows the live decider would ever ask the stroke head about.

    That is the index-up shape gate, or the letter classifier reading I (J) / an
    unsure X (Z) - the same two preconditions `MotionDecider` applies. Without
    this filter the offline number is dominated by hands the app never asks
    about: one MSL signer sweeps their K and X, and no shape gate opens on them.
    """
    import torch
    from model import load_model

    meta = json.loads(config.meta_path(profile).read_text(encoding="utf-8"))
    net = load_model(config.model_path(profile, must_exist=True),
                     meta["model"]["in_dim"], len(meta["labels"]),
                     tuple(meta["model"]["hidden"]), meta["model"]["dropout"])
    with torch.no_grad():
        logits = net(torch.from_numpy(aux["last"].astype(np.float32)))
        probs = torch.softmax(logits, 1).numpy()
    labels = meta["labels"]
    top1 = [labels[int(i)] for i in probs.argmax(1)]
    p1 = probs.max(1)
    display = np.asarray([(t == "I") or (t == "X" and p < GATE_X_P1)
                          for t, p in zip(top1, p1)], dtype=bool)
    shape = aux["gate"] >= GATE_SHAPE
    return display | shape


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the J/Z stroke head.")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=320)
    ap.add_argument("--target-precision", type=float, default=0.90)
    ap.add_argument("--report", default=str(config.REPORTS_DIR / "jz_model.md"))
    args = ap.parse_args()

    t0 = time.time()
    print("== J/Z stroke head ==")
    X, y, signer, clip, split, aux = build_dataset()
    dim = int(X.shape[1])
    ask = live_eligible(aux)
    print(f"  live gate opens on {int(ask.sum())}/{len(ask)} windows "
          f"({ask.mean()*100:.0f}%)")
    print(f"  feature dim {dim} (jz_features v{JF.FEATURE_VERSION}), "
          f"window {WIN_S:.2f}s")

    groups = sorted(set(signer))
    test_groups = sorted({s for s, sp in zip(signer, split) if sp == "test"})
    train_groups = [g for g in groups if g not in test_groups]
    print(f"  signers: {len(groups)} ({len(train_groups)} for CV, "
          f"{len(test_groups)} untouched: {', '.join(test_groups) or '-'})")

    # leave-one-signer-out on the CV signers, once per seed; the shipped head is
    # the average of those seeds, because a single seed's margins are luck.
    oof = np.zeros((len(y), len(CLASSES)), dtype=np.float64)
    seed_rows = []
    cv_only = np.isin(signer, train_groups) & ask
    cv_strokes = clip[cv_only]
    curves: dict = {}
    for sd in range(args.seeds):
        oof_sd = np.zeros_like(oof)
        for g in train_groups:
            tr = signer != g
            te = signer == g
            if len(set(y[tr])) < len(CLASSES) or te.sum() == 0:
                continue
            net, mean, std = fit(X[tr], y[tr], seed=sd, epochs=args.epochs)
            oof_sd[te] = predict(net, mean, std, X[te])
        oof = oof_sd if sd == 0 else (oof * sd + oof_sd) / (sd + 1)
        seed_thr = {}
        for name in LETTERS:
            seed_thr[name] = choose_thresholds(
                oof_sd[cv_only], y[cv_only], args.target_precision,
                groups=signer[cv_only], strokes=cv_strokes)[name]
        m_sd = _metrics(oof_sd[cv_only], y[cv_only], seed_thr)
        seed_rows.append((sd, seed_thr, m_sd))
        print(f"  seed {sd}: "
              + "  ".join(f"{k} P={m_sd[k]['precision']*100:.0f}%"
                          f" R={m_sd[k]['recall']*100:.0f}%" for k in LETTERS))

    # thresholds and every reported number come from the *ensemble* out-of-fold
    # probabilities, i.e. exactly the model that ships
    curves = {k: _curve(oof[cv_only], y[cv_only], k, signer[cv_only], cv_strokes)
              for k in LETTERS}
    thr = {name: choose_thresholds(oof[cv_only], y[cv_only], args.target_precision,
                                   groups=signer[cv_only], curve=curves[name],
                                   strokes=cv_strokes)[name]
           for name in LETTERS}
    cv_masked = _metrics(oof[cv_only], y[cv_only], thr)
    cv_stroke = _stroke_metrics(oof[cv_only], y[cv_only], cv_strokes, thr)
    cv_plain = _metrics(oof[cv_only], y[cv_only])
    m_ens = cv_masked
    print("  ensemble, per window: "
          + "  ".join(f"{k} P={m_ens[k]['precision']*100:.0f}%"
                      f" R={m_ens[k]['recall']*100:.0f}%" for k in LETTERS))
    print("  ensemble, per stroke: "
          + "  ".join(f"{k} P={cv_stroke[k]['precision']*100:.0f}%"
                      f" R={cv_stroke[k]['recall']*100:.0f}%"
                      f" ({cv_stroke[k]['tp']}/{cv_stroke[k]['strokes']})" for k in LETTERS))

    # signers never used in any fold - the only number that speaks to a new user
    held_sig = np.isin(signer, test_groups) if test_groups else np.zeros(len(y), bool)
    held = held_sig & ask
    held_m: dict = {}
    held_stroke: dict = {}
    if held_sig.any() and held.sum():
        held_probs = np.zeros((int(held.sum()), len(CLASSES)), dtype=np.float64)
        for sd in range(args.seeds):
            net_h, mean_h, std_h = fit(X[~held_sig], y[~held_sig], seed=sd,
                                       epochs=args.epochs)
            held_probs += predict(net_h, mean_h, std_h, X[held]) / args.seeds
        held_m = _metrics(held_probs, y[held], thr)
        held_stroke = _stroke_metrics(held_probs, y[held], clip[held], thr)

    # the shipped head: every signer, thresholds from the cross-validated folds
    nets = []
    mean = std = None
    for sd in range(args.seeds):
        net_sd, mean, std = fit(X, y, seed=sd, epochs=args.epochs)
        nets.append(net_sd.state_dict())
    meta = {
        "kind": "jz-stroke-head",
        "feature_version": JF.FEATURE_VERSION,
        "feature_dim": dim,
        "window_s": WIN_S,
        "steps": JF.STEPS,
        "labels": list(CLASSES),
        "thresholds": thr,
        "hidden": [128, 64],
        "dropout": 0.2,
        "ensemble": int(args.seeds),
        "train": {
            "windows": int(len(y)),
            "signers": int(len(groups)),
            "cv_signers": train_groups,
            "held_out_signers": test_groups,
            "corpus": "MSL dynamic alphabet (Zenodo 14689869), CC-BY-4.0",
            "note": ("Mexican SL strokes; J and Z are the same hook and zigzag "
                     "in ASL. Classifier handshapes are close, not identical."),
        },
        "cross_validation": {
            "scheme": "leave-one-signer-out",
            "target_precision": args.target_precision,
            "masked": cv_masked,
            "masked_strokes": cv_stroke,
            "argmax": cv_plain,
            "seeds": args.seeds,
            "per_seed": [{"seed": sd, "thresholds": t, "masked": m}
                         for sd, t, m in seed_rows],
        },
        "held_out_signers": held_m,
        "held_out_strokes": held_stroke,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.MODELS_DIR / "jz_motion.pt"
    torch = _torch()
    torch.save({"state_dicts": nets, "meta": meta,
                "mean": mean.astype(np.float32), "std": std.astype(np.float32)}, out)
    (config.MODELS_DIR / "jz_motion.meta.json").write_text(
        json.dumps(meta, indent=1, default=float), encoding="utf-8")
    print(f"  wrote {out} ({out.stat().st_size/1024:.0f} KB)  thresholds {thr}")

    spread = "  ".join(
        f"{k} recall {100*min(m[k]['recall'] for _s, _t, m in seed_rows):.0f}"
        f"-{100*max(m[k]['recall'] for _s, _t, m in seed_rows):.0f}%"
        for k in LETTERS)
    lines = [
        "# J and Z as model classes (MSL strokes -> shipped head)",
        "",
        f"Generated by `python src/train_jz.py --seeds {args.seeds}` on "
        f"{meta['trained_at']} in {time.time()-t0:.0f}s.",
        "",
        "The letter classifier is a per-frame model, so it cannot name a letter",
        "whose difference *is* a stroke. This is a second, trained head over a",
        f"{WIN_S:.2f}s window of canonicalised landmarks ({dim} features from",
        "`jz_features.py`, built by the same function offline and live). It",
        "replaces the hand-written trajectory rule as the decider; the shape",
        "gate is kept only as a precondition.",
        "",
        "## Cross-validated on unseen signers (leave-one-signer-out)",
        "",
        "**Per stroke** - the unit live decides on. A stroke fires when any of",
        "its gated windows clears the threshold; the cooldown then swallows the",
        "rest, so this is what a user experiences.",
        "",
        "| class | strokes | fired | precision | recall |",
        "|---|---|---|---|---|",
    ]
    for k in LETTERS:
        m = cv_stroke[k]
        lines.append(f"| {k} | {m['strokes']} | {m['fired']} | "
                     f"{100*m['precision']:.1f}% | {100*m['recall']:.1f}% |")
    lines += [
        "",
        "**Per window**, for comparison - a single weak window inside an",
        "otherwise good stroke is not an error live ever commits.",
        "",
        "| class | n | fired | precision | recall |",
        "|---|---|---|---|---|",
    ]
    for k in LETTERS:
        m = cv_masked[k]
        lines.append(f"| {k} | {m['n']} | {m['fired']} | "
                     f"{100*m['precision']:.1f}% | {100*m['recall']:.1f}% |")
    lines += [
        "",
        f"Shipped head = mean of {args.seeds} seeds; per-seed recall "
        f"spread on the same folds: {spread}.",
        "",
        f"Thresholds chosen at >= {100*args.target_precision:.0f}% pooled *window* precision,",
        "a floor of 0.50, and >= 60% precision for the worst signer: "
        + ", ".join(f"{k} {thr[k]:.2f}" for k in LETTERS) + ".",
        "A threshold of 1.01 means the head never fires that letter - refusing",
        "is preferred over guessing.",
        "",
        "## The whole curve, so the operating point is a choice and not a number",
        "",
    ]
    for name in LETTERS:
        lines += [f"**{name}**", "",
                  "| threshold | stroke fired | stroke precision | stroke recall | "
                  "window precision | window recall |",
                  "|---|---|---|---|---|---|"]
        for row in curves.get(name, []):
            lines.append(f"| {row['threshold']:.2f} | {row.get('s_fired', '-')} | "
                         f"{100*row.get('s_precision', row['precision']):.1f}% | "
                         f"{100*row.get('s_recall', row['recall']):.1f}% | "
                         f"{100*row['precision']:.1f}% | {100*row['recall']:.1f}% |")
        lines.append("")
    lines += [
        "Lowering a threshold buys recall at the price of firing on other",
        "letters. The stroke columns are the live decision; the window columns",
        "are what the head believes frame by frame, and they are always lower.",
        "",
        "## Signers never used in any fold",
        "",
    ]
    if held_m:
        lines += ["| class | strokes | fired | precision | recall |", "|---|---|---|---|---|"]
        for k in LETTERS:
            m = (held_stroke or {}).get(k) or dict(held_m[k], strokes="-")
            lines.append(f"| {k} | {m.get('strokes', '-')} | {m['fired']} | "
                         f"{100*m['precision']:.1f}% | {100*m['recall']:.1f}% |")
        lines += ["", "Same numbers per window:", "",
                  "| class | n | fired | precision | recall |", "|---|---|---|---|---|"]
        for k in LETTERS:
            m = held_m[k]
            lines.append(f"| {k} | {m['n']} | {m['fired']} | "
                         f"{100*m['precision']:.1f}% | {100*m['recall']:.1f}% |")
    else:
        lines.append("No signer was left out of the corpus, so this is empty.")
    lines += [
        "",
        "## Why the training corpus is Mexican SL",
        "",
        "There is no ASL J/Z motion corpus on this machine: the ASL photo",
        "datasets are single frames, so a J photograph contains no stroke. The",
        "MSL dynamic alphabet is real fingerspelled J and Z at 30 fps, and both",
        "letters are the same stroke in ASL (J hooks in front of the palm, Z",
        "zigzags in the air). The handshape differs slightly, which is why the",
        "shape gate still has to pass before this head is asked.",
        "",
        "Reproduce: `python src/train_jz.py`; measure the live path with",
        "`python src/live.py --selftest --letters J,Z`.",
        "",
    ]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
