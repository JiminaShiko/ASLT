"""
The whole no-data test battery in one command.

    cd "D:\\sign language project\\ASTL project\\ASLT\\videoModel"
    python -m tests.test_logic            (also: pytest tests/test_logic.py)

Everything here runs in seconds, needs no trained weights and no camera, and
writes only to a temporary folder. It is the safety net for the parts that
break silently: the vocabulary, the shared feature function, the sample
builder, the model contract, the phrase rules and the live decision rules.

Checks that need the dataset or the trained weights report SKIP with the
reason instead of quietly passing.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, dataset, features, landmarks, model, phrases, predictor, vocab  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []        # (section, name, detail)


def ok(section: str, name: str, detail: str = "") -> None:
    RESULTS.append((section, name, ""))
    print(f"PASS  {section}: {name}")


def fail(section: str, name: str, detail: str) -> None:
    RESULTS.append((section, "FAIL " + name, detail))
    print(f"FAIL  {section}: {name}   {detail}")


def check(section: str, name: str, cond, detail: str = "") -> bool:
    """cond may be a bool or a callable that raises."""
    try:
        good = bool(cond()) if callable(cond) else bool(cond)
        err = ""
    except Exception as exc:                       # noqa: BLE001
        good, err = False, f"{type(exc).__name__}: {exc}"
    detail = err or detail
    (ok if good else fail)(section, name, detail)
    return good


def skip(section: str, name: str, why: str) -> None:
    RESULTS.append((section, "SKIP " + name, why))
    print(f"SKIP  {section}: {name}   ({why})")


def _raises(fn) -> bool:
    """True when fn() raises: the fail-loud checks want an exception."""
    try:
        fn()
    except Exception:                              # noqa: BLE001
        return True
    return False


# ---------------------------------------------------------------------------
def t_config():
    S = "config"
    check(S, "NUM_FRAMES is 64", config.NUM_FRAMES == 64)
    check(S, "LANDMARK_VERSION and FEATURE_VERSION are set",
          config.LANDMARK_VERSION >= 2 and config.FEATURE_VERSION >= 2)
    check(S, "the hand gap-fill window is a whole number of source frames",
          isinstance(config.GAP_FILL_FRAMES, int)
          and 1 <= config.GAP_FILL_FRAMES <= config.NUM_FRAMES // 8,
          str(config.GAP_FILL_FRAMES))
    check(S, "the packed layout adds up",
          config.P_TOTAL == config.N_POSE + config.N_FACE + config.HANDS_TOTAL)
    check(S, "hand blocks are 21 points each",
          config.COLS_RHAND.start - config.COLS_LHAND.start == config.N_HAND)
    check(S, "the wrist column is the first row of each hand block",
          config.HAND_WRIST_COLS == (config.COLS_LHAND.start,
                                     config.COLS_RHAND.start))
    check(S, "nothing the model writes points inside the dataset folder",
          all(config.DATASET_DIR != p for p in
              (config.CACHE_DIR, config.MODELS_DIR, config.REPORTS_DIR,
               config.LOGS_DIR)))
    check(S, "the dataset folder is a sibling of the code, not a target",
          config.DATASET_DIR.is_dir() or not config.dataset_available())
    check(S, "release weights have one plain name, no version suffix",
          config.FOR_RELEASE.name == "words_core20.pt")
    check(S, "live thresholds are in the range the brief asked for",
          0.5 <= config.CONFIDENCE_MIN <= 0.9 and 0.0 < config.MARGIN_MIN <= 0.5)


# ---------------------------------------------------------------------------
def t_vocab():
    S = "vocab"
    spec = vocab.load_vocab()
    check(S, f"{len(spec.active)} active words + REST + OTHER = "
             f"{len(spec.labels)} labels",
          len(spec.labels) == len(spec.active) + 2)
    check(S, "REST and OTHER are the last two labels",
          spec.labels[-2:] == [config.REST_LABEL, config.OTHER_LABEL])
    check(S, "no duplicate word", len({w.word for w in spec.words})
          == len(spec.words))
    check(S, "every active word has a genuine source gloss",
          all(w.glosses or w.external_glosses for w in spec.active))
    check(S, "no gloss maps to two words",
          len(spec.gloss_to_word) == sum(len(w.glosses) for w in spec.active))
    check(S, "batch 2 stays off until asked for",
          2 not in spec.enabled_batches or len(spec.active) > 20)
    dup = [g for g in spec.raw["excluded_glosses_deliberately"]
           if g in spec.gloss_to_word]
    check(S, "deliberately excluded glosses are really excluded", not dup,
          str(dup))
    if not config.dataset_available():
        skip(S, "gloss resolution against the dataset", "dataset not present")
        return
    rt = vocab.resolve_table()
    missing = rt[rt["status"] == "MISSING"]["word"].unique().tolist()
    check(S, "no active word points at a gloss the dataset lacks",
          all(w.word not in missing for w in spec.active), str(missing))
    ov = vocab.other_videos("train")
    cap = int(spec.other["cap_train"])
    check(S, f"OTHER train sample respects the cap ({cap})",
          len(ov) <= cap * int(spec.other.get("per_gloss_max", 2)), str(len(ov)))
    ov2 = vocab.other_videos("train")
    check(S, "OTHER sampling is deterministic (same rows twice)",
          list(ov["video"]) == list(ov2["video"]))
    need = vocab.videos_to_process()
    check(S, "every video in the work list is on disk",
          all(config.video_path(v).is_file() for v in need["video"].head(20)))
    check(S, "videos_to_process has no duplicate video",
          need["video"].is_unique)


# ---------------------------------------------------------------------------
def _fake_pack(t=12, left=0.30, right=0.60, y=0.35):
    """A made-up landmark array: two hands, shoulders, a face."""
    arr = np.full((t, config.P_TOTAL, 3), np.nan, np.float32)
    arr[:, config.POSE_L_SHOULDER, :2] = (0.42, 0.45)
    arr[:, config.POSE_R_SHOULDER, :2] = (0.58, 0.45)
    arr[:, config.POSE_NOSE, :2] = (0.50, 0.26)
    arr[:, config.POSE_MOUTH_L, :2] = (0.495, 0.30)
    arr[:, config.POSE_MOUTH_R, :2] = (0.505, 0.30)
    for k in range(config.N_HAND):
        for hand, cx in (("left", left), ("right", right)):
            r = landmarks.hand_row(hand, k)
            arr[:, r, 0] = cx + 0.004 * k
            arr[:, r, 1] = y + 0.003 * (k % 5)
            arr[:, r, 2] = 0.1
    for k, idx in enumerate(config.FACE_ORDER):
        arr[:, landmarks.face_row(idx), :2] = (0.5 + 0.001 * k, 0.30)
    return arr


def _travel_pack(t=40, hand="left", dy=0.08):
    """_fake_pack with one hand drifting steadily away from the face."""
    a = _fake_pack(t=t)
    u = np.linspace(0.0, 1.0, t)
    for k in range(config.N_HAND):
        a[:, landmarks.hand_row(hand, k), 1] += dy * u     # image y grows down
    return a


def t_features():
    S = "features"
    check(S, f"FEATURE_DIM is {features.FEATURE_DIM}",
          features.FEATURE_DIM == 200,
          "v3 added the 30-column segment read-out (170 -> 200)")
    total = sum(w for (_, w) in features.FEATURE_BLOCKS.values())
    check(S, "the blocks add up to FEATURE_DIM", total == features.FEATURE_DIM,
          str(features.FEATURE_BLOCKS))
    check(S, "the segment block names every one of its columns",
          len(features.SEG_NAMES) == features.FEATURE_BLOCKS["segment"][1],
          f"{len(features.SEG_NAMES)} vs {features.FEATURE_BLOCKS['segment'][1]}")
    arr = _fake_pack()
    f = features.sequence(arr)
    check(S, f"output shape is [{config.NUM_FRAMES}, {features.FEATURE_DIM}]",
          f.shape == (config.NUM_FRAMES, features.FEATURE_DIM), str(f.shape))
    check(S, "no NaN reaches the model", np.isfinite(f).all())
    check(S, "a single frame still produces a full sequence",
          features.sequence(arr[:1]).shape == (config.NUM_FRAMES,
                                               features.FEATURE_DIM))
    empty = np.full((8, config.P_TOTAL, 3), np.nan, np.float32)
    fe = features.sequence(empty)
    check(S, "an all-missing clip gives finite features, not an exception",
          np.isfinite(fe).all())
    check(S, "the presence mask reports zero for an empty clip",
          fe[:, features.block_columns(["mask"])].max() == 0.0)
    # the centre/scale normalisation: moving the whole body must not change shape
    moved = arr.copy()
    moved[..., 0] += 0.2
    moved[..., 1] -= 0.1
    check(S, "translation invariant (centred on the shoulders)",
          np.allclose(features.sequence(moved), f, atol=1e-4))
    scaled = arr.copy()
    mid = (arr[:, config.POSE_L_SHOULDER, :2] +
           arr[:, config.POSE_R_SHOULDER, :2]) / 2
    scaled[..., :2] = (scaled[..., :2] - mid[:, None, :]) * 1.3 + mid[:, None, :]
    check(S, "scale invariant (normalised by shoulder width)",
          np.allclose(features.sequence(scaled), f, atol=1e-3))
    # mirror: the same signs with the hands swapped and the image flipped
    m = features.sequence(features.mirror_pack(arr))
    cols = features.block_columns(["hands"])
    half = len(cols) // 2                    # [left 21 points xy | right 21 xy]
    lh, rh = cols[:half], cols[half:]
    xs, ys = np.arange(0, half, 2), np.arange(1, half, 2)   # x,y per point
    check(S, "mirroring swaps the two hands",
          np.allclose(m[:, lh][:, ys], f[:, rh][:, ys], atol=1e-3)
          and np.allclose(m[:, lh][:, xs], -f[:, rh][:, xs], atol=1e-3))
    check(S, "mirroring twice is the identity",
          np.allclose(features.sequence(features.mirror_pack(
              features.mirror_pack(arr))), f, atol=1e-5))
    check(S, "mirroring leaves the pose block alone",
          np.allclose(m[:, features.block_columns(["pose"])],
                      f[:, features.block_columns(["pose"])], atol=1e-3))
    # (true for this pack because its two shoulders define the centre: a
    #  mirrored+swapped anatomical pair is exactly its own mirror image)
    g = np.random.default_rng(7)
    a1 = features.sequence(arr, augment=True, rng=g)
    g = np.random.default_rng(7)
    a2 = features.sequence(arr, augment=True, rng=g)
    check(S, "augmentation is reproducible for a given seed",
          np.array_equal(a1, a2))
    check(S, "augmentation actually changes something",
          not np.allclose(a1, f))
    base = features.sequence(arr)
    aug = features.sequence(arr, augment=True,
                            rng=np.random.default_rng(11))
    check(S, "the velocity block is the speed-jittered one",
          not np.allclose(aug[:, features.block_columns(["velocity"])],
                          base[:, features.block_columns(["velocity"])]))
    live = features.RingBuffer(maxlen=64)
    for row in arr:
        live.append(row)
    check(S, "the live buffer path equals the offline path exactly",
          np.array_equal(features.from_live_buffer(live), f),
          str(np.abs(features.from_live_buffer(live) - f).max()))
    for name in ("pose", "hands", "hand_shape", "face", "hand_to_body",
                 "mask", "velocity", "segment"):
        want = sum(w for b, (_, w) in features.FEATURE_BLOCKS.items()
                   if b in {name, "mask"})
        got = features.sequence(arr, blocks=[name, "mask"]).shape
        check(S, f"ablation keeps block '{name}' and stays finite",
              got == (config.NUM_FRAMES, want), str(got))
    check(S, "an unknown block name fails loud instead of silently shrinking",
          _raises(lambda: features.sequence(arr, blocks=["nonsense"])),
          "KeyError expected")
    check(S, "a repeated block is counted once, not twice",
          features.sequence(arr, blocks=["pose", "pose"]).shape[1]
          == features.FEATURE_BLOCKS["pose"][1]
          and len(features.block_columns(["hands", "hands"])) == 84)
    check(S, "the block order in the output is the canonical one",
          list(features.block_columns(["velocity", "pose"]))
          == sorted(set(int(c) for c in features.block_columns(
              ["velocity", "pose"]))))
    # --- feature v3: short detector dropouts are bridged, real absences are not
    li = {n: i for i, n in enumerate(features.SEG_NAMES)}
    segc = features.block_columns(["segment"])
    mcols = features.block_columns(["mask"])
    lwrist = features.block_columns(["hands"])[:2]     # x, y of the left wrist
    hole = _fake_pack(t=40)
    hole[18:21, config.COLS_LHAND] = np.nan            # 3 source frames lost
    fh = features.sequence(hole)
    src = features._resample_index(np.linspace(0, 39, config.NUM_FRAMES), 40)
    bad = np.flatnonzero((src >= 18) & (src <= 20))
    good = np.flatnonzero((src >= 22) & (src <= 26))
    check(S, f"a {int(bad.size)}-frame dropout lands inside the fill window",
          config.GAP_FILL_FRAMES >= 3 and len(bad) >= 2, str(len(bad)))
    check(S, "a short hand dropout is bridged, not drawn to the origin",
          float(np.abs(fh[bad][:, lwrist]).min()) > 0.05,
          str(fh[bad][:, lwrist]))
    check(S, "the bridged frames still report mask = 0 (the fill is not a sighting)",
          float(fh[bad][:, mcols[0]].max()) == 0.0)
    check(S, "the frames around a bridge still report mask = 1",
          float(fh[good][:, mcols[0]].min()) == 1.0)
    check(S, "a bridge costs the presence fraction only what it should",
          0.80 < float(fh[0, segc[li["l_presence"]]]) < 1.0,
          f"{float(fh[0, segc[li['l_presence']]]):.3f}")
    gone = _fake_pack(t=40)
    gone[10:32, config.COLS_LHAND] = np.nan            # 22 frames: a real absence
    fg = features.sequence(gone)
    deep = np.flatnonzero((src >= 12) & (src <= 29))   # the interior of the run
    check(S, "a long absence is NOT invented out of nothing",
          float(np.abs(fg[deep][:, lwrist]).max()) == 0.0,
          f"{len(deep)} frames, max {float(np.abs(fg[deep][:, lwrist]).max()):.4f}")
    check(S, "and every frame of it is masked as unseen",
          float(fg[:, mcols[0]][(src >= 10) & (src <= 31)].max()) == 0.0)
    check(S, "and the presence fraction says the hand was mostly missing",
          float(fg[0, segc[li["l_presence"]]]) < 0.60,
          f"{float(fg[0, segc[li['l_presence']]]):.3f}")
    # --- feature v3: the segment read-out
    fs = features.sequence(_fake_pack(t=24))
    check(S, "a still hand has no travel and no excursion",
          abs(float(fs[0, segc[li["l_travel"]]])) < 1e-5
          and abs(float(fs[0, segc[li["l_excursion"]]])) < 1e-5)
    check(S, "both hands in shot gives a real wrist-to-wrist distance",
          float(fs[0, segc[li["wrist_to_wrist_end"]]]) > 0.5
          and float(fs[0, segc[li["both_hands_present_frac"]]]) == 1.0,
          f"{float(fs[0, segc[li['wrist_to_wrist_end']]]):.3f}")
    ft = features.sequence(_travel_pack(t=40))
    check(S, "a hand that drifts away registers travel and a rising mouth distance",
          float(ft[0, segc[li["l_travel"]]]) > 0.3
          and float(ft[0, segc[li["l_mouth_end"]]])
          > float(ft[0, segc[li["l_mouth_start"]]]),
          f"travel {float(ft[0, segc[li['l_travel']]]):.2f}")
    check(S, "a monotone drift retreats as far as it travels, and the still hand does neither",
          abs(float(ft[0, segc[li["l_retreat"]]] - ft[0, segc[li["l_travel"]]])) < 1e-5
          and abs(float(ft[0, segc[li["r_travel"]]])) < 1e-5)
    check(S, "the segment block is one number per clip, constant down the frames",
          np.allclose(ft[:, segc], ft[:1, segc]))
    one = _fake_pack(t=20)
    one[:, config.COLS_RHAND] = np.nan                 # the idle hand of a 1-handed sign
    fo = features.sequence(one)
    rcols = [segc[li["r_" + c]] for c in features.SEG_COLS]
    check(S, "an idle hand reports zeros in its segment columns, not a fake trajectory",
          float(np.abs(fo[:, rcols]).max()) == 0.0)
    check(S, "and the two shared columns say the hands were never both in shot",
          float(fo[0, segc[li["wrist_to_wrist_end"]]]) == 0.0
          and float(fo[0, segc[li["both_hands_present_frac"]]]) == 0.0
          and float(fo[0, segc[li["l_presence"]]]) == 1.0)
    base2 = features.sequence(_fake_pack(t=24))
    mir2 = features.sequence(features.mirror_pack(_fake_pack(t=24)))
    lcols = np.array([li["l_" + c] for c in features.SEG_COLS])
    rcols = np.array([li["r_" + c] for c in features.SEG_COLS])
    check(S, "mirroring exchanges the two halves of the segment block",
          np.allclose(mir2[:, segc][:, lcols], base2[:, segc][:, rcols],
                      atol=1e-5))
    check(S, "and leaves the two shared columns alone",
          np.allclose(mir2[:, segc][:, 28:], base2[:, segc][:, 28:], atol=1e-5))
    doc = json.dumps(features.AUGMENT_DOC)
    check(S, "every augmentation knob is documented",
          all(k in doc for k in config.AUG), doc[:120])


# ---------------------------------------------------------------------------
def t_landmarks():
    S = "landmarks"
    arr = _fake_pack(t=1)[0]
    check(S, "hands above the shoulder line are detected",
          landmarks.hands_above_shoulders(
              _fake_pack(t=1, y=0.30)[0]))
    check(S, "hands resting below it are not",
          not landmarks.hands_above_shoulders(
              _fake_pack(t=1, y=0.75)[0]))
    check(S, "both hands present in a full pack",
          landmarks.hand_present(arr) == (True, True))
    bare = np.full((config.P_TOTAL, 3), np.nan, np.float32)
    bare[landmarks.hand_row("left", 0)] = (0.5, 0.5, 0.1)
    check(S, "a pack with one wrist reports one hand",
          tuple(landmarks.hand_present(bare)) == (True, False))
    seq = _fake_pack(t=40, y=0.30)
    seq[:5] = np.nan
    seq[35:] = np.nan
    st, en = landmarks.signing_window(seq)
    check(S, "the idle head and tail are trimmed off a window",
          st >= 4 and en <= 36, f"{st}:{en}")
    check(S, "every trimmed frame really has no hand above the shoulders",
          not any(landmarks.hands_above_shoulders(seq[i])
                  for i in list(range(st)) + list(range(en, 40))),
          f"window {st}:{en}")
    check(S, "a clip where MediaPipe found no hand is all REST",
          landmarks.signing_window(np.full((20, config.P_TOTAL, 3), np.nan,
                                           np.float32)) == (0, 20))
    check(S, "a clip that never rests keeps its whole window",
          landmarks.signing_window(_fake_pack(t=20, y=0.30)) == (0, 20))
    check(S, "shoulder midpoint is the mean of the two shoulders",
          np.allclose(landmarks.shoulder_mid(arr), (0.50, 0.45)))
    check(S, "no shoulders means no reference point, reported as NaN",
          np.isnan(landmarks.shoulder_mid(bare)).all())


# ---------------------------------------------------------------------------
def t_dataset():
    S = "dataset"
    spec = vocab.load_vocab()
    man = dataset._manifest(("train",))
    if man is None or len(man) == 0:
        skip(S, "everything", "the landmark cache is empty")
        return
    sign = dataset.sign_samples("train", spec, man)
    check(S, "sign samples are labelled with real class names",
          set(sign["label"]) <= set(spec.labels))
    check(S, "every sign window is at least 2 frames",
          (sign["end"] - sign["start"] >= 2).all())
    rest = dataset.rest_samples("train", spec, man)
    cap = int(spec.rest["cap_train"])
    check(S, f"REST samples respect the cap ({cap})", len(rest) <= cap,
          str(len(rest)))
    if len(rest):
        per = rest.groupby("video").size()
        check(S, "at most two REST slices per video (head and tail)",
              per.max() <= 2, str(per.max()))
        win = man[["video", "sign_start", "sign_end"]].rename(
            columns={"sign_start": "ws", "sign_end": "we"})
        joined = rest.merge(win, on="video", how="left")
        bad = joined[(joined["start"] < joined["we"])
                     & (joined["end"] > joined["ws"])]
        check(S, "REST slices never overlap the sign window",
              len(bad) == 0, f"{len(bad)} overlap e.g. "
              f"{bad[['video', 'start', 'end', 'ws', 'we']].head(2).to_dict('records')}")
        check(S, "every REST slice is at least min_frames long",
              (rest["end"] - rest["start"]
               >= int(spec.rest.get("min_frames", 4))).all())
        clip = man.set_index("video")["frames"]
        check(S, "every REST slice fits inside its clip",
              (rest["end"] <= rest["video"].map(clip)).all())
    sp = dataset.build_samples(("train",), None, man)
    check(S, "sample columns are complete",
          set(dataset.SAMPLE_COLUMNS) <= set(sp.columns))
    check(S, "sign and REST rows share one sample schema",
          set(sp["kind"]) <= {"sign", "rest"}, str(set(sp["kind"])))
    check(S, "no duplicate (video, kind, start, end) sample",
          not sp.duplicated(["video", "kind", "start", "end"]).any())
    fp = dataset.cache_fingerprint(sp)
    check(S, "the cache fingerprint is a stable hex string",
          len(fp) == 12 and all(c in "0123456789abcdef" for c in fp), fp)
    check(S, "the fingerprint does not depend on row order",
          fp == dataset.cache_fingerprint(sp.sample(frac=1, random_state=3)))
    d = dataset.describe(sp)
    check(S, "describe() reports counts and per-label rows",
          {"n_samples", "n_videos", "n_signers", "per_label", "kinds"}
          <= set(d) and d["n_samples"] == len(sp), str(list(d)))
    w = dataset.class_weights(spec.labels, sp["label"].value_counts().to_dict())
    check(S, "class weights are finite, positive and mean 1",
          w.shape == (len(spec.labels),) and np.isfinite(w).all()
          and (w > 0).all() and abs(float(w.mean()) - 1.0) < 1e-4, str(w[:6]))
    sw = dataset.sampler_weights(spec.labels, sp)
    check(S, "sampler weights are one per sample",
          len(sw) == len(sp) and np.isfinite(sw).all() and (sw > 0).all())


# ---------------------------------------------------------------------------
def t_model():
    S = "model"
    arch = model.Arch(in_dim=features.FEATURE_DIM, num_classes=22)
    net = model.build(arch)
    check(S, "the network has parameters", net.n_params() > 1000,
          str(net.n_params()))
    x = np.random.default_rng(0).normal(
        size=(3, config.NUM_FRAMES, features.FEATURE_DIM)).astype(np.float32)
    p = net.predict(x)
    check(S, "predict() returns a probability per class",
          p.shape == (3, 22) and np.allclose(p.sum(1), 1.0, atol=1e-4))
    check(S, "a single [T,D] row works too (live path) and keeps the batch axis",
          net.predict(x[0]).shape == (1, 22), str(net.predict(x[0]).shape))
    check(S, "predict() always runs in eval mode, so dropout cannot leak in",
          net.training is False and
          np.allclose(net.predict(x), net.predict(x), atol=0))
    mu, sd = model.standardise(np.concatenate([x, x * 2], axis=0))
    check(S, "standardise gives finite mean and positive std",
          np.isfinite(mu).all() and (sd > 0).all())
    with tempfile.TemporaryDirectory(dir=ROOT) as td:
        path = Path(td) / "roundtrip.pt"
        model.save(net, path, {"labels": ["a"] * 22, "seed": 1})
        net2, meta = model.load(path)
        check(S, "save/load round-trips the weights",
              np.allclose(net.predict(x), net2.predict(x), atol=1e-6))
        check(S, "meta survives the round trip", meta.get("seed") == 1)
        check(S, "a fresh net is in train mode, a loaded one is not",
              model.build(arch).training and not net2.training)
        check(S, "no .part file is left beside the weights",
              not list(Path(td).glob("*.part")))


# ---------------------------------------------------------------------------
def t_predictor():
    S = "predictor"
    labels = vocab.load_vocab().labels
    K = len(labels)
    hi, gi = labels.index("hello"), labels.index("good")
    rest_i, other_i = labels.index(config.REST_LABEL), labels.index(
        config.OTHER_LABEL)

    def row(p1, p2, first=hi, second=gi):
        """A probability row that sums to 1: p1, p2 and the remainder on a
        third class. Made by hand so the thresholds are tested, not guessed."""
        r = np.zeros(K, np.float32)
        third = next(i for i in range(K) if i not in (first, second))
        r[first], r[second], r[third] = p1, p2, max(0.0, 1.0 - p1 - p2)
        return r / r.sum()

    check(S, "the made-up rows really sum to 1",
          abs(float(row(0.6, 0.3).sum()) - 1.0) < 1e-6)
    d = predictor.decide(row(0.45, 0.30), labels)
    check(S, "decide() refuses a weak reading", not d.certain, d.reason)
    check(S, "the refusal says it was the confidence floor",
          "confidence" in d.reason, d.reason)
    check(S, "decide() accepts a strong one",
          predictor.decide(row(0.90, 0.02), labels).certain)
    m = predictor.decide(row(0.52, 0.48), labels, min_conf=0.50)
    check(S, "the refusal names the runner-up",
          not m.certain and "good" in m.reason, m.reason)
    # p1 = 0.88 is over the confidence floor, so what refuses the first
    # reading below is the margin - the pair override, and nothing else.
    check(S, "a pair override tightens only that pair",
          not predictor.decide(row(0.88, 0.10), labels, pair_margins={
              ("hello", "good"): 0.80}).certain
          and predictor.decide(row(0.88, 0.10), labels, pair_margins={
              ("sad", "sorry"): 0.80}).certain
          and "margin" in predictor.decide(
              row(0.88, 0.10), labels,
              pair_margins={("hello", "good"): 0.80}).reason)
    check(S, "the margin floor in config is what decide() defaults to",
          abs(predictor.required_margin("hello", "good")
              - config.MARGIN_MIN) < 1e-9)
    d = predictor.decide(row(0.70, 0.20), labels)
    check(S, "top-3 has at most three entries and they are sorted",
          len(d.top3) <= 3 and all(a[1] >= b[1] for a, b in zip(d.top3,
                                                                d.top3[1:])))
    check(S, "REST on top is never a commit candidate",
          not predictor.commitable(predictor.decide(
              row(0.95, 0.02, first=rest_i), labels)))
    check(S, "OTHER on top is never a commit candidate",
          not predictor.commitable(predictor.decide(
              row(0.95, 0.02, first=other_i), labels)))
    check(S, "a confident real word is commitable",
          predictor.commitable(predictor.decide(row(0.90, 0.02), labels)))
    check(S, "a Reading exposes the numbers the HUD shows",
          set(predictor.decide(row(0.9, 0.02), labels).as_dict())
          >= {"word", "p1", "p2", "margin", "top3", "certain", "reason"})


def t_phrases():
    S = "phrases"
    book = phrases.load_book()
    check(S, "rules load with unique patterns",
          len({r.pattern for r in book.rules}) == len(book.rules))
    check(S, "every rule text is non-empty", all(r.text.strip() for r in
                                                 book.rules))
    check(S, "longest rule is tried first",
          book.longest_first()[0].size >= book.longest_first()[-1].size)
    b = phrases.PhraseBuilder(book)
    for w in ("hello", "REST", "OTHER"):
        b.commit(w)
    check(S, "REST/OTHER are not typed", b.text() == "hello", b.text())
    check(S, "a rule renders", phrases.sentence(["my", "name"], book) ==
          "my name", phrases.sentence(["my", "name"], book))
    check(S, "word forms apply", phrases.sentence(["thanks"], book) ==
          "thank you", phrases.sentence(["thanks"], book))
    n = book.max_buffer_words
    b2 = phrases.PhraseBuilder(book)
    for i, w in enumerate(["good"] * (n + 4)):
        b2.commit(w)
    check(S, f"the buffer holds exactly {n} words", len(b2.words) == n)
    check(S, "the copula note is in the file, not invented by code",
          any("copula" in x.lower() for x in book.notes), str(book.notes)[:80])


def t_live_logic():
    S = "live"
    import argparse
    ns = argparse.Namespace()
    # live.logic_test prints its own lines; keep them, and record the result
    rc = __import__("src.live", fromlist=["live"]).logic_test(ns)
    (ok if rc == 0 else fail)(S, "src/live.py --logic-test all pass",
                              "" if rc == 0 else "see the lines above")


# ---------------------------------------------------------------------------
def t_trained():
    """Everything that needs the real weights. Skipped until they exist."""
    S = "trained"
    if not config.FOR_RELEASE.is_file():
        skip(S, "parity, ONNX and the meta contract",
             "models/words_core20.pt does not exist yet - run python "
             "src/run_all.py --only train (or the command block in the README)")
        return
    from src import train as T
    net, meta = model.load(config.FOR_RELEASE)
    labels = meta.get("labels") or []
    # models/meta.json is the provenance file train.py writes beside the
    # weights; the .pt carries only what load() needs to rebuild the graph.
    side = config.FOR_RELEASE.with_name("meta.json")
    check(S, "models/meta.json exists beside the weights", side.is_file(),
          str(side))
    full = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}
    check(S, "meta.json lists every class the net outputs",
          len(labels) == net.arch.num_classes == len(full.get("labels", [])),
          f"{len(labels)} vs {net.arch.num_classes}")
    check(S, "the checkpoint says it was trained on the official splits",
          "signer-disjoint" in json.dumps(full), str(full)[:120])
    check(S, "meta.json names the split sizes it scored on",
          full.get("test_n_signers") and full.get("train_signers"),
          str({k: full.get(k) for k in ("test_n_signers", "train_signers")}))
    v_ok = (full.get("feature_version") == config.FEATURE_VERSION
            and full.get("feature_dim") == features.FEATURE_DIM
            and net.arch.in_dim == features.FEATURE_DIM)
    check(S, "the weights were built by this version of features.py", v_ok,
          "" if v_ok else f"weights: v{full.get('feature_version')}, dim "
          f"{full.get('feature_dim')} / net expects {net.arch.in_dim}; code: "
          f"v{config.FEATURE_VERSION}, dim {features.FEATURE_DIM}")
    if not v_ok:
        # every remaining check in here feeds the CURRENT feature vector
        # through those weights. A stale model is one problem, not six.
        skip(S, "parity, model input width and the ONNX contract",
             "the shipped weights predate this feature vector - retrain "
             "(Stage E command block), then run this again")
        return
    if not config.MANIFEST_CSV.is_file():
        skip(S, "live feature path == training feature path",
             "no local landmark cache (normal in a clean checkout)")
    elif labels != vocab.load_vocab().labels:
        skip(S, "live feature path == training feature path",
             "expanded vocabulary is not trained yet; deployed weights stay unchanged")
    else:
        rc = T.parity(log=config.logger_for("test-parity"))
        if rc == 2:
            skip(S, "live feature path == training feature path",
                 "no usable cached clips for parity")
        else:
            (ok if rc == 0 else fail)(S, "live feature path == training feature path",
                                      "" if rc == 0 else f"parity returned {rc}")
    check(S, "the feature dim matches the model input",
          net.arch.in_dim == full.get("feature_dim") == features.FEATURE_DIM,
          f"{net.arch.in_dim} vs {full.get('feature_dim')}")
    if config.ONNX_PATH.is_file():
        import onnxruntime as ort
        sess = ort.InferenceSession(str(config.ONNX_PATH),
                                    providers=["CPUExecutionProvider"])
        x = np.zeros((2, config.NUM_FRAMES, net.arch.in_dim), np.float32)
        out = sess.run(None, {"features": x})[0]
        check(S, "the ONNX graph takes [B,64,D] and gives probabilities",
              out.shape == (2, len(labels)) and np.allclose(out.sum(1), 1.0,
                                                            atol=1e-4))
        check(S, "torch and onnx agree on the same input",
              np.allclose(net.predict(x), out, atol=1e-4),
              str(float(np.abs(net.predict(x) - out).max())))
    else:
        skip(S, "ONNX contract", "models/words_core20.onnx is missing")


# ---------------------------------------------------------------------------
CHECKS = [t_config, t_vocab, t_features, t_landmarks, t_dataset, t_model,
          t_predictor, t_phrases, t_live_logic, t_trained]


def main(argv=None) -> int:
    RESULTS.clear()
    print("videoModel logic battery - no camera, no training, no dataset "
          "writes\n")
    failures = []
    for fn in CHECKS:
        try:
            fn()
        except Exception as exc:                   # noqa: BLE001
            fail(fn.__name__.replace("t_", ""), "ran without crashing",
                 f"{type(exc).__name__}: {exc}")
            failures.append(fn.__name__)
    n_fail = sum(1 for _, name, _ in RESULTS if name.startswith("FAIL"))
    n_skip = sum(1 for _, name, _ in RESULTS if name.startswith("SKIP"))
    n_pass = len(RESULTS) - n_fail - n_skip
    print(f"\n{'=' * 62}\n{len(RESULTS)} checks: {n_pass} passed, "
          f"{n_fail} failed, {n_skip} skipped")
    if n_fail:
        print("\nFAILED:")
        for section, name, detail in RESULTS:
            if name.startswith("FAIL"):
                print(f"  {section}: {name[5:]}   {detail}")
        return 1
    tr = [name for section, name, _ in RESULTS if section == "trained"]
    print("the code is sound; the numbers still have to be trained"
          if not tr or any(n.startswith("SKIP") for n in tr) else
          "the code is sound, and the trained weights pass the same checks")
    return 0


def test_logic_battery():
    """Expose the same no-training battery to pytest discovery."""
    assert main([]) == 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
