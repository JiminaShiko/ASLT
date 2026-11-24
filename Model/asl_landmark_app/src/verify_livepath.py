"""End-to-end test of the shipped live path on real frames, no camera.

Feeds 20 FPS sequences through detect -> normalize -> model -> frame_call ->
Smoother and scores what the app commits, per letter, A-Z minus J/Z.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import hand_backend as hb  # noqa: E402
import landmark_store as LS  # noqa: E402
import live  # noqa: E402
import normalize  # noqa: E402

DT_MS = 50          # pretend 20 FPS (timers are in seconds)
CONF = 0.7          # the app defaults
MARGIN = 0.10
VOTE = 7
HOLD = 0.5


def pick_frames(df: pd.DataFrame, per: int, labels, rng) -> pd.DataFrame:
    """Up to `per` readable images per (dataset, letter), oldest path first."""
    out = []
    for (d, lab), grp in df.groupby(["dataset", "letter"]):
        if str(lab) not in labels or len(grp) < 3:
            continue
        idx = rng.choice(len(grp), size=min(per, len(grp)), replace=False)
        out.append(grp.iloc[idx])
    return pd.concat(out) if out else df.iloc[0:0]


def run_one(det, backend, labels, frames: pd.DataFrame, conf: float,
            margin: float, clock: list, recovery=None) -> dict:
    """Feed one (dataset, letter) block through the whole live path.

    `clock` is the shared MediaPipe timestamp (one continuous stream, like the
    app); `recovery` is the app's own, so the shipped path is measured.
    """
    import cv2

    smoother = live.Smoother(conf=conf, vote=VOTE, hold=HOLD)
    committed: list[str] = []
    held = None
    n_hand = n_low = n_close = 0
    n_recovered = 0
    top1 = []
    streak = 0
    t0 = time.perf_counter()
    for path in frames["image"].astype(str):
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.ndim != 3:
            continue
        clock[0] += DT_MS
        res, how = hb.detect_frame(det, bgr, clock[0], recovery,
                                   empty_streak=streak + 1)
        streak = streak + 1 if res.landmarks is None else 0
        if how in ("direct_0.3", "flip_0.3"):
            n_recovered += 1
        if res.landmarks is None:
            continue
        n_hand += 1
        feats = normalize.features(res.landmarks)
        call = live.frame_call(np.asarray(backend(feats)), labels, conf, margin)
        top1.append(call["conf"])
        n_low += int(call["low_conf"])
        n_close += int(call["close"])
        held_now, committed_now, _hc = smoother.step(call["letter"], call["conf"],
                                                     clock[0] / 1000.0)
        if held_now:
            held = held_now
        if committed_now:
            committed.append(committed_now)
    el = time.perf_counter() - t0
    return {"n_frames": len(frames), "n_hand": n_hand,
            "n_recovered": n_recovered,
            "committed": committed,
            # score the smoothed guess even without a commit: what is on screen
            "answer": committed[0] if committed else held,
            "committed_any": bool(committed),
            "low_conf_pct": 100 * n_low / max(n_hand, 1),
            "close_pct": 100 * n_close / max(n_hand, 1),
            "mean_top1": float(np.mean(top1)) if top1 else float("nan"),
            # loop FPS (the 20 FPS requirement) vs hand coverage: different things
            "frame_fps": len(frames) / max(el, 1e-9),
            "fps": n_hand / max(el, 1e-9)}


def pass_over(sel, backend, labels, args, recovery, tag: str) -> pd.DataFrame:
    """Run every selected block through the live path once (fresh detector)."""
    det = hb.create_detector(static=False)
    groups = sorted(set(zip(sel["dataset"], sel["letter"])))
    rows = []
    clock = [0]
    t_all = time.perf_counter()
    for i, (d, lab) in enumerate(groups, 1):
        frames = sel[(sel["dataset"] == d) & (sel["letter"] == lab)]
        r = run_one(det, backend, labels, frames, args.conf, args.margin, clock,
                    recovery)
        r.update({"dataset": d, "letter": str(lab)})
        rows.append(r)
    # 3 s gap so the VIDEO tracker drops the previous hand
        clock[0] += 3000
        if i % 12 == 0 or i == len(groups):
            print(f"   [{tag}] {i}/{len(groups)} groups, "
                  f"{time.perf_counter() - t_all:.0f}s elapsed", flush=True)
    det.close()
    res = pd.DataFrame(rows)
    res["ok"] = [c == l for c, l in zip(res["answer"], res["letter"])]
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=12, help="frames per letter per dataset")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "torch"],
                    help="onnx = the shipped exported model (default)")
    ap.add_argument("--profile", default=None, choices=sorted(config.PROFILES),
                    help="which trained profile to load (default: config's)")
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recover", type=int, default=1,
                    help="same retry as live.py: N consecutive empty frames "
                         "before an IMAGE-mode retry at --recover-conf, then a "
                         "mirrored one. 0 = plain video tracking only")
    ap.add_argument("--recover-conf", type=float, default=0.3)
    ap.add_argument("--recover-every", type=int, default=3,
                    help="cadence while the hand stays missing (see live.py)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = __import__("json").loads(
        config.meta_path(args.profile).read_text(encoding="utf-8"))
    backend, labels = live.build_backend(args.backend, profile=args.profile)
    print(f"== live-path A-Z test ==  backend={backend.name}  "
          f"profile={args.profile or config.PROFILE_DEFAULT}  "
          f"dim={meta['model']['in_dim']}  blocks="
          f"{'+'.join(meta['features']['blocks'])}")
    print(f"   gate {args.conf}  margin {args.margin}  vote {VOTE}  hold {HOLD}s  "
          f"frames at 20 FPS")

    df = LS.load_raw()
    rng = np.random.default_rng(args.seed)
    sel = pick_frames(df, args.per, labels, rng)
    if not len(sel):
        print("  nothing to test (no cached landmarks)")
        return 1
    recovery = hb.Recovery(conf=args.recover_conf, after=args.recover,
                           every=args.recover_every) if args.recover else None
    print(f"   detection: video tracking at {config.MIN_HAND_DETECTION_CONFIDENCE}"
          + (f" + retry at {args.recover_conf} after {args.recover} empty "
             f"frame(s), every {args.recover_every}th while missing"
             if recovery else " (recovery OFF)"))

    # same frames twice: the coverage gap is the retry alone (pass 1 = plain VIDEO)
    base = pass_over(sel, backend, labels, args, None, "tracking only")
    if recovery is not None:
        res = pass_over(sel, backend, labels, args, recovery, "tracking + retry")
        print("   " + recovery.summary())
        recovery.close()
    # detail from the shipped path; baseline pass only adds its hand count
        res = res.merge(base[["dataset", "letter", "n_hand", "frame_fps"]]
                        .rename(columns={"n_hand": "n_hand_base",
                                         "frame_fps": "frame_fps_base"}),
                        on=["dataset", "letter"], how="left")
    else:
        res = base.copy()
        res["n_hand_base"] = res["n_hand"]
        res["frame_fps_base"] = res["frame_fps"]

    fps = float(np.median(res["fps"]))
    ffps = float(np.median(res["frame_fps"]))
    ffps_base = float(np.median(res["frame_fps_base"]))
    n_frames = int(res["n_frames"].sum())
    n_hand = int(res["n_hand"].sum())
    n_hand_base = int(res["n_hand_base"].sum())
    n_rec = int(res["n_recovered"].sum())

    art = config.onnx_path(args.profile) if backend.name == "onnx" \
        else config.model_path(args.profile)
    lines = ["# Live-path A-Z test (no camera)\n",
             f"Backend: `{backend.name}` (`{art.name}`), "
             f"{meta['model']['in_dim']} features from "
             f"`{'+'.join(meta['features']['blocks'])}`. Frames are real public "
             f"dataset photographs, read BGR->RGB exactly as the webcam path "
             f"reads them, and fed at 20 FPS through `live.frame_call` and "
             f"`live.Smoother`. Nothing here uses a camera and nothing is saved "
             f"to disk.\n",
             "**This is a wiring test, not an accuracy test.** The model "
             "deployed by `live.py` and `app.py` is the final one, trained on "
             "every dataset in the cache, so every frame below is one it has "
             "already seen. A high score here proves the live path is connected "
             "correctly - colour order, video-mode timestamps, the shared "
             "`normalize.features()`, the gate, the smoother, the exported "
             "weights. It says nothing about transfer accuracy, which is the "
             "leave-one-dataset-out number in the profile's report. A *low* score here "
             "would be a bug, and that is the whole point.\n",
             f"Answered correctly per block: **{int(res['ok'].sum())}/{len(res)}** "
             f"({100 * res['ok'].mean():.1f}%); "
             f"{int(res['committed_any'].sum())}/{len(res)} blocks committed "
             f"inside the clip.\n",
             f"| measure | video tracking only | + retry (what ships) |",
             f"|---|---|---|",
             f"| loop speed, median per block (incl. decoding each frame from "
             f"disk, which a webcam does not cost) | {ffps_base:.1f} frames/s | "
             f"**{ffps:.1f} frames/s** ({'PASS' if ffps >= 20 else 'BELOW'} the "
             f"20 FPS target) |",
             f"| frames with a hand | {n_hand_base}/{n_frames} "
             f"({100 * n_hand_base / n_frames:.1f}%) | {n_hand}/{n_frames} "
             f"({100 * n_hand / n_frames:.1f}%) |",
             f"| usable detections/s | {float(np.median(base['fps'])):.1f} | "
             f"{fps:.1f} |",
             f"| retry | – | {recovery.summary() if recovery else 'disabled'} |",
             f"\nThe two columns are the same frames in the same process, so the "
             f"difference is the retry and nothing else. This is the only "
             f"hand-coverage figure in the project measured frame by frame "
             f"through the live path. It is not the same quantity as the "
             f"extractor's recall in the training report, which is offline **image** "
             f"recall over all 122k files and can afford five detections per "
             f"image because it runs once instead of thirty times a second. "
             f"Neither is a webcam measurement; only the self-test "
             f"(`python src/live.py --selftest`) is.\n"]
    need = int(round(1000 * (HOLD + 0.15) / DT_MS))
    if float(res["committed_any"].mean()) < 0.5:
        lines.append(f"NOTE: only {int(res['committed_any'].sum())}/{len(res)} clips "
                     f"were long enough for the {HOLD}s hold-to-commit timer to "
                     f"fire ({args.per} frames = "
                     f"{args.per * DT_MS / 1000:.1f}s of a {DT_MS}ms tick, and the "
                     f"letter only has to be held long enough after it first "
                     f"clears the gate), so most `answer` cells below are the "
                     f"letter the app is still holding rather than one it "
                     f"committed. The held letter still exercises every stage of "
                     f"the pipeline; for a committed-letter count re-run with "
                     f"`--per {max(need + 8, args.per * 2)}`.")
    for d in sorted(res["dataset"].unique()):
        sub = res[res["dataset"] == d].sort_values("letter")
        n_ok = int(sub["ok"].sum())
        lines.append(f"\n## live-path answer on `{d}` frames — "
                     f"{n_ok}/{len(sub)} letters reproduced (in-sample)\n")
        lines.append("| letter | frames | hand found | tracking only | via retry | "
                     "answer | correct | mean top1 | % below gate | % uncertain (gap) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _i, r in sub.iterrows():
            got = r["answer"] or "nothing"
            lines.append(
                f"| {r['letter']} | {r['n_frames']} | {r['n_hand']} | "
                f"{r['n_hand_base']} | {r['n_recovered']} | "
                f"{got} | {'yes' if r['ok'] else 'no'} | {r['mean_top1'] * 100:.0f}% | "
                f"{r['low_conf_pct']:.0f}% | {r['close_pct']:.0f}% |")
    lines.append("\n## Where the letters went, including the wrong ones\n")
    lines.append("| committed | expected | count |")
    lines.append("|---|---|---|")
    for (exp, got), n in Counter(zip(res["letter"],
                                     [c or "nothing" for c in res["answer"]]))\
            .most_common():
        lines.append(f"| {got} | {exp} | {n} |")
    lines.append("\nA block that commits *nothing* is the uncertain state working "
                 "as designed: the gate or the top1-top2 margin refused to vote, "
                 "which is better than a wrong letter on the screen. A block that "
                 "commits the wrong letter is a real error and should be read "
                 "against the profile report's per-letter table.\n")
    text = "\n".join(lines) + "\n"
    out = Path(args.out) if args.out else config.REPORTS_DIR / "live_path.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
