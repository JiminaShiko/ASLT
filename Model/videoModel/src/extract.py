"""Stage 2 - resumable MediaPipe Holistic landmark extraction into cache/manifest.csv.

Read hand_rate_seg (signing window), not hand_rate (whole clip). Never writes inside ASL Citizen/.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, landmarks, vocab

MANIFEST_COLUMNS = [
    "video", "split", "gloss", "word", "participant", "frames", "fps",
    "stride", "hand_rate", "hand_rate_seg", "sign_start", "sign_end",
    "rest_frames", "ms", "restart_ms", "bytes", "landmark_version",
    "status", "error",
]

_TRACKER: landmarks.Tracker | None = None
_OPTS: dict = {}


def landmark_file(video: str) -> Path:
    """The .npy for one video; ps: ids get a ps- prefix so stems cannot collide."""
    stem = config.safe_stem(video)
    if video.startswith("ps:"):
        stem = f"ps-{stem}"
    return config.LANDMARK_DIR / f"{stem}.npy"


def _init_worker(opts: dict) -> None:
    global _TRACKER, _OPTS
    _OPTS = opts
    _TRACKER = landmarks.Tracker(
        min_face_conf=opts["min_conf"], min_pose_conf=opts["min_conf"],
        min_landmarks_conf=opts["min_conf"])


def _work(job: tuple) -> dict:
    """Worker entry: extract one video, save it, return its manifest row."""
    video, split, gloss, word, participant = job
    row = dict(zip(MANIFEST_COLUMNS, [""] * len(MANIFEST_COLUMNS)))
    row.update(video=video, split=split, gloss=gloss, word=word,
               participant=participant, landmark_version=config.LANDMARK_VERSION)
    path = config.video_path(video)
    try:
        ex = landmarks.extract_video(path, tracker=_TRACKER,
                                     restart=bool(_OPTS["restart"]))
        arr = ex.landmarks
        if arr.shape != (ex.n_frames, config.P_TOTAL, 3):
            raise RuntimeError(f"unexpected packed shape {arr.shape}")
        out = landmark_file(video)
        # np.save appends ".npy" unless the name already ends with it
        tmp = out.with_name(f".{out.stem}.{os.getpid()}.part.npy")
        np.save(tmp, arr)
        tmp.replace(out)
        s, e = ex.window
        row.update(frames=ex.n_frames, fps=round(ex.fps, 3), stride=ex.stride,
                   hand_rate=round(ex.hand_rate, 4),
                   hand_rate_seg=round(
                       landmarks.hand_rate_window(arr, s, e), 4),
                   sign_start=s, sign_end=e,
                   rest_frames=ex.rest_frames, ms=round(ex.ms, 1),
                   restart_ms=round(ex.restart_ms, 1),
                   bytes=int(out.stat().st_size), status="ok", error="")
    except Exception as exc:                          # noqa: BLE001
        if _TRACKER is not None:
            _TRACKER.restart()
        row.update(status="error", frames=0,
                   error=f"{type(exc).__name__}: {exc}"[:400])
    return row


def read_manifest() -> pd.DataFrame:
    if config.MANIFEST_CSV.is_file():
        df = pd.read_csv(config.MANIFEST_CSV, dtype={"video": str,
                                                    "gloss": str,
                                                    "word": str})
        for c in MANIFEST_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df[MANIFEST_COLUMNS]
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def append_manifest(rows: list[dict]) -> pd.DataFrame:
    """Append-then-deduplicate so a killed run never loses finished work."""
    new = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df = pd.concat([read_manifest(), new], ignore_index=True)
    df = df.drop_duplicates("video", keep="last").reset_index(drop=True)
    df.to_csv(config.MANIFEST_CSV, index=False)
    errs = df[df["status"] == "error"]
    config.ERRORS_CSV.parent.mkdir(parents=True, exist_ok=True)
    errs.to_csv(config.ERRORS_CSV, index=False)
    return df


def pending(df: pd.DataFrame, todo: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Videos that still need MediaPipe: never extracted, errored, or stale."""
    if force:
        return todo
    done = set(df.loc[(df["status"] == "ok")
                      & (df["landmark_version"] == config.LANDMARK_VERSION),
                      "video"])
    files_ok = {v for v in done if landmark_file(v).is_file()}
    return todo[~todo["video"].isin(files_ok)]


def recompute_rates(log=None) -> dict:
    """Fill hand_rate_seg from the cache (for caches extracted before the column existed)."""
    log = log or config.logger_for("rates", config.LOGS_DIR
                                   / f"rates_{config.timestamp()}.log")
    df = read_manifest()
    if df.empty:
        log.warning("cache/manifest.csv is empty - nothing to recompute")
        return {"rows": 0, "updated": 0, "missing": 0}
    vals: list = []
    updated = missing = 0
    t0 = time.time()
    for r in df.itertuples(index=False):
        if str(r.status) != "ok":
            vals.append(None)
            continue
        p = landmark_file(r.video)
        if not p.is_file():
            vals.append(None)
            missing += 1
            continue
        try:
            arr = np.load(p, mmap_mode="r")
            v = round(float(landmarks.hand_rate_window(
                arr, r.sign_start, r.sign_end)), 4)
        except Exception as exc:                      # noqa: BLE001
            log.warning("could not read %s: %s", p.name, exc)
            vals.append(None)
            missing += 1
            continue
        vals.append(v)
        updated += 1
        if updated % 2000 == 0:
            log.info("  %5d rows done  %.0f/s", updated,
                     updated / max(time.time() - t0, 1e-9))
    df["hand_rate_seg"] = vals
    df.to_csv(config.MANIFEST_CSV, index=False)
    ok = df[df["status"] == "ok"].copy()
    for c in ("hand_rate", "hand_rate_seg"):
        ok[c] = pd.to_numeric(ok[c], errors="coerce")
    log.info("done: %d/%d rows filled in %.1f s (%d cached files missing)",
             updated, len(df), time.time() - t0, missing)
    log.info("clip-wide rate %.3f mean vs signing-window rate %.3f mean; "
             "below 0.50: %d clips-wide, %d by window",
             float(ok["hand_rate"].mean()), float(ok["hand_rate_seg"].mean()),
             int((ok["hand_rate"] < .5).sum()), int((ok["hand_rate_seg"] < .5).sum()))
    return {"rows": len(df), "updated": updated, "missing": missing,
            "manifest": df}


def run(splits=("train",), workers: int = 1, limit: int | None = None,
        force: bool = False, min_conf: float = 0.30, restart: bool = True,
        log=None) -> dict:
    log = log or config.logger_for("extract")
    config.ensure_dirs()
    config.require_dataset()
    todo = vocab.videos_to_process(splits)
    if limit:
        todo = todo.head(int(limit))
    df = read_manifest()
    todo = pending(df, todo, force)
    log.info("splits=%s workers=%d to_process=%d already_cached=%d "
             "min_conf=%.2f restart=%s",
             ",".join(splits), workers, len(todo), len(df), min_conf, restart)
    if todo.empty:
        log.info("nothing to do - the cache already covers this list")
        return {"processed": 0, "errors": 0, "ms_per_frame": 0.0, "seconds": 0.0}

    jobs = [tuple(r) for r in todo[["video", "split", "gloss", "word",
                                    "participant"]].itertuples(index=False,
                                                               name=None)]
    opts = {"min_conf": min_conf, "restart": int(restart)}
    t_start = time.time()
    rows: list[dict] = []
    n_err = 0
    frames = 0
    ms_total = 0.0
    log_every = max(10, workers * 5)

    def note(row: dict, i: int) -> None:
        nonlocal n_err, frames, ms_total
        rows.append(row)
        if row["status"] == "error":
            n_err += 1
            log.warning("ERROR %-46s %s", row["video"], row["error"])
        else:
            frames += int(row["frames"])
            ms_total += float(row["ms"])
        if i % log_every == 0 or i == len(jobs):
            el = time.time() - t_start
            rate = i / el if el > 0 else 0
            left = (len(jobs) - i) / rate if rate else 0
            log.info("  %5d/%5d  %.1f/s  eta %6.0f min  %.2f ms/frame  %d errors",
                     i, len(jobs), rate, left / 60,
                     ms_total / max(frames, 1), n_err)
            if i % (log_every * 10) == 0:
                append_manifest(rows)

    if workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(opts,)) as pool:
            for i, row in enumerate(pool.imap_unordered(_work, jobs, chunksize=4),
                                    1):
                note(row, i)
    else:
        _init_worker(opts)
        for i, job in enumerate(jobs, 1):
            note(_work(job), i)

    df = append_manifest(rows) if rows else read_manifest()
    el = time.time() - t_start
    mpf = ms_total / max(frames, 1)
    log.info("done: %d videos (%d errors) in %.1f s - %.2f ms/frame, "
             "%.2f s/video, %.1f frames/s effective",
             len(jobs), n_err, el, mpf,
             (ms_total / max(len(jobs), 1)) / 1000, frames / max(el, 1e-9))
    return {"processed": len(jobs), "errors": n_err, "ms_per_frame": mpf,
            "seconds": el, "frames": frames, "manifest": df}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic landmarks for the word "
                    "vocabulary into the resumable cache.")
    ap.add_argument("--splits", default="train,val,test",
                    help="comma list of train,val,test (default all three)")
    ap.add_argument("--workers", type=int, default=None,
                    help="MediaPipe processes (default: min(4, cpus-1))")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N videos (smoke test / partial runs)")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if the cache already has the video")
    ap.add_argument("--min-conf", type=float, default=0.30,
                    help="MediaPipe detection/landmarks confidence floor")
    ap.add_argument("--no-restart", action="store_true",
                    help="do not rebuild the detector between videos (faster, "
                         "but a clip can be smoothed with the previous one)")
    ap.add_argument("--log", default=None, help="log file (default logs/extract_<ts>.log)")
    ap.add_argument("--rates", action="store_true",
                    help="recompute hand_rate_seg from the cache (no "
                         "MediaPipe, no videos) and exit")
    args = ap.parse_args(argv)

    if args.rates:
        out = recompute_rates()
        df = out.get("manifest")
        if df is not None and len(df):
            ok = df[df["status"] == "ok"]
            print(f"\nhand detection, signing window only ({len(ok)} clips)")
            g = (ok.assign(seg=pd.to_numeric(ok["hand_rate_seg"],
                                             errors="coerce"),
                           full=pd.to_numeric(ok["hand_rate"],
                                              errors="coerce"))
                 .groupby("word")[["full", "seg"]].mean().sort_values("seg"))
            print(g.round(3).to_string())
        return 0

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    for s in splits:
        if s not in config.SPLIT_FILES:
            ap.error(f"unknown split '{s}'")
    workers = args.workers or max(1, min(4, config.cpu_count() - 1))
    log = config.logger_for("extract",
                            Path(args.log) if args.log
                            else config.LOGS_DIR / f"extract_{config.timestamp()}.log")
    out = run(splits, workers, args.limit, args.force, args.min_conf,
              not args.no_restart, log)
    df = out.get("manifest")
    if df is not None and len(df):
        ok = df[df["status"] == "ok"]
        print("\ncache summary")
        print(f"  videos ok        : {len(ok)}")
        print(f"  frames           : {int(ok['frames'].sum()):,}")
        print(f"  landmark bytes   : {int(ok['bytes'].sum()) / 1e6:.1f} MB "
              f"-> {config.LANDMARK_DIR}")
        print(f"  hand-detection   : {ok['hand_rate'].mean():.3f} mean "
              f"({(ok['hand_rate'] < 0.5).sum()} videos below 0.50)")
        print(f"  rest frames      : {int(ok['rest_frames'].sum()):,} trimmed "
              f"idle frames available for REST")
        if (df["status"] == "error").any():
            print(f"  errors           : {(df['status'] == 'error').sum()} "
                  f"- see {config.ERRORS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
