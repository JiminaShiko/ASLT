"""Step 2b - turn the J/Z clips into cached hand trajectories.

`src/verify_jz_trigger.py` can already measure the trajectory rule on real
movement, but it only reads the MSL cache that was built once by a throwaway
script. This is that step, kept: it walks the clips `src/download_jz_data.py`
put under `datasets/jz/clips/`, runs MediaPipe VIDEO over each one, and writes
`cache/jz_landmarks/<clip>.npz` with the SAME thirteen keys the MSL shards
carry, so the verifier, a future motion head and any analysis all read one
format:

    lm (N,21,3) float32  normalised image coords, one row per frame with a hand
    idx (N,) int32       source frame number of each kept row
    clip signer letter split rep  frames fps cadence w h sec

`cadence` is the integer milliseconds between kept rows that the live rule is
replayed at; `fps` is the clip's own measured rate.

Letters come from the filename. The MSL archive names every clip
`S<signer>-<letter>-<view>-<rep>.mp4`; anything without that pattern falls back
to a one-letter parent folder, and is skipped (with a count) when neither is
available - a corpus whose labels are whole sentences cannot be used here
without alignment, and guessing would poison the cache.

No camera, no training, nothing written outside `cache/jz_landmarks/`.

Run:
    python src/extract_jz_landmarks.py
    python src/extract_jz_landmarks.py --per-letter 100 --fps 30
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
import hand_backend  # noqa: E402

CLIP_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
MIN_ROWS = 8          # fewer kept frames than this is not a stroke
DEFAULT_FPS = 30.0


def parse_clip(path: Path) -> dict | None:
    """Signer / letter / view / repetition from a clip path, or None.

    Order of evidence: the MSL filename convention first, because it is
    explicit and carries the signer; then a one-letter parent folder, which is
    what a plain `A/`, `B/`, ... corpus looks like.
    """
    parts = path.stem.split("-")
    if len(parts) >= 3 and len(parts[1]) == 1 and parts[1].isalpha():
        letter = parts[1].upper()
        return {"letter": letter, "signer": parts[0],
                "view": parts[2],
                "rep": int(parts[3]) if parts[3].isdigit() else 0}
    parent = path.parent.name.strip()
    if len(parent) == 1 and parent.isalpha():
        return {"letter": parent.upper(), "signer": path.parent.parent.name,
                "view": "", "rep": 0}
    # A single alpha token anywhere in the name is the last resort; it is what
    # an LSA64-style `<subject>_<class>_<rep>` file needs.
    tokens = [t for t in path.stem.replace("_", "-").split("-")
              if len(t) == 1 and t.isalpha()]
    if len(tokens) == 1:
        return {"letter": tokens[0].upper(), "signer": path.parent.name,
                "view": "", "rep": 0}
    return None


def split_of(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root).parts
    except ValueError:
        return ""
    for part in rel[:-1]:
        low = part.lower()
        if low in ("train", "test", "val", "valid", "validation"):
            return low
    return ""


def read_clip(path: Path, detector, fps_hint: float | None) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or (fps_hint or DEFAULT_FPS)
    if not np.isfinite(fps) or fps <= 1.0 or fps > 240.0:
        fps = fps_hint or DEFAULT_FPS
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    step_ms = max(1, int(round(1000.0 / fps)))
    lms: list[np.ndarray] = []
    idx: list[int] = []
    i = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            result = hand_backend.detect_video(detector, frame, i * step_ms)
            if result.landmarks is None:
                continue
            lms.append(np.asarray(result.landmarks, dtype=np.float32))
            idx.append(i)
    finally:
        cap.release()
    if len(lms) < MIN_ROWS:
        return None
    return {"lm": np.stack(lms), "idx": np.asarray(idx, dtype=np.int32),
            "frames": i + 1, "fps": fps, "cadence": step_ms,
            "w": w, "h": h, "sec": (i + 1) / fps}


def main() -> int:
    ap = argparse.ArgumentParser(description="J/Z clips -> cached landmarks.")
    ap.add_argument("--clips-dir", default=str(config.JZ_CLIP_DIR),
                    help=f"where the clips are (default {config.JZ_CLIP_DIR})")
    ap.add_argument("--out-dir", default=str(config.JZ_LANDMARK_DIR),
                    help=f"where the .npz shards go (default {config.JZ_LANDMARK_DIR})")
    ap.add_argument("--per-letter", type=int, default=0,
                    help="cap clips per letter (0 = every clip)")
    ap.add_argument("--fps", type=float, default=None,
                    help="cadence to assume when a clip reports no frame rate")
    ap.add_argument("--force", action="store_true",
                    help="re-extract clips whose .npz already exists")
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir)
    out_dir = Path(args.out_dir)
    if not clips_dir.is_dir():
        raise SystemExit(f"{clips_dir} does not exist.\n"
                         "  fetch the clips first: python src/download_jz_data.py")
    clips = sorted(p for p in clips_dir.rglob("*")
                   if p.suffix.lower() in CLIP_SUFFIXES)
    if not clips:
        raise SystemExit(f"no clips under {clips_dir} - run "
                         "python src/download_jz_data.py")

    labelled = []
    skipped_unlabelled = 0
    for clip in clips:
        meta = parse_clip(clip)
        if meta is None:
            skipped_unlabelled += 1
            continue
        closed = config.LABELS + sorted(config.MOTION_LETTERS)
        if meta["letter"] not in closed:
            skipped_unlabelled += 1
            continue
        meta["split"] = split_of(clip, clips_dir)
        labelled.append((clip, meta))
    if skipped_unlabelled:
        print(f"  {skipped_unlabelled} clip(s) carry no letter label and are "
              "skipped - a corpus labelled by sentence needs alignment first")

    by_letter = collections.Counter(m["letter"] for _, m in labelled)
    print(f"  {len(labelled)} labelled clips: " +
          ", ".join(f"{k} {v}" for k, v in sorted(by_letter.items())))

    out_dir.mkdir(parents=True, exist_ok=True)
    config.ensure_dirs()
    detector = hand_backend.create_detector(static=False)
    taken: collections.Counter = collections.Counter()
    written = 0
    failed = 0
    started = time.time()
    try:
        for n, (clip, meta) in enumerate(labelled, 1):
            letter = meta["letter"]
            if args.per_letter and taken[letter] >= args.per_letter:
                continue
            dest = out_dir / (clip.stem + ".npz")
            if dest.is_file() and not args.force:
                taken[letter] += 1
                continue
            if n % 25 == 0 or n == len(labelled):
                rate = n / max(time.time() - started, 1e-6)
                print(f"  {n}/{len(labelled)} clips ({rate:.1f}/s)", flush=True)
            got = read_clip(clip, detector, args.fps)
            if got is None:
                failed += 1
                continue
            payload = {
                "lm": got["lm"], "idx": got["idx"],
                "clip": clip.stem, "signer": meta["signer"],
                "letter": letter, "split": meta["split"],
                "rep": np.int64(meta["rep"]), "frames": np.int64(got["frames"]),
                "fps": np.float64(got["fps"]), "cadence": np.int64(got["cadence"]),
                "w": np.int64(got["w"]), "h": np.int64(got["h"]),
                "sec": np.float64(got["sec"]),
            }
            np.savez_compressed(dest, **payload)
            taken[letter] += 1
            written += 1
    finally:
        detector.close()

    total = sum(1 for _ in out_dir.glob("*.npz"))
    print(f"\n  wrote {written} shard(s); {failed} clip(s) had too few hands; "
          f"{total} in {out_dir}")
    if total:
        print("  measure the trajectory rule on them with:\n"
              f"    python src/verify_jz_trigger.py --clips-dir {out_dir}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
