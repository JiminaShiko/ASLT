"""Convert selected sign videos into fixed-length RGB frame sequences."""

import csv
import glob
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd

from src.config import (
    ANNOTATION_FPS,
    ASL_CITIZEN_METADATA,
    ASL_CITIZEN_SPLITS_DIR,
    ASL_CITIZEN_VIDEOS,
    DATASETS_DIR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    NUM_FRAMES,
    PROCESSED_DIR,
    TARGET_WORDS,
    WLASL_METADATA,
    WLASL_VIDEOS,
    canonicalize_gloss,
    safe_filename,
)

VALID_SPLITS = {"train", "val", "test"}
VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".flv", ".3gp"}


def wlasl_records() -> list[dict]:
    if not WLASL_METADATA.is_file():
        raise FileNotFoundError(
            f"WLASL metadata not found at {WLASL_METADATA}. "
            "Supply the datasets and run `python -m src.download_data --manual` first."
        )
    data = json.loads(WLASL_METADATA.read_text(encoding="utf-8"))
    records = []
    for entry in data:
        gloss = canonicalize_gloss(entry.get("gloss", ""))
        if gloss not in TARGET_WORDS:
            continue
        for instance in entry.get("instances", []):
            video_id = str(instance.get("video_id", "")).strip()
            if not video_id:
                continue
            records.append(
                {
                    "dataset": "wlasl",
                    "video_id": video_id,
                    "gloss": gloss,
                    "split": str(instance.get("split", "unknown")).casefold(),
                    "signer_id": str(instance.get("signer_id", "")),
                    "video_path": find_video(WLASL_VIDEOS, video_id),
                    "frame_start": instance.get("frame_start", 1),
                    "frame_end": instance.get("frame_end", -1),
                    "annotation_fps": instance.get("fps", ANNOTATION_FPS),
                }
            )
    return records


def find_video(video_dir: Path, video_id: str) -> Path | None:
    # ASL Citizen stems contain spaces ("<id>-THANK YOU"); try the raw stem
    # first, then the sanitized form for ids that are not valid file names.
    stems = [video_id, safe_filename(video_id)]
    if stems[0] == stems[1]:
        stems.pop()
    for stem in stems:
        candidates = [
            path
            for path in video_dir.glob(glob.escape(stem) + ".*")
            if path.suffix.casefold()
            in {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".flv", ".3gp"}
        ]
        found = next((path for path in candidates if path.stat().st_size), None)
        if found is not None:
            return found
    return None


def csv_first(row: dict, *names: str) -> str:
    """Return the first nonempty cell among case-insensitive column names."""
    lowered = {str(key).strip().casefold(): str(value).strip() for key, value in row.items()}
    for name in names:
        value = lowered.get(name.casefold(), "")
        if value:
            return value
    return ""


def asl_citizen_records() -> list[dict]:
    if ASL_CITIZEN_SPLITS_DIR.is_dir():
        return _asl_citizen_split_records()
    return _asl_citizen_metadata_records()


def _asl_citizen_split_records() -> list[dict]:
    """Read the manual download layout: splits/{train,val,test}.csv plus
    videos/<id>-<GLOSS>.mp4 (ASL Citizen's released folder structure)."""
    records = []
    for split in VALID_SPLITS:
        split_path = ASL_CITIZEN_SPLITS_DIR / f"{split}.csv"
        if not split_path.is_file():
            continue
        with split_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                gloss = canonicalize_gloss(row.get("Gloss", ""))
                if gloss not in TARGET_WORDS:
                    continue
                video_file = csv_first(row, "Video file", "video file")
                if not video_file:
                    continue
                video_id = Path(video_file).stem
                records.append(
                    {
                        "dataset": "asl_citizen",
                        "video_id": video_id,
                        "gloss": gloss,
                        "split": split,
                        "signer_id": csv_first(row, "Participant ID", "participant id"),
                        "video_path": find_video(ASL_CITIZEN_VIDEOS, video_id),
                        "frame_start": None,
                        "frame_end": None,
                        "annotation_fps": None,
                    }
                )
    return records


def _asl_citizen_metadata_records() -> list[dict]:
    if not ASL_CITIZEN_METADATA.is_file():
        raise FileNotFoundError(
            f"ASL Citizen selected metadata not found at {ASL_CITIZEN_METADATA}. "
            "Supply the datasets and run `python -m src.download_data --manual` first."
        )
    frame = pd.read_csv(ASL_CITIZEN_METADATA, dtype=str).fillna("")
    required = {"video_id", "gloss", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"ASL Citizen metadata is missing required columns: "
            f"{sorted(required - set(frame.columns))}"
        )
    records = []
    for row in frame.to_dict(orient="records"):
        video_id = str(row["video_id"]).strip()
        if not video_id:
            continue
        records.append(
            {
                "dataset": "asl_citizen",
                "video_id": video_id,
                "gloss": canonicalize_gloss(row["gloss"]),
                "split": str(row["split"]).casefold(),
                "signer_id": str(row.get("signer_id", "")),
                "video_path": find_video(ASL_CITIZEN_VIDEOS, video_id),
                "frame_start": None,
                "frame_end": None,
                "annotation_fps": None,
            }
        )
    return records


def frame_interval(record: dict, frame_count: int, video_fps: float) -> tuple[int, int]:
    """Map WLASL's annotated 25-fps interval onto the file's frame indices."""
    if record["dataset"] != "wlasl":
        return 0, frame_count - 1

    annotation_fps = record.get("annotation_fps")
    try:
        annotation_fps = float(annotation_fps)
    except (TypeError, ValueError):
        annotation_fps = ANNOTATION_FPS
    if not math.isfinite(annotation_fps) or annotation_fps <= 0:
        annotation_fps = ANNOTATION_FPS

    try:
        frame_start = max(1, int(record.get("frame_start") or 1))
    except (TypeError, ValueError):
        frame_start = 1
    try:
        frame_end = int(record.get("frame_end") or -1)
    except (TypeError, ValueError):
        frame_end = -1

    start_seconds = (frame_start - 1) / annotation_fps
    end_seconds = (
        (frame_count - 1) / video_fps
        if frame_end < 0
        else max(0, frame_end - 1) / annotation_fps
    )
    start_index = max(0, min(frame_count - 1, round(start_seconds * video_fps)))
    end_index = max(
        start_index,
        min(frame_count - 1, round(end_seconds * video_fps)),
    )
    return start_index, end_index


def extract_frames(video_path: Path, record: dict) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Video cannot be opened: {video_path}")
        video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(video_fps) or video_fps <= 0:
            raise ValueError(f"Video has invalid FPS: {video_path}")

        frame_count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            frame_count += 1
        if frame_count == 0:
            raise ValueError(f"Video contains no decodable frames: {video_path}")
    finally:
        capture.release()

    start, end = frame_interval(record, frame_count, video_fps)
    indices = np.linspace(start, end, NUM_FRAMES).round().astype(np.int64)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Video cannot be reopened: {video_path}")

        frames = []
        next_sample = 0
        for index in range(int(indices[-1]) + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Could not decode frame {index} from {video_path}")
            while next_sample < len(indices) and indices[next_sample] == index:
                sampled = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled = cv2.resize(
                    sampled,
                    (FRAME_WIDTH, FRAME_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
                frames.append(sampled)
                next_sample += 1
        if next_sample != NUM_FRAMES:
            raise ValueError(f"Could not sample enough frames from {video_path}")
        return np.stack(frames).astype(np.uint8, copy=False)
    finally:
        capture.release()


def preprocess_record(record: dict) -> dict:
    """Decode and save one clip, returning its manifest row or failure."""
    base = {
        key: record[key]
        for key in ("dataset", "video_id", "gloss", "split", "signer_id")
    }
    video_path = record.get("video_path")
    if video_path is None or not Path(video_path).is_file():
        raise FileNotFoundError("Downloaded video file is missing.")
    if record["split"] not in VALID_SPLITS:
        raise ValueError(f"Unrecognized data split: {record['split']!r}")
    if record["gloss"] not in TARGET_WORDS:
        raise ValueError(f"Unexpected target gloss: {record['gloss']!r}")

    output_dir = PROCESSED_DIR / record["split"] / record["gloss"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        safe_filename(f"{record['dataset']}_{record['video_id']}") + ".npz"
    )
    frames = extract_frames(Path(video_path), record)
    temporary = output_path.with_name(output_path.name + ".part")
    with temporary.open("wb") as file:
        np.savez_compressed(file, frames=frames)
    temporary.replace(output_path)
    return {
        **base,
        "source_path": os.path.relpath(video_path, PROCESSED_DIR.parent),
        "processed_path": str(output_path.relative_to(PROCESSED_DIR)),
        "frames": NUM_FRAMES,
        "height": FRAME_HEIGHT,
        "width": FRAME_WIDTH,
        "channels": 3,
        "dtype": "uint8",
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def popsign_records() -> list[dict]:
    """Read clips downloaded by download_popsign_targets.py (optional extra data)."""
    manifests = sorted((DATASETS_DIR / "external").glob("popsign_*/download_manifest.csv"))
    records = []
    for manifest_path in manifests:
        with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                gloss = canonicalize_gloss(row.get("gloss", ""))
                split = str(row.get("split", "")).casefold()
                if gloss not in TARGET_WORDS or split not in VALID_SPLITS:
                    continue
                extracted_dir = row.get("extracted_dir", "")
                for relative in (row.get("files", "") or "").split("|"):
                    if not relative or Path(relative).suffix.casefold() not in VIDEO_SUFFIXES:
                        continue
                    video_path = manifest_path.parent / extracted_dir / relative
                    if not video_path.is_file():
                        continue
                    raw_id = PurePosixPath(extracted_dir, relative).as_posix()
                    digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:6]
                    video_id = f"{PurePosixPath(raw_id).with_suffix('')}_{digest}"
                    records.append(
                        {
                            "dataset": "popsign",
                            "video_id": video_id,
                            "gloss": gloss,
                            "split": split,
                            "signer_id": "",
                            "video_path": video_path,
                            "frame_start": None,
                            "frame_end": None,
                            "annotation_fps": None,
                        }
                    )
    return records


def main() -> None:
    records: list[dict] = []
    if WLASL_METADATA.is_file():
        records.extend(wlasl_records())
    else:
        print(f"WLASL metadata not found at {WLASL_METADATA}; skipping WLASL.")
    if ASL_CITIZEN_SPLITS_DIR.is_dir() or ASL_CITIZEN_METADATA.is_file():
        records.extend(asl_citizen_records())
    else:
        print("No ASL Citizen data found; skipping ASL Citizen.")
    popsign = popsign_records()
    if popsign:
        records.extend(popsign)
        print(f"PopSign: {len(popsign)} extra clips found under datasets/external "
              "(download_popsign_targets.py); they join their official splits.")
    else:
        print("No PopSign downloads found; training with WLASL and ASL Citizen only.")
    if not records:
        raise ValueError("No selected video records were found.")

    successful = []
    failures = []
    for index, record in enumerate(records, start=1):
        try:
            successful.append(preprocess_record(record))
        except (OSError, ValueError, cv2.error) as error:
            failures.append(
                {
                    "dataset": record["dataset"],
                    "video_id": record["video_id"],
                    "gloss": record["gloss"],
                    "split": record["split"],
                    "error": str(error),
                }
            )
        if index % 50 == 0 or index == len(records):
            print(
                f"Preprocessed {index}/{len(records)} records; "
                f"{len(successful)} succeeded, {len(failures)} failed."
            )

    manifest_fields = [
        "dataset",
        "video_id",
        "gloss",
        "split",
        "signer_id",
        "source_path",
        "processed_path",
        "frames",
        "height",
        "width",
        "channels",
        "dtype",
    ]
    write_csv(PROCESSED_DIR / "manifest.csv", successful, manifest_fields)
    write_csv(
        PROCESSED_DIR / "failed_videos.csv",
        failures,
        ["dataset", "video_id", "gloss", "split", "error"],
    )
    if successful:
        summary = (
            pd.DataFrame(successful)
            .groupby(["dataset", "split", "gloss"], as_index=False)
            .size()
            .rename(columns={"size": "video_count"})
        )
        summary.to_csv(PROCESSED_DIR / "summary.csv", index=False)
    else:
        raise RuntimeError(
            f"No videos were successfully preprocessed. See "
            f"{PROCESSED_DIR / 'failed_videos.csv'}."
        )
    print(
        f"Finished: {len(successful)} videos processed; {len(failures)} failed. "
        f"Manifest: {PROCESSED_DIR / 'manifest.csv'}"
    )


if __name__ == "__main__":
    main()
