"""Download selected WLASL clips and selected ASL Citizen videos."""

import argparse
import csv
import io
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from src.config import (
    ASL_CITIZEN_ARCHIVE,
    ASL_CITIZEN_DIR,
    ASL_CITIZEN_METADATA,
    ASL_CITIZEN_METADATA_DIR,
    ASL_CITIZEN_VIDEOS,
    DATASETS_DIR,
    MIN_ASL_CITIZEN_FREE_BYTES,
    TARGET_WORDS,
    WLASL_DIR,
    WLASL_METADATA,
    WLASL_VIDEOS,
    canonicalize_gloss,
    safe_filename,
)

WLASL_METADATA_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/"
    "start_kit/WLASL_v0.3.json"
)
WLASL_CUDA_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/"
    "start_kit/C-UDA-1.0.pdf"
)
ASL_CITIZEN_URL = (
    "https://download.microsoft.com/download/b/8/8/"
    "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
)
VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".flv", ".3gp"}
CSV_COLUMN_NAMES = {
    "videoid": "video_id",
    "videofile": "video_id",
    "filename": "video_id",
    "video": "video_id",
    "gloss": "gloss",
    "label": "gloss",
    "word": "gloss",
    "signerid": "signer_id",
    "participantid": "signer_id",
    "signer": "signer_id",
    "participant": "signer_id",
}


def download_file(url: str, destination: Path) -> None:
    """Download a small file atomically, reusing an existing nonempty file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "ASLT-Assignment3/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        with temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    if temporary.stat().st_size == 0:
        raise OSError(f"Download produced an empty file: {url}")
    temporary.replace(destination)


def remote_content_length(url: str) -> int | None:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "ASLT-Assignment3/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
        return None


def manual_asl_citizen_present() -> bool:
    """Detect the manually supplied ASL Citizen folder (split CSVs + videos)."""
    splits = ASL_CITIZEN_DIR / "splits"
    return (
        splits.is_dir()
        and any(splits.glob("*.csv"))
        and ASL_CITIZEN_VIDEOS.is_dir()
        and any(ASL_CITIZEN_VIDEOS.iterdir())
    )


def check_asl_citizen_space() -> None:
    ASL_CITIZEN_DIR.mkdir(parents=True, exist_ok=True)
    if ASL_CITIZEN_ARCHIVE.is_file() and ASL_CITIZEN_ARCHIVE.stat().st_size:
        return
    free_bytes = shutil.disk_usage(ASL_CITIZEN_DIR).free
    expected = remote_content_length(ASL_CITIZEN_URL)
    required = max(
        MIN_ASL_CITIZEN_FREE_BYTES,
        expected + 5_000_000_000 if expected is not None else 0,
    )
    if free_bytes < required:
        expected_text = (
            f"{expected / 1e9:.1f} GB archive plus 5 GB buffer"
            if expected is not None
            else "approximately 30 GB archive plus 5 GB buffer"
        )
        raise OSError(
            f"Not enough free space for ASL Citizen in {ASL_CITIZEN_DIR}. "
            f"Available: {free_bytes / 1e9:.1f} GB; required: "
            f"at least {required / 1e9:.1f} GB ({expected_text})."
        )


def download_resumable(url: str, destination: Path) -> None:
    """Download a large file with Range resume and a temporary partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "ASLT-Assignment3/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range)
            if not match or int(match.group(1)) != offset:
                raise OSError(f"Unexpected Range response: {content_range!r}")
            total = int(match.group(3)) if match.group(3) != "*" else None
            mode = "ab"
        else:
            offset = 0
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            mode = "wb"

        downloaded = offset
        last_report = time.monotonic()
        with partial.open(mode) as output:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 3:
                    if total:
                        print(
                            f"\rASL Citizen: {downloaded / 1e9:.2f} / "
                            f"{total / 1e9:.2f} GB",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\rASL Citizen: {downloaded / 1e9:.2f} GB",
                            end="",
                            flush=True,
                        )
                    last_report = now
    print()
    if total is not None and downloaded != total:
        raise OSError(
            f"Incomplete ASL Citizen download: {downloaded} of {total} bytes. "
            "Run the command again to resume."
        )
    if partial.stat().st_size == 0:
        raise OSError("ASL Citizen download produced an empty archive.")
    partial.replace(destination)


def load_wlasl_entries() -> list[dict]:
    download_file(WLASL_METADATA_URL, WLASL_METADATA)
    try:
        data = json.loads(WLASL_METADATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid WLASL metadata: {WLASL_METADATA}") from error
    if not isinstance(data, list):
        raise ValueError("WLASL metadata must be a list of gloss entries.")
    return data


def existing_video(video_dir: Path, video_id: object) -> Path | None:
    safe_id = safe_filename(video_id)
    for path in video_dir.glob(safe_id + ".*"):
        if path.suffix.casefold() in VIDEO_SUFFIXES and path.stat().st_size:
            return path
    return None


def download_direct_video(url: str, destination: Path) -> None:
    suffix = PurePosixPath(urllib.parse.urlparse(url).path).suffix.casefold()
    if suffix not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported direct-video URL extension: {suffix or '(none)'}")
    partial = destination.with_name(destination.stem + suffix + ".part")
    final = destination.with_name(destination.stem + suffix)
    if final.exists() and final.stat().st_size:
        return
    request = urllib.request.Request(
        url, headers={"User-Agent": "ASLT-Assignment3/1.0"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        with partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    if partial.stat().st_size == 0:
        raise OSError("Direct video download returned an empty file.")
    partial.replace(final)


def download_youtube_video(url: str, destination: Path) -> None:
    try:
        import yt_dlp
    except ImportError as error:
        raise RuntimeError(
            "yt-dlp is required for WLASL YouTube videos. "
            "Install it with: python -m pip install yt-dlp"
        ) from error
    options = {
        # Many WLASL sources predate MP4-only uploads and only expose adaptive
        # streams. Merge best video+audio (ffmpeg is bundled with the setup)
        # and remux legacy containers to MP4; fall back to pre-merged formats.
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "remuxvideo": "mp4",
        "outtmpl": str(destination.with_suffix(".%(ext)s")),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
        "fragment_retries": 2,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as error:
        raise OSError(f"yt-dlp could not retrieve the source video: {error}") from error
    if not info or info.get("_type") == "playlist":
        raise OSError("WLASL URL did not resolve to one video.")
    if existing_video(destination.parent, destination.stem) is None:
        raise OSError("yt-dlp completed without producing a recognized video file.")


def download_wlasl_videos(entries: list[dict]) -> None:
    WLASL_VIDEOS.mkdir(parents=True, exist_ok=True)
    target_set = set(TARGET_WORDS)
    samples = [
        (canonicalize_gloss(entry.get("gloss", "")), instance)
        for entry in entries
        if canonicalize_gloss(entry.get("gloss", "")) in target_set
        for instance in entry.get("instances", [])
    ]
    if not samples:
        raise ValueError("No proposal-vocabulary samples were found in WLASL.")

    failures: list[dict[str, str]] = []
    completed = 0
    for index, (gloss, instance) in enumerate(samples, start=1):
        video_id = str(instance.get("video_id", "")).strip()
        url = str(instance.get("url", "")).strip()
        split = str(instance.get("split", "unknown")).casefold()
        if not video_id or not url:
            failures.append(
                {"video_id": video_id, "gloss": gloss, "split": split, "error": "Missing video_id or URL"}
            )
            continue
        destination = WLASL_VIDEOS / safe_filename(video_id)
        if existing_video(WLASL_VIDEOS, video_id):
            completed += 1
            continue
        try:
            if "youtube.com" in url.casefold() or "youtu.be" in url.casefold():
                download_youtube_video(url, destination)
            else:
                download_direct_video(url, destination)
            completed += 1
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            RuntimeError,
        ) as error:
            failures.append(
                {
                    "video_id": video_id,
                    "gloss": gloss,
                    "split": split,
                    "error": str(error),
                }
            )
        if index % 20 == 0 or index == len(samples):
            print(f"WLASL: processed {index}/{len(samples)} videos.")
        time.sleep(0.25)

    failure_path = WLASL_DIR / "download_failures.csv"
    if failures:
        with failure_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=["video_id", "gloss", "split", "error"]
            )
            writer.writeheader()
            writer.writerows(failures)
    elif failure_path.exists():
        failure_path.unlink()
    print(
        f"WLASL: {completed} videos present or downloaded; "
        f"{len(failures)} unavailable (see {failure_path} if any)."
    )


def report_manual_wlasl_videos(entries: list[dict]) -> None:
    WLASL_VIDEOS.mkdir(parents=True, exist_ok=True)
    target_set = set(TARGET_WORDS)
    samples = [
        (canonicalize_gloss(entry.get("gloss", "")), instance)
        for entry in entries
        if canonicalize_gloss(entry.get("gloss", "")) in target_set
        for instance in entry.get("instances", [])
    ]
    if not samples:
        raise ValueError("No proposal-vocabulary samples were found in WLASL.")
    failures = []
    for gloss, instance in samples:
        video_id = str(instance.get("video_id", "")).strip()
        split = str(instance.get("split", "unknown")).casefold()
        if not video_id or not existing_video(WLASL_VIDEOS, video_id):
            failures.append(
                {
                    "video_id": video_id,
                    "gloss": gloss,
                    "split": split,
                    "error": "Video file was not found in the manually supplied WLASL videos folder.",
                }
            )

    failure_path = WLASL_DIR / "download_failures.csv"
    if failures:
        with failure_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=["video_id", "gloss", "split", "error"]
            )
            writer.writeheader()
            writer.writerows(failures)
    elif failure_path.exists():
        failure_path.unlink()
    print(
        f"WLASL: found {len(samples) - len(failures)}/{len(samples)} selected "
        f"videos in the manual folder; {len(failures)} missing."
    )
    if failures:
        print(f"WLASL: missing-video list: {failure_path}")


def normalized_csv_row(row: dict[str, str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for key, value in row.items():
        normalized_key = re.sub(r"[^a-z0-9]", "", (key or "").casefold())
        canonical_key = CSV_COLUMN_NAMES.get(normalized_key)
        if canonical_key and canonical_key not in mapped:
            mapped[canonical_key] = (value or "").strip()
    return mapped


def infer_split(filename: str) -> str:
    name = Path(filename).name.casefold()
    if "train" in name:
        return "train"
    if "valid" in name or "val" in name or "dev" in name:
        return "val"
    if "test" in name:
        return "test"
    return "unknown"


def read_asl_citizen_metadata(
    archive: zipfile.ZipFile,
) -> tuple[list[dict[str, str]], dict[str, zipfile.ZipInfo]]:
    video_infos = [
        info
        for info in archive.infolist()
        if PurePosixPath(info.filename).suffix.casefold() in VIDEO_SUFFIXES
        and not info.is_dir()
    ]
    member_by_stem: dict[str, zipfile.ZipInfo] = {}
    for info in video_infos:
        member_by_stem.setdefault(PurePosixPath(info.filename).stem.casefold(), info)

    selected_rows: list[dict[str, str]] = []
    for info in archive.infolist():
        if PurePosixPath(info.filename).suffix.casefold() != ".csv":
            continue
        split = infer_split(info.filename)
        try:
            with archive.open(info) as member:
                text = io.TextIOWrapper(member, encoding="utf-8-sig", newline="")
                reader = csv.DictReader(text)
                if not reader.fieldnames:
                    continue
                for raw_row in reader:
                    row = normalized_csv_row(raw_row)
                    if not row.get("video_id") or not row.get("gloss"):
                        continue
                    gloss = canonicalize_gloss(row["gloss"])
                    if gloss not in TARGET_WORDS:
                        continue
                    video_id = Path(row["video_id"]).stem
                    video_info = member_by_stem.get(video_id.casefold())
                    selected_rows.append(
                        {
                            "video_id": video_id,
                            "gloss": gloss,
                            "signer_id": row.get("signer_id", ""),
                            "split": split,
                            "archive_member": video_info.filename if video_info else "",
                        }
                    )
        except (UnicodeDecodeError, csv.Error) as error:
            raise ValueError(
                f"Could not read ASL Citizen metadata file {info.filename}."
            ) from error

    unique_rows: dict[str, dict[str, str]] = {}
    for row in selected_rows:
        key = row["video_id"].casefold()
        if key in unique_rows:
            previous = unique_rows[key]
            if any(previous[field] != row[field] for field in ("gloss", "split")):
                raise ValueError(
                    f"ASL Citizen video {row['video_id']} has conflicting "
                    "labels or split assignments."
                )
            continue
        unique_rows[key] = row
    return list(unique_rows.values()), member_by_stem


def extract_selected_asl_citizen_videos(
    archive: zipfile.ZipFile,
    rows: list[dict[str, str]],
) -> tuple[int, list[dict[str, str]]]:
    ASL_CITIZEN_VIDEOS.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []
    extracted = 0
    for row in rows:
        member_name = row["archive_member"]
        if not member_name:
            errors.append(
                {
                    "video_id": row["video_id"],
                    "gloss": row["gloss"],
                    "split": row["split"],
                    "error": "No matching video entry in archive",
                }
            )
            continue
        info = archive.getinfo(member_name)
        suffix = PurePosixPath(member_name).suffix.casefold()
        destination = ASL_CITIZEN_VIDEOS / (
            safe_filename(row["video_id"]) + suffix
        )
        if destination.exists():
            if destination.stat().st_size == info.file_size:
                extracted += 1
                continue
            raise OSError(
                f"Existing file has an unexpected size; refusing to overwrite: "
                f"{destination}"
            )

        temporary = destination.with_name(destination.name + ".part")
        copied = 0
        with archive.open(info) as source, temporary.open("wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                copied += len(chunk)
        if copied != info.file_size:
            temporary.unlink(missing_ok=True)
            raise OSError(
                f"Truncated video while extracting {member_name}: "
                f"{copied} of {info.file_size} bytes."
            )
        temporary.replace(destination)
        extracted += 1
    return extracted, errors


def download_asl_citizen() -> None:
    if not ASL_CITIZEN_ARCHIVE.is_file():
        download_resumable(ASL_CITIZEN_URL, ASL_CITIZEN_ARCHIVE)
    try:
        with zipfile.ZipFile(ASL_CITIZEN_ARCHIVE) as archive:
            rows, _ = read_asl_citizen_metadata(archive)
            if not rows:
                raise ValueError(
                    "No proposal-vocabulary metadata rows were found in the "
                    "ASL Citizen archive; check its layout and CSV headers."
                )
            ASL_CITIZEN_METADATA_DIR.mkdir(parents=True, exist_ok=True)
            with ASL_CITIZEN_METADATA.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "video_id",
                        "gloss",
                        "signer_id",
                        "split",
                        "archive_member",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)
            extracted, errors = extract_selected_asl_citizen_videos(
                archive, rows
            )
    except zipfile.BadZipFile as error:
        raise ValueError(
            f"{ASL_CITIZEN_ARCHIVE} is not a valid ZIP. Remove that specific "
            "corrupt archive and rerun the preparation command."
        ) from error

    if errors:
        failure_path = ASL_CITIZEN_DIR / "extraction_failures.csv"
        with failure_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=["video_id", "gloss", "split", "error"]
            )
            writer.writeheader()
            writer.writerows(errors)
    print(
        f"ASL Citizen: {extracted}/{len(rows)} selected videos extracted "
        f"to {ASL_CITIZEN_VIDEOS}."
    )
    if errors:
        print(
            f"ASL Citizen: {len(errors)} videos could not be matched; "
            f"see {ASL_CITIZEN_DIR / 'extraction_failures.csv'}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare WLASL and ASL Citizen data for the proposal vocabulary."
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help=(
            "Use locally supplied WLASL metadata/videos and ASL Citizen ZIP; "
            "do not download source data."
        ),
    )
    parser.add_argument(
        "--skip-asl-citizen",
        action="store_true",
        help=(
            "Only prepare WLASL; skip the ASL Citizen download and license "
            "check. Implied automatically when a manual ASL_Citizen folder "
            "with split CSVs and videos is present."
        ),
    )
    parser.add_argument(
        "--confirm-wlasl-cuda",
        action="store_true",
        help="Confirm that you read and accept the WLASL C-UDA terms.",
    )
    parser.add_argument(
        "--confirm-asl-citizen-license",
        action="store_true",
        help="Confirm that you read and accept the ASL Citizen license terms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skip_asl_citizen = args.skip_asl_citizen or manual_asl_citizen_present()
    if skip_asl_citizen:
        print(
            "ASL Citizen: manual dataset detected (or --skip-asl-citizen); "
            "skipping the ASL Citizen download."
        )
    if not args.confirm_wlasl_cuda or (
        not skip_asl_citizen and not args.confirm_asl_citizen_license
    ):
        raise SystemExit(
            "Before downloading, read both dataset licenses. Then rerun with "
            "--confirm-wlasl-cuda and --confirm-asl-citizen-license."
        )

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if args.manual:
        required_files = [WLASL_METADATA]
        if not skip_asl_citizen:
            required_files.append(ASL_CITIZEN_ARCHIVE)
        missing_files = [path for path in required_files if not path.is_file()]
        if missing_files:
            formatted = "\n".join(f"  - {path}" for path in missing_files)
            raise FileNotFoundError(
                "Manual mode requires these files to be supplied first:\n"
                f"{formatted}"
            )
    elif not skip_asl_citizen:
        check_asl_citizen_space()

    WLASL_DIR.mkdir(parents=True, exist_ok=True)
    if not args.manual:
        download_file(WLASL_CUDA_URL, WLASL_DIR / "C-UDA-1.0.pdf")
    entries = load_wlasl_entries()
    if args.manual:
        report_manual_wlasl_videos(entries)
    else:
        download_wlasl_videos(entries)
    if not skip_asl_citizen:
        download_asl_citizen()
    print(f"\nDataset preparation finished in {DATASETS_DIR}.")


if __name__ == "__main__":
    main()
