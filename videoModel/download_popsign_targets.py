"""Download PopSign isolated-sign archives for the project's 20 target glosses.

Run from VS Code's terminal with:
    python download_popsign_targets.py

By default, downloads only the `game/train` archives that have an exact label match.
Use `--splits train val test` for all official splits. The official site serves one
tar archive per sign and split; archives are safely extracted, then removed unless
`--keep-archives` is specified. No third-party Python packages are required.

PopSign v2.1 source and access guide:
    https://signdata.cc.gatech.edu/view/datasets/popsign_v2_1/index.html
    https://signdata.cc.gatech.edu/view/guides/downloading_popsign/index.html
"""

from __future__ import annotations

import argparse
import csv
import html.parser
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


DATASET = "popsign_v2_1"
CATEGORY = "game"  # Contains videos reviewed as the intended sign variant.
SPLITS = ("train", "val", "test")
BASE_URL = "https://signdata.cc.gatech.edu/data"
TARGET_ALIASES = {
    "eat": ("eat",),
    "family": ("family",),
    "friend": ("friend",),
    "good": ("good",),
    "hello": ("hello",),
    "help": ("help",),
    "name": ("name",),
    "sad": ("sad",),
    "school": ("school",),
    "drink": ("drink",),
    "thanks": ("thanks", "thank you", "thank-you"),
    "love": ("love",),
    "bye": ("bye", "goodbye", "good-bye"),
    "sorry": ("sorry",),
    "yes": ("yes",),
    "no": ("no",),
    "happy": ("happy",),
    "please": ("please",),
    "bad": ("bad",),
    "home": ("home",),
}


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", urllib.parse.unquote(value).casefold())


class DirectoryIndex(html.parser.HTMLParser):
    """Collect tar links from the public directory listing used by the official API."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = dict(attrs).get("href")
        if href and urllib.parse.urlparse(href).path.casefold().endswith(".tar"):
            self.hrefs.append(href)


def request_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ASLT-PopSign-target-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_index(dataset: str, split: str) -> dict[str, tuple[str, str]]:
    """Return normalized source-label -> (display label, absolute archive URL)."""
    directory = f"{BASE_URL}/{urllib.parse.quote(dataset)}/{CATEGORY}/{split}/"
    parser = DirectoryIndex()
    try:
        parser.feed(request_bytes(directory).decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Could not read PopSign's {split} index at {directory}: {error}") from error
    files: dict[str, tuple[str, str]] = {}
    for href in parser.hrefs:
        absolute = urllib.parse.urljoin(directory, href)
        filename = Path(urllib.parse.urlparse(absolute).path).name
        label = urllib.parse.unquote(filename[:-4])
        files.setdefault(normalize_label(label), (label, absolute))
    if not files:
        raise RuntimeError(
            f"No .tar archives were listed at {directory}. Check the dataset version "
            "or run with --dataset popsign_v1_0 if that is the version provided by the site."
        )
    return files


def matched_archives(dataset: str, splits: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selected: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for split in splits:
        index = fetch_index(dataset, split)
        for gloss, aliases in TARGET_ALIASES.items():
            found = None
            for alias in aliases:
                found = index.get(normalize_label(alias))
                if found:
                    break
            if found:
                source_label, url = found
                selected.append({"gloss": gloss, "source_label": source_label, "split": split, "url": url})
            else:
                missing.append({"gloss": gloss, "split": split})
    return selected, missing


def download(url: str, destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ASLT-PopSign-target-downloader/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if partial.stat().st_size == 0:
                raise OSError("Server returned an empty archive.")
            partial.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Download failed after {retries} attempts: {url}: {error}") from error
            time.sleep(attempt * 2)


def safe_extract(archive_path: Path, destination: Path) -> list[str]:
    """Extract regular files only and reject traversal paths and links."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    written = []
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in member.name
                    or (relative.parts and relative.parts[0].endswith(":"))):
                raise ValueError(f"Unsafe path in archive {archive_path.name}: {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported link or special file in archive: {member.name!r}")
            target = (destination / Path(*relative.parts)).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Archive member escapes its output folder: {member.name!r}")
        for member in members:
            target = destination / Path(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.is_file() and target.stat().st_size == member.size:
                    written.append(target.relative_to(destination).as_posix())
                    continue
                raise FileExistsError(f"Refusing to overwrite a different existing file: {target}")
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read archive member: {member.name}")
            temporary = target.with_name(target.name + ".part")
            with source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.replace(target)
            written.append(target.relative_to(destination).as_posix())
    return written


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET, help="PopSign dataset folder/version (default: popsign_v2_1).")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["train"],
                        help="Official split archives to fetch; defaults to train only to limit download size.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (defaults to datasets/external/<dataset-version>).")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded tar files after extraction.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt after showing matches.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output is None:
        args.output = Path(__file__).resolve().parent / "datasets" / "external" / args.dataset
    if len(set(args.splits)) != len(args.splits):
        print("Choose each split only once.", file=sys.stderr)
        return 2
    try:
        selected, missing = matched_archives(args.dataset, args.splits)
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Dataset: {args.dataset}/{CATEGORY}; matched {len(selected)} sign archives.")
    for item in selected:
        print(f"  {item['split']}: {item['gloss']} <- {item['source_label']}")
    if missing:
        print("No exact archive match:")
        for item in missing:
            print(f"  {item['split']}: {item['gloss']}")
    print(f"Output directory: {args.output.resolve()}")
    print("These are original participant videos and may require substantial disk space.")
    if not args.yes and input("Type DOWNLOAD to proceed: ").strip() != "DOWNLOAD":
        print("Cancelled; no archives downloaded.")
        return 0

    manifest_rows = []
    failures = []
    for item in selected:
        archive_path = args.output / "archives" / item["split"] / f"{item['source_label']}.tar"
        extract_dir = args.output / item["split"] / item["gloss"]
        try:
            print(f"Downloading {item['gloss']} ({item['split']})...")
            if not archive_path.is_file():
                download(item["url"], archive_path)
            files = safe_extract(archive_path, extract_dir)
            if not args.keep_archives:
                archive_path.unlink(missing_ok=True)
            manifest_rows.append({**item, "extracted_dir": str(extract_dir.relative_to(args.output)),
                                  "file_count": str(len(files)), "files": "|".join(files)})
            print(f"  Extracted {len(files)} files.")
        except (OSError, RuntimeError, ValueError, tarfile.TarError) as error:
            failures.append({**item, "error": str(error)})
            print(f"  Failed: {error}", file=sys.stderr)
    write_csv(args.output / "download_manifest.csv", manifest_rows,
              ["gloss", "source_label", "split", "url", "extracted_dir", "file_count", "files"])
    write_csv(args.output / "download_failures.csv", failures,
              ["gloss", "source_label", "split", "url", "error"])
    print(f"Done: {len(manifest_rows)} archives extracted; {len(failures)} failed.")
    print(f"Manifest: {args.output / 'download_manifest.csv'}")
    print("Review downloaded labels/variants before adding clips to model training.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
