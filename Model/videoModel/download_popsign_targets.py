"""Download PopSign isolated-sign clips for the project's target glosses.

Run from VS Code's terminal with:
    python download_popsign_targets.py

popsign_v2_1 (default) has no tar archives; the site serves one .mp4 per clip
in per-sign folders, so this tool walks the folder listings, picks a seeded
signer-spread subset (60 train + 20 val + 20 test clips per word by default),
prints the total size, and only then asks for confirmation. Downloads are
resumable via Range requests. popsign_v1_0 still works through the old
one-archive-per-sign branch.

PopSign v2.1 source and access guide:
    https://signdata.cc.gatech.edu/view/datasets/popsign_v2_1/index.html
    https://signdata.cc.gatech.edu/view/guides/downloading_popsign/index.html
"""

from __future__ import annotations

import argparse
import csv
import html.parser
import http.client
import re
import shutil
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

import random

from src import config


DATASET = "popsign_v2_1"
CATEGORY = "game"  # Contains videos reviewed as the intended sign variant.
SPLITS = ("train", "val", "test")
BASE_URL = "https://signdata.cc.gatech.edu/data"
SEED = 20261116      # must match config.SEED: vocab.external_jobs re-picks the
                     # same subset with the same round-robin, seed and order
DEFAULT_WORDS = ["bye", "please", "sleep", "chicken"]
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
    "sleep": ("sleep",),
    "chicken": ("chicken",),
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


def ssl_contexts() -> list[ssl.SSLContext]:
    """Default CA context, then the certifi bundle for servers with incomplete chains."""
    contexts = [ssl.create_default_context()]
    try:
        import certifi
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    return contexts


def urlopen_fallback(request: urllib.request.Request, timeout: int):
    """urlopen that retries with certifi when the default chain fails verification."""
    for context in ssl_contexts():
        try:
            return urllib.request.urlopen(request, timeout=timeout, context=context)
        except urllib.error.URLError as error:
            if not isinstance(error.reason, ssl.SSLCertVerificationError):
                raise
            print("Note: default certificate verification failed; retrying with the "
                  "certifi CA bundle.", file=sys.stderr)
            continue
    raise urllib.error.URLError("Certificate verification failed with all available CA bundles.")


def request_bytes(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ASLT-PopSign-target-downloader/1.0"})
    for attempt in range(1, retries + 1):
        try:
            with urlopen_fallback(request, timeout) as response:
                return response.read()
        except http.client.IncompleteRead as error:
            # The signdata server truncates big directory listings mid-body
            # (measured: the 563-folder train index stops at 91 KB of 97 KB).
            # The partial HTML still parses, so keep it instead of failing;
            # main_per_clip probes folders directly when a link is missing.
            print(f"Note: the server cut this page at {len(error.partial)} of "
                  f"{len(error.partial) + error.expected} bytes; using the "
                  f"partial listing.", file=sys.stderr)
            return error.partial
        except (TimeoutError, OSError, urllib.error.URLError) as error:
            if attempt == retries:
                raise
            print(f"Note: read interrupted ({error}); retrying {attempt}/{retries}...",
                  file=sys.stderr)
            time.sleep(attempt)
    raise RuntimeError("unreachable")


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


# ---------------------------------------------------------------------------
# per-clip branch (popsign_v2_1: one .mp4 per recording, no tar archives)
# ---------------------------------------------------------------------------
class FolderIndex(html.parser.HTMLParser):
    """Collect sign-folder links from a split's index page."""

    def __init__(self) -> None:
        super().__init__()
        self.folders: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        path = urllib.parse.urlparse(href).path
        if path.endswith("/") and path.rstrip("/") and path.count("/") <= 2:
            self.folders.append(urllib.parse.unquote(path.strip("/").split("/")[-1]))


class Mp4Index(html.parser.HTMLParser):
    """Collect clip-folder links from one sign folder's listing.

    Each recording is a subfolder named <signer>-<gloss>-<datetime>-<take>.mp4
    (the listing shows a trailing slash). The video file inside carries the
    same name again and is served with Accept-Ranges: bytes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        path = urllib.parse.urlparse(href).path
        if path.endswith("/") and path.rstrip("/").endswith(".mp4"):
            self.hrefs.append(href)


def _split_url(dataset: str, split: str) -> str:
    return f"{BASE_URL}/{urllib.parse.quote(dataset)}/{CATEGORY}/{split}/"


def fetch_split_folders(dataset: str, split: str) -> dict[str, str]:
    """normalized folder name -> folder name, from the split index page."""
    directory = _split_url(dataset, split)
    parser = FolderIndex()
    try:
        parser.feed(request_bytes(directory).decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Could not read PopSign's {split} index at {directory}: {error}") from error
    folders = {normalize_label(name): name for name in parser.folders}
    if not folders:
        raise RuntimeError(
            f"No sign folders were listed at {directory}. Check the dataset "
            "version or run with --dataset popsign_v1_0 for the tar branch.")
    return folders


def fetch_folder_clips(dataset: str, split: str, folder: str) -> list[tuple[str, str]]:
    """(file name, absolute video URL) for every recording in one sign folder.

    Each listing entry is a subfolder <name>.mp4/ and the video file inside
    repeats the name: <name>.mp4/<name>.mp4 (Range-probed: 206, resumable).
    """
    directory = (f"{BASE_URL}/{urllib.parse.quote(dataset)}/{CATEGORY}/{split}/"
                 f"{urllib.parse.quote(folder)}/")
    parser = Mp4Index()
    try:
        parser.feed(request_bytes(directory).decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Could not read the folder listing at {directory}: {error}") from error
    clips = []
    for href in parser.hrefs:
        folder_url = urllib.parse.urljoin(directory, href)
        path = urllib.parse.urlparse(folder_url).path.rstrip("/")
        name = Path(urllib.parse.unquote(path)).name      # keeps the .mp4 suffix
        clips.append((name, folder_url + urllib.parse.quote(name)))
    return clips


def pick_spread(clips: list[tuple[str, str]], want: int, split: str,
                gloss: str, dataset: str) -> list[tuple[str, str]]:
    """The same seeded signer round-robin src/vocab.external_jobs re-runs.

    Filenames are <signer>-<gloss>-<datetime>-<take>.mp4; the 4-char signer
    code is the first field. `gloss` must be the SERVER's gloss (e.g.
    "thankyou"): the seed string has to stay identical to the one
    vocab.external_jobs builds from its own gloss keys, otherwise extraction
    would pick a different subset than the download.
    """
    by_signer: dict[str, list[tuple[str, str]]] = {}
    for name, url in sorted(clips):
        signer = name[:4]
        if len(signer) == 4 and signer.isalnum():
            by_signer.setdefault(signer, []).append((name, url))
    order = sorted(by_signer)
    rng = random.Random(f"{SEED}:external:{dataset}:{split}:{gloss}")
    rng.shuffle(order)
    pools = [by_signer[s] for s in order]
    picked: list[tuple[str, str]] = []
    k = 0
    while len(picked) < want and any(pools):
        pool = pools[k % len(pools)]
        if pool:
            picked.append(pool.pop(0))
        k += 1
    return picked


def head_size(url: str, timeout: int = 30) -> int | None:
    request = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "ASLT-PopSign-target-downloader/1.0"})
    with urlopen_fallback(request, timeout) as response:
        length = response.headers.get("Content-Length")
        return int(length) if length else None


def download_clip(url: str, destination: Path, expected: int | None,
                  retries: int = 3) -> None:
    """One clip with Range-resume; the server sends Accept-Ranges: bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            if expected is not None and destination.is_file() \
                    and destination.stat().st_size == expected:
                return                              # already complete
            existing = destination.stat().st_size if destination.is_file() else 0
            if expected is not None and existing > expected:
                destination.unlink(missing_ok=True)  # corrupt: start over
                existing = 0
            headers = {"User-Agent": "ASLT-PopSign-target-downloader/1.0"}
            if existing:
                headers["Range"] = f"bytes={existing}-"
            request = urllib.request.Request(url, headers=headers)
            with urlopen_fallback(request, 120) as response, \
                    destination.open("ab" if existing else "wb") as output:
                if response.status == 200 and existing:
                    # the server ignored the Range header: restart the file
                    output.truncate(0)
                    output.seek(0)
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if expected is not None and destination.stat().st_size != expected:
                raise OSError(f"size {destination.stat().st_size} != expected "
                              f"{expected}; the server may have truncated")
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt == retries:
                raise RuntimeError(f"Download failed after {retries} attempts: {url}: {error}") from error
            time.sleep(attempt * 2)


def main_per_clip(args: argparse.Namespace) -> int:
    words = args.words or DEFAULT_WORDS
    unknown = [w for w in words if w not in TARGET_ALIASES]
    if unknown:
        print(f"Unknown target word(s): {unknown}. Known: {sorted(TARGET_ALIASES)}.",
              file=sys.stderr)
        return 2
    plan: list[dict] = []
    missing: list[dict[str, str]] = []
    for split in args.splits:
        try:
            folders = fetch_split_folders(args.dataset, split)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
        want = args.clips_per_word if split == "train" else args.heldout_clips
        for gloss in words:
            folder = None
            for alias in TARGET_ALIASES[gloss]:
                folder = folders.get(normalize_label(alias))
                if folder:
                    break
            clips: list[tuple[str, str]] | None = None
            if folder is not None:
                try:
                    clips = fetch_folder_clips(args.dataset, split, folder)
                except RuntimeError as error:
                    print(error, file=sys.stderr)
                    return 1
            else:
                # the split index can arrive truncated (the server cuts big
                # listings mid-body), so probe the alias folder directly
                for alias in TARGET_ALIASES[gloss]:
                    try:
                        probe = fetch_folder_clips(args.dataset, split, alias)
                    except RuntimeError:
                        continue
                    if probe:
                        folder, clips = alias, probe
                        break
            if folder is None or not clips:
                missing.append({"gloss": gloss, "split": split})
                continue
            # folder is the SERVER's gloss (e.g. "thankyou"); it names both
            # the destination folder and the subset seed, so vocab's walk and
            # round-robin (keyed on the same gloss) find exactly these clips
            picked = pick_spread(clips, want, split, folder, args.dataset)
            for name, url in picked:
                plan.append({"gloss": folder, "word": gloss, "split": split,
                             "file": name, "url": url})
    print(f"Dataset: {args.dataset}/{CATEGORY}; words: {', '.join(words)}; "
          f"{len(plan)} clips selected "
          f"(train {args.clips_per_word}, val/test {args.heldout_clips} per word).")
    if missing:
        print("No folder match:")
        for item in missing:
            print(f"  {item['split']}: {item['gloss']}")
    print("Measuring the exact total size (one HEAD per clip)...")
    total = 0
    unknown_size = 0
    for i, item in enumerate(plan, 1):
        try:
            size = head_size(item["url"])
        except (urllib.error.URLError, TimeoutError, OSError):
            size = None
        if size is None:
            unknown_size += 1
        else:
            total += size
            item["bytes"] = size
        if i % 50 == 0 or i == len(plan):
            print(f"  {i}/{len(plan)} clips sized")
    if unknown_size:
        print(f"WARNING: {unknown_size} clips answered no HEAD; the total is "
              "an underestimate.")
    print(f"Total: {len(plan)} clips, {total / 1e9:.2f} GB "
          f"(+{unknown_size} unsized).")
    print(f"Output directory: {args.output.resolve()}")
    if not args.yes and input("Type DOWNLOAD to proceed: ").strip() != "DOWNLOAD":
        print("Cancelled; no clips downloaded.")
        return 0

    rows: list[dict] = []
    failures: list[dict] = []
    for i, item in enumerate(plan, 1):
        dest = args.output / item["split"] / item["gloss"] / item["file"]
        try:
            if i % 20 == 0 or i == len(plan):
                print(f"Downloading {i}/{len(plan)}...")
            download_clip(item["url"], dest, item.get("bytes"))
            rows.append({"gloss": item["gloss"], "split": item["split"],
                         "file": item["file"], "url": item["url"],
                         "bytes": dest.stat().st_size,
                         "signer": item["file"][:4]})
        except (OSError, RuntimeError, ValueError) as error:
            failures.append({**item, "error": str(error)})
            print(f"  Failed: {item['file']}: {error}", file=sys.stderr)
    write_csv(args.output / "download_manifest.csv", rows,
              ["gloss", "word", "split", "file", "url", "bytes", "signer"])
    write_csv(args.output / "download_failures.csv", failures,
              ["gloss", "split", "file", "url", "error"])
    print(f"Done: {len(rows)} clips downloaded; {len(failures)} failed.")
    print(f"Manifest: {args.output / 'download_manifest.csv'}")
    print("The pipeline picks the same subset back up automatically: run")
    print("  python -m src.extract --splits train,val,test --workers 4")
    return 1 if failures else 0


def download(url: str, destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ASLT-PopSign-target-downloader/1.0"})
            with urlopen_fallback(request, 120) as response, partial.open("wb") as output:
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
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=None,
                        help="Official splits to fetch; tar branch defaults to "
                             "train, the per-clip branch to train val test.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (defaults to datasets/external/<dataset-version> in this videoModel folder).")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded tar files after extraction.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt after showing matches.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the selected subset; no video downloads, HEAD sizing or file writes.")
    parser.add_argument("--words", nargs="+", default=None,
                        help=f"Words to fetch in the per-clip branch (default: "
                        f"{' '.join(DEFAULT_WORDS)}; focused communication signs).")
    parser.add_argument("--clips-per-word", type=int, default=60,
                        help="Train clips per word, signer-spread (default 60).")
    parser.add_argument("--heldout-clips", type=int, default=20,
                        help="Val/test clips per word for the ps_test group (default 20).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output is None:
        args.output = config.EXTERNAL_DIR / args.dataset
    if len(set(args.splits or [])) != len(args.splits or []):
        print("Choose each split only once.", file=sys.stderr)
        return 2
    if args.dataset == "popsign_v2_1":
        if args.splits is None:
            args.splits = ["train", "val", "test"]
        return main_per_clip(args)
    if args.splits is None:
        args.splits = ["train"]
    return main_tar(args)


def main_tar(args: argparse.Namespace) -> int:
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
