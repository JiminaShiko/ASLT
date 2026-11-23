"""Download the J/Z motion corpora (see src/config.py:MOTION_DATASETS).

msl_dynamic is NOT ASL: it trains the J/Z trajectory stage only, never the
24-class shape model. Nothing is trained here and no frame is written.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

MANIFEST = config.DATA_DIR / "jz_datasets.json"
CLIP_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
USER_AGENT = "ASLT-jz-downloader/1.0 (+research project)"
CHUNK = 1 << 20


def url_for(spec: dict, filename: str) -> str:
    if spec["kind"] == "zenodo":
        return (f"https://zenodo.org/api/records/{spec['record']}"
                f"/files/{urllib.parse.quote(filename)}/content")
    return spec["url"]


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def fetch(url: str, dest: Path, expected: int | None = None) -> None:
    """Download with HTTP Range resume. A partial file keeps its .part name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if dest.is_file() and (expected is None or dest.stat().st_size == expected):
        print(f"  [cached] {dest.name} ({human(dest.stat().st_size)})")
        return
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"  [resume] {dest.name} at {human(have)}")
    else:
        print(f"  [get]    {url}")
    request = urllib.request.Request(url, headers=headers)
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            total = response.headers.get("Content-Length")
            total = int(total) + have if total else expected
            mode = "ab" if have and response.status == 206 else "wb"
            if mode == "wb":
                have = 0
            with open(part, mode) as fh:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
                    have += len(block)
                    if total:
                        pct = 100.0 * have / total
                        sys.stdout.write(
                            f"\r           {pct:5.1f}%  {human(have)} / {human(total)}")
                        sys.stdout.flush()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"download failed: HTTP {exc.code} for {url}\n"
                         "  the Zenodo record or the TTIC page may have moved; "
                         "check the URL in src/config.py:MOTION_DATASETS") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(f"download failed: {type(exc).__name__}: {exc}\n"
                         f"  the partial file is kept at {part} and the next "
                         "run resumes it") from exc
    print()
    part.replace(dest)
    print(f"  [done]   {dest.name} in {time.time() - started:.0f}s "
          f"({human(dest.stat().st_size)})")


def unpack(archive: Path, dest: Path, force: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if not force and next((p for suffix in CLIP_SUFFIXES
                           for p in dest.rglob("*" + suffix)), None) is not None:
        print(f"  [cached] {dest.name}/ already holds clips")
        return
    suffix = archive.suffix.lower()
    print(f"  [unpack] {archive.name} -> {dest}")
    if suffix == ".7z":
        try:
            import py7zr
        except ImportError as exc:  # pragma: no cover - dependency message
            raise SystemExit(
                "extracting a .7z archive needs py7zr:\n"
                "  pip install py7zr\n"
                f"or unpack {archive} yourself into {dest}") from exc
        with py7zr.SevenZipFile(archive, "r") as zf:
            zf.extractall(dest)
    elif suffix in (".tgz", ".tar", ".gz"):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest)  # noqa: S202 - trusted, public dataset archives
    elif suffix == ".zip":
        shutil.unpack_archive(str(archive), str(dest))
    else:
        raise SystemExit(f"do not know how to unpack {archive.name}")


def clip_count(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in CLIP_SUFFIXES)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the J/Z motion corpora.")
    ap.add_argument("--only", nargs="+", default=None, metavar="ID",
                    help="download only these ids (default: every non-optional "
                         "corpus, i.e. msl_dynamic)")
    ap.add_argument("--force", action="store_true",
                    help="re-download and re-extract even if already present")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep the .7z/.tgz after a successful extraction")
    ap.add_argument("--list", action="store_true",
                    help="print the registry and exit")
    args = ap.parse_args()

    by_id = {d["id"]: d for d in config.MOTION_DATASETS}
    if args.list:
        for spec in config.MOTION_DATASETS:
            print(f"{spec['id']:<14} {spec['size_gb']:>6.1f} GB  "
                  f"{spec['license']:<12} letters: {spec['letters']}")
            print(f"{'':<14} {spec['why']}")
        print(f"\ndestination: {config.JZ_CLIP_DIR}")
        return 0

    wanted = args.only or [d["id"] for d in config.MOTION_DATASETS
                           if d["id"] == "msl_dynamic"]
    unknown = [w for w in wanted if w not in by_id]
    if unknown:
        raise SystemExit(f"unknown corpus {unknown}; known: {sorted(by_id)}")

    config.ensure_dirs()
    config.JZ_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== Step 1b: J/Z motion data ==\n   into {config.JZ_CLIP_DIR}")
    results = []
    for did in wanted:
        spec = by_id[did]
        print(f"\n-- {did} ({spec['size_gb']:.1f} GB, {spec['license']}) --")
        raw = config.JZ_DIR / did / "raw"
        entry = {"id": did, "license": spec["license"], "why": spec["why"],
                 "letters": spec["letters"], "status": "unknown",
                 "archives": [], "clips": 0, "path": None}
        try:
            if spec["kind"] == "zenodo":
                for name in spec["files"]:
                    archive = raw / name
                    fetch(url_for(spec, name), archive)
                    entry["archives"].append(name)
                    unpack(archive, config.JZ_CLIP_DIR, force=args.force)
                    if not args.keep_archives:
                        archive.unlink(missing_ok=True)
            else:
                name = Path(spec["url"]).name
                archive = raw / name
                fetch(url_for(spec, name), archive)
                entry["archives"].append(name)
                unpack(archive, config.JZ_CLIP_DIR, force=args.force)
                if not args.keep_archives:
                    archive.unlink(missing_ok=True)
            entry["status"] = "ok"
        except SystemExit as exc:
            entry["status"] = "failed"
            entry["note"] = str(exc)
            print(f"  [warn] {did}: {exc}")
        entry["clips"] = clip_count(config.JZ_CLIP_DIR)
        entry["path"] = config.JZ_CLIP_DIR.relative_to(config.ROOT).as_posix()
        results.append(entry)
        if entry["status"] == "ok":
            print(f"  [ok] {did}: {entry['clips']} clips under {entry['path']}")

    mg_path = MANIFEST
    previous: list = []
    if mg_path.is_file():
        try:
            previous = json.loads(mg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = []
    keep = [r for r in previous if r.get("id") not in {x["id"] for x in results}]
    mg_path.write_text(json.dumps(keep + results, indent=2), encoding="utf-8")
    print(f"\nwrote {mg_path.relative_to(config.ROOT)}")

    total = clip_count(config.JZ_CLIP_DIR)
    if not total:
        raise SystemExit("no clips arrived - see the warnings above")
    print(f"{total} clips are ready. Next:\n"
          "  python src/extract_jz_landmarks.py\n"
          "  python src/verify_jz_trigger.py --clips-dir cache/jz_landmarks")
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
