"""Step 1 - acquire the datasets.

Everything here is idempotent and offline-friendly:

* if a dataset is already in the kagglehub cache we use it and never re-download;
* if it is missing we try to download it with kagglehub;
* if we have no Kaggle credentials we print a short how-to instead of a stack
  trace, skip that dataset, and continue with whatever is available.

Run:  python src/download_data.py [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MANIFEST = config.DATA_DIR / "datasets.json"


# ---------------------------------------------------------------------------
# Kaggle cache handling
# ---------------------------------------------------------------------------
def kaggle_cache_root() -> Path:
    try:
        from kagglehub.cache import get_cache_folder

        return Path(get_cache_folder()) / "datasets"
    except Exception:
        return config.KAGGLEHUB_CACHE_DIR / "datasets"


def find_cached_version(ref: str) -> Path | None:
    """Return the newest *complete* cached copy of `owner/name`, if any."""
    owner, name = ref.split("/")
    base = kaggle_cache_root() / owner / name
    if not base.is_dir():
        return None
    marker_parent = base  # `<owner>/<name>/<version>.complete`
    versions_dir = base / "versions"
    if not versions_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for entry in versions_dir.iterdir():
        if not entry.is_dir() or not entry.name.isdigit():
            continue
        ver = int(entry.name)
        if not (marker_parent / f"{ver}.complete").exists():
            # kagglehub only writes the marker once extraction succeeded
            continue
        if best is None or ver > best[0]:
            best = (ver, entry)
    return best[1] if best else None


def has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    for p in (
        Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json",
        Path.home() / ".kaggle" / "kaggle.json",
    ):
        if p.is_file():
            return True
    return False


CREDENTIAL_HELP = """
Kaggle credentials are not configured, so datasets that are not already in the
local kagglehub cache cannot be downloaded.

  1. Open https://www.kaggle.com/settings -> "Create New Token" (downloads kaggle.json)
  2. Put the file at  C:\\Users\\<you>\\.kaggle\\kaggle.json
     (or set the KAGGLE_USERNAME and KAGGLE_KEY environment variables instead)

Already-cached datasets work without credentials, so this step still completes.
""".strip()


# ---------------------------------------------------------------------------
# Image discovery + label mapping
# ---------------------------------------------------------------------------
def normalize_label(folder: str) -> str | None:
    """Map any dataset's folder name onto a single uppercase letter label."""
    key = folder.strip().lower().replace("-", "_")
    if key in config.IGNORED_LABELS:
        return None
    if len(key) == 1 and key.isalpha():
        up = key.upper()
        return up if up in config.ALL_LETTERS else None
    # e.g. "letter_a", "a", "A1"
    if key.startswith("letter_") and len(key) == 8:
        return key[-1].upper()
    return None


def iter_letter_images(root: Path, spec: dict, max_depth: int = 2):
    """Yield (image_path, letter, session_id) for a dataset root.

    Two layouts are supported, selected by spec["letter_from"]:

      "top_folder"  root/<LETTER>/[<subgroup>/]image   (grassknoted, danrasband)
      "sub_folder"  root/<GROUP>/<LETTER>/image        (massey: the *inner*
                    folder is the letter and the outer one is the capture
                    group / signer - see the note in config.DATASETS)

    session_id is the grouping used for honest blocked splits; it must be
    unique inside this dataset root, so it is namespaced by root folder name.
    """
    if spec.get("letter_from") == "sub_folder":
        for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for letter_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
                letter = normalize_label(letter_dir.name)
                if letter is None:
                    continue
                sid = f"{root.name}/{group_dir.name}"
                for f in sorted(letter_dir.iterdir()):
                    if _is_image(f, spec):
                        yield f, letter, sid
        return

    for letter_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        letter = normalize_label(letter_dir.name)
        if letter is None:
            continue
        if spec["session_from"] == "signer_folder":
            for sub in sorted(p for p in letter_dir.iterdir() if p.is_dir()):
                for f in sorted(sub.iterdir()):
                    if _is_image(f, spec):
                        yield f, letter, f"{letter_dir.name}/{sub.name}"
            for f in sorted(letter_dir.iterdir()):
                if _is_image(f, spec):
                    yield f, letter, letter_dir.name
        else:
            files = [f for f in sorted(letter_dir.iterdir()) if _is_image(f, spec)]
            for f in files:
                yield f, letter, letter_dir.name
            # one extra level (grassknoted/asl_alphabet_test nests again)
            for sub in sorted(p for p in letter_dir.iterdir() if p.is_dir()):
                for f in sorted(sub.iterdir()):
                    if _is_image(f, spec):
                        yield f, letter, f"{letter_dir.name}/{sub.name}"


def _is_image(path: Path, spec: dict) -> bool:
    if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
        return False
    prefix = spec.get("image_prefix")
    return path.name.startswith(prefix) if prefix else True


def locate_roots(ref_path: Path, spec: dict) -> list[Path]:
    """Find the real folder holding the letter directories (handles nesting)."""
    found: list[Path] = []
    for rel in spec["roots"]:
        cand = ref_path / rel if rel else ref_path
        if not cand.is_dir():
            continue
        # grassknoted ships asl_alphabet_train/asl_alphabet_train - go one deeper
        inner = cand / cand.name
        if inner.is_dir() and any(normalize_label(p.name) for p in inner.iterdir() if p.is_dir()):
            cand = inner
        found.append(cand)
    return found


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
def acquire(spec: dict, force: bool) -> dict:
    ref = spec["ref"]
    result = {"id": spec["id"], "ref": ref, "role": spec["role"],
              "license": spec["license"], "why": spec["why"],
              "status": "unknown", "path": None, "n_images": 0,
              "n_bytes": 0, "letters": {}, "note": ""}

    cached = None if force else find_cached_version(ref)
    if cached is not None:
        result.update(status="cached", path=str(cached),
                      note="already in kagglehub cache, no download performed")
        print(f"  [cached] {ref}")
    else:
        if not has_kaggle_credentials():
            result.update(status="skipped_no_credentials",
                          note="no local Kaggle credentials and no cached copy")
            print(f"  [skip]   {ref} - not cached and no Kaggle credentials")
            return result
        print(f"  [download] {ref} ...")
        try:
            import kagglehub

            t0 = time.time()
            path = Path(kagglehub.dataset_download(ref, force_download=force))
            result.update(status="downloaded", path=str(path),
                          note=f"took {time.time() - t0:.0f}s")
        except Exception as exc:  # dead link, gated dataset, network, ...
            result.update(status="skipped_download_failed",
                          note=f"{type(exc).__name__}: {str(exc)[:180]}")
            print(f"  [skip]   {ref} -> {type(exc).__name__}: {str(exc)[:160]}")
            return result

    roots = locate_roots(Path(result["path"]), spec)
    if not roots:
        result.update(status="skipped_bad_layout",
                      note=f"no letter folders found under {spec['roots']}")
        print(f"  [skip]   {ref} - unexpected folder layout")
        return result

    letters: dict[str, int] = {}
    n_bytes = 0
    for root in roots:
        for img, letter, _session in iter_letter_images(root, spec):
            letters[letter] = letters.get(letter, 0) + 1
            try:
                n_bytes += img.stat().st_size
            except OSError:
                pass
    result.update(n_images=sum(letters.values()), n_bytes=n_bytes, letters=letters,
                  roots=[str(r) for r in roots])
    return result


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


# ---------------------------------------------------------------------------
# MediaPipe model asset
# ---------------------------------------------------------------------------
def ensure_hand_model(quiet: bool = False) -> Path:
    config.ensure_dirs()
    if config.HAND_MODEL_PATH.is_file() and config.HAND_MODEL_PATH.stat().st_size > 1_000_000:
        if not quiet:
            print(f"  [ok] hand model already present ({human(config.HAND_MODEL_PATH.stat().st_size)})")
        return config.HAND_MODEL_PATH
    for legacy in config.LEGACY_HAND_MODEL_CANDIDATES:
        if legacy.is_file():
            shutil.copy2(legacy, config.HAND_MODEL_PATH)
            if not quiet:
                print(f"  [ok] reused hand model from {legacy.name} (read-only copy)")
            return config.HAND_MODEL_PATH
    for url in config.HAND_MODEL_URLS:
        try:
            if not quiet:
                print(f"  [download] {url}")
            urllib.request.urlretrieve(url, config.HAND_MODEL_PATH)  # noqa: S310
            if config.HAND_MODEL_PATH.stat().st_size > 1_000_000:
                return config.HAND_MODEL_PATH
        except Exception as exc:
            if not quiet:
                print(f"  [warn] {type(exc).__name__}: {exc}")
    raise SystemExit(
        "Could not obtain hand_landmarker.task. Download it manually from\n"
        f"  {config.HAND_MODEL_URLS[0]}\n"
        f"and save it to {config.HAND_MODEL_PATH}"
    )


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Download/locate the ASL datasets.")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if a cached copy exists")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    print("== Step 1: datasets ==")
    print(f"cache root: {kaggle_cache_root()}")
    print(f"Kaggle credentials present: {has_kaggle_credentials()}")
    if not has_kaggle_credentials():
        print(CREDENTIAL_HELP)

    results = []
    creds_missing_shown = False
    for spec in config.DATASETS:
        res = acquire(spec, args.force)
        results.append(res)
        if res["status"] == "skipped_no_credentials" and not creds_missing_shown:
            creds_missing_shown = True

    usable = [r for r in results if r["n_images"] > 0]
    print("\n-- summary --")
    for r in results:
        cov = len(r["letters"])
        rng = ""
        if r["letters"]:
            keys = sorted(r["letters"])
            rng = f" (covers {''.join(keys)})"
        print(f"  {r['id']:<12} {r['status']:<24} {r['n_images']:>7} imgs  "
              f"{cov:>2} letters{rng}  {human(r['n_bytes'])}")
        print(f"      license: {r['license']}")
        if r["note"]:
            print(f"      note:    {r['note']}")

    if not usable:
        raise SystemExit("No dataset is available locally and none could be "
                         "downloaded. Fix Kaggle credentials (see above) and re-run.")

    missing = [r["id"] for r in results if r["n_images"] == 0]
    if missing:
        print(f"\n  WARNING: continuing without {missing}")

    # Store project-relative paths whenever possible.  The manifest is tracked,
    # while the data itself is local and ignored; absolute paths would make a
    # pull on another machine point back to this user's drive and username.
    for result in results:
        for key in ("path", "roots"):
            value = result.get(key)
            if not value:
                continue
            paths = value if isinstance(value, list) else [value]
            stored = []
            for item in paths:
                path = Path(item).resolve()
                try:
                    stored.append(path.relative_to(config.ROOT).as_posix())
                except ValueError:
                    stored.append(str(path))
            result[key] = stored if isinstance(value, list) else stored[0]

    MANIFEST.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {MANIFEST.relative_to(config.ROOT)}")

    print("\n-- MediaPipe hand model --")
    print(f"  {ensure_hand_model(quiet=args.quiet)}")

    letters_present = set().union(*[set(r["letters"]) for r in usable])
    gaps = sorted(set(config.LABELS) - letters_present)
    if gaps:
        print(f"\n  letters with NO data anywhere: {''.join(gaps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
