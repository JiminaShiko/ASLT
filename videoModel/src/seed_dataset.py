"""Build a compact, versioned replay/evaluation set from processed clips.

Usage: python -m src.seed_dataset --per-class-train 20 --per-class-val 5 --per-class-test 5
The seed manifest retains source IDs, split/signer metadata, checksums and preprocessing shape.
Only processed training clips are replayed; validation/test anchors remain held out.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.config import FRAME_HEIGHT, FRAME_WIDTH, NUM_FRAMES, PROCESSED_DIR, safe_filename

SEED_DIR = PROCESSED_DIR / "seed"
MANIFEST_FIELDS = ["dataset", "video_id", "gloss", "split", "signer_id", "source_path",
                   "processed_path", "frames", "height", "width", "channels", "dtype", "sha256"]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def select_diverse(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Deterministic round-robin across signers, then stable hash ordering."""
    by_signer: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_signer[row.get("signer_id", "") or "unknown"].append(row)
    for signer, items in by_signer.items():
        items.sort(key=lambda row: hashlib.sha256(
            f"{row['dataset']}:{row['video_id']}".encode()).hexdigest())
    selected = []
    signers = sorted(by_signer)
    while signers and len(selected) < limit:
        remaining = []
        for signer in signers:
            if by_signer[signer] and len(selected) < limit:
                selected.append(by_signer[signer].pop(0))
            if by_signer[signer]:
                remaining.append(signer)
        signers = remaining
    return selected


def build_seed(manifest: Path, per_class: dict[str, int], seed_dir: Path = SEED_DIR) -> dict:
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # Carry forward seeds that have disappeared from the raw manifest. The processed
    # seed clips remain the durable replay source after source videos are removed.
    previous_manifest = seed_dir / "manifest.csv"
    if previous_manifest.is_file():
        with previous_manifest.open(encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    unique = {}
    for row in rows:
        key = (row.get("dataset", ""), row.get("video_id", ""), row.get("split", ""))
        if key not in unique or str(row.get("processed_path", "")).startswith("seed/"):
            unique[key] = row
    rows = list(unique.values())
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split") in {"train", "val", "test"}:
            groups[(row["gloss"], row["split"])].append(row)
    selected = []
    for (gloss, split), records in sorted(groups.items()):
        quota = per_class[split]
        selected.extend(select_diverse(records, quota))
    seed_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in selected:
        source = PROCESSED_DIR / row["processed_path"]
        if not source.is_file():
            continue
        relative = Path("seed") / row["split"] / safe_filename(row["gloss"]) / (
            safe_filename(f"{row['dataset']}_{row['video_id']}") + ".npz")
        destination = PROCESSED_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        output_rows.append({**row, "processed_path": relative.as_posix(), "sha256": digest(destination)})
    with (seed_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    retained = {row["processed_path"] for row in output_rows}
    for stale in (PROCESSED_DIR / "seed").glob("**/*.npz"):
        if stale.relative_to(PROCESSED_DIR).as_posix() not in retained:
            stale.unlink()
    size = sum((PROCESSED_DIR / row["processed_path"]).stat().st_size for row in output_rows)
    report = {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "preprocessing": {"frames": NUM_FRAMES, "height": FRAME_HEIGHT, "width": FRAME_WIDTH,
                                "channels": 3, "dtype": "uint8", "container": "npz"},
              "quotas_per_class": per_class,
              "records": len(output_rows), "bytes": size,
              "classes": sorted({row["gloss"] for row in output_rows}),
              "selection": "deterministic signer-balanced round-robin"}
    (seed_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DIR / "manifest.csv")
    parser.add_argument("--output", type=Path, default=SEED_DIR)
    parser.add_argument("--per-class-train", type=int, default=20)
    parser.add_argument("--per-class-val", type=int, default=5)
    parser.add_argument("--per-class-test", type=int, default=5)
    args = parser.parse_args()
    if min(args.per_class_train, args.per_class_val, args.per_class_test) < 0:
        parser.error("seed quotas must be nonnegative")
    report = build_seed(args.manifest, {"train": args.per_class_train, "val": args.per_class_val,
                                        "test": args.per_class_test}, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
