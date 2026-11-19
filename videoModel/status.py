"""
What is done, what is missing, and the one command that comes next.

    cd "D:\\sign language project\\ASTL project\\ASLT\\videoModel"
    python status.py                 human sections
    python status.py --json          the same numbers as one machine-readable blob
    python status.py --quiet         only the NEXT line (run_all.py uses this)

Reads only small files: the three split CSVs, the manifest, meta.json, the
checkpoint header and reports/. It never opens a video and never writes
anything, so it is safe to run in the middle of a training job.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# keep the third-party startup chatter out of a status report
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("KMP_WARNINGS", "0")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, extract, features, vocab  # noqa: E402


def _n(x) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def section_env() -> dict:
    out = dict(config.env_summary())
    out["cpus"] = config.cpu_count()
    out["free_gb_on_this_folder"] = round(config.free_bytes(config.ROOT) / 1e9, 1)
    out["dataset_read_only"] = str(config.DATASET_DIR)
    return out


def section_dataset() -> dict:
    if not config.dataset_available():
        return {"present": False, "path": str(config.DATASET_DIR),
                "hint": "put the ASL Citizen download at "
                        f"{config.DATASET_DIR} (see README)"}
    tables = {s: vocab.split_table(s) for s in ("train", "val", "test")}
    all_rows = pd.concat(tables.values(), ignore_index=True)
    vids = (list(config.VIDEOS_DIR.glob("*.mp4"))
            if config.VIDEOS_DIR.is_dir() else [])
    st = vocab.stats()
    return {"present": True,
            "path": str(config.DATASET_DIR),
            "rows_per_split": {s: int(len(t)) for s, t in tables.items()},
            "videos_per_split": {s: int(t["video"].nunique())
                                 for s, t in tables.items()},
            "signers_per_split": {s: int(t["participant"].nunique())
                                  for s, t in tables.items()},
            "videos_on_disk": len(vids),
            "glosses_in_splits": int(all_rows["gloss"].nunique()),
            "licence_file": config.LICENSE_FILE.is_file(),
            **{k: v for k, v in st.items()
               if isinstance(v, (int, float, str)) and k != "words"}}


def section_vocab() -> dict:
    spec = vocab.load_vocab()
    try:
        rt = vocab.resolve_table()
    except Exception as exc:                       # noqa: BLE001
        return {"resolve_table_error": str(exc)[:160],
                "labels": spec.labels,
                "n_active": len(spec.active)}
    thin = []
    for w in spec.active:
        sub = rt[rt["word"] == w.word]
        tv = int(sub["train_videos"].sum()) if len(sub) else 0
        ts = int(sub["train_signers"].max()) if len(sub) else 0
        if tv < w.min_videos:
            thin.append({"word": w.word, "train_videos": tv,
                         "train_signers": ts, "min_videos": w.min_videos})
    # >1 gloss row for a word means merged ASL-LEX variants (EAT1/EAT2 ...)
    gloss_variants = {str(k): int(v) for k, v in rt["word"].value_counts().items()
                      if k in spec.labels}
    return {"enabled_batches": list(spec.enabled_batches),
            "active_words": [w.word for w in spec.active],
            "n_active": len(spec.active),
            "labels": spec.labels,
            "gloss_variants_merged": gloss_variants,
            "skipped": [{"word": w.word, "batch": w.batch, "status": w.status,
                         "why": w.note or "no gloss, or batch not enabled"}
                        for w in spec.skipped],
            "thin_words_below_min_videos": thin,
            "rest_cap": spec.rest, "other_cap": spec.other}


def needed_videos() -> tuple:
    """Rows src/extract.py must cover, or the reason it cannot be answered."""
    try:
        return vocab.videos_to_process(), ""
    except Exception as exc:                       # noqa: BLE001
        return None, str(exc)[:160]


def section_cache() -> dict:
    man = extract.read_manifest()
    need, err = needed_videos()
    out = {"manifest_rows": _n(len(man)),
           "cached_ok": 0, "cached_failed": 0,
           "needed": _n(len(need)) if need is not None else None,
           "pending": None,
           "needed_error": err or None,
           "landmark_version": config.LANDMARK_VERSION,
           "feature_version": config.FEATURE_VERSION,
           "feature_dim": features.FEATURE_DIM,
           "num_frames": config.NUM_FRAMES,
           "per_split": {}, "hand_rate_mean": None, "gb_on_disk": 0.0}
    if man.empty:
        out["pending"] = out["needed"]
        return out
    ok = man[man["status"].astype(str) == "ok"]
    out["cached_ok"] = _n(len(ok))
    out["cached_failed"] = _n(len(man) - len(ok))
    if need is not None:
        done = set(ok["video"].astype(str))
        out["pending"] = int(len(set(need["video"].astype(str)) - done))
        for s in ("train", "val", "test"):
            sub = ok[ok["split"].astype(str) == s]
            out["per_split"][s] = {
                "cached": _n(len(sub)),
                "needed": _n(len(need[need["split"].astype(str) == s]))}
    if len(ok):
        out["hand_rate_mean"] = round(float(np.nanmean(
            ok["hand_rate"].astype(float))), 3)
        # the signing-window rate is the one that answers "can MediaPipe see
        # the sign at all"; the clip-wide one includes the idle head and tail
        if "hand_rate_seg" in ok.columns:
            seg = pd.to_numeric(ok["hand_rate_seg"], errors="coerce")
            out["hand_rate_seg_mean"] = (None if seg.notna().sum() == 0
                                         else round(float(seg.mean()), 3))
            out["hand_rate_seg_below_50"] = _n(int((seg < .5).sum()))
        out["rest_frames_total"] = _n(ok["rest_frames"].sum())
        out["gb_on_disk"] = round(sum(
            f.stat().st_size for f in config.LANDMARK_DIR.glob("*.npy")) / 1e9, 2)
    return out


def section_model() -> dict:
    out = {"release_weights": config.FOR_RELEASE.is_file(),
           "meta_json": config.META_JSON.is_file(),
           "onnx": config.ONNX_PATH.is_file(),
           "holistic_task": config.HOLISTIC_TASK.is_file(),
           "checkpoint": None, "smoke_leftover": (
               (config.CKPT_DIR / "smoke").is_dir())}
    if config.META_JSON.is_file():
        try:
            m = json.loads(config.META_JSON.read_text(encoding="utf-8"))
            out["trained"] = {k: m.get(k) for k in
                              ("epochs_ran", "best_val_top1", "labels",
                               "feature_dim", "feature_version",
                               "feature_blocks", "seed",
                               "trained_at", "num_classes")}
        except Exception as exc:                       # noqa: BLE001
            out["meta_json_error"] = str(exc)
    if config.CKPT_DIR.is_dir():
        for f in sorted(config.CKPT_DIR.glob("*.pt")):
            try:
                obj = torch_load_header(f)
                out["checkpoint"] = {"file": f.name,
                                     "epoch": obj.get("epoch"),
                                     "best_val_top1": obj.get("best_val_top1"),
                                     "seed": obj.get("seed"),
                                     "fingerprint": str(obj.get("fingerprint"))[:12]}
                break
            except Exception as exc:                   # noqa: BLE001
                out["checkpoint"] = {"file": f.name, "unreadable": str(exc)[:80]}
    return out


def torch_load_header(path: Path) -> dict:
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj if isinstance(obj, dict) else {}


def section_reports() -> dict:
    files = sorted(p.name for p in config.REPORTS_DIR.glob("*")) if (
        config.REPORTS_DIR.is_dir()) else []
    out = {"files": files, "words_summary": None}
    js = config.REPORTS_DIR / "words_summary.json"
    if js.is_file():
        try:
            s = json.loads(js.read_text(encoding="utf-8"))
            ev = s.get("eval") or {}          # train.py nests the splits here
            out["words_summary"] = {
                "test_top1": (ev.get("test") or {}).get("top1_all"),
                "test_top5": (ev.get("test") or {}).get("top5_all"),
                "val_top1": (ev.get("val") or {}).get("top1_all"),
                "macro_test_words": (ev.get("test") or {}).get("macro_top1_words"),
                "other_as_word_rate": ((ev.get("test") or {}).get("background")
                                       or {}).get("other_as_word_rate"),
                "epochs_ran": s.get("epochs_ran")}
        except Exception as exc:                       # noqa: BLE001
            out["words_summary_error"] = str(exc)
    return out


def next_step(status: dict) -> tuple[str, str]:
    """(command, why) - the one thing to run next. Empty command = done."""
    c, m, d = status["cache"], status["model"], status["dataset"]
    extract_cmd = "python -m src.extract --splits train,val,test --workers 4"
    if not d.get("present"):
        return ("", f"the dataset is not at {config.DATASET_DIR} - put it there "
                    "first (README, step 1)")
    if c.get("needed_error"):
        return ("", f"the video list cannot be built: {c['needed_error']}")
    if not c["cached_ok"]:
        return (extract_cmd, "nothing has been extracted yet")
    if c["pending"]:
        return (extract_cmd, f"{c['pending']} of {c['needed']} videos are not "
                             "extracted - it is resumable, run it as often "
                             "as you like")
    if not m["release_weights"]:
        return ("python -m src.train --epochs 60",
                "the cache is complete and there are no trained weights yet")
    # The weights record what feature vector built them. If the code has moved
    # on since, the live demo refuses to read them (it says "uncertain" with a
    # retrain reason), and quoting the old report as the current result would
    # be wrong too - so the next command is the retrain, not the report.
    w = m.get("trained") or {}
    wdim, wver = w.get("feature_dim"), w.get("feature_version")
    if wdim is not None and (wdim != features.FEATURE_DIM
                             or wver != features.FEATURE_VERSION):
        return ("python -m src.train --seeds 3",
                f"the weights were built by feature v{wver} at {wdim} numbers, "
                f"this code builds v{features.FEATURE_VERSION} at "
                f"{features.FEATURE_DIM} - retrain (the cache is reused, no "
                "re-extraction), then re-export the ONNX graph")
    if not m["onnx"]:
        return ("python -m src.export_onnx",
                "weights exist but the ONNX graph does not")
    if "words_summary.json" not in status["reports"]["files"]:
        return ("python -m src.train",
                "no reports/words_summary.json: training never finished")
    return ("python check_results.py",
            "everything is in place - this prints the summary and writes "
            "REPORT_WORDS.md")


def next_command(status: dict) -> str:
    cmd, why = next_step(status)
    return f"{cmd}  # {why}" if cmd else f"# nothing to run: {why}"


def collect() -> dict:
    return {"env": section_env(), "dataset": section_dataset(),
            "vocab": section_vocab(), "cache": section_cache(),
            "model": section_model(), "reports": section_reports()}


def print_human(status: dict) -> None:
    def show(title, obj):
        print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))
        for k, v in obj.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2}")
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                print(f"  {k}:")
                for row in v[:24]:
                    print(f"    {row}")
                if len(v) > 24:
                    print(f"    ... {len(v) - 24} more")
            elif isinstance(v, list) and len(v) > 8:
                print(f"  {k}: {len(v)} items: {v[:8]} ...")
            else:
                print(f"  {k}: {v}")

    show("environment", status["env"])
    show("dataset (read only)", status["dataset"])
    show("vocabulary", status["vocab"])
    show("landmark cache", status["cache"])
    show("model", status["model"])
    show("reports", status["reports"])
    print("\n=== NEXT " + "=" * 54)
    print(status["next"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Project status (read only).")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true",
                    help="only the next command, nothing else")
    ap.add_argument("--next", action="store_true",
                    help="the next command and the reason, two lines")
    a = ap.parse_args(argv)
    st = collect()
    cmd, why = next_step(st)
    st["next"] = next_command(st)
    if a.quiet:
        print(cmd or "# done")
        return 0
    if a.next:
        print(cmd or "(nothing to run)")
        print(why)
        return 0
    if a.json:
        print(json.dumps(st, indent=1, default=str))
        return 0
    print_human(st)
    print(f"\n{'-' * 68}\nNothing here was written to disk. The dataset is read "
          f"by path only:\n  {config.DATASET_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
