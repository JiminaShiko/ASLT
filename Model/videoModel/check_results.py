"""
Stage 8 - read the compact results and say what to fix.

    cd "D:\\sign language project\\ASTL project\\ASLT\\videoModel"
    python check_results.py              pasteable summary on screen
    python check_results.py --write      also write REPORT_WORDS.md
    python check_results.py --json       machine-readable

It reads ONLY the small files training leaves behind:
reports/words_summary.json, models/meta.json, reports/per_word.csv,
reports/confused_pairs.csv, reports/seed_runs.csv, cache/manifest.csv and
cache/extract_errors.csv. It never loads a video, a landmark or the weights,
it never re-runs training, and it does not import torch (the sweep table
comes from the torch-free src/seedreport.py).

The verdict lines at the bottom are the point: they name the single next
action, based on what the numbers actually show.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.seedreport import compare_table, latest_fair_tags, fair_table, candidate_name  # noqa: E402


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001
        return {"_error": str(exc)}


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:                              # noqa: BLE001
        return None


def _pct(x) -> str:
    try:
        return f"{100.0 * float(x):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(x) -> str:
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return "n/a"


def gather() -> dict:
    out: dict = {"missing": [], "files": {}, "candidate": candidate_name()}
    summary = _read_json(config.REPORTS_DIR / "words_summary.json")
    meta = _read_json(config.META_JSON)
    if summary is None:
        out["missing"].append("reports/words_summary.json - training has not "
                              "finished (or wrote to the smoke folder)")
    if meta is None:
        out["missing"].append(f"{config.META_JSON} - no completed model in this output scope")
    out["summary"], out["meta"] = summary, meta
    out["per_word"] = _read_csv(config.REPORTS_DIR / "per_word.csv")
    out["pairs"] = _read_csv(config.REPORTS_DIR / "confused_pairs.csv")
    out["per_word_val"] = _read_csv(config.REPORTS_DIR / "per_word_val.csv")
    out["pairs_val"] = _read_csv(config.REPORTS_DIR / "confused_pairs_val.csv")
    # Never display leftover split CSVs from an older promotion as this run.
    for split, suffix in (("test", ""), ("val", "_val")):
        if split not in ((summary or {}).get("eval") or {}):
            out[f"per_word{suffix}"] = out[f"pairs{suffix}"] = None
    out["manifest"] = _read_csv(config.MANIFEST_CSV)
    out["errors"] = _read_csv(config.ERRORS_CSV)
    for key, path in (("words_summary", config.REPORTS_DIR / "words_summary.json"),
                      ("meta", config.META_JSON),
                      ("per_word", config.REPORTS_DIR / "per_word.csv"),
                      ("confused_pairs", config.REPORTS_DIR / "confused_pairs.csv"),
                      ("manifest", config.MANIFEST_CSV)):
        out["files"][key] = str(path) if path.is_file() else None
    return out


def headline(d: dict) -> dict:
    """The handful of numbers that decide whether this is usable."""
    s, m = d.get("summary") or {}, d.get("meta") or {}
    ev = (s.get("eval") or {})
    test, val = ev.get("test") or {}, ev.get("val") or {}
    train_ev = ev.get("train") or {}
    ps = ev.get("ps_test") or {}
    b = test.get("background") or {}
    data = s.get("data") or {}
    return {
        "candidate": d.get("candidate") or s.get("candidate"),
        "label_set_id": s.get("label_set_id"),
        "evaluation_id": s.get("evaluation_id"),
        "commit_policy": s.get("commit_policy"),
        "groups": {split: stats.get("groups", {}) for split, stats in ev.items()},
        "commit_metrics": {split: stats.get("gate", {}) for split, stats in ev.items()},
        "smoke": bool(s.get("smoke")),
        "epochs_ran": s.get("epochs_ran"),
        "best_val_top1": s.get("best_val_top1"),
        "test_top1": test.get("top1_all"),
        "test_top5": test.get("top5_all"),
        "test_macro_words": test.get("macro_top1_words"),
        "test_words_only_top1": test.get("top1_words_only"),
        "val_top1": val.get("top1_all"),
        # the memorisation pair: the released weights on their own training
        # split vs val, words-only on both (older runs have no train eval)
        "train_words_only_top1": (train_ev.get("top1_words_only")
                                  or m.get("train_words_top1")),
        "val_words_only_top1": val.get("top1_words_only"),
        "n_test_samples": test.get("n_samples"),
        "n_test_videos": test.get("n_videos"),
        "n_test_signers": test.get("n_signers"),
        "n_train_samples": m.get("train_samples"),
        "n_train_signers": m.get("train_signers"),
        # the PopSign held-out group (absent when no external data was used)
        "ps_test_top1": ps.get("top1_all"),
        "ps_test_words": ps.get("top1_words_only"),
        "ps_test_n": ps.get("n_samples"),
        "ps_test_videos": ps.get("n_videos"),
        "ps_test_signers": ps.get("n_signers"),
        "n_train_ps_videos": (data.get("train") or {}).get("n_ps_videos"),
        "split_check": s.get("split_check"),
        "rest_as_word_rate": b.get("rest_as_word_rate"),
        "other_as_word_rate": b.get("other_as_word_rate"),
        "word_as_extra_rate": b.get("word_as_extra_rate"),
        "feature_dim": s.get("feature_dim"),
        "feature_blocks": s.get("feature_blocks"),
        "labels": s.get("labels") or m.get("labels"),
        "params": m.get("params"),
        "seed": s.get("seed"),
        "seconds": s.get("seconds"),
        "sec_per_epoch": s.get("sec_per_epoch"),
    }


def weakest(per: pd.DataFrame | None, n: int = 6) -> list[str]:
    if per is None or "top1" not in per.columns:
        return []
    words = per[~per["label"].isin([config.REST_LABEL, config.OTHER_LABEL])]
    w = words[words["n_samples"] >= 5].dropna(subset=["top1"])
    w = w.nsmallest(n, "top1")
    return [f"{r['label']} {_pct(r['top1'])} ({int(r['n_samples'])} samples, "
            f"{int(r['signers'])} signers)" for _, r in w.iterrows()]


def top_pairs(pairs: pd.DataFrame | None, n: int = 8) -> list[str]:
    if pairs is None or pairs.empty or "pair" not in pairs.columns:
        return []
    return [f"{r['pair']}: {int(r['swapped'])} swapped of "
            f"{int(r['n_a']) + int(r['n_b'])} ({_pct(r['swap_rate'])})"
            for _, r in pairs.head(n).iterrows()]


def hand_rate_per_word(man: pd.DataFrame | None) -> list[tuple[str, float, int, float]]:
    """Mean hand-detection rate per word: signing window first, whole clip second.

    hand_rate_seg is the number that matters. The clip-wide hand_rate counts
    the pre- and post-roll of a sentence clip, where the hands are legitimately
    in the speaker's lap, so on its own it reads ~0.44 and looks like a
    detection ceiling that the measured per-word accuracy contradicts. Old
    manifests without the segment column fall back to the clip-wide number,
    and the table says so.
    """
    if man is None or man.empty or "hand_rate" not in man.columns:
        return []
    ok = man[man["status"].astype(str) == "ok"]
    if ok.empty or "word" not in ok.columns:
        return []
    seg = "hand_rate_seg" in ok.columns
    full = pd.to_numeric(ok["hand_rate"], errors="coerce")
    segv = (pd.to_numeric(ok["hand_rate_seg"], errors="coerce") if seg
            else full)
    g = pd.DataFrame({"word": ok["word"].astype(str), "seg": segv,
                      "full": full}).groupby("word").agg(
        seg=("seg", "mean"), full=("full", "mean"), n=("seg", "size"))
    rows = [(str(w), float(r["seg"]), int(r["n"]), float(r["full"]))
            for w, r in g.iterrows() if np.isfinite(r["seg"])]
    return sorted(rows, key=lambda x: x[1])


def rates_are_cached(man: pd.DataFrame | None) -> bool:
    """True when the manifest carries the signing-window rate."""
    return (man is not None and "hand_rate_seg" in man.columns
            and pd.to_numeric(man["hand_rate_seg"], errors="coerce").notna().any())


def gate_point(d: dict) -> str:
    """Validation-only policy evidence. Never search thresholds on test rows."""
    val = ((d.get("summary") or {}).get("eval") or {}).get("val") or {}
    gate = val.get("gate") or {}
    policy = gate.get("policy") or {}
    if not policy:
        return ("No saved validation commit-policy metrics. No threshold is recommended; "
                "historical confidence-only CSVs do not establish hedge-aware commits.")
    return (f"VAL only, fixed p1 >= {_num(policy.get('confidence_min'))}, "
            f"margin >= {_num(policy.get('margin_min'))}, with saved pair/hedge rules: "
            f"word commit share {_pct(gate.get('typed_share'))}, "
            f"correct-commit recall {_pct(gate.get('correct_commit_recall'))}, "
            f"commit precision {_pct(gate.get('typed_precision'))}. "
            "Any future policy selection must use validation only, then be frozen "
            "before final test scoring. These are offline clips, not live accuracy.")


def verdict(d: dict, h: dict) -> list[str]:
    """What to do next, from the numbers on the page - not from optimism."""
    v: list[str] = []
    if d["missing"]:
        v += [f"BLOCKED: {m}" for m in d["missing"]]
        scope = f" --candidate {h['candidate']}" if h.get("candidate") else ""
        v.append("After extracting the active vocabulary's train,val cache, run: "
                 f"python -m src.train{scope} --seeds 3 --eval-splits val,test,ps_test")
        return v
    if h["smoke"]:
        return ["SMOKE ONLY: scratch numbers are not evaluation results."]
    if h["val_top1"] is None:
        return ["No genuine validation results: no configuration or gate recommendation."]
    v.append("Recommendations below use VAL only. Test/ps_test remain final, "
             "fixed-policy measurements, not configuration-selection inputs.")
    v.append(gate_point(d))
    val_ev = ((d.get("summary") or {}).get("eval") or {}).get("val") or {}
    t = float(h["val_top1"])
    macro = val_ev.get("macro_top1_words")
    if macro is not None and t - float(macro) > 0.15:
        v.append("VAL pooled accuracy exceeds macro word accuracy: inspect the "
                 "validation per-word counts before choosing further experiments.")
    per = d.get("per_word_val")
    if per is not None:
        zero = per[(~per["label"].isin([config.REST_LABEL, config.OTHER_LABEL]))
                   & (per["n_samples"] >= 5) & (per["top1"] <= 0.05)]
        if len(zero):
            v.append("Never recognised at all: "
                     + ", ".join(zero["label"].astype(str)) +
                     " - check their gloss mapping and video counts before "
                     "touching the architecture.")
    if d.get("errors") is not None and len(d["errors"]):
        v.append(f"{len(d['errors'])} videos failed extraction "
                 "(cache/extract_errors.csv). Under ~1% is normal; read the "
                 "reasons before ignoring them.")
    man = d.get("manifest")
    val_man = man[man["split"].eq("val")] if man is not None and "split" in man else None
    hr = hand_rate_per_word(val_man)
    if hr:
        if not rates_are_cached(val_man):
            v.append("The manifest only carries the clip-wide hand rate, "
                     "which counts the idle head and tail of a sentence clip "
                     "and so reads low for everything. Fill the window one: "
                     "python -m src.extract --rates (about a minute, no "
                     "MediaPipe, no videos).")
        else:
            rate = {w: s for w, s, _n, _f in hr}
            full = {w: f for w, _s, _n, f in hr}
            v.append(f"Hand detection over the SIGNING WINDOW averages "
                     f"{np.mean(list(rate.values())):.2f} (clip-wide it is "
                     f"{np.mean(list(full.values())):.2f}): the low clip "
                     "number is the hands resting between signs, not a "
                     "detection failure, and it is not the ceiling.")
            per = d.get("per_word_val")
            bg = {config.REST_LABEL, config.OTHER_LABEL}
            acc = ({} if per is None or "top1" not in per.columns else
                   dict(zip(per["label"].astype(str), per["top1"])))
            not_det = sorted(w for w in rate
                             if w not in bg and w in acc and acc[w] is not None
                             and float(acc[w]) < 0.60 and rate[w] >= 0.80)
            if not_det:
                v.append("MediaPipe is NOT what limits "
                         + ", ".join(not_det[:8]) + ": the hand is visible in "
                         "over 80% of their signing window and the model still "
                         f"misses them ({', '.join(f'{w} {100 * float(acc[w]):.0f}%' for w in not_det[:8])}). "
                         "That is a features-and-data problem - the segment "
                         "read-out in features v3 and more signers - not "
                         "camera framing and not more epochs.")
            low = [w for w, s, _n, _f in hr if s < 0.75]
            if low:
                v.append("Genuinely hard to see in under 75% of the window: "
                         + ", ".join(low[:8]) + " - framing helps these, and "
                         "only these.")
    if t < 0.35:
        v.append(f"VAL {_pct(t)} across {len(h['labels'] or [])} classes remains low. "
                 "One validation-only experiment with the cache complete: "
                 "an ablation run (--blocks pose,hands) to see whether the "
                 "face and mask blocks are helping or just noise.")
    elif t < 0.60:
        s = d.get("summary") or {}
        hist = s.get("history") or []
        ran, want = int(s.get("epochs_ran") or 0), int(s.get("epochs_requested") or 0)
        best_ep = len(hist)
        if hist:
            vals = [h.get("val_top1") or 0.0 for h in hist]
            best_ep = int(max(range(len(vals)), key=lambda i: vals[i])) + 1
            last_tr = hist[-1].get("train_top1") or 0.0
            if last_tr - float(vals[-1]) > 0.30:
                v.append(f"Train {_pct(last_tr)} vs val {_pct(float(vals[-1]))}: "
                         "the model has memorised the training signers. More "
                         "epochs cannot fix that - more signers, or stronger "
                         "regularisation, can.")
        v.append(f"VAL {_pct(t)}: further validation is needed. "
                 + (f"Early stopping fired (best epoch {best_ep} of {ran} run, "
                    f"{want} asked), so the run had levelled off - longer "
                    "training is not the lever. The seed-to-seed spread is "
                    "already measured in reports/seed_runs.md; the remaining "
                    "lever is more signers (docs/EXTERNAL_DATA.md)."
                    if ran < want else
                    "it ran every epoch it was asked for and was still "
                    "improving at the end, so a longer run with --fresh is "
                    "worth trying; the seed-to-seed spread is in "
                    "reports/seed_runs.md and the data lever is "
                    "docs/EXTERNAL_DATA.md."))
    else:
        v.append(f"VAL {_pct(t)} is an offline result only. Check core20 retention "
                 "and new-word coverage separately; it does not establish live accuracy.")
    v.append("Nothing was re-run to produce this page. Every number came from "
             "the files training wrote.")
    return v


def render(d: dict) -> str:
    h = headline(d)
    L: list[str] = []
    A = L.append
    A(f"# Word recognizer - candidate {h['candidate']} results" if h.get("candidate")
      else "# Word recognizer - offline results")
    if h.get("candidate"):
        A("Candidate outputs only; not deployed. Shipped weights and published reports are unchanged.")
    A("Offline held-out clip evaluation, not measured live/webcam accuracy.")
    A("")
    A(f"Written by check_results.py on {config.timestamp()}. "
      "Read-only: it re-ran nothing.")
    A("")
    if h["smoke"]:
        A("> **SMOKE RUN - these numbers are not results.** They only prove "
          "the code path works end to end.")
        A("")
    A("## Headline")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| test top-1 (official signer-disjoint) | {_pct(h['test_top1'])} |")
    A(f"| test top-5 | {_pct(h['test_top5'])} |")
    A(f"| test macro over words | {_pct(h['test_macro_words'])} |")
    A(f"| test top-1 with REST/OTHER removed from the argmax | "
      f"{_pct(h['test_words_only_top1'])} |")
    A(f"| words-only top-1: train vs val (memorisation gap) | "
      f"{_pct(h['train_words_only_top1'])} vs "
      f"{_pct(h['val_words_only_top1'])} |")
    A(f"| best val top-1 | {_pct(h['best_val_top1'])} |")
    A(f"| test size | {_num(h['n_test_samples'])} samples / "
      f"{_num(h['n_test_videos'])} videos / {_num(h['n_test_signers'])} signers |")
    A(f"| train size | {_num(h['n_train_samples'])} samples / "
      f"{_num(h['n_train_signers'])} signers |")
    A(f"| epochs | {_num(h['epochs_ran'])} (best val {_pct(h['best_val_top1'])}) |")
    A(f"| features | {_num(h['feature_dim'])} dims, blocks "
      f"{h['feature_blocks']}, {config.NUM_FRAMES} frames |")
    A(f"| classes | {len(h['labels'] or [])} |")
    A(f"| label-set identity | {h['label_set_id'] or 'unknown (historical)'} |")
    A(f"| evaluation identity | {h['evaluation_id'] or 'unknown (historical)'} |")
    A(f"| parameters | {_num(h['params'])} |")
    A(f"| seed / seconds | {h['seed']} / {_num(h['seconds'])} "
      f"({_num(h['sec_per_epoch'])} s per epoch) |")
    A("")
    A("Every test number above is from signers the model never saw. A random "
      "split would read far higher and would not be reported here.")
    A("")
    if h["ps_test_top1"] is not None:
        A("## Held-out groups (never pooled)")
        A("")
        A("| group | videos | top-1 | words-only top-1 |")
        A("|---|---|---|---|")
        A(f"| ASL Citizen official test signers (offline) | "
          f"{_num(h['n_test_videos'])} | {_pct(h['test_top1'])} | "
          f"{_pct(h['test_words_only_top1'])} |")
        A(f"| PopSign val+test signers (second, independent style) | "
          f"{_num(h['ps_test_videos'])} | {_pct(h['ps_test_top1'])} | "
          f"{_pct(h['ps_test_words'])} |")
        A("")
        A(f"Training saw {_num(h['n_train_ps_videos'])} PopSign clips on top of "
          "the ASL Citizen train set. Each row is the signer-disjoint split of "
          "its own corpus; a number that pooled them would be meaningless and "
          "is not reported. Neither row is a measurement of live webcam accuracy.")
        A("")
    if h["split_check"] is not None:
        A("## Signer-disjointness check")
        A("")
        if h["split_check"]:
            for p in h["split_check"]:
                A(f"- {p}")
        else:
            A("- no problem found: no (source, signer) pair appears in train "
              "and in a held-out split")
        A("")
    A("## Retention and expansion (full-classifier argmax)")
    A("")
    A("Core20 is active batch 1; new_words is the active later batches. "
      "Expanded-total accuracy is not an improvement over an old smaller-vocabulary baseline.")
    source = (d.get("summary") or {}).get("weak6_source")
    A(f"Weak-six membership: {source or 'not recorded'}; fixed from VAL for every final split.")
    A("")
    A("| split | group | samples | words scored | top-1 | macro | correct-commit recall | labels |")
    A("|---|---|---|---|---|---|---|---|")
    for split, groups in h["groups"].items():
        for name, stats in groups.items():
            A(f"| {split} | {name} | {_num(stats.get('n_samples'))} | "
              f"{_num(stats.get('n_words_scored'))} | {_pct(stats.get('top1'))} | "
              f"{_pct(stats.get('macro_top1'))} | {_pct(stats.get('correct_commit_recall'))} | "
              f"{', '.join(stats.get('labels', []))} |")
    A("")
    A("## Fixed commit policy (offline clips)")
    A("")
    A("Policy is saved with this run; test is not re-thresholded by this report. "
      "Missing historical policy metrics are not reconstructed from rounded CSVs.")
    A(f"Saved policy: `{json.dumps(h['commit_policy'], sort_keys=True)}`")
    A("")
    A("| split | commits | precision | word commit share | correct-commit recall | hedge refusals | REST slip | OTHER slip |")
    A("|---|---|---|---|---|---|---|---|")
    for split, gate in h["commit_metrics"].items():
        if gate:
            A(f"| {split} | {_num(gate.get('n_committed'))} | {_pct(gate.get('typed_precision'))} | "
              f"{_pct(gate.get('typed_share'))} | {_pct(gate.get('correct_commit_recall'))} | "
              f"{_num(gate.get('n_hedge_refused'))} | {_pct(gate.get('rest_slip'))} | "
              f"{_pct(gate.get('other_slip'))} |")
    A("")
    A("## Background classes (raw argmax, before commit policy)")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| REST predicted as a word | {_pct(h['rest_as_word_rate'])} |")
    A(f"| OTHER predicted as a word | {_pct(h['other_as_word_rate'])} |")
    A(f"| a real word predicted as REST/OTHER | {_pct(h['word_as_extra_rate'])} |")
    A("")
    A("## Weakest words (test description only, not selection)")
    A("")
    wk = weakest(d.get("per_word"))
    A("\n".join(f"- {x}" for x in wk) if wk else "- nothing readable in "
      "reports/per_word.csv")
    A("")
    A("## Most confused pairs")
    A("")
    tp = top_pairs(d.get("pairs"))
    A("\n".join(f"- {x}" for x in tp) if tp else "- none recorded")
    A("")
    A("## Hand detection per word (worst window rate first)")
    A("")
    hr = hand_rate_per_word(d.get("manifest"))
    if hr:
        cached = rates_are_cached(d.get("manifest"))
        A("| word | signing window | whole clip | videos |")
        A("|---|---|---|---|")
        for w, s, n, f in hr[:24]:
            A(f"| {w} | {s:.3f}{'' if cached else ' (=clip)'} | {f:.3f} | {n} |")
        A("")
        A("The window column is the one to read: it is measured between "
          "sign_start and sign_end, where the sign actually happens. The clip "
          "column is what the extractor reports for the whole sentence clip, "
          "idle head and tail included. A word can only be missed because of "
          "detection if the window column is low - and per "
          "reports/predictions.csv it usually is not."
          if cached else
          "- this manifest has no hand_rate_seg column; every number shown is "
          "the clip-wide one. Run python -m src.extract --rates.")
    else:
        A("- no cached videos to measure")
    A("")
    A("## Extraction errors")
    A("")
    if d.get("errors") is not None and len(d["errors"]):
        A(f"- {len(d['errors'])} rows in cache/extract_errors.csv")
        for _, r in d["errors"].head(10).iterrows():
            A(f"  - {r.get('video')}: {str(r.get('error'))[:110]}")
    else:
        A("- none recorded")
    A("")
    A("## Seed sweep: baseline vs new")
    A("")
    A(compare_table())
    A("")
    fair = latest_fair_tags()
    if fair:
        A(fair_table(*fair))
        A("")
    A("Two precision numbers appear in the tables: \"typed precision, real "
      "words only\" scores the gate's commits on clips that really are "
      "vocabulary words, while \"typed precision, everything typed\" also "
      "counts the commits it makes over REST/OTHER clips - all of which are "
      "wrong. These are offline commit-policy measurements, not a promise of "
      "live segmentation or sentence accuracy.")
    A("")
    A("Each row is the mean over the seeds of one config, with its half-range. "
      "Configs are chosen on the val columns only; the test columns are "
      "reported, never used to decide. Comparisons require explicit matching "
      "label-set and evaluation identities; unknown historical identities do not qualify.")
    A("")
    A("## What to fix")
    A("")
    for x in verdict(d, h):
        A(f"- {x}")
    A("")
    A("## Files")
    A("")
    for k, v in d["files"].items():
        A(f"- {k}: {v or 'missing'}")
    A("")
    if h["n_train_ps_videos"]:
        A(f"Trained on ASL Citizen ({config.LICENSE_FILE.name}: non-commercial "
          f"research use) plus {_num(h['n_train_ps_videos'])} PopSign v2.1 "
          "clips (CC BY 4.0). No webcam frame was used for training or "
          "calibration, and the live demo writes nothing to disk.")
    else:
        A("Trained on ASL Citizen only, under the terms in "
          f"{config.LICENSE_FILE.name}: non-commercial research use. No webcam "
          "frame was used for training or calibration, and the live demo writes "
          "nothing to disk.")
    A("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Read the results (read only).")
    ap.add_argument("--candidate", help="read/write only the named candidate outputs")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true",
                    help="write REPORT_WORDS.md in the selected output scope")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    try:
        candidate_root = config.use_candidate(a.candidate) if a.candidate else None
    except ValueError as exc:
        ap.error(str(exc))
    out = Path(a.out) if a.out else (
        config.REPORTS_DIR / "REPORT_WORDS.md" if candidate_root is not None
        else ROOT / "REPORT_WORDS.md")
    if candidate_root is not None and not out.resolve().is_relative_to(candidate_root.resolve()):
        ap.error("candidate --out must stay inside its candidate directory")
    d = gather()
    h = headline(d)
    if a.json:
        print(json.dumps({"headline": h, "missing": d["missing"],
                          "weakest": weakest(d.get("per_word")),
                          "pairs": top_pairs(d.get("pairs")),
                          "hand_rate": hand_rate_per_word(d.get("manifest")),
                          "verdict": verdict(d, h)}, indent=1, default=str))
    else:
        print(render(d))
    if a.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(d), encoding="utf-8")
        print(f"\nwrote {out}", file=sys.stderr)
    return 1 if d["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
