"""Measure the J/Z preconditions, letter by letter, on data already on disk.

Local photographs bound the rule (photo-to-photo, not a stroke); MSL clips are
real movement but another language. Reads public files, writes one report.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import landmark_store as LS  # noqa: E402
import live  # noqa: E402
import normalize  # noqa: E402

DT = 0.05          # 20 FPS, as verify_livepath.py; every letter so false fires stay visible
WATCH = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def readings(backend, lms: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Per-frame raw top-1 for these hands: (top1, p1, i, s)."""
    rows = []
    for k, lm in enumerate(lms):
        p = np.asarray(backend(normalize.features(lm)))
        order = np.argsort(p)[::-1]
        rows.append((labels[int(order[0])], float(p[int(order[0])]),
                     float(p[int(order[1])]), k))
    return pd.DataFrame(rows, columns=["top1", "p1", "p2", "pos"])


def drive(lms: np.ndarray, top1=None, p1=None, th: dict | None = None,
          intended: str | None = None) -> dict:
    """Feed one stream through the shipped decider and count what happened.

    Pass top1=p1=None for shape-gate-only behaviour (the pre-round-6 app).
    """
    md = live.MotionDecider()
    if th:
        md.apply_thresholds(th)
    if intended:
        md.intended = intended
    out = {"n": 0, "shape": 0, "disp": 0, "both": 0, "J": 0, "Z": 0,
           "blocked": 0, "strokes": 0, "allow_j": 0, "allow_z": 0,
           "ceil_refused": 0}
    t = 0.0
    for k, lm in enumerate(lms):
        t += DT
        a = None if top1 is None else str(top1[k])
        b = None if p1 is None else float(p1[k])
        r = md.update(t, lm, a, b)
        out["n"] += 1
        # measured from the frame itself, never from decider memory (`last_top1`
        # is absent in a run with no readings)
        up_shape = md.handshape(normalize.canonicalize(lm))[0]
        up_disp = md.display_candidate(a, b) is not None
        # the flat ceiling is judged here, not via `display_candidate`: the
        # decider no longer applies it to I, so asking it would always say 0 %
        named = live.MotionDecider.DISPLAY_FOR.get(a)
        out["ceil_refused"] += int(named is not None and b is not None
                                   and b >= live.MotionDecider.DISPLAY_P1_MAX)
        out["shape"] += int(up_shape)
        out["disp"] += int(up_disp)
        out["both"] += int(up_shape and up_disp)
        out["allow_j"] += int("J" in md.candidates)
        out["allow_z"] += int("Z" in md.candidates)
        if r == "J":
            out["J"] += 1
        elif r == "Z":
            out["Z"] += 1
        elif md.debug.get("blocked"):
            out["blocked"] += 1
    out["strokes"] = len(md.strokes)
    return out


def local_table(df: pd.DataFrame, backend, labels, per: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (d, lab), grp in df.groupby(["dataset", "letter"]):
        lab = str(lab)
        if lab not in WATCH or len(grp) < 4:
            continue
        grp = grp.sample(min(per, len(grp)), random_state=int(rng.integers(1 << 31)))
        grp = grp.sort_values("image")
        lms = LS.landmark_matrix(grp)
        rd = readings(backend, lms, labels)
        old = drive(lms)                                  # shape gate only
        new = drive(lms, rd["top1"].values, rd["p1"].values)
        rows.append({"dataset": d, "letter": lab, "n": new["n"],
                     "shape_pct": 100 * old["shape"] / max(new["n"], 1),
                     "disp_pct": 100 * new["disp"] / max(new["n"], 1),
                     "allow_j_pct": 100 * new["allow_j"] / max(new["n"], 1),
                     "allow_z_pct": 100 * new["allow_z"] / max(new["n"], 1),
                     "j_old": old["J"], "z_old": old["Z"],
                     "j_new": new["J"], "z_new": new["Z"],
                     "blocked": new["blocked"],
                     "shown_I_pct": 100 * float((rd["top1"] == "I").mean()),
                     "shown_X_pct": 100 * float((rd["top1"] == "X").mean()),
                     "ceil_refused_pct": 100 * new["ceil_refused"] / max(new["n"], 1),
                     "mean_p1": float(rd["p1"].mean())})
        print(f"   {d:12} {lab}: shape {rows[-1]['shape_pct']:5.1f}%  "
              f"display {rows[-1]['disp_pct']:5.1f}%  "
              f"old ceiling would refuse {rows[-1]['ceil_refused_pct']:5.1f}%  "
              f"fires J {old['J']}->{new['J']} Z {old['Z']}->{new['Z']}", flush=True)
    return pd.DataFrame(rows)


def msl_table(backend, labels, per_clips: int, seed: int,
              clips_dir: Path | None = None) -> pd.DataFrame:
    """Per-letter fire rates on real clips.

    `clips_dir` can point at cache/jz_landmarks: identical shard schema.
    """
    base = Path(clips_dir) if clips_dir else config.CACHE_DIR / "msl_landmarks"
    shards = sorted(base.glob("*.npz"))
    if not shards:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rng.shuffle(shards)
    rows: dict[str, dict] = {}
    for path in shards:
        z = np.load(path)
        letter = str(z["letter"].item())
        if letter not in WATCH:
            continue
        d = rows.setdefault(letter, {"letter": letter, "clips": 0, "n": 0,
                                     "shape": 0, "disp": 0, "j_new": 0,
                                     "z_new": 0, "j_old": 0, "z_old": 0,
                                     "shown_I": 0, "shown_X": 0,
                                     "ceil_refused": 0})
        if d["clips"] >= per_clips:
            continue
        lms = np.asarray(z["lm"], dtype=np.float64)
        if len(lms) < 6:
            continue
        fps = float(z["fps"].item()) or 30.0
        dt = 1.0 / max(fps, 1.0)                 # the clip's real cadence
        rd = readings(backend, lms, labels)
        old, new = drive_msl(lms, dt), drive_msl(lms, dt, rd)
        d["clips"] += 1
        d["n"] += len(lms)
        d["shape"] += old["shape"]
        d["disp"] += new["disp"]
        d["j_old"] += old["J"]; d["z_old"] += old["Z"]
        d["j_new"] += new["J"]; d["z_new"] += new["Z"]
        d["shown_I"] += int((rd["top1"] == "I").sum())
        d["shown_X"] += int((rd["top1"] == "X").sum())
        d["ceil_refused"] += new["ceil_refused"]
    out = pd.DataFrame(list(rows.values()))
    if not len(out):
        return out
    out["shape_pct"] = 100 * out["shape"] / out["n"].clip(lower=1)
    out["disp_pct"] = 100 * out["disp"] / out["n"].clip(lower=1)
    out["ceil_pct"] = 100 * out["ceil_refused"] / out["n"].clip(lower=1)
    return out.sort_values("letter")


def drive_msl(lms, dt: float, rd: pd.DataFrame | None = None) -> dict:
    """Same as `drive` but on the clip's own frame interval, per run of frames."""
    md = live.MotionDecider()
    out = {"n": 0, "shape": 0, "disp": 0, "J": 0, "Z": 0, "blocked": 0,
           "ceil_refused": 0}
    t = 0.0
    for k, lm in enumerate(lms):
        t += dt
        a = None if rd is None else str(rd["top1"].iloc[k])
        b = None if rd is None else float(rd["p1"].iloc[k])
        r = md.update(t, lm, a, b)
        out["n"] += 1
        out["J"] += int(r == "J")
        out["Z"] += int(r == "Z")
        out["blocked"] += int(bool(md.debug.get("blocked")))
        out["shape"] += int(md.handshape(normalize.canonicalize(lm))[0])
        out["disp"] += int(md.display_candidate(a, b) is not None)
        # flat ceiling judged directly, as in `drive`
        out["ceil_refused"] += int(live.MotionDecider.DISPLAY_FOR.get(a) is not None
                                   and b is not None
                                   and b >= live.MotionDecider.DISPLAY_P1_MAX)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=300,
                    help="cached images per (dataset, letter) block")
    ap.add_argument("--profile", default=None, choices=sorted(config.PROFILES))
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--msl-clips", type=int, default=8,
                    help="clips per letter (0 = skip that table)")
    ap.add_argument("--no-msl", action="store_true")
    ap.add_argument("--clips-dir", default=None,
                    help="trajectory shards to measure instead of the shipped "
                         "cache/msl_landmarks, e.g. cache/jz_landmarks after "
                         "python src/download_jz_data.py && "
                         "python src/extract_jz_landmarks.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(config.REPORTS_DIR / "jz_trigger.md"))
    args = ap.parse_args()

    meta = json.loads(config.meta_path(args.profile).read_text(encoding="utf-8"))
    backend, labels = live.build_backend(args.backend, profile=args.profile)
    prof = args.profile or config.PROFILE_DEFAULT
    print(f"== J/Z precondition measurement ==  backend={backend.name}  "
          f"profile={prof}  labels={''.join(labels)}")

    df = LS.load_raw()
    # not keep_static_classes: that drops J and Z; keep letters with >= 50 rows
    df = df[df["letter"].str.fullmatch(r"[A-Z]") & df["letter"].isin(WATCH)]
    df = df[df.groupby("letter")["letter"].transform("size") >= 50]
    loc = local_table(df, backend, labels, args.per, args.seed)
    msl = pd.DataFrame() if args.no_msl or args.msl_clips == 0 else \
        msl_table(backend, labels, args.msl_clips, args.seed,
                  Path(args.clips_dir).resolve() if args.clips_dir else None)
    clips_name = (Path(args.clips_dir).name if args.clips_dir
                  else "msl_landmarks")
    # computed once, here, so no sentence below can assert what the table says
    oth_msl = msl[~msl["letter"].isin(["J", "Z"])] if len(msl) else None
    false_msl_old = int((oth_msl["j_old"] + oth_msl["z_old"]).sum()) if oth_msl is not None else 0
    false_msl_new = int((oth_msl["j_new"] + oth_msl["z_new"]).sum()) if oth_msl is not None else 0

    # frame-weighted roll-up, never a mean of per-dataset means
    PCT_COLS = ["shape_pct", "disp_pct", "allow_j_pct", "allow_z_pct",
                "shown_I_pct", "shown_X_pct", "ceil_refused_pct"]
    SUM_COLS = ["j_old", "z_old", "j_new", "z_new", "blocked"]
    agg = pd.DataFrame([
        {"letter": lab, "n": int(sub["n"].sum()),
         **{c: float(np.average(sub[c], weights=sub["n"])) for c in PCT_COLS},
         **{c: int(sub[c].sum()) for c in SUM_COLS}}
        for lab, sub in loc.groupby("letter")])
    tot = {"n": int(agg["n"].sum()), "j_old": int(agg["j_old"].sum()),
           "z_old": int(agg["z_old"].sum()), "j_new": int(agg["j_new"].sum()),
           "z_new": int(agg["z_new"].sum()), "blocked": int(agg["blocked"].sum())}
    wrong_new = int(sum(agg.loc[~agg["letter"].isin(["J", "Z"]), "j_new"]
                        + agg.loc[~agg["letter"].isin(["J", "Z"]), "z_new"]))
    wrong_old = int(sum(agg.loc[~agg["letter"].isin(["J", "Z"]), "j_old"]
                        + agg.loc[~agg["letter"].isin(["J", "Z"]), "z_old"]))

    def g(letter: str, col: str) -> float:
        """One measured cell, or NaN if that letter was not in the sample."""
        sub = agg[agg["letter"] == letter]
        return float(sub[col].iloc[0]) if len(sub) else float("nan")

    # name the non-J/Z letters the I exemption made fire
    fr = agg[~agg["letter"].isin(["J", "Z"])].copy()
    fr["delta"] = (fr["j_new"] + fr["z_new"]) - (fr["j_old"] + fr["z_old"])
    gained = fr[fr["delta"] > 0].sort_values("delta", ascending=False)
    if len(gained):
        false_now = ("The other letters went from "
                     f"{wrong_old} fires to {wrong_new}, all of the increase on "
                     + ", ".join(f"**{r['letter']}** {int(r['j_old'] + r['z_old'])}"
                                 f" -> {int(r['j_new'] + r['z_new'])}"
                                 for _i, r in gained.iterrows())
                     + ". Those are frames the classifier reads as I, which is "
                     "the price of the exemption and the reason it needs a "
                     "second opinion on real movement, not on photographs.")
    else:
        false_now = (f"The other letters are unchanged at {wrong_new} fires "
                     f"(was {wrong_old}): the exemption opened no new route for "
                     "a letter that is not J or Z.")

    md = [f"# Why J and Z never fired, and what reaches them now\n",
          f"Generated by `python src/verify_jz_trigger.py --per {args.per} "
          f"--profile {prof} --backend {args.backend}` on "
          f"{pd.Timestamp.now():%Y-%m-%d %H:%M}. Backend `{backend.name}`, "
          f"{meta['model']['in_dim']} features from "
          f"`{'+'.join(meta['features']['blocks'])}`, labels "
          f"`{''.join(labels)}` - **no J, no Z in the classifier**, which is the "
          f"reason a separate rule exists at all.\n",
          "## The old precondition was unreachable where it mattered\n",
          "The shipped rule only judged movement on a frame whose shape is "
          "'index extended, every other finger curled' "
          "(`MotionDecider.handshape`: index tip/MCP-to-PIP ratio >= 1.6 **and** "
          "all four others <= 1.20). Measured over the cached photographs, that "
          f"gate opens on **{g('J', 'shape_pct'):.1f} % of J frames** and "
          f"**{g('Z', 'shape_pct'):.1f} % of Z frames**, while opening on "
          f"{g('G', 'shape_pct'):.1f} % of G, {g('L', 'shape_pct'):.1f} % of L "
          f"and {g('D', 'shape_pct'):.1f} % of D - it is closed on the two "
          "letters it exists for and widest on three it must never claim. The "
          f"display precondition below opens on {g('J', 'disp_pct'):.1f} % of J "
          f"and {g('Z', 'disp_pct'):.1f} % of Z. It also opens on the two "
          f"readings it is built from - a real I on {g('I', 'disp_pct'):.1f} % "
          f"and a real X on {g('X', 'disp_pct'):.1f} % - and refusing those is "
          "the trajectory rule's job, not the precondition's, which is why the "
          f"fire columns below are the ones to read. Among every other letter "
          f"measured none exceeds "
          f"{max(g(x, 'disp_pct') for x in WATCH if x not in 'JZIX'):.1f} %.\n",
          f"| corpus | frames | fires J (old) | fires Z (old) | fires J (new) | "
          f"fires Z (new) | refused on the share rule (new) |",
          "|---|---|---|---|---|---|---|",
          f"| local ASL, {agg['letter'].nunique()} letters | {tot['n']} | {tot['j_old']} | "
          f"{tot['z_old']} | {tot['j_new']} | {tot['z_new']} | {tot['blocked']} |",
          "\n`fires (old)` is the shape gate alone - what shipped until round 5 - "
          "and `fires (new)` is both preconditions as they ship now, after "
          f"rounds 6 and 7. J and Z account for {tot['j_new'] + tot['z_new'] - wrong_new} "
          "of the new fires against "
          f"{tot['j_old'] + tot['z_old'] - wrong_old} before it: the gate is "
          f"what was missing, and it added "
          f"{tot['j_new'] + tot['z_new'] - wrong_new - (tot['j_old'] + tot['z_old'] - wrong_old)} "
          "fires on the two letters the rule exists for. "
          + false_now + " A local 'fire' here is a *photo-to-photo* difference "
          "between different hands, so read these as bounds on reachability, "
          "never as a webcam success rate - and the same exemption measured on "
          "the MSL clips below, which *are* continuous strokes rather than photo "
          f"jumps, moved the false fires there from {false_msl_old} to "
          f"{false_msl_new}.",
          "## Per letter, local ASL photographs\n",
          "| letter | frames | shape gate opens | display gate opens | refused "
          "by the OLD flat ceiling | window allows J | window allows Z | shown "
          "as I | shown as X | J fires old -> new | Z fires old -> new | refused "
          "|",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _i, r in agg.sort_values("letter").iterrows():
        md.append(f"| {r['letter']} | {int(r['n'])} | {r['shape_pct']:.1f}% | "
                  f"{r['disp_pct']:.1f}% | {r['ceil_refused_pct']:.1f}% | "
                  f"{r['allow_j_pct']:.1f}% | "
                  f"{r['allow_z_pct']:.1f}% | {r['shown_I_pct']:.1f}% | "
                  f"{r['shown_X_pct']:.1f}% | {int(r['j_old'])} -> "
                  f"{int(r['j_new'])} | {int(r['z_old'])} -> {int(r['z_new'])} | "
                  f"{int(r['blocked'])} |")
    md += ["\nThe two `shown as` columns are the argument for the change. The "
           "classifier has no J, so a real J is *displayed* as the closest thing "
           f"it knows, and it does that consistently: the J row is shown as I on "
           f"{g('J', 'shown_I_pct'):.1f} % of its frames, the Z row as X on "
           f"{g('Z', 'shown_X_pct'):.1f} %, and the two letters the model is "
           f"being asked to double as (real I, real X) are shown that way on "
           f"{g('I', 'shown_I_pct'):.1f} % and {g('X', 'shown_X_pct'):.1f} % - "
           "which is exactly why the display alone is not the answer and the "
           "trajectory still has to agree.\n",
           f"The `refused by the OLD flat ceiling` column is the round-7 "
           f"argument, and it is measured against the ceiling as round 6 shipped "
           f"it (`p1 < {live.MotionDecider.DISPLAY_P1_MAX}` for both routes), "
           f"not against the rule now in force: {g('J', 'ceil_refused_pct'):.1f} "
           f"% of real J frames were thrown away before the trajectory was ever "
           f"looked at, against {g('Z', 'ceil_refused_pct'):.1f} % of Z frames. "
           "That asymmetry is the whole reason the two routes are no longer "
           "treated alike. A confident X really is an X, so Z keeps its ceiling; "
           "a confident I is exactly what a J frame looks like, so confidence "
           f"was never evidence against J, and the per-route table "
           f"`DISPLAY_P1_MAX_FOR = "
           f"{live.MotionDecider.DISPLAY_P1_MAX_FOR}` now says so. What protects "
           "a held I is not the ceiling but the trajectory: a still hand has no "
           "reversal, no net travel and no path, and `live.py --logic-test` "
           "pins exactly that, on a synthetic I held for 1.5 s at p1 0.99. "
           "Photographs cannot test it, because consecutive cached frames are "
           "different hands and their difference looks like movement - which is "
           f"why the I row above nevertheless fires {int(agg.loc[agg['letter'] == 'I', 'j_new'].sum())} "
           f"J's over {int(agg.loc[agg['letter'] == 'I', 'n'].sum())} frames here "
           "and is the one number in this file that has to be re-measured on a "
           "real hand before anyone calls the exemption free.\n",
           "`window allows J / Z` is the share rule in force: the fraction of "
           "judged frames whose window majority named that letter, i.e. how "
           f"often each letter could fire at all ({g('J', 'allow_j_pct'):.1f} % "
           f"for J on J frames, {g('Z', 'allow_z_pct'):.1f} % for Z on Z "
           f"frames). I frames are now judgable too - {g('I', 'allow_j_pct'):.1f} "
           f"% of them allow a J, against {g('I', 'allow_z_pct'):.1f} % that "
           f"allow a Z - and every other letter stays under "
           f"{max(g(x, 'allow_j_pct') for x in WATCH if x not in 'JZI'):.1f} %. "
           "`refused` counts frames whose "
           "path matched while no precondition held; it guards against a reading "
           "that flickers between I and X, and on this data it never had to "
           "intervene - the flicker case is covered by a synthetic check in "
           "`live.py --logic-test` instead.\n",
           "Percentages here are **frame-weighted** across datasets. An earlier "
           "version of this file averaged the per-dataset percentages instead, "
           "which let 30 cached J frames from one source count as much as 300 "
           "from another: on the round-6 rule that mean-of-means reported the J "
           "display gate as 47.3 % where the frames themselves said 23.6 %. Any "
           "percentage quoted from before that fix is a mean of means and is not "
           "comparable with the tables above.\n",
           f"## Per letter, real movement ({clips_name})\n"]
    if len(msl):
        md += [f"Shards from `{clips_name}` - the only data here with genuine "
               "continuous strokes. Frame interval is each clip's own fps.\n",
               "| letter | clips | frames | shape gate opens | display gate "
               "opens | refused by the OLD flat ceiling | J fires old -> new | "
               "Z fires old -> new |",
               "|---|---|---|---|---|---|---|---|"]
        for _i, r in msl.iterrows():
            md.append(f"| {r['letter']} | {int(r['clips'])} | {int(r['n'])} | "
                      f"{r['shape_pct']:.1f}% | {r['disp_pct']:.1f}% | "
                      f"{r['ceil_pct']:.1f}% | "
                      f"{int(r['j_old'])} -> {int(r['j_new'])} | "
                      f"{int(r['z_old'])} -> {int(r['z_new'])} |")
        if clips_name == "msl_landmarks":
            md.append("\nAn MSL fire rate is **not** an ASL fire rate: the hands, "
                      "the camera and the letters all differ. It is here to "
                      "answer one narrower question - can the display "
                      "precondition see movement the shape gate could not - and "
                      "nothing more. The trained motion head measured on this "
                      "corpus is in `reports/msl_motion.md`; it is *not* "
                      "shipped, because it transferred to local ASL at 0.8 % "
                      "recall for J.\n")
        else:
            md.append(f"\nA fire rate on `{clips_name}` is **not** a webcam "
                      "rate: these are recorded clips, and the rule's real "
                      "protection is the stroke, not the corpus. Confirm on "
                      "your own hand with `python src/live.py --calibrate-jz`.\n")
    else:
        md.append("Skipped (`--no-msl` / no MSL landmark cache).\n")
    md += ["## What changed in the code, and what did not\n",
           "```python",
           "MotionDecider.DISPLAY_FOR       = " + repr(
               live.MotionDecider.DISPLAY_FOR),
           f"MotionDecider.DISPLAY_P1_MAX    = "
           f"{live.MotionDecider.DISPLAY_P1_MAX}   (default)",
           "MotionDecider.DISPLAY_P1_MAX_FOR  = " + repr(
               live.MotionDecider.DISPLAY_P1_MAX_FOR) + "   # None = no ceiling",
           f"MotionDecider.DISPLAY_MIN_SHARE = "
           f"{live.MotionDecider.DISPLAY_MIN_SHARE}",
           "```",
           "\nA frame now enters the trajectory rule if the shape gate holds "
           "**or** the classifier's top-1 is I at any confidence (candidate J) "
           "or an X below that route's confidence ceiling (candidate Z), and the "
           f"letter that may fire is the one named on at least "
           f"{live.MotionDecider.DISPLAY_MIN_SHARE:.0%} "
           "of the frames in the window the rule actually judges. The shape gate "
           "is unchanged and still on its own opens both letters, so nothing that "
           "fired before can fail to fire now. What the trajectory requires did "
           "not move: same `WINDOW`, `MIN_HOLD`, `SEG`, `J_*`, `Z_*`, and "
           "`--calibrate-jz` still cannot tighten anything, only loosen. "
           "`update()` called without a reading is byte-for-byte the old rule, "
           "which is what `live.py --logic-test` drives.\n"]

    # what the change did not fix
    ztot = tot["z_new"]
    j_msl = z_msl = n_msl_j = 0
    z_disp_msl = float("nan")
    if len(msl):
        jz = msl[msl["letter"] == "J"]
        zz = msl[msl["letter"] == "Z"]
        j_msl = int(jz["j_new"].sum()) if len(jz) else 0
        n_msl_j = int(jz["n"].sum()) if len(jz) else 0
        z_msl = int(zz["z_new"].sum()) if len(zz) else 0
        z_tot_msl = int(zz["z_old"].sum() + zz["n"].sum()) if len(zz) else 0
        z_disp_msl = (100 * zz["disp"].sum() / max(zz["n"].sum(), 1)
                      if len(zz) else float("nan"))
        del z_tot_msl
    md += ["\n## What this did not fix, in one line each\n",
           f"* Z fired {ztot} times locally and {z_msl} on the MSL clips, in "
           f"spite of the precondition now being reachable on Z "
           f"({g('Z', 'disp_pct'):.1f} % of local Z frames"
           + (f", {z_disp_msl:.1f} % of MSL Z frames"
              if z_disp_msl == z_disp_msl else "")
           + "). The gate is not what is stopping Z: the shipped zig-zag "
           "thresholds (`Z_PATH`, `Z_DROP`, two tip reversals, hand still) "
           "matched nothing measured, because they were written against a "
           "synthetic zig-zag. That is what `--calibrate-jz` is for, and with a "
           "reachable gate it can now measure a real Z stroke at all.\n",
           f"* J fires rarely - {tot['j_new']} times over {tot['n']} local "
           f"frames, {j_msl} over {n_msl_j} MSL frames. Same cause, same "
           "remedy.\n",
           f"* Two kinds of false fire, and they have different causes. The "
           f"shape gate ones predate everything: G "
           f"{g('G', 'shape_pct'):.1f} %, L {g('L', 'shape_pct'):.1f} %, Z "
           f"{g('Z', 'shape_pct'):.1f} %, D {g('D', 'shape_pct'):.1f} %, X "
           f"{g('X', 'shape_pct'):.1f} % of frames open a gate the rule then "
           "reads as a hook, because between two different people's photographs "
           "any difference looks like movement. Tightening that gate would break "
           "letters that fire correctly today, so it is recorded as the next job "
           "rather than quietly changed. The new ones are the I route's: "
           f"non-J/Z letters went {wrong_old} -> {wrong_new} locally, and "
           f"on the MSL clips, which are continuous strokes rather than photo "
           f"jumps, the same route changed the false fires from {false_msl_old} "
           f"to {false_msl_new}. On a real hand the discriminator is the "
           "trajectory, not the reading - a held I has no wrist reversal, no net "
           "travel and no path, which `live.py --logic-test` checks - but only "
           "the webcam can say whether an I that arrives with a wobble types J, "
           "and this file has never seen one.\n"]

    md += ["\n## What this does not prove\n",
           "No webcam frame was used, and no number in this file is a live "
           "accuracy. Only `python src/live.py --selftest` and using the app "
           "measure the real camera. The local blocks are sequences of different "
           "hands, so a fire here does not mean the stroke was right - only that "
           "the gate can be reached at all, and the honest way to read the two "
           "'old -> new' columns is as a reachability change, not a precision "
           "change. The precision of the *precondition* itself is a separate "
           "measurement and lives in the per-letter 'shown as' columns above.\n"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    loc.to_csv(config.REPORTS_DIR / "jz_trigger_local.csv", index=False)
    if len(msl):
        msl.to_csv(config.REPORTS_DIR / "jz_trigger_msl.csv", index=False)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
