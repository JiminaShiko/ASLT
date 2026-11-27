# ASL alphabet recogniser — current numbers

Rounds 3–6, 2026-11-15 and 2026-11-21. Short version of the measurements; the
full generated report is `reports/report_all.md`. One model ships; the other
profiles' weights and reports (`standard`, `round2`) were **deleted on
2026-11-27**, because `all` won every comparison and keeping them only invited
the wrong file to load. The shipped weights are the 2026-11-21 run, which added
the `rushil` dataset. Every
accuracy here is **leave-one-dataset-out**: the
image was scored by a model that never saw its dataset. Nothing in this file is a
live accuracy claim — the only measurement of a particular hand is the self-test,
`python src/live.py --selftest`.

## Headline (profile `all`, the shipped default)

**Cross-dataset accuracy, POOLED (headline): 68.7% [68.4–69.0% 95% CI] (85,641/124,663 held-out images)**

This is the 2026-11-21 model with `rushil` added (full generated tables:
`reports/report_all.md`).

| held-out dataset | images | accuracy | note |
|---|---|---|---|
| danrasband | 712 | 93.4% | small-n, does not carry the headline |
| grassknoted | 61,562 | 79.9% | |
| massey | 60,085 | 56.3% | the weakest dataset, and the minority style |
| rushil | 2,304 | 86.5% | 23 letters only; carries the headline |

- Macro (mean of the four, unweighted): **79.0%**. Pooled is the headline.
- Held-out capture group (massey A+B → C+D+E — unseen hands, matched style):
  **85.7%** over 35,893 images, the closest dataset-side proxy for a webcam.
- On the 103,208 images round 1's single-pass detector could find at all:
  **70.9%**, against round 1's 67.6%.

The Stage sections below are the 2026-11-15 run (three datasets, 67.0% pooled).
Their numbers are kept as the record of each decision it made; where a line
reads like the shipped model's accuracy, the headline above is the current one.

## Stage C — the two profiles, one table (2026-11-15 run)

`all` trains on every signer equally. `standard` gives the flagged minority-style
cells (massey M/N/P) weight 0, so the model never sees that way of signing them.
Same 71 features, same folds, same seed, paired image by image. The `standard`
weights, report and comparison script were deleted on 2026-11-27; the table
below is kept as the measurement that decided it.

| measure | all | standard |
|---|---|---|
| **pooled cross-dataset (headline)** | **67.0%** | 66.0% |
| danrasband / grassknoted / massey | 94.4 / 79.5 / 53.8% | 94.2 / 77.5 / 53.8% |
| M | 3.2% | 2.5% |
| N | 0.4% | 0.3% |
| P | 40.0% | 39.8% |
| T | 16.8% | 9.4% |
| C | 36.8% | 37.2% |
| D | 40.1% | 41.2% |
| G | 52.9% | 52.8% |
| Q | 55.3% | 51.6% |

**The two columns answer different questions, so this is not a league table.** The
down-weighted cells only exist in folds where massey is in *training*, so
massey-as-test is untouched by design — its column is identical, digit for digit,
in both profiles — and the entire difference lands on the grassknoted fold, where
those letters kept their own M/N/P. Read on the cells it was meant to rescue,
removing the minority style bought nothing: grassknoted went M 3.3% → 1.7%,
N 0.2% → 0.0%, P 71.4% → 71.1%, and T 31.9% → 17.0% as collateral damage. `all` is
therefore the only shipped model. Round 2's halved-weights model pooled at 65.9%
and, like `standard`, has been deleted — retrain it with `--style-weight 0.5` if
that experiment is ever wanted again.

## Stage B — why T fails

Full evidence: `reports/letter_diagnosis.md` (`python src/diagnose_letter.py
--letters T,N,M,P --profile all`), contact sheets in `reports/samples/`.

1. Not a detection-path artifact: out-of-fold T scores 17.3% on a first-try
   detection (n=4,759) against 10.8% after upscale+pad and 8.3% after the mirror.
2. Not garbage landmarks: T's thumb jitter is 0.22/0.28 palm units and its length
   1.20/1.01 (grassknoted/massey) — ordinary, and the contact sheets show a
   plausible tucked thumb in all three datasets.
3. Not a style conflict: T's cross-dataset mean-shape gap is 4.1 units, under the
   alphabet median of 4.6 and level with letters that work (A 3.3, Y 2.6, V 4.3),
   while M/N/P/Q sit at 13.9–17.7.
4. It is a shape collision inside *every* dataset: T's nearest other letter is L
   in grassknoted (crowding 0.231) and danrasband (0.377), A in massey (0.440) —
   and the errors are exactly those (grassknoted T→L 92%, massey T→A 86%).
5. An index laid across the palm *is* a legal reading of both L and T at these
   angles, and nothing in the 71 features says "the thumb is pressed over the
   folded index".
6. So nothing was applied: re-weighting is the wrong lever (Stage C), `also_drop`
   stays empty, and T remains in both profiles' training data.
7. T is nonetheless 5× better under `all` (16.8%) than under round 2's weighting
   (3.1%) — its score moves with the training mix, which is itself evidence that
   the decision boundary, not the data volume, is what is missing.
8. Least destructive fix the evidence supports: one thumb-over-index occlusion
   feature, or more T frames. Both are round-4 work.

## Stage A — what the live path actually runs

`src/live.py` does **not** run the extractor's five-rung ladder per frame. It uses
MediaPipe VIDEO tracking plus a bounded retry: after 1 consecutive frame with no
hand it re-tries that frame once in IMAGE mode at confidence 0.3, then every 3rd
empty frame, then on a mirrored copy (`src/hand_backend.py:Recovery`). Upscale+pad
is deliberately absent — a resize plus a second full detection per frame is not
affordable at 30 FPS. Measured frame by frame through the real code path on an
idle CPU (`reports/live_path.md`, `--backend onnx`): loop speed
**103.6 → 85.0 frames/s** with the retry on, hand coverage **84.3% → 98.3%** of
frames, 17.9 ms per try (17.8 ms amortised). Both sides clear the ≥20 FPS target,
so the retry ships.

The two coverage numbers are different quantities, and an earlier version of these
reports conflated them:

- **Offline image recall: 74.0% → 88.5%** of the dataset images (106,906 →
  127,791 hands). The extractor may spend five detections per image because it
  runs once, offline.
- **Live frame coverage: 84.3% → 98.3%** of camera frames, through the live path,
  where it may not. Neither is an accuracy figure: a frame with no hand is not a
  low score, it is no answer at all.

## What is still weak

Cross-dataset, profile `all`: **N 0.4%**, **M 3.2%**, **T 16.8%**, C 36.8%,
D 40.1%, P 40.0%. On massey alone, also G 12.1%, Q 8.2%, C 14.4%, O 35.3%,
V 41.8%, K 47.4%, R 47.9%. M and N collide with *each other* inside every dataset
(crowding 0.10–0.46), which no weighting can separate; the massey-specific
collapse on C/D/G/K/O/Q/R/V is a style gap that deleting the minority style has
now been shown not to fix.

## Round 5 — the refusal-margin measurement (2026-11-15)

Triggered by a live test: *"great model, but J, P and Z are not correct."* The
three letters have three different causes, and only one of them is a threshold.

**J and Z** are not in the per-frame classifier. Round 5 answered them with a
hand-written trajectory rule; rounds 6-7 fixed what that rule could not see, and
the rule is now the *fallback*. A trained stroke head (`models/jz_motion.pt`,
written by `src/train_jz.py`) reads a 1.2 s window of landmarks and is what names
J and Z in both front ends. Measured on signers it never saw, restricted to the
windows the live gate opens: J 100% precision / 92% recall, Z 94% / 96%
(`reports/jz_model.md`). Counted the way live commits - one stroke fires if any of
its windows clears the threshold - that is J 69/70 strokes and Z 52/53, with the
head shipped as a 5-seed ensemble so no single seed's margins decide it. The two
signers held out of every fold give J 6/6 strokes and Z 4/7, which is the number
to expect on a brand-new hand. The shape/display preconditions are kept in front of it,
because they are what stops a moving hand that is not J or Z from being asked at
all. `--calibrate-jz` still measures the fallback rule's thresholds from three J
and three Z of your own signing - in memory, nothing written.

**P** is a data problem, and the margin experiment is what proved it. The plan
was to raise the top1−top2 gap required for the collisions the OOF data shows
(P↔N 30.6%, Q↔N 35.2%, M↔S 35.0% of their frames). Doing the measurement first
killed that plan: those are not near-ties. In ~90% of true-P frames P is not even
the runner-up (median gap over 0.4), so a gap rule cannot reach them — setting
NP/NQ/MS to 0.25–0.30 moves the wrong-window rate by **0.1 points**. What is
shipped instead is a table for the four pairs that *are* near-ties, found by
replaying the live 4-of-7 vote window over all 122,359 held-out frames (now
dumped with `p1,p2,runner_up`):

| rule | wrong held letters | correct held letters |
|---|---|---|
| flat gap 0.10 (before) | 13,094 | 40,134 |
| `PAIR_MARGINS` = `AS/GH/MN/PQ` at 0.50 | **10,905** (−17%) | 39,786 (−0.9%) |

6.3 wrong letters deleted per correct one lost. The same 3.1-point drop bought by
raising the flat gate costs ~4× more correct letters, which is why the table
exists instead of a bigger default. Per letter: P 54.1% → 36.8% wrong, Q 30.3% →
16.4%, H 39.4% → 33.3%, S 36.7% → 31.7%, A 44.3% → 39.6%; it holds inside each
fold separately (grassknoted 11.6% → 8.1%, massey 39.1% → 35.6%). It is a
`max()` with `--margin`, so a pair entry can only ever refuse more, and
`--pair-margins none` (or the sidebar checkbox) turns it off for an on-camera A/B.

The same dump says why M and N deserve a look on your own camera: when the model
says M or N on an unseen hand it is right 9.7% and 0.5% of the time (top-1
precision — their per-letter accuracy is 3.2% and 0.4%, a different denominator),
and still only 14.9% and 0.0% at p1 ≥ 0.85. Confidence does not rescue them,
which is the signature of a style gap rather than a borderline frame. Hard-refusing a top-1 of M or N is
nearly free at window level (22.5% wrong, correct −0.06%) and was deliberately
**not** shipped: it would make two letters untypeable on a model that scores
94.9% and 90.5% on them in-sample.

## Round 6 — why J and Z could not fire, and what reaches them now (2026-11-15)

The other half of the same live report. Round 5 made a missing J *explain
itself*; this round asked whether the rule was ever looking at those letters, and
the answer was no.

**The old precondition was unreachable where it mattered.** The shipped gate only
judged movement on a frame whose *shape* is "index up, everything else curled".
Measured over 15,772 cached photographs of all 26 letters (`reports/jz_trigger.md`,
by `python src/verify_jz_trigger.py --per 300`), that gate opens on **0.0 % of J
frames** — a real J never passes it — and on 20.0 % of Z, while opening widest on
the letters it must never claim: G 28.1 %, L 20.4 %, Q 13.6 %, D 11.2 %. That is
why 15 of the 17 local fires it produced were on letters other than J or Z, and
those false fires predate this change.

**What ships instead: a second precondition, from the classifier's own reading.**
The 24-class model has no J, so it answers a real J with its nearest neighbour —
and it does it consistently: 96.3 % of J frames are displayed as I, 94.2 % of Z
frames as X. A frame now enters the trajectory rule if the shape gate holds **or**
the top-1 is I (candidate J) or an X below 0.90 confidence (candidate Z), with the
letter only allowed to fire when that reading covers ≥ 60 % of the window actually
judged. That opens on 47.3 % of J and 94.2 % of Z and on nothing else above 8.7 %.

| corpus | frames | J fires (shape gate only) | J fires (+ display) | Z fires (shape gate only) | Z fires (+ display) |
|---|---|---|---|---|---|
| local ASL photographs, 26 letters | 15,772 | 17 | **27** | 0 | **1** |
| MSL dynamic clips (real movement) | 3,301 | 12 | **15** | 0 | 0 |

All 11 added local fires are on J or Z; the 15 false fires on other letters are
**unchanged**, because the shape gate was not touched and they all come through
it. The trajectory thresholds did not move either — same `WINDOW`, `MIN_HOLD`,
`SEG`, `J_*`, `Z_*` — and `--calibrate-jz` still cannot tighten anything, only
loosen. A fire here is a *photo-to-photo* difference: these are bounds on
reachability, not a webcam success rate, and nothing in this section is one.

**What is still blocking Z.** The display gate now sees Z (94.2 % of local Z
frames, 35.1 % of MSL Z frames) and Z still fires once in 15,772 frames and never
on the clips. The gate is not what stops Z — the shipped zig-zag thresholds were
written against a synthetic stroke and matched nothing measured. That is exactly
the question `--calibrate-jz` exists for, and for the first time it can record a
real Z stroke at all: with a reading the guided calibrator keeps the stroke,
without one it kept none.

**The alternative that was tried and rejected.** A motion classifier on real
video: the MSL dynamic corpus (622 clips, 50,022 landmark frames) trained a
102-dim motion vector per 1.2 s window to leave-one-signer-out precision of
**J 91.9 %, Z 85.5 %** (98.1 % / 93.1 % with the static head added) — and the same
head scored **0.8 % recall for J** on local ASL, i.e. chance. It learned MSL
hands, not fingerspelling motion, so it is documented in `reports/msl_motion.md`
and not shipped, and the rule stands. Per-hand wrist-path magnitude explains why
the rule is still conservative: the same 1.2 s window measures 9.08 palm units of
wrist travel on MSL J but 1.25 on local ASL J — the local "J" photographs are a
static index-up pose, so the movement the rule wants is not in them.

**Verification.** `python src/live.py --logic-test` is now **84 checks** (was 73;
87 after round 7):
the original 73 still pass unchanged, which is the regression proof, since
`update()` without a reading is byte-for-byte the round-5 rule. The 11 new ones
pin the limits of the display path — a confident X is an X, a reading outside
`{I, X}` is no candidate, a reading flickering between I and X is refused by the
window share, a straight transit still types nothing — and check, by parsing the
source with `ast`, that both front-ends hand the decider the frame's own reading
(5/5 call sites in `live.py`, 2/2 in `app.py`), so the two cannot drift apart.
`python src/verify_livepath.py --per 20` still reproduces **71/72** blocks through
the whole live path, so the wiring is unaffected. No dataset was retrained, no
new file enters the model, and nothing is saved that was not saved before.

## Round 7 — the confident-I problem: why J was still hard after calibrating (2026-11-15)

Round 6 shipped, the app was run on a real camera, and the report was *"everything
is fine now but J is a little difficult"* — with the read-out saying `J/Z: neither
gate met`, **after** a guided `--calibrate-jz` session. That string is only
reachable when no frame in the whole process has ever opened either
precondition, so the calibration had loosened thresholds on a rule that was
still never being shown a J. The blocker was the precondition, not the stroke.

**The measurement that found it.** Real J frames are not read as unsure. Over the
cached photographs (frame-weighted, `--per 300`), **93.3 % of J frames are read as
I**, at median confidence **0.95**, and **72.7 % of all J frames sit at or above
round 6's flat `p1 < 0.90` ceiling** — refused before the trajectory was ever
computed. Z is the opposite: 89.4 % of Z frames are read as X at median 0.50, and
only **0.6 %** were refused by the ceiling. The single ceiling treated those two
routes as one rule, and it was right about X ("a confident X is an X") and wrong
about I, because a J *is* a confident I — same shape, same frame; only movement
separates them.

**What ships.** `DISPLAY_P1_MAX_FOR = {'J': None, 'Z': 0.90}`: the I → J route has
no confidence requirement, the X → Z route keeps it. Nothing else moved — the
shape gate, the ≥ 60 % window share, and all ten trajectory thresholds are
exactly as round 5 left them, and the display constants remain outside `DEFAULTS`
so `--calibrate-jz` still cannot reach them.

| corpus | frames | J display gate (round 6) | J display gate (now) | Z display gate (unchanged) | J fires on MSL J clips | false J on MSL clips |
|---|---|---|---|---|---|---|
| local ASL photographs | 15,772 | 23.6 % | **93.3 %** | 89.4 % | – | – |
| MSL dynamic clips (real movement) | 3,301 | 48.0 % of all frames | **53.7 %** | – | **1 → 2 of 25** | **41 → 41** |

**What it cost, said out loud.** On the photographs the non-J/Z fires went **15 →
38, all 23 of the increase on the I row (0 → 23)**. That is not a live false-fire
rate: consecutive cached frames are *different hands*, and the difference between
two photographs looks like movement to any trajectory rule — the same artifact
that made G fire J 7 times before this change existed. On MSL clips, which are
continuous strokes, the same change added **no** false fires (12 → 12). The real
protection is the trajectory: a still I has no wrist reversal, no net travel and
no path, and `--logic-test` now drives exactly that case. Whether an I that
arrives with a sideways wobble types a J on a given signer's hand is a webcam
question this project has not yet been able to answer, and `reports/jz_trigger.md`
says so in as many words.

**A reporting bug fixed on the way.** The per-letter roll-up in
`verify_jz_trigger.py` averaged the per-dataset percentages instead of weighting
them by frames, so 30 cached J frames from one source counted as much as 300 from
another: it reported the J display gate as 47.3 % where the frames said 23.6 %,
and it also carried a hard-coded sentence claiming the change "touched no false
fire". Both the aggregation and that sentence are now computed from the data, and
the report has a `refused by the OLD flat ceiling` column so this asymmetry stays
measurable rather than remembered.

**Verification.** `python src/live.py --logic-test` is **87 checks** (84 before
this round): one round-6 check was replaced because it pinned the behaviour round
7 removes ("a confident I is an I: no candidate"), and three were added — a
confident I under a hook fires J, a confident X under a zig-zag still does not
fire Z, an I held still at p1 0.99 types nothing, and the exemption table covers
J only while staying unreachable by calibration. `python
src/verify_livepath.py` reproduces **70/72** blocks; the two misses are `N → M`
(danrasband, 58 % of its frames below the confidence gate) and one `T` block that
committed nothing, both 24-class model behaviour, and that script never
instantiates `MotionDecider`, so round 7 cannot have moved it. `app.py` boots
headless (`HTTP 200`, `/_stcore/health` → `ok`, no traceback). No retraining, no
new file on disk, nothing saved from a camera.

## Round 8 — the webcam closed itself: Google's STUN server against a 10-second budget (2026-11-15)

**The report from the camera.** "It opens. The webcam is worked but the problem is
when I click start the webcam open for a sec than everything got lag/stop and the
webcam close again and the stop btn becomes start again" — and on the rerun attempt,
"the error shown and screen got white".

**How it was caught.** `app.py` was run unchanged under a throwaway launcher that
forced `streamlit_webrtc`, `aiortc` and `aioice` to log at DEBUG into a file, and
the user pressed START. 1 755 lines, three attempts, each about ten seconds:

```
20:57:29.5  create a new worker                        (the click)
20:57:30.9  setRemoteDescription(offer)                (camera opened, offer arrived)
20:57:36.1 / 37.2 / 38.6  STUN BINDING -> 74.125.250.129:19302, retransmitted, unanswered
20:57:39.8  Stopping WebRTC worker                     (10.3 s after the click)
```

**Cause.** `app.py` pinned both ends of the peer connection to
`stun:stun.l.google.com:19302`. That network never answers a STUN query, so ICE
gathering never finished and the SDP answer was never produced; `streamlit_webrtc`
calls `process_offer(..., timeout=10)`, and at ten seconds it stops the worker —
releasing the camera, so STOP becomes START again — and raises
`SignallingTimeoutError`, which is the error box and the half-drawn white page. The
`InvalidStateError('RTCIceTransport is closed')` tasks further down the same log are
consequence, not cause: that is the connect task waking up on a transport the
teardown had already closed.

**Ruled out, with the evidence that ruled them out.** Frame errors: `recv()` swallows
them and paints the red "webcam is still running" box, which never appeared. The
orphaned-context rule in `_reset_context`: fragment reruns finish with
`FINISHED_FRAGMENT_RUN_SUCCESSFULLY` and do not increment `script_run_count`, so the
two-run rule cannot fire. CPU saturation: real, and it explains the lag, but it
cannot explain a clock that stops at almost exactly ten seconds every time.

**What shipped** (both halves approved before writing):

| change | why |
|---|---|
| host-only ICE: `RTC_CONFIGURATION = RTCConfiguration({"iceServers": []})` **and** a matching `frontend_rtc_configuration`, swapped together by `rtc_pair(use_stun)` behind a sidebar **"Use a public STUN server"** checkbox, off by default | the browser and the server are the same machine, so host candidates are enough. Both ends have to agree, and the list has to be explicitly empty: `{}` or a missing `iceServers` key makes the library put Google straight back (`_resolve_server_rtc_configuration` fills in only when `iceServers is None`, `component.py:431`; same rule on the frontend at `:287`) |
| `media_stream_constraints` now asks for 640×480 at **24 fps**, and "Work every Nth frame" defaults to **2** instead of 1 | one processed frame measures ~50 ms here, so a 30 fps camera hands the pipeline 33 ms of budget with 50 ms of work and the surplus is decoded and thrown away. Stride 2 is ~26 ms, which fits |

**Measured without a camera or a browser** — two aiortc peers in one process,
synthetic 64×48 frames in RAM, the exact configs the app now uses:

| ICE config | gathering | SDP answer ready | against the 10 000 ms budget |
|---|---|---|---|
| `{"iceServers": []}` — what ships | **5 ms** | **9 ms** | fits, with ~1 000× to spare |
| Google STUN — what `app.py` used to send | 5 009 ms | **10 035 ms** | **blown. This is the crash** |

10 035 ms against a 10 000 ms timeout is the bug reproduced without a browser: aioice
waits its full five seconds per host candidate for a STUN reply that never comes.
Both peers still reached `checking → completed` and `connectionState=connected` on
host candidates alone, which is the claim the fix rests on. (The same harness did not
manage to push an encoded frame between two local aiortc peers in either
configuration — a limitation of the synthetic track, identical in both columns, and
not evidence about the app, whose browser path demonstrably delivered frames: the
picture appeared before the teardown.)

**The price, stated.** MediaPipe VIDEO costs 45–52 ms whatever the resolution (240 px
45.7 ms vs 1280 px 49.9 ms), so shrinking the picture cannot buy server-side speed and
the CLI ceiling stays ~20 fps — `live.py --benchmark` reports 19.5 fps. That is why the
web page runs the hand tracker in the browser: the server then only classifies 21
landmarks, and the page can send a frame every 28 ms instead of every 40 ms. Stride 2
still halves what the stroke head sees (12 hand positions a second instead of 24), so
if J or Z stops firing, drop back to 1.

**Verification.** `python src/live.py --logic-test` is still **87 checks** (this round
touched only `app.py`, and none of `MotionDecider`); `python src/verify_livepath.py
--profile all --backend onnx` is still **70/72** blocks; `app.py` boots headless under
the DEBUG launcher, renders the live tab (`The frontend state is neither playing nor
signalling (key=asl-landmark-live)`), answers `HTTP 200`, and logs **zero STUN lines**.
Still unproven, and unprovable from here: the user's own press of START — the browser
half of the pairing, and how the calmer stream feels. No image from a camera is
recorded or written anywhere; the launcher's file is text about the connection.

**Documentation split (same day, before the push).** `README.md` was 1,093 lines and
mixed the quick start with every measured table, so it was rewritten as the short
guide (about / numbers / commands / guide) and the long content moved verbatim to
`docs/DETAILS.md`, with a banner there stating that `src/...` and `reports/...`
paths in it are relative to `asl_landmark_app/`, one level up. The README keeps the
plain-text look of the group's `README.txt` demo — caps headings, `key : value`
blocks — but every aligned block is fenced: single newlines inside a markdown
paragraph are soft breaks, so an unfenced block renders as one run-on line on
GitHub. The four dangling `see README` references (`src/config.py`,
`src/normalize.py`, `src/live.py`, `app.py`) now point at `docs/DETAILS.md`, and
each number the README claims was re-measured against disk rather than inherited:
2.92 GB of datasets from `data/datasets.json` (5.57 GB once `rushil` was added
on 2026-11-21), `cache/` 76 MB,
`reports/oof_all.csv` 23 MB, `models/hand_landmarker.task` 7,819,105 bytes, and
**52 files / 1.52 MB** offered by `git ls-files -o --exclude-standard` under the
new `.gitignore`. The privacy grep printed in the README was run and returns no
matches.

## Reproducing



```
python src/train.py && python src/export.py
python src/diagnose_letter.py --letters T,N,M,P --profile all --out reports/letter_diagnosis.md
python src/verify_livepath.py --per 20 --backend onnx --profile all
python src/verify_jz_trigger.py --per 300 --profile all --backend torch
        #   ^ the round-6 and round-7 tables above: what each gate opens and what
        #     fires, per letter, on cached photographs + cached MSL trajectories,
        #     plus how much of J the old flat confidence ceiling threw away.
        #     No camera. Percentages are frame-weighted across datasets.
python src/live.py --logic-test                      # 87 checks, no camera
python src/live.py --selftest --profile all,standard    # your camera; saves nothing
python src/train.py --profile all --folds-only --no-save --dump-preds reports/oof_all.csv
        #   ^ the dump behind the round-5 margin table (p1, p2 and the runner-up
        #     letter for every held-out image). Reading PAIR_MARGINS starts here.
```
