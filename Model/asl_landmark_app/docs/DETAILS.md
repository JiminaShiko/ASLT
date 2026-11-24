# DETAILS — the long reference (moved out of README.md on 2026-11-15)

README.md is the short guide: what it is, the numbers, the commands, the app.
This file is everything else — the full measured tables and what each one means,
the stage-by-stage pipeline with every script flag, the normalisation maths, the
head-to-head with the removed photo model, the complete troubleshooting table, the
verification recipes and the reproducibility notes. Paths like `src/live.py` and
`reports/live_path.md` are relative to `asl_landmark_app/`, i.e. one level up
from this file. The round-by-round narrative lives in ../REPORT.md.

---

# ASL alphabet — live webcam recogniser (landmark + MLP)

Webcam frame → MediaPipe Hands (21 landmarks) → the same normalisation used in
training → small MLP → smoothed letter output. Everything is trained from
public datasets; **you never have to record or add a single image of your own.**

```
python run_all.py        # download -> extract -> train -> report -> export
python src/live.py       # start recognising (console window)
streamlit run app.py     # browser UI: live webcam, photo upload, results
```

---

## Measured results (profile `all`)

**Current shipped model (2026-11-21, `rushil` added): 68.7 % [68.4–69.0 %]
pooled over 124,663 held-out images, macro 79.0 %, same-yardstick-as-round-1
70.9 % (`reports/report_all.md`). The tables on this page are the 2026-11-15
three-dataset run and are kept as that run's record.**

The 2026-11-15 seed-0 model was retrained to regenerate the out-of-fold dump
(`reports/oof_all.csv`); it reproduced the round-3 headline exactly, so the table
is unchanged apart from this note. What round 5 added is on the *decision* layer,
not the model — see *What round 5 changed*.

| what | number | meaning |
|---|---|---|
| **cross-dataset, leave-one-dataset-out, POOLED** (2026-11-15, 3 datasets) | **67.0 %** [66.7–67.2 %] — danrasband 94.4 % (712 imgs, small-n), grassknoted 79.5 %, massey 53.8 % | the honest headline for profile `all` on that run: every test image was predicted by a model that never saw its dataset. The shipped 2026-11-21 model is 68.7 % with rushil added |
| the other profile, `standard` | 66.0 % pooled; grassknoted M 3.3 → 1.7 %, N 0.2 → 0.0 %, T 31.9 → 17.0 % | removing the minority-style cells rescued nothing where it was meant to, so `all` ships — see `python src/profile_compare.py` and REPORT.md |
| same yardstick as round 1 | 69.2 % on the 101,932 images round 1's detector could find at all | the only cell comparable with round 1's 67.6 %, and inside the run-to-run band: no measurable change |
| **held-out capture group** (massey A+B → C+D+E) | **85.7 %** over 35,893 images | unseen hands, matched imaging style — the closest *dataset-side* proxy for your webcam. It is still not a webcam measurement |
| detection recall, offline images | 74.0 % → **88.5 %** (106,906 → 127,791 hands) | the extractor's ladder: +20,885 usable images at no annotation cost. Not the same quantity as live frame coverage — read the next row and the section below it |
| live frame coverage, and its cost | 84.3 % → **98.3 %** of frames; loop **103.6 → 85.0 frames/s**; 17.9 ms per retry (17.8 amortised) | measured frame by frame through the webcam code path on an idle CPU, on charger (`reports/live_path.md`); the same build measured 27–33 FPS on battery in an earlier run, and both clear the ≥20 target |
| optimistic block split | 82.6 / 96.0 / 96.3 % per dataset | **not** a live estimate; the same hands appear on both sides |
| ONNX parity | max \|Δlogit\| 1.91e-06, argmax agreement 100 %, accuracy 97.40 % on both backends over 2,000 cached hands (`standard`: 91.10 %) | `--backend onnx` is equivalent and ~6× faster on the classifier (0.232 → 0.036 ms); MediaPipe is the cost, not this MLP |
| weakest letters, cross-dataset | **N 0.4 %**, **M 3.2 %**, **T 16.8 %**, C 36.8 %, D 40.1 %, P 40.0 %; on massey also C 14.4 %, G 12.1 %, Q 8.2 %, O 35.3 %, V 41.8 % | T→L/A, M↔N, P→N. `reports/letter_diagnosis.md` says which of those is a shape collision and which is a style gap |

### Coverage, and why the two coverage numbers are not the same quantity

Round 1's 67.6 % pooled was measured on a cache that was **missing 20,885 hands**
— the flat palms, the backlit silhouettes, the small crops. The extractor's retry
ladder finds them, so they are now in training *and* in the test set. That is why
the pooled headline sits on a bigger denominator rather than on a worse model:
scored only on the images round 1 could also score, the shipped profile is at
**69.2 %**. Whether the recovered images also moved the *model* was measured with
a paired, seeded control (`--train-how direct_0.5`, which leaves the test rows
alone and withholds the rescued images from training only); `reports/recovered_data_cost.md`
put the effect on the average at 0.2 points in either direction, which is what a
re-roll of this pipeline costs, so the honest reading is "no measurable change".

| training rows | pooled, all 122,359 | round 1's 101,932 | rescued 20,427 |
|---|---|---|---|
| every recovered image | 65.9 % | 68.1 % | 55.4 % |
| round 1's mix only (`--train-how direct_0.5`) | 66.1 % | 68.2 % | 55.4 % |

*(that control ran on round 2's weighting; it is the control that answers the
question, so it is quoted as measured rather than re-derived).*

What the recovered hands buy is coverage, and there are **two** coverage numbers,
measured on different denominators — conflating them is the mistake an earlier
version of this file made:

- **Offline image recall: 74.0 % → 88.5 %** over 122k files. The extractor may
  spend five detections on one image because it runs once, offline.
- **Live frame coverage: 84.3 % → 98.3 %** of frames, measured frame by frame
  through the actual webcam code path in `reports/live_path.md`. The live path
  cannot afford that ladder, so `src/hand_backend.py:Recovery` retries an IMAGE
  detection at confidence 0.3 only once the VIDEO tracker has lost the hand, then
  on a mirrored copy, and only every 3rd empty frame. It is not the full ladder:
  upscale+pad is deliberately absent, and a try costs 17.9 ms (17.8 ms
  amortised), so the loop goes 103.6 → 85.0 frames/s — both sides of that clear
  the ≥20 FPS target, which is why the retry ships.

A frame with no hand is not a low score, it is no answer at all — that is the
argument for keeping both, and neither is an accuracy claim.

What did not move is the transfer ceiling, and the per-dataset row says where it
sits: **massey is 53.8 %**, by a wide margin the worst. Training on danrasband
alone scores 79.6 % on grassknoted and adding massey's 60,085 images gives 79.5 %
— 0.1 points, so "massey poisons the other folds" is not a measured effect. What
remains solid is that massey tests worst and signs differently: Stage B's contact
sheets show it signing M, N and P another way, and its landmark quality is worse.
More independent-signer data like danrasband (only 712 usable images) is the fix,
not a bigger MLP — Stage E showed that directly, and round 3 showed the other
candidate fix (teaching the model to ignore one style) does not work either.

### What round 2 changed, and what each change bought

| change | evidence | measured effect |
|---|---|---|
| detection retry ladder (lower confidence, upscale + pad, flip, mapping landmarks back through the inverse) | Stage C table in `reports/report_all.md` | recall 74.0 → **88.5 %**, +20,885 usable hands, no new labels — and **no** change in cross-dataset accuracy, which took a paired control run to see (`reports/recovered_data_cost.md`) |
| cache versioning + sidecar guard | `cache/extract_meta.json` | a stale round-1 cache can no longer be trained on silently |
| thumb-relative feature block | `reports/ablation_features.md` (2 runs per set, measured before the seed fix) | K 43.7 → 58.9 % in that sweep, and K 58.3 % in the shipped `all` model; pooled +0.2, i.e. inside the noise band — promoted for costing nothing, not for winning |
| style divergence rebuilt as cluster + median-multiple | `reports/report_all.md` Stage B | identifies exactly M, N, P instead of flagging all 24 letters; the old split-half screen was wrong and says so in the report |
| minority-style down-weight (0.5) applied by default **in round 2** | `reports/ablation_styleweight_off.md` | **zero** on the pooled number — kept then as the least destructive response to Stage B, not as a score trick. Round 3 trained it as a profile and it lost (table below) |
| capacity and augmentation swept instead of assumed | `reports/ablation_models.md`, `ablation_aug1.md`, `ablation_aug2.md` | nothing to gain: wider MLP −0.5, boosting −2.1, stronger augmentation −0.3, all inside noise |
| uncertain state + top-3 in both front-ends, one shared `frame_call()` | `live.py --logic-test` | wrong letters become refusals: 14 % of frames below the gate, 4 % near-ties go quiet instead of voting (round 5 raised the bar for four more pairs — see *What round 5 changed*) |
| feature/model width guard at startup | `normalize.feature_dim()`, `live.build_backend()` | a 71-feature stream can never silently hit a 57-feature model again |
| training seeded end to end — init, sampler *and* augmentation | two identical `--folds-only` runs now diff to zero; the profile's `.meta.json` records `seed` + `reproduce_command` | found by accident while re-running the final training: the same command had been giving 63.9 % and 64.6 %. Nothing in the score moved, but the +0.2 and −0.5 effects in Stages D and E are now labelled for what they are |
| live path verified without a camera | `reports/live_path.md` | 71 of 72 letter blocks reproduced through the real code path at a median 85.0 frames/s — wiring, not accuracy |

The M/N/T collapse is a **transfer** failure, not a capacity failure, and that is
measured rather than asserted: `python src/insample_check.py` runs the shipped
`all` model back over the exact 122,359 landmark vectors it trained on and scores
**90.5 % on N, 94.9 % on M, 95.2 % on T** (96.7 % overall) — against 0.4 %, 3.2 %
and 16.8 % in the cross-dataset column. The features do
separate those letters — the boundary learned from one set of hands lands in the
wrong place for another. `python src/insample_check.py --profile standard` shows
the other side of the same fact: with massey's M/N/P cells withheld from
training, that model scores **0.0 % on massey N in its own training data** — it is
not worse at that style, it has never seen it. Details in `REPORT.md` and
`reports/report_all.md`.

### What round 3 changed, and what each change bought

| change | evidence | measured effect |
|---|---|---|
| two shipped profiles — `all` (every style at weight 1.0) and `standard` (massey M/N/P at weight 0) — with one `--profile` flag through `train.py`, `live.py` and `app.py` | `python src/profile_compare.py`, `reports/report_all.md`, `reports/report_standard.md` | `all` **67.0 %** vs `standard` 66.0 % pooled. The down-weight did not rescue the letters it aimed at (grassknoted M 3.3 → 1.7 %, N 0.2 → 0.0 %, T 31.9 → 17.0 %), so `all` is the default and `standard` stays selectable |
| T diagnosed without changing the model | `reports/letter_diagnosis.md` | neither a style gap nor a detection-path artifact: T's cross-dataset shape gap (4.12 units) sits *below* the alphabet median (4.57), while inside each dataset its nearest neighbour is the letter it is actually answered as — L in grassknoted (crowding 0.23), A in massey (0.44). M↔N collide inside all three datasets, which is why no weight and no feature moved them |
| a bounded live retry instead of the offline ladder | `reports/live_path.md` | hand coverage 84.3 → **98.3 %** of frames at 103.6 → **85.0 frames/s** (17.9 ms per try, 17.8 amortised) — clears the ≥20 FPS target, so it ships. Upscale+pad is deliberately absent from the live path |
| the two coverage denominators separated, at the source | `coverage_sentence()` in `src/train.py`, README, `reports/report_*.md`, `REPORT.md` | offline image recall (74.0 → 88.5 % of 144,446 files) and live frame coverage (84.3 → 98.3 %) are different quantities and are now stated separately everywhere; the earlier wording that called the first one "frames" is gone from every document |
| guided live self-test, `--selftest`, saves nothing | `python src/live.py --selftest --profile all,standard` | the only measurement that involves *your* hands: per-letter hit rate, most common wrong letter, uncertain share, no-hand share, then a winner between profiles |
| `src/diagnose_letter.py` plus the `--dump-preds` OOF CSVs | `reports/nearest_shapes.csv`, `reports/thumb_audit.csv` (the 44 MB `reports/oof_*.csv` dumps were pruned in the 2026-11-15 cleanup; `oof_all.csv` was regenerated in round 5 with three more columns, `oof_standard.csv` is still rebuild-on-demand — both scripts print the one command that does it) | the whole Stage B diagnosis re-runs from CSV — no retraining, no camera, once the dump exists |

### What round 5 changed, and what each change bought

Round 5 started from a live test report — *"great model, but J, P and Z are not
correct"* — and each of those three turned out to need a different answer.

| change | evidence | measured effect |
|---|---|---|
| J/Z fail **loudly**: a held J/Z handshape whose trajectory does not fire now prints which criterion missed and by how much, instead of letting the 24-class model vote for its nearest neighbour (usually I) | `live.py --logic-test` (`hint()` is empty exactly when a rule is satisfied, and names the binding criterion when it is not) | the silent-wrong-letter case is gone from both front-ends; the message is generated from the same `_criteria()` table as the firing decision, so the explanation cannot describe a rule that did not run |
| `--calibrate-jz`: three J and three Z of *your* signing derive this session's thresholds, with per-key floors and caps | `live.py --logic-test` (a shrunken stroke that the shipped defaults miss fires at rate 1.0 after calibration, baseline 0.0) | J/Z recall is the one part of the project that depends on your style, and the thresholds were written against synthetic templates. It records landmarks in RAM for six strokes, prints a reproducible `--jz KEY=VALUE,...`, and **writes nothing** — your photos still never reach disk |
| per-pair refusal margins `PAIR_MARGINS` = `AS/GH/MN/PQ` at 0.50, chosen by replaying the live vote window over all 122,359 out-of-fold frames | `reports/oof_all.csv` (now with `p1,p2,runner_up`), the comment block above `PAIR_MARGINS` in `src/live.py` | wrong held-letter windows **13,094 → 10,905** (24.6 % → 21.5 %) for **0.9 %** of the correct ones — 6.3 wrong letters deleted per correct one lost. Getting the same drop with a flat margin costs ~4× more, so the table exists rather than a bigger default. Per letter: P 54.1 → 36.8 % wrong, Q 30.3 → 16.4 %, H, S and A also improve, and it holds inside each fold separately |

**What the margins did *not* fix, and why that is the interesting result.** P↔N,
Q↔N and M↔S were the collisions this was scoped to repair. They are not in the
table, because they are not near-ties: on unseen hands the true letter is not
even the *runner-up* in ~90 % of those frames (only 9.9 % of true-P frames have
P in second place, median gap over 0.4), so no top1−top2 rule can see them.
Raising the NP/NQ/MS margins to 0.25–0.30 moves the global wrong-window rate by
**0.1 points** — noise. The dump also says why those letters are hard rather than
ambiguous: when the model *says* M or N it is right only 9.7 % and 0.5 % of the
time out of fold (top-1 precision; their per-letter accuracy is 3.2 % and 0.4 %,
a different denominator), and still only 14.9 % and 0.0 % when it is most
confident (p1 ≥ 0.85). Hard-refusing a
top-1 of M or N would in fact be nearly free at window level, but it would make
two letters untypeable on the live model, which scores 94.9 % and 90.5 % on them
*in-sample*, so it is recorded here as a measured option and deliberately not
shipped. The fix for P, N and Q is data this model has never seen — more
independent signers — not a gate.

### What round 6 changed, and what each change bought

Round 6 started from the other half of the same live report: *J and Z do not
fire*. Round 5 made that failure explain itself; this round went after **why the
rule never even looked at those letters**, and tried the thing the README used to
name as the proper fix.

| change | evidence | measured effect |
|---|---|---|
| J/Z get a **second precondition**: the trajectory rule now also judges a frame the 24-class model reads as I (candidate J) or as an unsure X, `p1 < 0.90` (candidate Z), with the letter only allowed to fire if that reading covers ≥ 60 % of the window being judged. The shape gate is untouched and still opens both letters on its own | `python src/verify_jz_trigger.py --per 300` → `reports/jz_trigger.md`, 15,772 cached photographs over all 26 letters + 3,301 frames of real MSL movement | the old gate was **backwards**: it opened on 0.0 % of J frames and 20.0 % of Z while opening on 28.1 % of G, 20.4 % of L and 13.6 % of Q. The display precondition opens on 47.3 % of J and 94.2 % of Z and on nothing else above 8.7 %, because the model has no J and answers a real J as I (96.3 % of J frames) and a real Z as X (94.2 %). Fires moved 17 → 27 for J and **0 → 1** for Z locally, and 0 → 3 on the MSL J clips (12 → 15 across all MSL letters) — all 11 added local fires on J or Z, **the 15 pre-existing false fires unchanged**, because none of them come through the new path |
| the MSL dynamic corpus (Zenodo 14689869, 622 front-view clips of J K Ñ Q X Z) was extracted to landmarks and a motion classifier was trained on it, then **deliberately not shipped** | the numbers are in `reports/msl_motion.md`; what survives of the corpus is `cache/msl_landmarks/` (14 MB) + `msl_landmarks.npz` (11 MB) + `msl_windows.pkl` (2.9 MB), which `python src/verify_jz_trigger.py` reads. The ~10 GB archive + clips and the throwaway training scripts are deleted, so the trajectories can be re-scored but not re-extracted | leave-one-signer-out precision on MSL is J 91.9 % / Z 85.5 % (98.1 % / 93.1 % with the static head), but the same head scores **0.8 % recall for J** on this project's local ASL photographs, i.e. chance — it learned MSL hands, not fingerspelling motion. The trajectory rule stays, and `--calibrate-jz` stays the per-person fix |
| calibration and the overlay now receive the same per-frame reading the typer uses, so a Z stroke can be *recorded* at all | `python src/live.py --logic-test` — 84 checks at this round (87 since round 7), 11 of them about the display gate, including an AST check that both front-ends pass the reading (`live.py` 5/5 call sites, `app.py` 2/2) | with no reading, the shipped Z gate records 0 of the synthetic Z strokes; with one it records them and reports how much of each stroke the model supported. The app's read-out says which precondition is actually in force (`shape gate: index extension 1.12 (needs >= 1.6)` vs `display gate: the model reads this frame as X at 0.71`), instead of quoting a gate that did not open the watch |

Nothing was retrained, no dataset was added to the model, and the only new files
are the two reports. `update()` called without a reading is byte-for-byte the
round-5 rule — that is what the pre-existing 73 checks still drive, and they all
still pass.

**What this did not fix.** Z still fires almost nowhere (1 local fire, 0 on the
MSL clips): the precondition is reachable now, so what blocks Z is the shipped
zig-zag thresholds themselves, which were written against a synthetic stroke.
That is a `--calibrate-jz` question, and for the first time that command has a
gate wide enough to measure a real Z through. The 15 false fires on G/L/Q/T/X are
also untouched, and are a shape-gate question, not a display one.

> Numbers in the table above are the round-6 measurement and are kept as the
> decision was made on them. Both have since been re-measured: the per-letter
> roll-up in `reports/jz_trigger.md` is now frame-weighted (the 47.3 % J figure
> was a mean of per-dataset means, and over the same frames the round-6 rule
> opened on 23.6 %), and round 7 then removed the confidence ceiling from the I
> route, so the gate as it ships now opens on 93.3 % of J and 89.4 % of Z. The
> current numbers are always the regenerated report's, never this table's.

### What round 7 changed: J was still hard after calibrating

Round 6 shipped, the app was tested on a real camera, and the report back was
"everything is fine now but J is a little difficult" - with the read-out saying
`J/Z: neither gate met` *after* a guided calibration. That message can only
appear when no frame in the process has ever opened either precondition, which
pointed at the precondition rather than the thresholds calibration had just
loosened, and the measurement agreed.

| change | evidence | measured effect |
|---|---|---|
| the display precondition's confidence ceiling is now **per candidate**: `DISPLAY_P1_MAX_FOR = {'J': None, 'Z': 0.90}`. A frame the model reads as I is a J candidate at any confidence; a frame it reads as X still has to look unsure | `python src/verify_jz_trigger.py --per 300` → the new `refused by the OLD flat ceiling` column, plus a sweep of 25 MSL J clips scored with the shipped `MotionDecider.criteria()` | on the cached photographs 94 % of real J frames are read as I at median p1 0.95, and **72.7 % of all J frames sat above the old flat 0.90** - the precondition refused them before the trajectory was ever looked at. It opens on 93.3 % of J now (was 23.6 %). Z is untouched by design: only 0.6 % of its frames were ever refused by the ceiling, because a real Z is read as X at median p1 0.50, which is genuinely unsure. On real movement (MSL J clips) the precondition goes 48.0 % → 53.7 % of frames and J clips firing 1/25 → 2/25, with the false fires unchanged at 12 |
| the report's per-letter percentages are **frame-weighted**, and the stale "touched no false fire" sentence is gone | same regeneration; the roll-up no longer averages dataset rows | it reported the J display gate as 47.3 % when 30 frames from one source and 300 from another were being counted equally. The honesty matters in both directions: the table now shows non-J/Z local fires went **15 → 38, all of it on the I row (0 → 23)**, which is the price of the exemption, and says plainly that photographs cannot settle it because consecutive cached frames are different hands |

The reason the exemption is not simply "now everything types J" is that J and I
are the same *frame*; only the *movement* separates them, and that is the
trajectory rule's job. A held I has no wrist reversal, no net travel and no
path, and `live.py --logic-test` now pins exactly that - a synthetic I held at
p1 0.99 types nothing, while the same confident reading under a sideways hook
types J. What photographs cannot test, and what only the camera can, is whether
an I that arrives with a back-and-forth wobble types a J; `--calibrate-jz` can
tighten nothing (it only loosens), so if that turns out to happen the fix is a
trajectory threshold, not the ceiling.

Nothing was retrained and no threshold moved: `WINDOW`, `MIN_HOLD`, `COOLDOWN`,
`SEG`, `J_*` and `Z_*` are exactly as round 5 shipped them, and `--calibrate-jz`
still cannot reach the display constants at all. The check count went 84 → 87.

### What round 8 changed: the webcam opened, lagged, and closed itself after 10 s

Reported from a real camera: press **START**, the picture appears for about a second,
the page lags, the camera releases, STOP turns back into START, and a retry ends with
an error box on a half-white page.

`app.py` had pointed **both** ends of the peer connection at Google's public STUN
server (`stun:stun.l.google.com:19302`). On the network this app is used on that
server is never answered — the app's own debug log shows BINDING requests
retransmitted at 36.1 / 37.2 / 38.6 s after the click, then `Stopping WebRTC worker`
at 39.8 s. Unanswered STUN stalls ICE gathering, so the SDP answer never finishes, so
`streamlit_webrtc`'s `process_offer(..., timeout=10)` kills the worker (camera
released) and raises `SignallingTimeoutError` (the error on the page).

| change | measured effect |
|---|---|
| ICE is **host-only on both ends** by default (`rtc_pair(False)` → `{"iceServers": []}` for aiortc *and* the browser), with a sidebar **"Use a public STUN server"** checkbox to bring STUN back on both sides at once | two local aiortc peers with the shipped config: gathering **5 ms**, SDP answer **9 ms**. With Google's server on the same machine: gathering **5 009 ms**, answer **10 035 ms** — over the 10 000 ms budget, which is the crash reproduced with no camera and no browser |
| the camera is asked for 24 fps and **"Work every Nth frame" defaults to 2** | one processed frame costs ~50 ms here, so a 30 fps stream is 33 ms of budget against 50 ms of work; stride 2 is ~26 ms. MediaPipe VIDEO is flat in resolution (45.7 ms at 240 px, 49.9 ms at 1280 px), so this is the only lever — and the ceiling stays ~20 fps |

The cost of stride 2 is temporal: the J/Z rule sees 12 hand positions a second instead
of 24, and its binding criterion on real movement is the reversal count, so the slider
help says to return to 1 (or re-run `--calibrate-jz`) if J or Z stops firing.

Nothing else moved — the model, the features, the gate and every threshold are as
round 7 shipped them: 87 logic checks, 70/72 live-path blocks, and a headless boot of
the edited app that logs **no STUN traffic at all**.

### Head-to-head against the photo model, on data neither model had seen

The photo model (a MobileNetV2 image classifier) was removed from the repo on
2026-11-23; the numbers below are kept as historical evidence. Both models scored
on the **same held-out image lists neither had trained on**,
with the photo model's own 29-class output renormalised over the 24 letters
(so nobody wins by predicting "space"/"del"). The landmark fold was trained on
the two datasets the test set came from — the same leave-one-out protocol used
for the headline number. The photo model's advertised 93.25 % is a
same-distribution split (mostly synthetic renders) and is not comparable.

| held-out set (both models blind) | landmark MLP | photo MobileNetV2 |
|---|---|---|
| Massey, 1,440 images (60/letter) | **56.2 %** | 17.5 % |
| danrasband ASL test set, 683 images | **93.3 %** | 58.4 % |

*Dated 2026-11-13: these two rows were measured with the round-1 model and the
round-1 cache (57 features, 74 % detection recall), and have not been re-run
since. The margin is ~40 and ~35 points, so a 2-point move in the landmark
number changes the conclusion for neither.*

Landmark wins are largest exactly where the photo model blurs internally
similar shapes (F +95, L +95, B +92, A +88, E +83 points); the photo model is
ahead only on N and P — the same two letters the cross-dataset table flags as
this pipeline's weakest. The raw per-letter logs behind those numbers
(`reports/head_to_head_massey.log`, `head_to_head_danrasband.log`) were pruned
in the 2026-11-15 folder cleanup; the surviving per-letter rows are the ones
quoted above, and re-running the comparison needs both models' own eval scripts
from outside this folder.

---

## Privacy / what this project does NOT do

- **No tool here records, saves, or trains on your webcam images.** `src/live.py`
  and `app.py` read frames, draw on them in RAM, and throw them away. There is no
  `imwrite`, no `VideoWriter`, no landmark dump, no log file anywhere in the live
  path. `src/live.py` prints `nothing was written to disk` when it exits, and that
  is literal — see *Verifying the no-save claim* for the two commands that check
  it.
- The J/Z calibration is the only thing in the live path that *remembers*
  anything, and what it remembers is numbers: three J and three Z of wrist/tip
  positions as floats in `MotionDecider.strokes`, capped at 40 strokes, cleared
  when the process ends. It exists so `--calibrate-jz` can print the thresholds it
  derived, and it has no save path — the removed photo model's
  `calibrate_camera.py` (it photographed your hand onto disk) was deliberately
  not ported.
- Training data comes only from the public datasets listed below, which are read
  **in place** from the kagglehub cache.
- Input images are converted **BGR → RGB and stay colour**. `hand_backend.bgr_to_rgb()`
  raises instead of accepting a grayscale frame, because MediaPipe's palm model
  was trained on colour images.
- J and Z are **not** in the 24-class model. They are moving letters, so a
  single frame carries almost no information about them; see *J and Z* below.

### Verifying the no-save claim

You do not have to take the sentence above on trust — two greps, both read-only:

```bash
grep -n "imwrite\|VideoWriter\|to_csv\|open(" src/live.py src/hand_backend.py src/normalize.py
grep -rn "imwrite\|VideoWriter\|save\|to_csv\|open(" app.py
```

The first prints **nothing**: the live path contains no write call, and no `open`
at all. In `app.py` the only `Image.open()` calls decode bytes the browser has
already handed over (the upload widget, and `st.camera_input`'s own in-memory
buffer — shown back to you, never written), the only other matches are help
strings and the label of that selfie-mirroring checkbox.

The files this project *does* write are all offline-stage outputs, and none of
them can ever receive a webcam frame: `train.py` (weights, `.meta.json`, the
profile report, the confusion figure, and `--dump-preds` when you name a CSV),
`export.py` (the ONNX copy), `extract_landmarks.py` (the landmark cache) and
`download_data.py`.

A third check is behavioural, and it is the one that would catch a path a `grep`
misses: the round-5 verification battery walks the whole folder before and after
driving `frame_call`, the smoother, the motion rule, the calibrator and `app.py`
over real dataset hands, and fails if any file in the project was created,
changed or touched. It passed — 20 battery checks on top of the 64 logic checks,
with 533 of 576 real frames still casting a vote and exactly one frame whose
verdict the table changed.

## Requirements

Python 3.11+ with the pinned packages. Install the merged file from the
repository root (the folder that contains it):

```bash
python -m pip install -r requirements.txt
```

Tested on **Python 3.13.15, CPU only, Windows**, with
`mediapipe==1.2.2 opencv-python==5.0.0.93 numpy==2.5.3 torch==2.14.1`.

Two deliberate deviations from the original plan, both forced by the machine
this was built on — recorded here rather than hidden:

1. **`numpy<2` is not installable here.** Only Python 3.13/3.14 is present, and
   numpy 1.26 has no cp313 wheels; mediapipe 1.2.2, opencv 5.0 and torch 2.14 all
   require numpy ≥ 2 anyway. Nothing in this codebase depends on numpy-1
   behaviour.
2. **The legacy `mp.solutions.hands` API no longer exists** in mediapipe ≥ 1.0,
   so `src/hand_backend.py` uses the Tasks API (`mediapipe.tasks.python.vision.
   HandLandmarker`). The mapping is exact: the brief's
   `static_image_mode=True` → `RunningMode.IMAGE` for still images (dataset
   extraction) and `RunningMode.VIDEO` with an explicit timestamp for the
   webcam loop. `max_num_hands=1` → `num_hands=1`, and
   `min_detection_confidence=0.5` → `min_hand_detection_confidence=0.5` plus
   `min_hand_presence_confidence=0.5`.

You do **not** need a Kaggle account if the datasets are already in the
kagglehub cache (they are on this machine). If one is missing and must be
downloaded, `src/download_data.py` stops with a four-line instruction instead of
a stack trace:

1. <https://www.kaggle.com/settings> → *Create New Token* (downloads `kaggle.json`)
2. put it in `C:\Users\<you>\.kaggle\kaggle.json`
3. re-run `python src/download_data.py`

## Datasets used

| id | source | role | licence | what it gives us |
|---|---|---|---|---|
| `grassknoted` | [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | primary | GPL 2.0 | 104 000 files, 4 000 per letter, all of them real 200×200 JPEG photographs of a hand at arm's length (checked in Stage B — the widely repeated claim that most of them are "synthetic MNIST-style renders" is wrong: ~13 000 unique colours per image, noisy borders, no flat background). What *is* true is that a subset are badly underexposed backlit silhouettes, which is where most of our detection failures come from |
| `massey` | [mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out](https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out) | cross-dataset | not stated on the page (Massey University academic set) | 65 774 colour crops, 24 letters × 5 capture groups, different hands/camera/room from grassknoted |
| `danrasband` | [danrasband/asl-alphabet-test](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test) | extra holdout | CC0 | 780 real 200×200 photos, 30 per letter |

Deliberately **not** used: *Sign Language MNIST* — 28×28 grayscale, so
MediaPipe cannot find a hand in it and there are no landmarks to normalise.

One further corpus is read for **measurement only, never for training**: the
Mexican Sign Language dynamic front-view set (Zenodo 14689869, CC-BY-4.0, 622
clips of real continuous strokes covering J K Ñ Q X Z, 18 train signers + 2
disjoint test signers) — the only data here with genuine J and Z *movement*, used
to answer one question: can the J/Z precondition see motion at all, and does a
model trained on real motion transfer to ASL. It was unpacked (2.1 GB archive +
8.1 GB of mp4s), run through MediaPipe VIDEO to 50,022 cached landmark frames,
and then the archive and the clips were deleted with the user's approval. What is
left is `cache/msl_landmarks/` (14 MB), `cache/msl_landmarks.npz` (11 MB),
`cache/msl_landmarks.csv` and `cache/msl_windows.pkl` (2.9 MB), which is what
`python src/verify_jz_trigger.py` reads. It is a different language, it changed no
shipped accuracy number, and the head trained on it is documented as **rejected**
in `reports/msl_motion.md` (0.8 % J recall on local ASL). Nothing from it reaches
the classifier's training set.

Signer id / dataset / folder id are stored per sample, which is what makes the
honest splits possible.

> **Massey folder layout, because it is a trap.** The outer folder (`A`–`E`) is
> *not* the letter — it is one of five capture groups. The inner folder (`a`–`y`,
> no `j`, no `z`) is the letter. Confirmed on 360 sampled files: the `n` in
> `color_<n>_<frame>.png` equals `ord(letter) - ord('a')`, and it always matches
> the *inner* folder. Getting this backwards turns cross-dataset accuracy into
> exactly chance, which is how we caught it.

## The pipeline, stage by stage

| script | what it does | output |
|---|---|---|
| `src/download_data.py` | resolves every dataset from the kagglehub cache (no re-download if present), prints size + licence + per-letter counts, copies the hand model into `models/` | `data/datasets.json` |
| `src/extract_landmarks.py` | runs MediaPipe Hands over the images (12-worker `ProcessPoolExecutor`). Round 2's **detection recovery ladder**: any image that fails at the brief's confidence 0.5 is retried at 0.3, then upscaled to a ≥480 px short side with a 12 % replicated border, then from a horizontally flipped copy, and the winning attempt's landmarks are mapped back to the original frame before storing. Every row records which attempt found it (`how`). Frames with no hand at all are dropped and counted per class/dataset | `cache/landmarks.parquet`, `cache/skipped_stats.csv`, `cache/extract_meta.json` |
| `src/inspect_data.py` | the Stage B diagnosis: contact sheets of 12 sampled images per dataset per letter with the MediaPipe skeleton drawn on top, the between-dataset mean-shape distance for every letter against the split-half distance inside a dataset, the per-dataset mirror rate, and a provenance check on the image files themselves | `reports/samples/contact_*.png`, `cache/divergence.csv`, and with `--nearest` / `--thumb-audit` the per-dataset shape-collision table (`reports/nearest_shapes.csv`) and the thumb-landmark sanity table (`reports/thumb_audit.csv`) |
| `src/verify_livepath.py` | tests the **live pipeline end to end without a camera**: real public-dataset photographs are read BGR→RGB exactly as the webcam path reads them, pushed through `hand_backend.detect_video` → `normalize.features` → `live.frame_call` → `live.Smoother` at a simulated 20 FPS, and the committed letter per clip is compared with the label. Nothing is saved except the report. Privacy-safe by construction: it never opens a capture device | `reports/live_path.md` |
| `src/ablate_features.py` | the Stage D feature ablation: re-runs the leave-one-dataset-out folds once per feature set in a fresh subprocess (`ASL_FEATURE_BLOCKS`), which is the same code path as the headline number, and writes one comparison table | `reports/ablation_features.md` |
| `src/insample_check.py` | the transfer-vs-capacity check: runs one **shipped** profile's `.pt` (default `models/asl_all.pt`, `--profile` to choose) back over the exact landmark vectors it trained on and prints per-letter in-sample accuracy, so a letter that is 0.4 % cross-dataset and 90.5 % here is provably a data problem, not a feature problem. Refuses to run if the sidecar's feature width and `normalize.feature_dim()` disagree. Reads only, writes nothing | stdout |
| `src/ablate_models.py` | the Stage E capacity check: the same folds for MLP 128-64, MLP 256-128 and HistGradientBoosting, plus single-frame latency so a winner that cannot hold 20 FPS is visible as one | `reports/ablation_models.md` |
| `src/verify_jz_trigger.py` | the round-6 J/Z measurement: replays the cached photographs (all 26 letters, oldest path first) and the cached MSL dynamic trajectories through `MotionDecider` twice — once as round 5 shipped it (shape gate only), once with the display precondition — and reports, per letter, how often each gate opened, what the classifier displayed that frame as, how often the window majority allowed J or Z, and how many times each rule actually fired. Reads only; never opens a camera | `reports/jz_trigger.md`, `reports/jz_trigger_local.csv`, `reports/jz_trigger_msl.csv` |
| `src/normalize.py` | **the** shared normaliser: wrist at origin, divided by the wrist→middle-MCP bone length, hands in the other 2-D chirality family mirrored to the canonical one (see `chirality_family`), then the feature vector | imported, not run |
| `src/augment.py` | landmark-only augmentation (rotation, 3-D tilt, scale, noise, landmark dropout, mirroring) | imported, not run |
| `src/train.py` | class-balanced MLP with early stopping + the whole honest evaluation, for one `--profile` | `models/<profile>.pt`, `models/<profile>.meta.json`, that profile's report, `reports/confusion_matrix*.png`, and `--dump-preds` writes the leave-one-dataset-out prediction of every cached image |
| `src/live.py` | the webcam app with smoothing and the `--selftest` guided drill | nothing on disk |
| `app.py` | the Streamlit browser UI: live webcam, photo upload, camera snapshot, results tables — imports `live.py`'s backend and smoother classes so the two front-ends cannot disagree | nothing on disk |
| `src/export.py` | ONNX export + parity check against PyTorch, for one `--profile` | `models/<profile>.onnx` |
| `src/profile_compare.py` | round 3 Stage C: builds the paired profile table straight out of the dumped OOF predictions — no retraining, no re-scoring | stdout |
| `src/diagnose_letter.py` | round 3 Stage B: for the letters you name, prints what they get confused with in each held-out dataset, splits their accuracy by the detection path that found the hand, and pulls in the within-dataset shape-collision and thumb-landmark tables — all from files already on disk, so a camera is never involved | `reports/letter_diagnosis.md` with `--out` |
| `run_all.py` | runs the stages above as subprocesses | — |

Useful flags:

```bash
python run_all.py --quick                 # small smoke run, all stages
python run_all.py --force-extract         # rebuild the landmark cache
python run_all.py --skip-download --skip-extract   # retrain only
python run_all.py --epochs 120 --live     # longer training, then open the camera
python src/train.py --max-rows-per-ds 2000 # cap per dataset (speed)
python src/train.py --folds-only --no-save  # just the honest score, writes nothing
python src/train.py --folds-only --no-save --train-how direct_0.5
        #   ^ narrow the folds' TRAINING rows to one detection path while leaving
        #     the test rows alone. Used to separate "we trained on 20,885 harder
        #     images" from "we changed the features": it is the only way to replay
        #     round 1's data mix against today's code.
python src/train.py --style-weight 1.0     # off: Stage B's divergent cells keep weight 1
        #   ^ since round 3 this is not a global default any more — the *profile*
        #     decides it (`all` = 1.0, `standard` = 0, `round2` = 0.5). Leaving it
        #     out means "use the profile's value"; giving it explicitly still wins.
        #     The measured effect on the pooled number is zero either way.
python src/train.py --folds-only --no-save --dump-preds reports/oof_all.csv
        #   ^ the fastest way to rebuild the prediction dump: `--folds-only` skips
        #     the final full-data model, and since round 5 an explicit
        #     --dump-preds is honoured even under --no-save (you named the file).
python src/verify_livepath.py --per 20      # whole live path, real frames, no camera
python src/verify_jz_trigger.py --per 300   # what reaches J and Z, per letter, no camera
python src/train.py --seed 7                # a different draw, same data
        #   ^ since 2026-11-14 this seeds everything random (weight init, sampler,
        #     augmentation), so a plain re-run is bit-identical. It is not a knob to
        #     shop for a better score: seed 0 is what ships, and it was picked
        #     before the fix existed, i.e. it was never chosen for its number.
```

### Normalisation (step 3) — one implementation, used everywhere

`src/normalize.py::features()` is the only feature builder in the project.
`src/train.py`, `src/export.py` and `src/live.py` all call it, so training and
inference cannot drift apart. There is also a batch version,
`features_from_normalized_batch()`, used for speed inside the training loop — it
is asserted to produce bit-identical output to the single-sample path.

For each detected hand:

1. subtract the wrist (landmark 0) from all points;
2. divide by `‖wrist → middle-finger MCP (landmark 9)‖` (with an epsilon guard),
   so the pose is scale-invariant;
3. canonicalise chirality. MediaPipe's `handedness` label assumes a *mirrored*
   (selfie) image, which raw OpenCV frames are not, so trusting it silently
   flips half the dataset. Instead we use the sign of the 2-D cross product of
   `wrist→index-MCP` × `wrist→pinky-MCP`, and mirror when it is positive.
   `mirror_hand()` only negates *x*: landmark ids are anatomical, so a flipped
   hand keeps its ids.
4. build the feature vector from blocks. The default is `xy` (42) +
   `tip_dists` (10 pairwise fingertip distances) + `curl` (5 finger-curl
   angles) + `thumb` (14) = **71 dims**, chosen by the Stage D ablation.
   Three further blocks exist and are compared in
   `reports/ablation_features.md`:
   - `xyz` — 63 dims including MediaPipe's z. Documented as being on an
     *arbitrary* scale, and measured across our datasets the palm-width /
     wrist-MCP ratio is 0.67 for grassknoted and danrasband but 1.05 for
     Massey, so it does not transfer.
   - `thumb` — 14 thumb-relative features: the thumb tip's distance to the
     index/middle/ring MCP and PIP joints, to the pinky MCP, to the palm
     centre and to the other four fingertips, plus two *signed* perpendicular
     offsets of the thumb tip from the knuckle line and the fingertip line.
     That last pair is the "thumb across the palm vs thumb outside" relation,
     which is what A/S/U and P/Q/R differ by and which no distance captures.
   - `orient` — 8 image-plane orientation features: the unit direction of
     `wrist→middle-MCP`, the unit direction of the knuckle line, their dot and
     cross products, the pinky-MCP/index-MCP length ratio and the sine of the
     palm opening angle (plus the 3-D unit palm normal when `xyz` is on).
     The hand is deliberately **not** rotation-normalised: P and K, and Q and
     G, are the same finger pattern at a different wrist angle, so the angle is
     signal. These features only make it linearly available instead of forcing
     the MLP to re-derive sin/cos from 42 raw numbers.

   Set them with `ASL_FEATURE_BLOCKS=xy,tip_dists,curl,thumb,orient` (the
   ablation scripts do exactly that, in a subprocess, so the sweep uses the
   identical code path).

   **Why `thumb` is on and the other two are not.** Every set in the sweep was
   run twice, `--seed` changed, which put the run-to-run noise band on the pooled
   number at roughly ±1–2 points — so a +0.2 gain is *not* evidence of a better
   model and was never claimed as one. `xyz` costs 1.4 points (outside the band,
   and it contradicts the measurement above: it hands the model a per-dataset
   fingerprint). `orient` buys nothing on the headline and costs K 6.5 points.
   `thumb` is the only set that is not worse on any watched letter, and the number
   that carried the decision was K 43.7 → 58.9 % — **with the caveat that
   per-letter noise in the same table reaches 14 points and that sweep ran before
   the seed fix, so the pair is a lead, not a measurement.** The half of it the
   shipped model can check is the destination: K is 60.9 % pooled cross-dataset
   here, consistent with 58.9 %. Re-running the five sets under the pinned seed is
   a round-3 job. The block is kept on the rule the brief set — best pooled set
   unless it costs a strong letter more than 3 points — and for the structural
   reason: the thumb is the one digit whose relationship to the other fingers
   carries a letter, and fingertip-to-fingertip distances cannot express it.
   `normalize.feature_dim()` reads the width back off the real
   function, and `live.build_backend()` refuses to start if the model on disk
   expects a different width, so a feature change can never silently meet a
   stale model.

### Training and the honest score (step 4)

Model: `71 → 128 → 64 → 24`, BatchNorm + Dropout 0.25, label smoothing 0.05,
AdamW, `WeightedRandomSampler` class balancing, ReduceLROnPlateau, early
stopping on a **blocked** validation slice. The width was checked, not assumed:
`reports/ablation_models.md` runs the same folds at 256-128 and against
HistGradientBoosting, and both land inside the noise band (−0.5 and −2.1
points), i.e. the learner is not the limit. Augmentation was measured the same
way — strength 2.0 against the default 1.0 gives 65.5 % vs 65.8 %, so it was
left alone (`reports/ablation_aug1.md`, `ablation_aug2.md`).

Evaluation policy — this is the part that matters:

- **Primary = cross-dataset.** Leave-one-dataset-out (train on two, test on the
  third) *and* every pairwise direction, both reported. A random split is never
  presented as the score.
- **Held-out capture group** inside Massey (train on groups A–C, test on D–E):
  unseen hands in a matched imaging style — the closest available proxy for your
  webcam.
- **Contiguous-block split** for the photo datasets, clearly labelled *OPTIMISTIC
  in the profile's report*. Blocks are rank buckets inside each class run, so no whole
  letter is ever held out (which is what made an earlier version of this split
  report a meaningless 0 %).

Augmentation is deliberately strong (±15° in-plane rotation, ±10° 3-D tilt, scale
0.9–1.1, Gaussian noise, 30 % landmark dropout replaced by the finger-chain mean,
random mirroring) because there is no self-recorded data to make up for.

**Reproducibility.** `train_one()` seeds three things now: the augmentation stream
(numpy), the layer initialisation (`torch.manual_seed`) and the
`WeightedRandomSampler` (its own `torch.Generator`). It used to seed only the
first, and the other two drew from torch's unseeded global RNG — so **running the
same command twice gave two different models**: pooled cross-dataset 63.9 % and
64.6 % from one identical pair of invocations. That 0.7-point swing is larger than
the +0.2 the thumb-feature promotion was decided on, so it is not a footnote: the
fix is in, and `python src/train.py` is now bit-for-bit repeatable on this machine
(verified by running `--folds-only` twice and diffing the two
`ABLATION_RESULT` lines). Each profile's `models/<prefix>.meta.json` records
`seed` and the exact `reproduce_command`, and torch here is CPU-only, so there is
no GPU nondeterminism left to blame. Two rules follow, and both are applied in
`REPORT.md` and the profile reports: any gap under ~1 pooled point is written as
"no measurable
difference", and every ablation table says plainly that its rows were measured
before the fix, so the digits are one draw each even though the conclusions were
chosen to survive that.

### Live app (step 5)

```bash
python src/live.py                       # torch model
python src/live.py --backend onnx        # exported model, usually a bit faster
python src/live.py --conf 0.6 --hold 0.35
python src/live.py --margin 0.05         # commit on closer calls (riskier)
python src/live.py --pair-margins none   # drop the measured per-pair margins
python src/live.py --profile standard    # which trained model to load
python src/live.py --calibrate-jz        # 3 J + 3 Z of your signing -> this session's J/Z thresholds
python src/live.py --jz J_PATH=0.62      # replay one of those sessions (printed at its end)
python src/live.py --selftest            # guided self-test on YOUR camera (below)
python src/live.py --logic-test          # smoothing logic checks, no camera
python src/live.py --benchmark 300       # FPS on in-memory frames, no camera
python src/live.py --selftest --profile all,standard   # both models, then compare
```

Keys: `SPACE` commit the held letter now (no wait), `BACKSPACE` delete,
`C` clear the buffer, `Q` quit. The overlay shows the letter, the **top-3
letters with their probabilities**, the hold-progress bar, live FPS, the text
typed so far, and — when a frame refuses to vote — the reason it refused.

Smoothing is a small state machine (`Smoother` in `src/live.py`), verified by
`--logic-test` (64 checks):

- a frame must be ≥ `--conf` (default 0.70) to vote at all;
- and its top1−top2 gap must be ≥ `--margin` (default 0.10). Failing the gap
  test draws **UNCERTAIN** and casts no vote, because two nearly tied letters
  means one finger is being read differently from frame to frame — exactly the
  N/S, K/V and T/L cases the confusion matrix showed. Failing only the
  confidence test silences the frame *without* the uncertain label, and that
  separation is deliberate: with 24 classes a genuine 45 %/40 % tie is *also*
  under a 0.70 gate, so an "uncertain" state keyed to confidence alone would
  essentially never fire. Both behaviours are asserted in the selftest;
- for the four pairs the out-of-fold gaps say 0.10 is not enough (`AS`, `GH`,
  `MN`, `PQ` — `PAIR_MARGINS`), the required gap is raised to 0.50 *for that
  pair only*, and the frame says which rule bound it
  (`no vote: only 47 pts ahead of S - A/S needs 50`). The rule is a `max()`,
  never a `min()`: a pair entry can only refuse more, and `--pair-margins none`
  turns the whole table off so you can A/B it on camera;
- the letter shown is the majority of the last `--vote` (default 7) frames;
- a letter must be held for `--hold` (default 0.5 s) before it is typed;
- the same letter cannot be typed twice in a row — you must change shape or
  drop the hand, otherwise "HELLO" would come out as "HELO";
- no hand for a moment resets the vote and releases the repeat lock;
- average FPS is printed on exit;
- the two chirality checks added in round 4: `hand_backend.swap_label()` is the
  one definition of "the same hand in a flipped image" (the live `flip_0.3`
  rung now uses it, so it can no longer disagree with the extraction ladder),
  and `normalize.chirality_family()` flips with `x -> 1-x` and is undone by
  `canonicalize()`;
- the J/Z rule and its calibration (rounds 5, 6 and 7): `MotionDecider.hint()`
  is empty exactly when one of the two trajectory rules is satisfied, and
  otherwise names the criterion furthest from firing — so a held J shape can no
  longer fall through to the static model's nearest neighbour. `--calibrate-jz`
  moves the stroke thresholds only inside their own floors and caps, and never
  past the shipped default in the strict direction. Round 6 added the second
  precondition (the frame's own top-1 reading: I → candidate J, an unsure X →
  candidate Z, ≥ 60 % of the judged window in agreement) and, because a
  precondition that is not in force must not be the one described, the read-out
  now names which gate opened the watch. Round 7 removed the confidence ceiling
  from the I → J route, because a confident I is what a J frame looks like; the
  X → Z route keeps it. Both front-ends pass the reading, and an AST check over
  the real source files fails the test if either stops. See *J and Z*.

`app.py` calls the same `live.frame_call()` for its decision, so the browser
and the console window cannot disagree about a frame.

### Model profiles (`--profile`)

One command trains one profile; the name is a short recipe stored in
`src/config.py:PROFILES`, and every front end (`live.py`, `app.py`,
`verify_livepath.py`, `export.py`) resolves the weights through it, so nothing
can quietly run a different model than the one you asked for.

| profile | what it trains on | files |
|---|---|---|
| `all` | every dataset and every signer at weight 1.0 | `models/asl_all.pt` / `.onnx` / `.meta.json` |
| `standard` | the same, except the flagged minority-style cells (massey M/N/P) at weight 0 | `models/asl_standard.*` |
| `round2` | those cells halved (0.5) instead of removed — the model shipped before round 3 | `models/asl_mlp.*` — **pruned 2026-11-15**, retrain with `python src/train.py --profile round2` to get it back |

`--dump-preds` writes the leave-one-dataset-out prediction for every cached
image, which is what the per-letter, per-dataset and per-detection-path tables in
the profile reports — and `python src/diagnose_letter.py --letters T,N,M` — are
computed from (`python src/insample_check.py --preds
reports/oof_all.csv --confuse T,N,M --by-path`).

`reports/oof_all.csv` is on disk (~23 MB). Since round 5 each row also carries
`p1`, `p2` and `runner_up` — the top score, the runner-up score and which letter
that was — because that is exactly what `frame_call()` can see, and it is the
measurement `PAIR_MARGINS` was derived from. `reports/oof_standard.csv` (the same
for the other profile) was pruned in the 2026-11-15 folder cleanup; rebuilding
either one is a full retrain of that profile, and reproduces the shipped numbers
because the seed is recorded in the profile's `.meta.json`:

```bash
python src/train.py --profile all --dump-preds reports/oof_all.csv
python src/train.py --profile standard --dump-preds reports/oof_standard.csv
```

Add `--folds-only` to either command to skip the final full-data model — you only
want the held-out predictions, and `--dump-preds` is honoured under `--no-save`
too because you named the file.

Nothing else in the folder needs them: `live.py`, `app.py`, `insample_check.py`
without `--preds`, the ONNX export and both shipped models all work as they are.

The two profiles measure different things, so their headline numbers are not a
league table: in `standard` the held-out massey fold has had its M/N/P removed
from training on purpose, so massey's M/N/P cells there are expected to look
worse, and the letters it was meant to rescue (in the *other* datasets) are the
ones to read.

### Live self-test (`--selftest`, nothing saved)

Every other number in this project comes from other people's recorded hands.
It is the only measurement of *your* signing, and it is the thing that decides
which profile to use. (`--drill` is the same command, kept as an alias.)

```bash
python src/live.py --selftest                          # 24 static letters, default profile
python src/live.py --selftest --letters M,N,T          # just the problem letters
python src/live.py --selftest --profile all,standard   # both, back to back, then compare
```

For each letter: the letter and a countdown appear on the video, you hold the
sign for ~3 s (tune with `--drill-hold`, prep time with `--drill-countdown`), and
the app counts the raw per-frame top-1
prediction while you hold. It then moves on. At the end it prints, per profile:

| column | meaning |
|---|---|
| frames | frames counted during the hold |
| hand% | share of them where MediaPipe found a hand at all |
| hit% | share whose top-1 was the letter you were signing |
| below-gate% | share under the confidence gate (would not vote) |
| uncertain% | share whose top1−top2 gap was under the margin that applied to them — `--margin`, or the pair's own raised value when `PAIR_MARGINS` covers that pair |
| wrong | the most common wrong letter, if any |

and, when more than one profile ran, a side-by-side table of hit rates with the
winner per letter. Keys: `Q` stop and print the report, `S` skip this letter,
`SPACE` redo it.

**Which profile to choose.** `all` is the default because it is the better model
on *everyone's* data — 67.0 % against 66.0 % pooled cross-dataset — and because
removing the minority style did not rescue the letters it was meant to rescue.
`standard` is there for one specific situation: you sign M, N and P the way the
massey signer does, and you would rather the model learned that shape cleanly
than averaged against the other two. No dataset table can tell you which of those
you are, because none of them contains your hands. Run

```bash
python src/live.py --selftest --letters M,N,T,P --profile all,standard
```

once and use whichever profile printed the higher hit rate for you. That
comparison measures one person's signing, so for that person it outranks the
pooled table. Until you run it, use `all` — and to make it stick, set
`ASL_PROFILE=standard` in the environment or pass `--profile standard` (the
browser app has the same choice in its sidebar).

Privacy: the self-test reads the webcam frame by frame and keeps only counters in
memory. It writes no frame, no video, no landmark and no log — there is no code
path in it that opens a file for writing.

### Browser app (Streamlit)

```bash
streamlit run app.py
```

Four tabs:

- **Live Webcam** — the same `Smoother` state machine as the console app,
  running inside a `streamlit-webrtc` video handler. Shows the held letter with
  a hold-progress bar, the typed text, top-5 probability bars, the **J/Z watch**
  readout — which precondition is in force this frame (`shape gate: index
  extension 1.12 (needs >= 1.6)` or `display gate: the model reads this frame as
  X at 0.71`) plus the live path/reversal/drop numbers — and, when a J or Z is
  watched but no trajectory fired, the reason
  (`J held but not fired: net travel 0.11 needs 0.30`). When a frame is refused a
  vote the overlay says why, in two different places: the big tag becomes
  **UNCERTAIN**, and under the top-3 bars it prints
  `no vote: only 4 pts ahead of S` for a near tie,
  `no vote: only 47 pts ahead of S - A/S needs 50` when a measured pair raised the
  bar, or `no vote: under the 0.70 gate` for low confidence — the reasons are
  never collapsed into one message, and the number it prints is the one the frame
  actually had to clear. `Calibrate J / Z` runs the same guided six-stroke
  session as `--calibrate-jz` (progress in the sidebar, then an expander showing
  what changed and what each stroke measured); it stops recording if you cancel.
  The frame loop is a `st.fragment(run_every=0.2)` so the page refreshes ~5×/s
  without re-running the whole script.
- **Upload Photo** — one frame through the *identical* MediaPipe → normalise →
  MLP path, reporting the letter, confidence, top-5 bars, and whether the hand
  was mirrored to the canonical chirality.
- **Camera Snapshot** — grabs the current webcam frame into memory and runs the
  photo path on it. The frame is never written anywhere; it exists for the same
  reason `live.py`'s overlay does — to see what the pipeline actually saw.
- **Model & Results** — parses the selected profile's report (`reports/report_<profile>.md`) and shows the honest scores, the
  three split tables, a per-class accuracy chart, plus buttons that shell out to
  `live.py --logic-test` / `--benchmark` and one that shows `reports/live_path.md`,
  the A–Z run of the real frames through the webcam code path. It *displays* that
  report rather than re-running it, because the run takes a couple of minutes of
  MediaPipe and would block the page; the button says so and gives the command.

Sidebar: a **profile** selector (only the profiles with weights on disk are
offered, and the file it loaded is printed under it), torch/onnx backend switch,
confidence/vote/hold sliders, a checkbox for the measured per-pair refusal gaps
(with an expander giving the table, the 6.3:1 efficiency and the three pairs the
measurement *rejected*), the J/Z motion toggle, landmark overlay, mirror, a
"process every Nth frame" throttle and a feature-vector inspector. Under
`weights:` an ONNX that is older than the `.pt` it was exported from is flagged
with the exact `src/export.py` command that fixes it, because the Runtime switch
would otherwise quietly serve yesterday's model.
The Results tab reads the selected profile's own report
(`reports/report_all.md`, `report_standard.md`; a `round2` run would write
`report_round2.md`, whose copy was pruned along with that profile's weights) and
passes `--profile` to the selftest and benchmark buttons, so the browser can
never show one model's numbers while running another's.

Two deliberate absences:

- **No `calibrate_camera.py`.** The removed photo model had a calibration tab that
  captures images of *your* hand to disk. This project's brief forbids recording
  anything, so it is not ported — there is nothing to calibrate in the model,
  which never sees pixels, only landmark geometry. The one thing that genuinely
  is per-person, the J/Z motion thresholds, is calibrated in memory instead:
  `Calibrate J / Z` keeps a few hundred floats for six strokes, applies numbers to
  the running process, and disappears when it exits. It has no save path.
- No image is ever written to disk by the browser app, same as `live.py`
  (`grep`-checkable, see *Verifying the no-save claim*).

Dependencies added for this: `streamlit==1.64.0 streamlit-webrtc==0.79.0
av==17.1.0` (already in the repo-root `requirements.txt`).

### ONNX export (step 6)

```bash
python src/export.py --samples 3000 --opset 17        # profile in config default
python src/export.py --profile all                    # one profile's weights
```

Exports `models/<profile>.onnx` and compares it with PyTorch on 3 000 held-out
real landmark vectors, printing max |Δlogit| and whether the argmax ever
differs. `live.py --backend onnx` refuses to start if that file is missing.
Latest run: max |Δlogit| **1.9e-06**, argmax agreement **100 %**, identical
accuracy (96.53 %) on both backends over 3 000 real hands, 0.034 ms/frame for
ONNX against 0.248 ms for PyTorch. (These latency digits are wall-clock on a
laptop and move with power state; the parity numbers do not.)

### Verifying the live path without a camera

```bash
python src/verify_livepath.py --per 20 --backend onnx
```

This exists because "the webcam works on my machine" is not evidence, and
because a test that needs a person in front of a camera cannot be re-run by CI
or by you. It takes real photographs out of the public datasets, reads them
`cv2.imdecode(..., IMREAD_COLOR)` → BGR→RGB exactly as the webcam path does, and
pushes them through `hand_backend.detect_video` → `normalize.features` →
`live.frame_call` → `live.Smoother` on a 50 ms simulated tick, then reports what
the pipeline actually answered per letter, how often a hand was found, how many
frames fell under each gate, and the median frame rate. Nothing touches a
capture device and only the markdown report is written.

The catch, stated up front: the deployed model trained on **every** dataset, so
every frame here is one it has seen. 71 of 72 letter blocks answered correctly and
a median 85.0 frames/s are therefore a *wiring and latency* result, not an
accuracy result — its value is that a wrong letter, a missing hand or a sub-20
FPS number would all be bugs. `REPORT.md`'s leave-one-dataset-out table is the
accuracy. One
honest limit in the report itself: at `--per 20` only 36 of the 72 blocks are long
enough for the 0.5 s hold-to-commit timer to fire, so most `answer` cells are the
letter the app is still *holding*, not one it committed; it exercises every stage
of the pipeline either way, and `--per 40` raises the committed count.

### The gate costs coverage, and that is the trade it is for

Across those 1 440 frames: MediaPipe's video path found a hand in 84.3 % of them,
the bounded retry takes that to 98.3 %, and the ladder that recovers the rest of
the *offline* images (upscale + pad) runs in the extractor, not live, because it
costs five detections per frame. 12.3 % of frames fell below the confidence gate
(up to 60 % on T in danrasband) and 2.2 % were near-ties (up to 20 % on T). The app goes quiet on
those instead of guessing, which is why the live rate of *typed letters* will look
lower than the classifier accuracy: on M and N especially it refuses far more often
than it is wrong — the one letter this run got wrong, N on danrasband, was answered
M on frames that scored 69 % mean top-1 and 47 % below the gate.

### Proving the weak letters are a transfer problem, not a broken model

```bash
python src/insample_check.py            # per-letter in-sample accuracy
python src/insample_check.py --profile standard --letters NMTP
python src/insample_check.py --all --csv
```

This is the check that stops "N scores 0.4 %, so the network cannot do N". It
loads the selected profile's `.pt` (default `models/asl_all.pt`) exactly as
`live.py` loads it, runs it over the
**same 122,359 landmark vectors it trained on**, and prints the per-letter
number. On the shipped model: N 90.5 %, M 94.9 %, T 95.2 %, P 95.2 %, overall
96.7 % —
against 0.4 %, 3.2 %, 16.8 % and 40.0 % in `REPORT.md`'s cross-dataset column.
The features express the letters; the boundary just does not generalise to hands
it never saw. It writes nothing, opens no camera, and refuses to run if
the sidecar's feature width disagrees with `normalize.feature_dim()`.
`--all --csv` dumps the whole alphabet if you want to chart it.

### What is in git, and what is generated

`.gitignore` keeps the folder committable: **50 files, 1.6 MB** instead of 698
files and 116 MB. Committed — every `src/*.py`, `app.py`, `run_all.py`,
`data/datasets.json`, both trained models with their
`meta.json` (`models/asl_all.pt/.onnx`, `asl_standard.pt/.onnx`, ~80 KB each, so
the app runs straight after a clone), and the written reports including the two
confusion-matrix PNGs. Ignored — `cache/` (the extracted landmark store: 629 files,
76 MB), `reports/oof_*.csv` (23 MB of per-frame predictions), `reports/samples/`
(the contact sheets), `models/hand_landmarker.task` (MediaPipe's own 7.5 MB file,
downloaded by `src/download_data.py`), plus `__pycache__/`, `*.log` and any
`kaggle.json`. So a fresh clone needs:

```
# from the repository root (the merged requirements.txt lives there)
pip install -r requirements.txt

# then from Model/asl_landmark_app
python src/download_data.py        # datasets + models/hand_landmarker.task
python src/extract_landmarks.py    # ~20 min, rebuilds cache/landmarks.parquet
streamlit run app.py
```

Apart from the install, only `download_data.py` stands between a clone and
`streamlit run app.py`: the trained weights are committed, so neither landmark
extraction nor retraining is required. What the app does need is `hand_landmarker.task`,
MediaPipe's own file, which `download_data.py` fetches from Google's storage bucket.

The bulk dataset downloads stay outside this folder (the Kaggle cache under your
user profile, and the MSL corpus on the drive with the most free space), so
nothing here ever puts ~1 GB of images in your history. To keep the large caches
out of git *permanently* nothing was ever staged; if you do want to version them,
that needs `git lfs`, not a force-add.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `kagglehub` download error or the credential box appears | the dataset is not in the cache and there is no token → follow the 3 lines printed by `download_data.py`, or work offline with whatever is cached (a dataset that fails is skipped and logged, the run continues) |
| `No module named mediapipe.tasks` | mediapipe < 0.10 → install the repo-root `requirements.txt` |
| `cv2.error` about a grayscale image | you fed a single-channel frame; the pipeline requires colour BGR from OpenCV, do not convert to gray |
| hand model missing | `models/hand_landmarker.task` is copied by `download_data.py`; re-run it |
| webcam opens a black window | another app holds the camera; close it, or `--camera 1` |
| `live.py` starts but prints `camera stream ended` immediately | some Windows camera drivers open under MSMF but never deliver frames (this machine's USB2.0 HD UVC WebCam does). `live.py` now tests one real frame per backend and falls back to DirectShow automatically (`open_camera()`); if you still see it, close Teams/Zoom/browser tabs holding the camera |
| browser tab: video box stays black / stuck on "connecting" | the browser grabs the camera itself (not OpenCV): allow the camera prompt or click the address-bar camera icon → Allow → reload; close other apps holding the camera; check Windows Settings > Privacy & security > Camera > "let desktop apps access your camera"; if two camera devices exist, pick the working one in the browser's site settings |
| browser tab: the picture runs for a while and then closes itself | fixed 2026-11-15. `app.py` took its MediaPipe VIDEO clock from a module global (`TS_EPOCH = time.monotonic()`), so any **full script rerun** — pressing a button, moving a slider, resizing the window, a reconnecting browser — rewound that clock while the *cached* landmarker still remembered a larger timestamp. The next frame then raised `ValueError: Input timestamp must be monotonically increasing.` (reproduced: the very first frame after the rewind, and every frame until real time caught up), `streamlit-webrtc` re-raises a worker-thread exception from the track and ends the track, and the browser closes the video element — which is what "the camera closed on its own" was. The epoch now lives in the same `@st.cache_resource` as the detector, `recv()` clamps every timestamp to be strictly greater than the one it last sent, and **no per-frame failure is allowed to leave `recv()`**: it is counted, named in a red box above the read-out, and the untouched frame passes through instead (after a second of consecutive failures a fresh landmarker is swapped in). If the box appears, press STOP then START; if the camera dies with **no** box, the cause is outside this app — a backgrounded tab that Chrome throttles, or ICE reaching `disconnected` when the network or Windows sleeps, on which state the library stops the worker itself |
| browser tab: the picture is there but the read-out says "The webcam is not running" | the panel used to trust one flag — `live_stream.state.playing`, which the *browser component* sends the server — so when that component reloaded or the tab was throttled the flag went false while frames kept flowing, and the two looked identical from the page. `live_panel()` now measures its own heartbeat (frames counted between its runs) and tells the three cases apart: component not attached / stream genuinely stopped / flag lost but frames arriving, and it always prints `N frames/s reaching Python right now`. Chrome's console line *"Your app is having trouble loading the streamlit_webrtc.component.webrtc_streamer component"* belongs to this symptom: the assets are served fine (`/component/streamlit_webrtc.component.webrtc_streamer/assets/index-*.js` → 200, 548,257 bytes), so it is a client-side readiness timeout — cold first load, a backgrounded tab, or a resume from sleep. Keep the tab in the foreground, reload with Ctrl+Shift+R, then START |
| console `live.py`: `ValueError: Input timestamp must be monotonically increasing.` on the first frame after a J/Z calibration | fixed 2026-11-15. The same MediaPipe rule as the row above, but reached a different way: `live.py` has three loops over **one** VIDEO landmarker — typing, `--calibrate-jz` and the `--selftest` drill — and each computed its timestamps from its own `time.monotonic()` epoch. Calibration ran, ended, and the typing loop restarted at 0 ms on a detector that had already seen thousands, so `--calibrate-jz` was always followed by a crash on frame 1 (cancelling with ESC crashed too — the handover was the bug, not the strokes). There is now one epoch per process (`live.stamp_ms()`), and as a backstop `hand_backend.detect_video` keeps the high-water mark *on the detector object* and nudges any stamp that would rewind, so no future loop can poison a live landmarker. Guarded by 5 checks in `--logic-test`, including one that feeds a second loop's clock starting at 0 |
| browser tab: START opens the camera for about a second, the page lags, the camera closes, STOP turns back into START, and a retry shows an error on a white page | fixed 2026-11-15, and it was never the camera or the model. `app.py` pointed **both** ends of the peer connection at `stun:stun.l.google.com:19302`; on the network this app is used on that server is never answered (the app's own debug log: BINDING requests retransmitted 36.1 / 37.2 / 38.6 s after the click). Unanswered STUN stalls ICE gathering, so the SDP answer is never produced, so `streamlit_webrtc`'s `process_offer(..., timeout=10)` stops the worker at ten seconds — releasing the camera — and raises `SignallingTimeoutError`, which is the error box. ICE is now **host-only on both ends** (`rtc_pair(False)`), with a sidebar **"Use a public STUN server"** checkbox that turns it back on for both ends together if you ever open the app from another network. Measured with two local aiortc peers and no camera: answer ready in **9 ms** instead of **10 035 ms**, and the peers connected on host candidates alone. The same row used to point at "ICE reaching `disconnected` when the network sleeps" — that path still exists, but it was not this bug |
| FPS below ~15 | `--backend onnx`, `--width 480 --height 360`, `--no-landmarks` (skip drawing) |
| letters type too slowly / too eagerly | raise/lower `--hold`; `--conf` trades accuracy for responsiveness |
| J or Z never fires | two different causes, and the read-out separates them since round 6. **The watch never opened**: neither precondition held — no index-up shape *and* no reading that was I (any confidence since round 7) or an unsure X — and `--motion-debug` prints `J/Z: neither gate met (index-up shape, or a reading that was not I / not an unsure X)`; `reports/jz_trigger.md` says how often each gate opens per letter. **The watch opened but nothing fired**: the trajectory rule did not match, and the message names the binding criterion with the number it saw against the number it needed (`J held but not fired: net travel 0.11 needs 0.30`). The shipped thresholds came from synthetic strokes, so run `python src/live.py --profile all --calibrate-jz --motion-debug` — three J and three Z of your signing, and the typing loop now starts on the same clock instead of crashing (see the row above) — or `--no-motion` to leave them out of typing entirely. |
| J appears when you sign I | the round-7 exemption means a frame the model reads as I is worth *judging*, and only the trajectory can then separate the two: a J has to hook sideways and come back, and a still I has no reversal, no net travel and no path. If your I arrives with a sideways wobble it can match, and `--motion-debug` will show the J path numbers on a frame you meant as I. The fix is a trajectory threshold (`SEG`, `J_NET`, `J_PATH` via `--jz`), not the ceiling; photographs cannot settle this one, only your camera can. |
| J or Z fires *as I* (or as X) | the motion rule did not fire and the static classifier answered its nearest neighbour, which is what it is for: the model has no J and no Z, so a true J reads as I on 96 % of frames and a true Z as X on 94 %. That reading is now the *precondition* for the trajectory rule rather than a wrong letter, but it only becomes a J or a Z if the movement matches too. If your J moves and still types I, the path thresholds are the problem — calibrate as above |
| a letter you *were* signing goes quiet for a second (usually P/Q, A/S, M/N or G/H) | that is `PAIR_MARGINS`: those four pairs get a 0.50 gap instead of the flat 0.10, because the out-of-fold data says a close second letter means a wrong letter there. The refusal names itself (`A/S needs 50`). Compare on camera with `--pair-margins none` or the sidebar checkbox — measured on 122,359 held-out frames, the table removes 6.3 wrong letters per correct one it costs |
| MediaPipe never finds a hand for a specific letter | look at `cache/skipped_stats.csv`; flat palms and backlit silhouettes are the usual casualties of the palm detector. Round 2's recovery ladder (`upscale → pad → flip`, all as *retries*) is what the `kept_recovered` column counts |
| `landmarks.parquet was extracted with cache_version=1 but the code now expects 2` | deliberate guard in `landmark_store.load_raw()`: the loader refuses a cache that does not match `config.CACHE_VERSION`, because the ladder changes which images get landmarks and on what geometry. Re-run `python src/extract_landmarks.py --force` (~20 min, 12 workers) |
| extraction finishes and then dies with `OSError: [Errno 22] Invalid argument` on the parquet | something held a handle on the cache while it was being truncated (this happened once, after 20 minutes of work). The write now goes to `landmarks.parquet.tmp` and is renamed into place with retries, so the old cache survives; make sure `app.py`/`live.py` and any shell sitting in `cache/` are closed and re-run |

## Limitations (read before trusting the numbers)

- **Cross-dataset accuracy is the honest ceiling**, and it is far below the
  optimistic split. Live use should expect that gap, not the big number.
- **massey is the ceiling.** It holds the LODO fold at 53.8 %, against 77.4 % on
  grassknoted and 94.0 % on danrasband. Round 2 also suspected it of *dragging the
  other folds down* — training on danrasband alone scored 84.8 % on grassknoted
  while adding massey's 60,085 images gave 75.9 %. Under the seeded pipeline that
  contrast narrows to 79.6 % against 77.4 %, i.e. 2.2 points inside a 2.5-point
  seed-to-seed band, so **the poisoning claim is no longer supported by the
  numbers** and only massey's own worst-in-class score is. Stage B down-weighted
  the three cells it could prove are signed differently (M, N, P) and measured the
  effect at zero on the headline, so a per-letter weight is not the lever. The
  lever is more data from independent signers — and that is a round-3 decision,
  not something to hide in a footnote here.
- grassknoted's per-letter images are real photographs, **not** synthetic
  renders — round 1 called them that and was wrong. They are small (often
  ~200 px), underexposed silhouettes against bright windows, which is why they
  were the hardest to detect and why some of its letters stay weak.
- Massey crops are ~120×106 tight hand boxes, so MediaPipe sees a hand pressed
  against the frame border; detection and landmark quality there are worse than
  on full photographs.
- danrasband is only 30 photos per letter (712 usable images), which is why it
  is flagged **small-n** and excluded from the headline despite scoring 94.0 %.
- The 2-D-only feature space ignores finger depth, which is exactly what
  separates some letters in profile views. Feeding MediaPipe's z in was tried
  and cost 1.4 points (`reports/ablation_features.md`): its scale is not
  comparable between datasets.
- **Apparent chirality is ambiguous in 2-D, and it is the first suspect for the
  M/N failure.** `normalize.should_mirror()` uses the sign of the palm
  cross-product `(wrist→index-MCP) × (wrist→pinky-MCP)`. That separates the two
  2-D mirror patterns reliably, and `normalize.chirality_family()` names the
  two groups. Cross-tabbed against MediaPipe's own `handedness` field over all
  127,791 cached hands, the two tests sort hands into the same pair of groups
  on 98.4 % of danrasband, 91.1 % of grassknoted and 87.0 % of massey rows -
  but MediaPipe's *words* are the other way round on these unmirrored frames,
  because its label is defined for a selfie-mirrored image. So neither field
  may be read as anatomy, and neither reaches a feature vector: the cross
  product cannot tell a right hand seen from the back from a left hand seen
  from the palm, which land on opposite sides of the mirror. M and N are usually
  photographed from the side, one finger apart, and they are the two worst
  letters cross-dataset (**N 0.3 %**, **M 2.0 %** after round 2) while scoring
  **89.6 % and 93.9 %** on those very same landmark vectors in-sample —
  reproducible with `python src/insample_check.py`, which re-runs the shipped
  71-feature model over the cache and writes nothing. So the evidence says
  *transfer*, not
  *capacity*, and the mirror choice is the cheapest thing to test first.
- The datasets sit on opposite sides of that mirror before canonicalisation
  (grassknoted is 6.2 % mirrored, massey 85.4 %), so any residual chirality
  problem shows up exactly where it hurts: in the cross-dataset number.
- **The round-5 refusal margins are measured on other people's hands.** The 0.50
  gaps for `AS/GH/MN/PQ` come from replaying the live vote window over the
  leave-one-dataset-out predictions of 122,359 public images, so what they
  predict is *the wrong-letter rate a stranger would see*, not yours. On 576 real
  cached hands re-run through `frame_call` for the verification battery the table
  changed the verdict on exactly 1 frame — the effect is concentrated in the
  windows where the model is nearly tied, and whether your signing lands in those
  windows is a webcam question. `--pair-margins none` / the sidebar checkbox exist
  so you can A/B it, and the same is true of the J/Z calibration, which is the
  only part of round 5 that measures you.
- One hand only (`num_hands=1`); no signer-independent body context.
- J and Z are absent from the classifier by design, and rounds 6 and 7 did not
  change that: they made them **reachable** by the trajectory rule, not learned.
  On 15,772 real cached frames the precondition now opens on 93.3 % of J and
  89.4 % of Z (the shape gate alone opens on 0.0 % and 22.7 %), J fires 62 times
  and Z once, and none of those counts come from a webcam — so the honest
  one-line status is still "the gate is no longer the thing stopping J and Z;
  the synthetic stroke thresholds are", which is a `--calibrate-jz` question on
  your own hands. Two things that round opened are genuinely unresolved:
  23 of those fires are on **I** frames, which is the cost of letting a
  confident I be judged at all and is not measurable on photographs; and Z
  fires once, because `Z_PATH`/`Z_DROP` were written against a synthetic
  zig-zag. Nothing here measures live J/Z recall or live J-on-I false fires, and
  no number in `reports/jz_trigger.md` should be read as either.

## J and Z

J and Z are fingerspelled *with movement*: one frame of J is "index up, the rest
curled", which is exactly a stiff I/L/D, and one frame of Z is the same shape
mid-stroke. No single-frame classifier can recover them. The honest alternative
— a small GRU over motion trained on real fingerspelling video — was tried in
round 6 on the public Mexican Sign Language dynamic corpus, worked on its own
language (J 91.9 % / Z 85.5 % leave-one-signer-out precision) and transferred to
local ASL at **0.8 % recall for J**, so it is measured in `reports/msl_motion.md`
and not shipped. It cannot be trained on your camera, either, by the rules of
this project.

So `src/live.py` adds them back with a **trajectory rule** (`MotionDecider`),
which is the closest alternative:

| letter | what the rule looks for (all distances in palm units, so camera distance does not matter) |
|---|---|
| gate | either precondition, held ≥ 0.40 s: **shape** — index extended (tip-MCP ≥ 1.6 × pip-MCP), middle/ring/pinky curled (≤ 1.2); or **display** — the 24-class model's top-1 is I (candidate J, any confidence) or an X below 0.90 confidence (candidate Z), and that reading has to cover ≥ 60 % of the window being judged |
| **J** | the hand hooks: the *wrist* sweeps sideways ≥ 0.30 and back once (≥ 1 direction reversal, total path ≥ 0.75) while the index stays put relative to the palm |
| **Z** | the hand stays put (wrist ≤ 0.45) and the *index tip* draws the zig-zag: ≥ 2 reversals, path ≥ 0.90, plus ≥ 0.30 of vertical progress |

The display precondition was added in round 6 because the shape gate, on real
photographs, was **closed on the two letters it exists for**: on the
frame-weighted numbers it opened on 0.0 % of J frames and 22.7 % of Z while
opening on 40.2 % of G, 17.8 % of L and 16.0 % of D. The classifier has no J, so
it answers a real J with its nearest neighbour and does it consistently — 93.3 %
of J frames are shown as I, 89.4 % of Z frames as X — which is a truer
description of those two letters than a finger-ratio rule. Round 6 gated that
reading on it looking *unsure* (`p1 < 0.90`) on both routes; round 7 dropped the
ceiling for I only, after measuring that it was refusing **72.7 % of all real J
frames** — a J is a confident I, so confidence was never evidence against it —
while Z kept it, because only 0.6 % of Z frames were refused and an unsure X
really is the signal there. The gate as it ships opens on 93.3 % of J and 89.4 %
of Z, and on the two letters it borrows (a real I on 95.2 %, a real X on 5.2 %)
which the trajectory then has to refuse; no other letter exceeds 0.5 %. The
numbers are per letter in `reports/jz_trigger.md`, regenerated by
`python src/verify_jz_trigger.py --per 300` (no camera). The trajectory rule is
unchanged, so reaching it is all this altered, and deliberately so: a confident
X is still an X, and a straight transit between two letters still has zero
reversals.

Direction reversals are counted with 0.18-palm hysteresis, so ordinary landmark
jitter cannot produce one, and a straight transit between two letters has zero
reversals so it does not look like J. After a letter fires there is a 1.1 s
cooldown. The rule was written conservative — a missing J beat a J that was
never signed — and round 7 spent some of that margin on purpose, because a
letter that never fires is not conservative, it is broken. What still holds the
line is the trajectory: with the I route wide open, a *still* I is the case that
has to type nothing, and it is checked as one.

```bash
python src/live.py --motion-debug   # show the live measurements on the overlay
python src/live.py --no-motion      # static A-Z (minus J/Z) only
python src/live.py --calibrate-jz   # measure the ten thresholds against your J and Z
python src/live.py --jz J_NET=0.21,J_PATH=0.62   # reuse those numbers next time
```

`python src/live.py --logic-test` covers the rule with scripted trajectories —
still hand, straight transit, hook, zig-zag, the same hook with an open hand,
pure landmark jitter, cooldown spacing, a second hook after the cooldown — plus
the round-5 additions: a held shape whose rule cannot fire must *say* which
criterion missed (and `hint()` must be empty exactly when a rule is satisfied),
and a deliberately shrunken J and Z that the shipped defaults miss must fire at
100 % after calibration while the baseline on those same strokes is 0 %. Round 6
adds 11 more: the display path alone must turn a hook into a J and a zig-zag into
a Z, a confident I/X or a reading outside `{I, X}` must not, a reading that
flickers between I and X must be refused by the 60 % window share, and both
front-ends must pass the frame's own reading to the decider (checked by AST on
the real source, 5/5 call sites in `live.py`, 2/2 in `app.py`). Round 7 replaces
one of those 11 and adds three more (87 checks in total): the same hook under a
confident I at p1 0.99 must fire J, a confident X must still not fire Z, an I
held still for 1.5 s must type nothing even with the reading wide open, and the
exemption table must cover J only, keep Z on the shipped ceiling, and stay
outside `DEFAULTS` so `--calibrate-jz` cannot reach it. All of it runs without a
camera.

Two things round 5 changed about how this behaves when it fails. A J or Z
handshape that is held but produces no trajectory used to fall through to the
24-class model, which answered with the nearest letter it does have — usually I —
confidently and silently. It now prints the binding criterion with the number it
saw and the number it needed, from the same `_criteria()` table the decision
itself is computed from, so the explanation cannot describe a rule that did not
run.

**Be clear about the status of this piece:** the shape of the *shipped*
thresholds comes from how the letters are drawn in the synthetic test strokes,
not from fitted data, so recall varies with signing style — which is precisely
what `--calibrate-jz` exists to fix, per session, on your own hands. Even after
calibrating it is a rule, not a model. Round 6 is the measurement of what that
costs: with the gate now reachable, J fires 27 times in 15,772 local frames and
3 times in 566 MSL frames, and Z fires **once** — the precondition is no longer
the thing stopping Z, the zig-zag thresholds are.

