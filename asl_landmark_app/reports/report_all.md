# ASL landmark recogniser - training report

Generated 2026-11-15 13:43:48 · model `asl_all` · 24 classes (A-Z minus J, Z) · input 71 dims

## Headline number

**Cross-dataset accuracy, POOLED (headline): 67.0% [66.7–67.2% 95% CI] (81,954/122,359 held-out images)**

**Cross-dataset accuracy, MACRO (mean of the 3 held-out datasets, unweighted): 75.9%**

**Same yardstick as round 1 (pooled, scored only on the 83% of these images that round 1's single-pass detector could find at all): 69.2% (70,493/101,932)**

Round 1 headlined 67.6% pooled on a cache that was missing 20,885 hands. Stage C put them back, and they are not neutral: the recovered images are the flat palms, the silhouettes and the small crops, i.e. the frames a real webcam is most likely to give you. The line directly above this one scores only the images round 1 could also score, so it is the only cell in this report that may be compared with 67.6% — and on that like-for-like subset this model is at 69.2% against round 1's 67.6%, a 1.6-point difference. Per the Reproducibility section that is inside the run-to-run band, so it is neither a measured improvement nor a measured regression.
What moved the headline is the denominator. Whether the recovered images also moved the *model* was then measured with `--train-how direct_0.5`, which leaves the test rows alone and withholds the ladder-rescued images from training only: same seed, same 71 features, same style weight, same folds.


| training rows | pooled, all 122,359 | round 1's 101,932 | ladder-rescued 20,427 |
|---|---|---|---|
| every recovered image (this model) | 67.0% | 69.2% | 56.1% |
| round 1's data mix only (control) | 66.1% | 68.2% | 55.4% |


Two paired runs on identical held-out images, and the largest gap in the table is 0.9 points, which is the size of one re-roll of this pipeline. So the recovered images neither helped nor hurt the average.
What the ladder buys is coverage, and the two coverage numbers are not the same thing. **Offline:** the extractor finds a hand in 88.5% of dataset images instead of 74.0% - that is image recall over 144,446 files, where five detections per image cost nothing. **Live:** the app cannot afford that ladder per frame, so it retries a frame at confidence 0.3 only when tracking has just lost the hand, and it finds a hand in 98.3% of frames instead of 84.3%, measured frame by frame in `reports/live_path.md` at 85.0 frames/s for the whole pipeline. A frame with no hand produces nothing at all, so coverage is the part worth paying for; accuracy is the table above.
The **pooled (micro)** number is the headline: every held-out image counts exactly once, so it estimates "a random image from hands the model never trained on", which is what the webcam will actually give you. The **macro** mean weights each *dataset* equally, so a small holdout can carry it far beyond the truth — an earlier version of this report headlined the macro mean (75.9%) while the pooled figure was 67.0%.

Datasets under 2000 held-out images are flagged **small-n** and do not carry the headline: danrasband. They are still used IN TRAINING for every other fold and for the final model; only their role as a *measuring stick* is limited. Excluding them, the pooled cross-dataset accuracy is **66.8%** over 121,647 images [66.6–67.1%] and the macro mean of the headline-bearing datasets is **66.7%**.

This is leave-one-dataset-out: every test sample was predicted by a model trained only on the *other* datasets, i.e. on hands, cameras and backgrounds it never saw. Treat it as the expected live accuracy ceiling.

| held-out dataset | test images | letters covered | accuracy | carries headline? |
|---|---|---|---|---|
| danrasband | 712 | 24 (ABCDEFGHIKLM…) | 94.4% | **no — small-n** |
| grassknoted | 61562 | 24 (ABCDEFGHIKLM…) | 79.5% | yes |
| massey | 60085 | 24 (ABCDEFGHIKLM…) | 53.8% | yes |

All numbers are over the 24-letter static alphabet (`ABCDEFGHIKLMNOPQRSTUVWXY`); J and Z are moving letters and are handled by the trajectory rule in `src/live.py`, not by this classifier.

## Per-letter accuracy for each held-out dataset

A letter can look strong in the pooled table and still be dataset-specific. Each column is a model that never saw that dataset; `–` means no test images for that letter, and a bracketed number is a small test count (< 100 images), so read it loosely.

| letter | danrasband | grassknoted | massey | spread | worst dataset |
|---|---|---|---|---|---|
| **A** | 100% (30) | 93.8% | 87.0% | 13 pts | **massey 87.0%** |
| **B** | 100% (30) | 96.0% | 91.5% | 9 pts | **massey 91.5%** |
| **C** | 87% (30) | 61.6% | 14.4% | 72 pts | **massey 14.4%** |
| **D** | 100% (30) | 50.6% | 28.1% | 72 pts | **massey 28.1%** |
| **E** | 100% (30) | 86.7% | 83.4% | 17 pts | **massey 83.4%** |
| **F** | 100% (30) | 91.4% | 93.7% | 9 pts | **grassknoted 91.4%** |
| **G** | 100% (30) | 90.8% | 12.1% | 88 pts | **massey 12.1%** |
| **H** | 100% (30) | 88.0% | 98.8% | 12 pts | **grassknoted 88.0%** |
| **I** | 100% (25) | 94.9% | 57.7% | 42 pts | **massey 57.7%** |
| **K** | 100% (28) | 69.1% | 47.4% | 53 pts | **massey 47.4%** |
| **L** | 100% (30) | 99.1% | 97.5% | 2 pts | **massey 97.5%** |
| **M** | 93% (30) | 3.3% | 2.1% | 91 pts | **massey 2.1%** |
| **N** | 37% (30) | 0.2% | 0.2% | 37 pts | **massey 0.2%** |
| **O** | 100% (29) | 95.6% | 35.3% | 65 pts | **massey 35.3%** |
| **P** | 90% (30) | 71.4% | 2.4% | 88 pts | **massey 2.4%** |
| **Q** | 97% (30) | 93.1% | 8.2% | 88 pts | **massey 8.2%** |
| **R** | 100% (30) | 93.0% | 47.9% | 52 pts | **massey 47.9%** |
| **S** | 97% (30) | 93.5% | 86.1% | 11 pts | **massey 86.1%** |
| **T** | 73% (30) | 31.9% | 1.1% | 72 pts | **massey 1.1%** |
| **U** | 100% (30) | 92.4% | 71.7% | 28 pts | **massey 71.7%** |
| **V** | 93% (30) | 93.8% | 41.8% | 52 pts | **massey 41.8%** |
| **W** | 100% (30) | 96.2% | 97.4% | 4 pts | **grassknoted 96.2%** |
| **X** | 100% (30) | 91.0% | 85.0% | 15 pts | **massey 85.0%** |
| **Y** | 100% (30) | 97.0% | 90.9% | 9 pts | **massey 90.9%** |

Read the spread column, not just the averages: a letter whose datasets disagree by 30+ points is signed or imaged differently across them, which is a data problem, not a model problem. The contact sheets in `reports/samples/` (Stage B) are the visual evidence for that call.

## Stage B: are the datasets signing the same letter?

Measured on the cached landmarks, not on the images: for every letter the mean normalised hand shape of each dataset is compared with every other, in wrist->middle-MCP bone units. Three datasets and 24 letters give one distance per pair per letter, and averaging a pair over the alphabet is what makes the *structural* call possible: two datasets that are close on average are one style, and the dataset that sits outside that pair is the second style.

Reproduce with `python src/inspect_data.py --per-dataset 12`, which also writes the contact sheets in `reports/samples/` and `cache/divergence.csv`.

**The cluster:** `danrasband` and `grassknoted` sit 4.4 units apart on average; `massey` sits 6.8 and 7.8 away from them. So there are two styles in this corpus, not three, and `massey` is the minority one - which matches the contact sheets, where the two cluster members share a 200x200 tile, the same arm's-length framing and the same handshapes, while the minority dataset is a different signer at a different distance. (Whether the cluster members are literally one person or one re-packaged source cannot be settled from the images; what can be settled is that they behave as a single style.)

**The per-letter conflicts** (a letter is flagged when the minority style is >= 2.0x the median minority distance of 6.2 units over the alphabet; the full ranking is shown so the cut is visible, not hidden):

| letter | minority | dist to cluster | x median | cluster pair dist | flagged |
|---|---|---|---|---|---|
| M | massey | 25.9 | 4.19x | 13.5 | yes |
| N | massey | 22.4 | 3.62x | 9.4 | yes |
| P | massey | 21.3 | 3.44x | 6.2 | yes |
| H | massey | 9.6 | 1.55x | 1.5 |  |
| Q | massey | 9.6 | 1.55x | 12.1 |  |
| C | massey | 9.6 | 1.55x | 6.8 |  |
| G | massey | 9.1 | 1.48x | 2.8 |  |
| R | massey | 7.4 | 1.20x | 3.8 |  |
| D | massey | 7.3 | 1.19x | 6.5 |  |
| X | massey | 6.4 | 1.03x | 2.8 |  |
| O | massey | 6.4 | 1.03x | 4.0 |  |
| I | massey | 6.2 | 1.00x | 2.7 |  |

The ratio to the *split-half* distance inside one dataset, which an earlier version of this report thresholded on, is deliberately not used here: a dataset agrees with itself to ~0.2 units no matter what, so on this cache 24 of 24 letters clear ratio 5 and the screen says 'everything is a different sign'. Between-dataset distance measures capture geometry as well as handshape, so it has to be read against the typical between-dataset distance, which is what the x-median column does.

**What the contact sheets actually show** (12 random images per dataset per letter, MediaPipe skeleton drawn on top, grey = no hand found):

- **Two of the three datasets behave as one style.** The cluster pair shares a fixed 200x200 tile, an arm's-length framing and, letter by letter, the same handshape; their shapes land on top of each other for almost every letter. The small member of that pair (~30 images per letter) cannot carry a headline number on its own, which is why Stage A excludes it from the pooled headline.
- **The minority dataset really does sign M, N and P differently.** For M and N its signer holds the fingers splayed rather than pressed together; for P the open hand faces the camera instead of the side. Those are the three letters that clear the cut by a wide margin, and they are exactly where the round-1 accuracy collapsed (N 0.3%, M 5.8%, P 35.5%). The next tier - H 1.55x, Q 1.55x, C 1.55x, G 1.48x - sits between 1.2x and the 2.0x cut and is left alone; `--divergent-ratio 1.5` would pull them in, and whether that is a help or a handicap is measured in Stage D rather than assumed.
- **K, V, T, D and the rest are the same handshape in all three datasets.** Their weakness is a *rotation* problem - K and V are the same two-finger V at different wrist angles, G and H differ only by which finger is up - plus poor detection recall, not a label conflict. That is what the Stage D orientation features are for.
- **B is a recall problem, not a style problem**: 11 of 12 sampled minority-dataset B frames are a flat palm filling the frame, which the palm detector refuses. See the Stage C before/after table.
- **The "synthetic renders" claim about grassknoted is wrong as worded.** The files are real 200x200 JPEG photo tiles (~13k unique colours, noisy borders, no pure-black background): a hand at arm's length against a ceiling, skylight or wall. What *is* true is that a subset of them are **backlit silhouettes** - the hand almost black against a bright window. Those are legitimately labelled but give MediaPipe nothing to find, or a garbage skeleton, and they account for most of grassknoted's dropped images and several of its weak letters.

**Choice applied.** Deleting the minority style of M/N/P would raise the pooled number only by making the model blind to a real, common way of signing those letters, and it is our only independent-signer dataset. The least destructive option is therefore a *training weight*, not a deletion: every image stays in the cache and in every evaluation, and the minority style of a flagged letter is down-weighted. This run used `--style-weight 1.0` (weight 1.0 = no reweighting; all styles are trained at full weight).
The weight is applied inside `WeightedRandomSampler` in `train_one()`, so it changes sampling frequency only - no rows are removed from `cache/landmarks.parquet`, and the held-out test sets are untouched, so the cross-dataset numbers stay comparable across settings. Whether the weight actually helps is an empirical question and is measured, not assumed: `reports/ablation_features.md` reports the same feature set at weight 1.0 and at the chosen weight, because down-weighting one dataset helps the fold that tests the *other* dataset and can hurt the fold that tests the down-weighted one.
## Pairwise cross-dataset (direction matters)

| train on | test on | shared letters | test n | accuracy |
|---|---|---|---|---|
| danrasband | grassknoted | 24 | 61562 | 79.6% |
| danrasband | massey | 24 | 60085 | 54.7% |
| grassknoted | danrasband | 24 | 712 | 92.7% |
| grassknoted | massey | 24 | 60085 | 50.6% |
| massey | danrasband | 24 | 712 | 77.1% |
| massey | grassknoted | 24 | 61562 | 73.5% |

## Held-out capture group (unseen hands, same dataset)

| dataset | train groups | test groups | test images | accuracy |
|---|---|---|---|---|
| massey | dataset5/A+dataset5/B | dataset5/C+dataset5/D+dataset5/E | 35893 | 85.7% |

This is the number closest to what the webcam will do for you: the model has never seen those specific hands, cameras or backgrounds, but the letter distribution and imaging style match.

## Optimistic reference splits — NOT the headline

- within-dataset, **contiguous-block** split of `danrasband` (190 blocks, 184 held-out images): **82.6%** — optimistic, because the same signer, camera and background appear in both halves.
- within-dataset, **contiguous-block** split of `grassknoted` (192 blocks, 15231 held-out images): **96.0%** — optimistic, because the same signer, camera and background appear in both halves.
- within-dataset, **contiguous-block** split of `massey` (960 blocks, 15116 held-out images): **96.3%** — optimistic, because the same signer, camera and background appear in both halves.
- final model, contiguous-block validation slice: **96.4%** — also optimistic, same reason.

## Per-class accuracy (pooled cross-dataset predictions)

| letter | test images | correct | accuracy |
|---|---|---|---|
| A | 4695 | 4254 | 90.6% |
| B | 4993 | 4677 | 93.7% |
| C | 5063 | 1864 | 36.8% |
| D | 5285 | 2117 | 40.1% |
| E | 4856 | 4135 | 85.2% |
| F | 5530 | 5115 | 92.5% |
| G | 5485 | 2901 | 52.9% |
| H | 5345 | 4996 | 93.5% |
| I | 5146 | 3932 | 76.4% |
| K | 5693 | 3321 | 58.3% |
| L | 5517 | 5424 | 98.3% |
| M | 4953 | 159 | 3.2% |
| N | 4575 | 19 | 0.4% |
| O | 4791 | 3229 | 67.4% |
| P | 4210 | 1682 | 40.0% |
| Q | 4286 | 2369 | 55.3% |
| R | 5578 | 3912 | 70.1% |
| S | 5255 | 4736 | 90.1% |
| T | 5150 | 863 | 16.8% |
| U | 5277 | 4353 | 82.5% |
| V | 5444 | 3711 | 68.2% |
| W | 5253 | 5085 | 96.8% |
| X | 4926 | 4341 | 88.1% |
| Y | 5053 | 4759 | 94.2% |

## Weakest letters

- **N**: 0.4% (19/4575)
- **M**: 3.2% (159/4953)
- **T**: 16.8% (863/5150)
- **C**: 36.8% (1864/5063)
- **P**: 40.0% (1682/4210)
- **D**: 40.1% (2117/5285)

## Most-confused pairs

| true | predicted | errors | fraction of that class |
|---|---|---|---|
| T | A | 2255 | 43.8% |
| G | H | 2362 | 43.1% |
| M | S | 1725 | 34.8% |
| C | B | 1702 | 33.6% |
| T | L | 1611 | 31.3% |
| P | N | 1287 | 30.6% |
| Q | N | 1308 | 30.5% |
| D | X | 1508 | 28.5% |

## Confusion matrix

![confusion](reports/confusion_matrix_all.png)

## Stage C: detection recall, before and after the recovery ladder

The extractor now walks a ladder of forward image transforms and stops at the first one that finds a hand: `direct_0.5` (round 1, unchanged) -> `direct_0.3` -> `upscale_pad_0.3` (small sides raised to >=480 px plus a 12% replicated border) -> `flip_0.3` -> `flip_upscale_pad_0.3`. Landmarks from a transformed copy are mapped back through the exact inverse before they reach the cache, and the handedness label is flipped when the flip path fired. "before" below is attempt 1 alone, i.e. round 1.

| dataset | images selected | recall before (attempt 1) | recall after ladder | gain | recovered images | still dropped |
|---|---|---|---|---|---|---|
| danrasband | 780 | 95.3% | 99.0% | +3.7 pts | 29 | 8 |
| grassknoted | 78,000 | 77.3% | 85.8% | +8.5 pts | 6,652 | 11,066 |
| massey | 65,666 | 69.9% | 91.5% | +21.6 pts | 14,204 | 5,581 |

Recovered images per letter (top 10), and whether the recovery landed where round 1 was blind:

| letter | recovered | still dropped |
|---|---|---|
| B | 1,907 | 765 |
| W | 1,651 | 777 |
| M | 1,561 | 787 |
| A | 1,260 | 1,011 |
| C | 1,130 | 883 |
| S | 1,079 | 559 |
| N | 1,064 | 1,149 |
| O | 1,059 | 895 |
| U | 982 | 406 |
| E | 980 | 855 |

Which retry step found them: `upscale_pad_0.3` 12,913, `direct_0.3` 6,454, `flip_upscale_pad_0.3` 775, `flip_0.3` 743.
Detections still fail most often on: P (1623), Q (1416), N (1149), A (1011), O (895), C (883) — flat palms filling the frame (massey B), curled/edge-on hands, and the backlit silhouette photographs inside grassknoted, where the hand is almost black against a bright window. The Stage B section corrects the round-1 wording, which called those silhouettes "synthetic renders"; they are real underexposed photos.

## Stage D: which features actually help

Five feature sets, each trained twice on the same folds with `--seed` changed, so the *noise* column is run-to-run variation and not sampling error. Full table: `reports/ablation_features.md`.

Caveat that applies to every number in the two tables below: these sweeps were run **before the seed bug described in "Reproducibility" was found**, so each row is a single draw from an unseeded pipeline and carries at least the ±0.7 points that re-running one identical command produced. The conclusions below are written to survive that — none of them rests on a gap smaller than the band — but the digits themselves will not be exactly reproduced.

| feature set | dim | pooled | noise | vs baseline |
|---|---|---|---|---|
| `xy+tip_dists+curl` (round-1 set) | 57 | 65.3% | 1.6 | +0.0 |
| + MediaPipe z | 78 | 64.0% | 2.5 | -1.4 |
| **+ thumb relations** | **71** | **65.5%** | **0.4** | **+0.2** |
| + palm orientation | 65 | 65.3% | 1.5 | +0.0 |
| thumb + orientation | 79 | 64.2% | 0.7 | -1.1 |

**The 71-dim thumb set is what this model trains on** (`xy + tip_dists + curl + thumb`). Three things decided it:

* z coordinates hurt (-1.4, outside the noise). MediaPipe documents z as an arbitrary scale, and measured across our datasets it is not comparable: the palm-width / wrist-to-MCP ratio is 0.67 for grassknoted and danrasband but 1.05 for massey. Feeding it in teaches the model a per-dataset signature.
* thumb relations are the only addition that never made a watched letter worse, and the number that carried the decision is K 43.7% -> 58.9%. Read it with two things in mind: per-letter noise in that same table reaches 14 points, and the sweep predates the seed fix, so the pair is a lead rather than a measurement. The half of it that this run can check is the destination - K is 58.3% pooled cross-dataset here, which is at least consistent with 58.9%. Re-measuring all five sets under the pinned seed is a round-3 task. The reason the block is believed at all is structural: the thumb is the one digit whose *relationship to the other fingers* carries a letter (A/S/U, F, P/Q/R), and plain fingertip-to-fingertip distances do not express it.
* the headline gain over the 57-dim baseline is +0.2, which is **inside every noise band measured** — the 1.6-point seed-to-seed spread of that table and the 0.7 points a single repeated command moved — so it is not evidence of a better model on its own. It is promoted on the brief's stated rule — best pooled set unless it costs a currently strong letter more than 3 points — and it is the only set that costs nothing.

N stays at 0.3-0.6% under every one of the five sets. No feature can fix a letter whose datasets disagree about what it looks like; see Stage B.

## Stage E: is the bottleneck the learner?

Same folds, same 71 features, three learners. Full table: `reports/ablation_models.md`.

| learner | pooled | macro | 1-frame latency | classifier FPS |
|---|---|---|---|---|
| **MLP 128-64 (used)** | **65.3%** | **74.6%** | 0.2 ms | 4552 |
| MLP 256-128 | 64.8% | 74.5% | 0.1 ms | 6726 |
| HistGradientBoosting | 63.2% | 72.4% | 6.7 ms | 150 |

Doubling the width moves the pooled number by -0.5 and gradient boosting, a much stronger learner on tabular data, by -2.1 — the first inside every noise band (the 0.7 points from re-running one identical command, the up to 2.5 points between seeds), the second at the edge of the larger one. Neither buys anything, so the honest reading is: **the learner is not the limit.** We keep 128-64 because it is the best measured, the smallest artifact, and 0.2 ms per frame — the live budget is spent on MediaPipe, not on this MLP, and the requirement is >=20 FPS for the whole pipeline (measured in `reports/live_path.md`).

One per-letter result is worth recording even though it does not change the choice: boosting reaches T 37.5% where both MLPs sit at 1.6-5.0%, and N 6.3% where the MLPs sit at 0.3-1.7%. Those are the two letters whose distinguishing evidence is a *threshold on a single dimension* (how far the fingertips sit from the palm), which trees read directly and an MLP has to re-derive. Per-letter noise is large enough (T moves by up to 9.6 points between repeats) that this is a lead, not a result.

**Augmentation was measured, not raised on faith.** The brief said to increase it only if the cross-dataset gap is overfitting rather than data. Running the chosen set twice at strength 1.0 and twice at 2.0: 65.8% (spread 1.8) against 65.5% (spread 0.3) — no gain, `reports/ablation_aug1.md` and `reports/ablation_aug2.md`. The gap between folds is therefore not the model memorising its training data; it is the datasets themselves, which is what the per-dataset row in the headline section shows.

## Stage F: what the live app does with an unsure frame

A 65% cross-dataset model is wrong about one third of the time, so the live app is built to say *I am not sure* rather than to guess. One shared function, `src/live.py::frame_call()`, decides what a frame is allowed to contribute, and both front-ends (console and Streamlit) call it:

* it shows the **top three** letters with their probabilities, not one bare letter, so a 0.44 N / 0.41 S frame looks like the coin-flip it is;
* two independent reasons can silence a frame, and the overlay names whichever one fired — `low_conf` (top-1 under `--conf`, default 0.70: the model does not think it is looking at a letter) and `close` (top-1 minus top-2 under `--margin`, default 0.10: two letters are nearly equally good). They are kept separate because with 24 classes a genuine tie is *also* under the confidence gate, so reporting only the gate would mean the uncertain state never appears;
* an uncertain frame casts **no vote**. That matters more than it sounds: the smoother commits on a majority over a 7-frame window, so if ties voted they would quietly majority-rule the wrong letter into the text buffer.

The measured cost of the gate is in `reports/live_path.md`: the share of held-out frames that fall under the gate and the share that are ties, per letter. Weak letters are visible there as *silence*, which is the correct behaviour for a model that cannot tell N from S.

## Normalisation actually used

```json
{
  "origin_landmark": 0,
  "scale_landmarks": [
    0,
    9
  ],
  "scale_eps": 1e-06,
  "canonical_handedness": "Right",
  "mirror_left_to_right": true
}
```
```json
{
  "blocks": [
    "xy",
    "tip_dists",
    "curl",
    "thumb"
  ],
  "dim": 71,
  "coord_space": "2d_image_plane",
  "fingertips": [
    4,
    8,
    12,
    16,
    20
  ],
  "finger_chains": [
    [
      1,
      2,
      3,
      4
    ],
    [
      5,
      6,
      7,
      8
    ],
    [
      9,
      10,
      11,
      12
    ],
    [
      13,
      14,
      15,
      16
    ],
    [
      17,
      18,
      19,
      20
    ]
  ]
}
```

`src/normalize.py::features()` is called by training AND `src/live.py`; there is exactly one implementation.

## Training config

```json
{
  "name": "asl_all",
  "in_dim": 71,
  "seed": 0,
  "hidden": [
    128,
    64
  ],
  "dropout": 0.25,
  "label_smoothing": 0.05,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "batch": 256,
  "epochs": 80,
  "fold_epochs": 30,
  "patience": 12,
  "aug_strength": 1.0,
  "style_weight": 1.0,
  "divergent_ratio": 2.0,
  "divergent_cells": []
}
```

## Reproducibility

Every number in this file comes from one run of `python src/train.py` at seed 0, and that command reproduces them exactly on the same machine. It did not used to. `train_one()` seeded the augmentation stream but the layer initialisation and the `WeightedRandomSampler` were still drawing from torch's unseeded global RNG, so two identical invocations produced two different models: pooled cross-dataset **63.9%** and **64.6%** — a 0.7-point swing from nothing but re-running the command, bigger than the +0.2 that Stage D used to promote the thumb features. That is now pinned (`torch.manual_seed` plus an explicit sampler generator); a repeat of the same command is bit-identical, verified by running `--folds-only` twice and diffing.

Two consequences for how to read this report. First, any difference below roughly one pooled point — Stage D's 71-vs-57 comparison, Stage E's wider-MLP row — is not evidence of anything on its own, and the tables say so rather than ranking them. Second, the round-1 comparison and the `--train-how` control are now genuinely same-seed experiments: the only thing that differs between them is the rows they were told to use.
