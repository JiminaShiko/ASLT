# What the recovered images cost — and what they actually bought

Round 2's pooled headline moved **down** against round 1 (67.6 % → **65.9 %**)
while three things changed at once: the extractor found 20,885 more hands, the
features gained the thumb block, and Stage B's three divergent cells were
down-weighted. One number cannot separate those, so this experiment changes only
the first.

> **This file replaces an earlier version that claimed a trade.** That version
> said the recovered images were worth **+3.4 points** on the hard frames and
> cost **-1.8** on the easy ones. That claim was wrong and is retracted below.

## The experiment

`--train-how direct_0.5` restricts the leave-one-dataset-out folds' **training**
rows to the images a single-pass detector finds on the first try — round 1's data
mix — and leaves the test rows alone. Scored on the same test rows, it is
round 1's data against today's code: same seed, same folds, same 71 features,
same 0.5 style weight. The only difference is whether the model ever saw a
ladder-rescued image during training.

```bash
cd "D:/sign language project/ASTL project/ASLT/asl_landmark_app"
python src/train.py --folds-only --no-save --train-how direct_0.5
```

## Result: nothing measurable, in either direction

| training rows | pooled, all 122,359 test images | round 1's 101,932 | ladder-rescued 20,427 |
|---|---|---|---|
| every recovered image (what ships) | **65.9 %** (80,691) | **68.1 %** (69,374) | **55.4 %** (11,317) |
| round 1's mix only (control) | **66.1 %** (80,861) | **68.2 %** (69,538) | **55.4 %** (11,323) |
| difference | -0.2 pts | -0.1 pts | ±0.0 pts |

The rescued-group counts are derived, not measured directly: they are the
difference of the two published cells, which is exact because both runs score the
same held-out images in the same folds.

**Reading.** Training on 20,885 extra hard images changed the cross-dataset score
by two tenths of a point, in the wrong direction, and the hard group it was
supposed to help is identical to one decimal place. Against the noise floor this
is a null result: before the seed fix, running *one identical command twice* moved
the pooled headline by 0.7 points and the rescued-group score by 2.5 (54.8 % and
52.3 % from two draws of the same configuration). Every difference in the table
above is smaller than that.

So the sentence that survives is: **the recovered images neither helped nor hurt
the model; they changed the measuring stick.** The headline fell versus round 1
because 20,427 images that score 55.4 % are now inside the test set, and on the
101,932 images round 1 actually had, this model scores 68.1 % against round 1's
67.6 % — half a point, also inside the band, and therefore not a claim of
improvement either.

## What the ladder does buy

Not an average. Two other things, both worth having on their own terms:

- **Coverage at inference.** Detection recall went 74.0 % → 88.5 %, and a frame
  in which MediaPipe finds no hand contributes nothing to the smoother — no
  letter, no confidence, no vote. That is a 14.5-point gain in how often the app
  can answer at all, which is the quantity a user actually experiences.
- **An honest denominator.** The flat palms, the backlit silhouettes and the small
  crops are now scored and reported per letter instead of being silently filtered
  out of the test set by a detector failure. Round 1's 67.6 % was measured on the
  easy 83 %; the 65.9 % here is measured on everything.

Keeping the recovered images in *training* is still the right call — it costs
nothing measurable, it is the same distribution the webcam will produce, and a
model that has seen the hard shapes is the one you want when the ladder rescues a
hand at runtime. It is just not supported by a score change, and this file used to
say that it was.

## The one exception worth following up: T

| letter, pooled cross-dataset | control | shipped | difference |
|---|---|---|---|
| **T** | **14.3 %** | **3.1 %** | **-11.2** |
| M | 6.0 % | 2.0 % | -4.0 |
| K | 56.8 % | 60.9 % | +4.1 |
| G | 51.5 % | 52.0 % | +0.5 |
| N | 0.5 % | 0.3 % | -0.2 |

T is signed with the index crossed over the middle — the geometry most likely to
be mis-landed by an upscaled, padded or mirrored detection — so a model trained on
the rescued rows learns a noisier T than one that never saw them. An 11-point gap
is only just outside the 9.6-point per-letter spread Stage D measured, and that
spread was itself measured on unpaired, unseeded runs, so this is a **lead, not a
result**. Note also that three of the five watched letters did not move at all,
which is what the pooled null result above predicts. The cheap round-3 test is a
*training weight* on `how != direct_0.5` (keep the coverage, damp the damage)
rather than the all-or-nothing row filter used here; it has not been run, so it is
not applied.

## How to re-derive everything on this page

Both numbers in the results table are one command apart, and both are seeded, so
they reproduce exactly on this machine:

```bash
python src/train.py --folds-only --no-save                     # shipped mix
python src/train.py --folds-only --no-save --train-how direct_0.5   # control
```

`models/meta.json` records the seed and the literal `reproduce_command` of the
delivered artifact.

Raw lines, kept here so the numbers above are auditable without re-running
anything:

```
# control  (2026-11-14 14:49, python src/train.py --folds-only --no-save --train-how direct_0.5)
ABLATION_RESULT blocks=xy+tip_dists+curl+thumb dim=71 pooled=66.0850 macro=75.3933 \
  n=122359 r1=68.2200 M=6.0166 N=0.4590 T=14.3301 G=51.4676 K=56.7715
# per-fold: danrasband 94.7%, grassknoted 78.9%, massey 52.6%

# shipped  (2026-11-14 17:22, python src/train.py)
HEADLINE  pooled cross-dataset accuracy: 65.9% [65.7-66.2%] (80,691/122,359)
          macro (mean of 3 datasets)   : 75.1%
          same test set as round 1 (83% of these images): 68.1% (69,374/101,932)
# per-fold: danrasband 94.0%, grassknoted 77.4%, massey 53.8%
# pooled per letter (REPORT.md): M 2.0%, N 0.3%, T 3.1%, G 52.0%, K 60.9%
```

The two runs are not identical in cost: the shipped report also trains the final
all-data model and the pairwise/group/block estimators, so its fold models are the
same draws the control reproduces. That is what makes the table above a paired
comparison rather than two accidents.

