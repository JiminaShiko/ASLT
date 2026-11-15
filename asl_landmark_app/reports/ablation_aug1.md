# Stage D - feature ablation

Same protocol as the headline number: leave-one-dataset-out, every test image scored by a model that never saw its dataset. Pooled = micro over all folds (the headline); macro = mean of the per-dataset accuracies. **Each row is the mean of 2 independent runs** (`--seed` changed - the data, the folds and the epochs are fixed), so the spread column is run-to-run noise, not sampling error. Since 2026-11-14 `--seed` controls the weight init, the sampler and the augmentation together; sweeps recorded here that predate that fix had only the augmentation seeded, so their repeats were unpaired draws. Measured cost of re-running one identical command at that time: 0.7 pooled points - which is why nothing below is decided on a gap smaller than the spread column.

| # | feature set | dim | pooled | spread | vs baseline | macro | M | N | T | G | K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | + thumb features | 71 | 65.8% | 1.8 | +0.0 | 74.9% | 3.1 (3-4) | 0.3 (0-0) | 4.1 (2-6) | 54.3 (54-54) | 52.0 (51-53) |

Run-to-run noise: pooled moved at most 1.8 points across repeats of the same feature set. Per letter the noise is far larger: M up to 0.8 points, N up to 0.0 points, T up to 3.6 points, G up to 0.1 points, K up to 2.2 points - the letters trade confusions with each other, so a single letter number from a single run is not evidence of anything. Only differences larger than the pooled spread are treated as real.

**Best by pooled cross-dataset accuracy: 3 + thumb features** (65.8%, 71 dims).

Rule from the brief: promote the best pooled set unless it costs a currently strong letter more than 3 points.
