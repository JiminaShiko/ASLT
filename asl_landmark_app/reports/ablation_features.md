# Stage D - feature ablation

Same protocol as the headline number: leave-one-dataset-out, every test image scored by a model that never saw its dataset. Pooled = micro over all folds (the headline); macro = mean of the per-dataset accuracies. **Each row is the mean of 2 independent runs** (`--seed` changed - the data, the folds and the epochs are fixed), so the spread column is run-to-run noise, not sampling error. Since 2026-11-14 `--seed` controls the weight init, the sampler and the augmentation together; sweeps recorded here that predate that fix had only the augmentation seeded, so their repeats were unpaired draws. Measured cost of re-running one identical command at that time: 0.7 pooled points - which is why nothing below is decided on a gap smaller than the spread column.

| # | feature set | dim | pooled | spread | vs baseline | macro | M | N | T | G | K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline (57) | 57 | 65.3% | 1.6 | +0.0 | 74.3% | 2.3 (2-2) | 0.3 (0-0) | 7.3 (7-8) | 49.5 (49-50) | 43.7 (37-50) |
| 2 | + z coords | 78 | 64.0% | 2.5 | -1.4 | 73.4% | 5.7 (5-6) | 0.4 (0-0) | 15.6 (11-20) | 40.4 (38-43) | 50.5 (48-53) |
| 3 | + thumb features | 71 | 65.5% | 0.4 | +0.2 | 74.6% | 4.8 (4-5) | 0.4 (0-0) | 8.4 (6-11) | 51.7 (51-53) | 58.9 (59-59) |
| 4 | + orientation features | 65 | 65.3% | 1.5 | +0.0 | 74.3% | 4.7 (4-5) | 0.3 (0-0) | 6.4 (3-10) | 51.7 (49-54) | 37.2 (35-39) |
| 5 | thumb + orientation | 79 | 64.2% | 0.7 | -1.1 | 74.0% | 8.6 (6-12) | 0.6 (0-1) | 2.7 (2-4) | 49.9 (50-50) | 47.8 (41-55) |

Run-to-run noise: pooled moved at most 2.5 points across repeats of the same feature set. Per letter the noise is far larger: M up to 6.1 points, N up to 0.6 points, T up to 9.6 points, G up to 4.7 points, K up to 14.0 points - the letters trade confusions with each other, so a single letter number from a single run is not evidence of anything. Only differences larger than the pooled spread are treated as real.

**Best by pooled cross-dataset accuracy: 3 + thumb features** (65.5%, 71 dims).

Sets that cost a baseline letter more than 3 points: 2 loses G (9.2), 4 loses K (6.5), 5 loses T (4.6)

Rule from the brief: promote the best pooled set unless it costs a currently strong letter more than 3 points.
