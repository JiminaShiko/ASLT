# Stage D - feature ablation

Same protocol as the headline number: leave-one-dataset-out, every test image scored by a model that never saw its dataset. Pooled = micro over all folds (the headline); macro = mean of the per-dataset accuracies. **Each row is the mean of 3 independent runs** (`--seed` changed - the data, the folds and the epochs are fixed), so the spread column is run-to-run noise, not sampling error. Since 2026-11-14 `--seed` controls the weight init, the sampler and the augmentation together; sweeps recorded here that predate that fix had only the augmentation seeded, so their repeats were unpaired draws. Measured cost of re-running one identical command at that time: 0.7 pooled points - which is why nothing below is decided on a gap smaller than the spread column.

| # | feature set | dim | pooled | spread | vs baseline | macro | M | N | T | C | D | G | Q | A | B | F | H | L | S | W | X | Y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | + thumb features | 71 | 66.1% | 2.3 | +0.0 | 75.1% | 4.3 (3-5) | 0.4 (0-0) | 13.5 (9-17) | nan (nan-nan) | nan (nan-nan) | 53.1 (50-57) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) |
| 6 | shipped + thumb-cross | 79 | 64.1% | 1.1 | -2.0 | 73.7% | 3.3 (2-4) | 0.3 (0-0) | 7.3 (4-13) | nan (nan-nan) | nan (nan-nan) | 50.3 (49-52) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) | nan (nan-nan) |

Run-to-run noise: pooled moved at most 2.3 points across repeats of the same feature set. Per letter the noise is far larger: M up to 2.0 points, N up to 0.1 points, T up to 8.9 points, G up to 7.0 points - the letters trade confusions with each other, so a single letter number from a single run is not evidence of anything. Only differences larger than the pooled spread are treated as real.

**Best by pooled cross-dataset accuracy: 3 + thumb features** (66.1%, 71 dims).

Sets that cost a baseline letter more than 3 points: 6 loses T (6.2)

Rule from the brief: promote the best pooled set unless it costs a currently strong letter more than 3 points.
