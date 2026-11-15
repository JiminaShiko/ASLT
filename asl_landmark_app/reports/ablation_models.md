# Stage E - model capacity check

Features: `xy+tip_dists+curl+thumb` (71 dims). Protocol: leave-one-dataset-out, 122,359 test images.

| learner | pooled | macro | M | N | T | G | K | P | C | 1-frame latency | live FPS | fold time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp 128-64 | 65.3% | 74.6% | 5.6 | 0.3 | 1.6 | 50.0 | 45.7 | 41.2 | 38.0 | 0.2 ms | 4552 | 290s |
| mlp 256-128 | 64.8% | 74.5% | 3.3 | 1.7 | 5.0 | 50.4 | 54.0 | 39.1 | 44.2 | 0.1 ms | 6726 | 321s |
| hgb | 63.2% | 72.4% | 5.5 | 6.3 | 37.5 | 48.7 | 59.4 | 38.0 | 34.7 | 6.7 ms | 150 | 130s |

| learner | danrasband | grassknoted | massey |
|---|---|---|---|
| mlp 128-64 | 93.7% | 77.8% | 52.2% |
| mlp 256-128 | 94.7% | 78.3% | 50.6% |
| hgb | 91.3% | 77.6% | 48.2% |

**Best by pooled accuracy: `mlp 128-64` (65.3%).**

Widening the MLP moved the pooled number by -0.5 points (65.3% -> 64.8%).

Reading: if neither a wider MLP nor gradient boosting buys more than about a point, the ceiling is the **features and the data**, not the learner - which is what the Stage D table is for. The band to compare against is measured in Stage D (`reports/ablation_features.md`: same feature set, same data, same folds, repeated runs), where it is 1.6-2.5 pooled points, and it is widened by the 0.7 points that re-running one identical command cost before the seed fix of 2026-11-14. Each row here is a single run, so anything inside that band is not a win. The per-letter cells are read the same way: they move by tens of points between runs and are recorded for orientation, not as results. The live requirement is >=20 FPS for the whole frame pipeline; MediaPipe landmark detection itself costs far more than the classifier, so a classifier that needs more than ~10 ms per frame is already too slow to be worth it.
