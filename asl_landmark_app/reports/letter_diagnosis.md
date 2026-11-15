# Letter diagnosis: T, N, M, P (profile `all`)

122,359 out-of-fold predictions over 3 held-out datasets, 71-dim features (`xy+tip_dists+curl+thumb`).

## 1. What the letter actually gets called, per held-out dataset

Out-of-fold predictions, so every row was produced by a model that never saw that dataset. `top confusions` is the share of the letter's images that landed on each wrong answer.

| letter | held-out dataset | images | accuracy | top confusions |
|---|---|---|---|---|
| T | danrasband | 30 | 73.3% (n=30) | P 50%, L 25%, Y 25% |
| T | grassknoted | 2,554 | 31.9% (n=2,554) | L 92%, A 4%, Y 1% |
| T | massey | 2,566 | 1.1% (n=2,566) | A 86%, M 3%, S 3% |
| N | danrasband | 30 | 36.7% (n=30) | M 95%, S 5% |
| N | grassknoted | 1,897 | 0.2% (n=1,897) | P 65%, Q 30%, A 1% |
| N | massey | 2,648 | 0.2% (n=2,648) | A 45%, S 40%, U 7% |
| M | danrasband | 30 | 93.3% (n=30) | A 50%, N 50% |
| M | grassknoted | 2,337 | 3.3% (n=2,337) | P 42%, Q 38%, A 6% |
| M | massey | 2,586 | 2.1% (n=2,586) | S 68%, A 24%, E 2% |
| P | danrasband | 30 | 90.0% (n=30) | Q 100% |
| P | grassknoted | 2,252 | 71.4% (n=2,252) | Q 60%, D 12%, Y 10% |
| P | massey | 1,928 | 2.4% (n=1,928) | N 68%, M 21%, X 3% |

## 2. Accuracy split by how the hand was found

`direct` is the plain first detection (0.5, then the 0.3 retry the extractor ladder runs before anything fancier). If a letter were failing because the ladder handed the model worse landmarks, its `direct` column would be healthy and the ladder columns would collapse. The reverse means the detection path is not the problem.

| letter | all | direct (first try) | upscale+pad | flipped |
|---|---|---|---|---|
| T | 16.8% (n=5,150) | 17.3% (n=4,759) | 10.8% (n=379) | 8.3% (n=12) |
| N | 0.4% (n=4,575) | 0.5% (n=3,932) | 0.2% (n=565) | 0.0% (n=78) |
| M | 3.2% (n=4,953) | 3.4% (n=3,979) | 2.4% (n=883) | 1.1% (n=91) |
| P | 40.0% (n=4,210) | 42.5% (n=3,730) | 18.0% (n=417) | 34.9% (n=63) |
| *all letters* | 67.0% (n=122,359) | 67.9% (n=108,241) | 61.4% (n=13,393) | 33.9% (n=725) |

## 3. Is the shape even separable?

`crowding` is the distance from this letter's mean hand to the closest other letter's mean, divided by the median distance to any other letter (units: summed joint distance, a hand is ~1 palm across). Near 0 means two letters share one region of feature space *within the same dataset*, which no re-weighting can separate - a different problem from one dataset disagreeing about how a letter looks. Rows below are the letters asked about, out of every letter in that dataset.

From `python src/inspect_data.py --nearest`:

| dataset | letter | n | nearest | dist | median_other | crowding |
|---|---|---|---|---|---|---|
| danrasband | N | 30 | M | 1.68 | 16.14 | 0.104 |
| danrasband | M | 30 | N | 1.68 | 14.79 | 0.113 |
| danrasband | P | 30 | Q | 9.66 | 31.06 | 0.311 |
| danrasband | T | 30 | L | 2.82 | 7.5 | 0.377 |
| grassknoted | M | 1500 | N | 3.03 | 26.59 | 0.114 |
| grassknoted | N | 1500 | M | 3.03 | 24.37 | 0.124 |
| grassknoted | T | 1500 | L | 2.38 | 10.33 | 0.231 |
| grassknoted | P | 1500 | H | 13.76 | 25.3 | 0.544 |
| massey | N | 1500 | M | 3.1 | 7.32 | 0.423 |
| massey | T | 1500 | A | 3.2 | 7.28 | 0.44 |
| massey | M | 1500 | N | 3.1 | 6.7 | 0.463 |
| massey | P | 1500 | Q | 7.14 | 15.42 | 0.463 |

### Is the thumb landmark garbage?

`in_palm%` is the share of frames where the thumb tip lands inside the palm polygon (expected for a tucked thumb, so T and M should not be 0); `thumb_len` is tip-to-wrist in palm units (a real thumb is ~1-1.3, a hallucinated one wanders); `jitter` is how much the thumb moves between two frames of the *same* sign (a stable landmark is < 0.4).

From `python src/inspect_data.py --thumb-audit T,N,M`:

| letter | dataset | n | in_palm% | tip_knuckle | tip_little | thumb_len | jitter |
|---|---|---|---|---|---|---|---|
| M | grassknoted | 2000 | 4.3 | 0.989 | 1.311 | 0.688 | 0.806 |
| M | massey | 2000 | 3.85 | 0.445 | 0.961 | 0.947 | 0.327 |
| M | danrasband | 30 | 33.333 | 0.698 | 0.871 | 0.42 | 0.502 |
| N | grassknoted | 1897 | 0.685 | 0.958 | 1.384 | 0.533 | 0.568 |
| N | massey | 2000 | 0.8 | 0.338 | 1.204 | 0.982 | 0.306 |
| N | danrasband | 30 | 20.0 | 0.712 | 0.851 | 0.337 | 0.373 |
| P | grassknoted | 2000 | 0.1 | 0.963 | 0.787 | 1.032 | 0.576 |
| P | massey | 1928 | 6.224 | 0.474 | 1.208 | 0.682 | 0.592 |
| P | danrasband | 30 | 0.0 | 0.95 | 0.486 | 1.051 | 0.191 |
| T | grassknoted | 2000 | 0.15 | 0.615 | 1.46 | 1.197 | 0.22 |
| T | massey | 2000 | 0.05 | 0.352 | 1.299 | 1.005 | 0.277 |
| T | danrasband | 30 | 0.0 | 0.55 | 1.268 | 1.029 | 0.197 |
