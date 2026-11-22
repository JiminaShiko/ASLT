# Live-path A-Z test (no camera)

Backend: `onnx` (`asl_all.onnx`), 71 features from `xy+tip_dists+curl+thumb`. Frames are real public dataset photographs, read BGR->RGB exactly as the webcam path reads them, and fed at 20 FPS through `live.frame_call` and `live.Smoother`. Nothing here uses a camera and nothing is saved to disk.

**This is a wiring test, not an accuracy test.** The model deployed by `live.py` and `app.py` is the final one, trained on every dataset in the cache, so every frame below is one it has already seen. A high score here proves the live path is connected correctly - colour order, video-mode timestamps, the shared `normalize.features()`, the gate, the smoother, the exported weights. It says nothing about transfer accuracy, which is the leave-one-dataset-out number in the profile's report. A *low* score here would be a bug, and that is the whole point.

Answered correctly per block: **92/95** (96.8%); 51/95 blocks committed inside the clip.

| measure | video tracking only | + retry (what ships) |
|---|---|---|
| loop speed, median per block (incl. decoding each frame from disk, which a webcam does not cost) | 49.0 frames/s | **60.2 frames/s** (PASS the 20 FPS target) |
| frames with a hand | 1625/1873 (86.8%) | 1830/1873 (97.7%) |
| usable detections/s | 38.1 | 59.8 |
| retry | – | recovery(conf=0.3, after=1, every=3, flip=on): 248 empty frames, retried on 234 (94.4% of them), found 205 hands (9 via mirror), 25.5 ms per try, 24.1 ms amortised |

The two columns are the same frames in the same process, so the difference is the retry and nothing else. This is the only hand-coverage figure in the project measured frame by frame through the live path. It is not the same quantity as the extractor's recall in the training report, which is offline **image** recall over all 122k files and can afford five detections per image because it runs once instead of thirty times a second. Neither is a webcam measurement; only the self-test (`python src/live.py --selftest`) is.


## live-path answer on `danrasband` frames — 23/24 letters reproduced (in-sample)

| letter | frames | hand found | tracking only | via retry | answer | correct | mean top1 | % below gate | % uncertain (gap) |
|---|---|---|---|---|---|---|---|---|---|
| A | 20 | 20 | 17 | 3 | A | yes | 92% | 5% | 5% |
| B | 20 | 20 | 17 | 3 | B | yes | 90% | 10% | 5% |
| C | 20 | 20 | 19 | 1 | C | yes | 75% | 35% | 10% |
| D | 20 | 20 | 19 | 1 | D | yes | 95% | 5% | 0% |
| E | 20 | 20 | 18 | 2 | E | yes | 84% | 25% | 0% |
| F | 20 | 20 | 20 | 0 | F | yes | 96% | 0% | 0% |
| G | 20 | 20 | 15 | 5 | G | yes | 84% | 25% | 0% |
| H | 20 | 20 | 20 | 0 | H | yes | 97% | 0% | 0% |
| I | 20 | 20 | 15 | 5 | I | yes | 93% | 5% | 0% |
| K | 20 | 20 | 19 | 1 | K | yes | 92% | 5% | 0% |
| L | 20 | 20 | 18 | 2 | L | yes | 88% | 10% | 10% |
| M | 20 | 19 | 13 | 6 | M | yes | 84% | 16% | 5% |
| N | 20 | 19 | 14 | 5 | M | no | 73% | 37% | 26% |
| O | 20 | 20 | 16 | 4 | O | yes | 90% | 15% | 0% |
| P | 20 | 20 | 17 | 3 | P | yes | 94% | 5% | 5% |
| Q | 20 | 20 | 16 | 4 | Q | yes | 81% | 25% | 25% |
| R | 20 | 20 | 18 | 2 | R | yes | 82% | 30% | 5% |
| S | 20 | 19 | 19 | 0 | S | yes | 89% | 5% | 0% |
| T | 20 | 20 | 19 | 1 | T | yes | 63% | 60% | 5% |
| U | 20 | 20 | 19 | 1 | U | yes | 78% | 25% | 0% |
| V | 20 | 19 | 15 | 4 | V | yes | 83% | 16% | 0% |
| W | 20 | 20 | 20 | 0 | W | yes | 88% | 15% | 5% |
| X | 20 | 20 | 13 | 7 | X | yes | 91% | 10% | 5% |
| Y | 20 | 20 | 16 | 4 | Y | yes | 92% | 5% | 5% |

## live-path answer on `grassknoted` frames — 24/24 letters reproduced (in-sample)

| letter | frames | hand found | tracking only | via retry | answer | correct | mean top1 | % below gate | % uncertain (gap) |
|---|---|---|---|---|---|---|---|---|---|
| A | 20 | 20 | 14 | 6 | A | yes | 92% | 5% | 0% |
| B | 20 | 20 | 13 | 7 | B | yes | 94% | 5% | 0% |
| C | 20 | 17 | 11 | 6 | C | yes | 91% | 6% | 0% |
| D | 20 | 20 | 18 | 2 | D | yes | 92% | 5% | 0% |
| E | 20 | 20 | 16 | 4 | E | yes | 87% | 15% | 5% |
| F | 20 | 20 | 16 | 4 | F | yes | 84% | 25% | 0% |
| G | 20 | 19 | 14 | 5 | G | yes | 86% | 16% | 11% |
| H | 20 | 18 | 14 | 4 | H | yes | 91% | 11% | 6% |
| I | 20 | 20 | 14 | 6 | I | yes | 94% | 5% | 0% |
| K | 20 | 20 | 16 | 4 | K | yes | 88% | 15% | 0% |
| L | 20 | 19 | 14 | 5 | L | yes | 93% | 5% | 0% |
| M | 20 | 19 | 11 | 8 | M | yes | 79% | 26% | 26% |
| N | 20 | 17 | 9 | 8 | N | yes | 90% | 6% | 6% |
| O | 20 | 18 | 13 | 5 | O | yes | 90% | 11% | 0% |
| P | 20 | 20 | 16 | 4 | P | yes | 90% | 10% | 5% |
| Q | 20 | 20 | 13 | 7 | Q | yes | 86% | 20% | 5% |
| R | 20 | 20 | 18 | 2 | R | yes | 83% | 25% | 0% |
| S | 20 | 19 | 14 | 5 | S | yes | 88% | 11% | 5% |
| T | 20 | 20 | 15 | 5 | T | yes | 88% | 15% | 5% |
| U | 20 | 20 | 11 | 9 | U | yes | 95% | 0% | 0% |
| V | 20 | 19 | 13 | 6 | V | yes | 95% | 5% | 0% |
| W | 20 | 19 | 14 | 5 | W | yes | 93% | 5% | 5% |
| X | 20 | 19 | 18 | 1 | X | yes | 79% | 26% | 11% |
| Y | 20 | 20 | 13 | 7 | Y | yes | 94% | 5% | 0% |

## live-path answer on `massey` frames — 24/24 letters reproduced (in-sample)

| letter | frames | hand found | tracking only | via retry | answer | correct | mean top1 | % below gate | % uncertain (gap) |
|---|---|---|---|---|---|---|---|---|---|
| A | 20 | 20 | 18 | 2 | A | yes | 79% | 30% | 15% |
| B | 20 | 20 | 20 | 0 | B | yes | 97% | 0% | 0% |
| C | 20 | 20 | 20 | 0 | C | yes | 92% | 10% | 0% |
| D | 20 | 20 | 18 | 2 | D | yes | 90% | 5% | 5% |
| E | 20 | 20 | 20 | 0 | E | yes | 89% | 10% | 5% |
| F | 20 | 20 | 20 | 0 | F | yes | 96% | 0% | 0% |
| G | 20 | 20 | 20 | 0 | G | yes | 87% | 15% | 0% |
| H | 20 | 20 | 20 | 0 | H | yes | 96% | 0% | 0% |
| I | 20 | 20 | 20 | 0 | I | yes | 94% | 5% | 5% |
| K | 20 | 20 | 20 | 0 | K | yes | 84% | 20% | 5% |
| L | 20 | 20 | 20 | 0 | L | yes | 97% | 0% | 0% |
| M | 20 | 20 | 20 | 0 | M | yes | 96% | 5% | 0% |
| N | 20 | 20 | 20 | 0 | N | yes | 92% | 10% | 5% |
| O | 20 | 20 | 20 | 0 | O | yes | 82% | 25% | 5% |
| P | 20 | 18 | 13 | 5 | P | yes | 87% | 17% | 6% |
| Q | 20 | 20 | 16 | 4 | Q | yes | 90% | 5% | 5% |
| R | 20 | 19 | 19 | 0 | R | yes | 90% | 5% | 0% |
| S | 20 | 20 | 20 | 0 | S | yes | 93% | 5% | 0% |
| T | 20 | 20 | 20 | 0 | T | yes | 85% | 20% | 0% |
| U | 20 | 20 | 20 | 0 | U | yes | 93% | 5% | 0% |
| V | 20 | 20 | 20 | 0 | V | yes | 93% | 5% | 0% |
| W | 20 | 20 | 20 | 0 | W | yes | 97% | 0% | 0% |
| X | 20 | 20 | 20 | 0 | X | yes | 87% | 15% | 5% |
| Y | 20 | 20 | 20 | 0 | Y | yes | 94% | 5% | 0% |

## live-path answer on `rushil` frames — 21/23 letters reproduced (in-sample)

| letter | frames | hand found | tracking only | via retry | answer | correct | mean top1 | % below gate | % uncertain (gap) |
|---|---|---|---|---|---|---|---|---|---|
| A | 20 | 20 | 20 | 0 | A | yes | 82% | 15% | 0% |
| B | 20 | 20 | 20 | 0 | B | yes | 97% | 0% | 0% |
| C | 20 | 20 | 20 | 0 | C | yes | 94% | 0% | 0% |
| D | 20 | 20 | 20 | 0 | D | yes | 96% | 0% | 0% |
| E | 20 | 20 | 20 | 0 | E | yes | 97% | 0% | 0% |
| F | 20 | 20 | 20 | 0 | F | yes | 91% | 0% | 0% |
| G | 17 | 17 | 17 | 0 | nan | no | 55% | 94% | 6% |
| I | 20 | 20 | 20 | 0 | I | yes | 93% | 5% | 0% |
| K | 20 | 20 | 20 | 0 | K | yes | 96% | 0% | 0% |
| L | 20 | 20 | 20 | 0 | L | yes | 97% | 0% | 0% |
| M | 20 | 20 | 20 | 0 | M | yes | 82% | 20% | 5% |
| N | 13 | 13 | 13 | 0 | N | yes | 78% | 23% | 8% |
| O | 20 | 20 | 20 | 0 | O | yes | 93% | 0% | 0% |
| P | 20 | 1 | 0 | 1 | nan | no | 38% | 100% | 100% |
| Q | 11 | 11 | 10 | 1 | Q | yes | 95% | 0% | 0% |
| R | 20 | 20 | 19 | 1 | R | yes | 99% | 0% | 0% |
| S | 20 | 20 | 20 | 0 | S | yes | 90% | 5% | 5% |
| T | 20 | 20 | 20 | 0 | T | yes | 96% | 0% | 0% |
| U | 20 | 20 | 20 | 0 | U | yes | 93% | 0% | 0% |
| V | 20 | 20 | 20 | 0 | V | yes | 94% | 0% | 0% |
| W | 20 | 20 | 20 | 0 | W | yes | 98% | 0% | 0% |
| X | 12 | 12 | 12 | 0 | X | yes | 71% | 42% | 0% |
| Y | 20 | 20 | 20 | 0 | Y | yes | 98% | 0% | 0% |

## Where the letters went, including the wrong ones

| committed | expected | count |
|---|---|---|
| A | A | 4 |
| B | B | 4 |
| C | C | 4 |
| D | D | 4 |
| E | E | 4 |
| F | F | 4 |
| I | I | 4 |
| K | K | 4 |
| L | L | 4 |
| M | M | 4 |
| O | O | 4 |
| Q | Q | 4 |
| R | R | 4 |
| S | S | 4 |
| T | T | 4 |
| U | U | 4 |
| V | V | 4 |
| W | W | 4 |
| X | X | 4 |
| Y | Y | 4 |
| G | G | 3 |
| H | H | 3 |
| P | P | 3 |
| N | N | 3 |
| M | N | 1 |
| nan | G | 1 |
| nan | P | 1 |

A block that commits *nothing* is the uncertain state working as designed: the gate or the top1-top2 margin refused to vote, which is better than a wrong letter on the screen. A block that commits the wrong letter is a real error and should be read against the profile report's per-letter table.

