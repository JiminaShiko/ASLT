# Live-path A-Z test (no camera)

Backend: `onnx` (`asl_all.onnx`), 71 features from `xy+tip_dists+curl+thumb`. Frames are real public dataset photographs, read BGR->RGB exactly as the webcam path reads them, and fed at 20 FPS through `live.frame_call` and `live.Smoother`. Nothing here uses a camera and nothing is saved to disk.

**This is a wiring test, not an accuracy test.** The model deployed by `live.py` and `app.py` is the final one, trained on every dataset in the cache, so every frame below is one it has already seen. A high score here proves the live path is connected correctly - colour order, video-mode timestamps, the shared `normalize.features()`, the gate, the smoother, the exported weights. It says nothing about transfer accuracy, which is the leave-one-dataset-out number in the profile's report. A *low* score here would be a bug, and that is the whole point.

Answered correctly per block: **90/95** (94.7%); 0/95 blocks committed inside the clip.

| measure | video tracking only | + retry (what ships) |
|---|---|---|
| loop speed, median per block (incl. decoding each frame from disk, which a webcam does not cost) | 83.3 frames/s | **74.0 frames/s** (PASS the 20 FPS target) |
| frames with a hand | 809/950 (85.2%) | 924/950 (97.3%) |
| usable detections/s | 73.3 | 74.0 |
| retry | – | recovery(conf=0.3, after=1, every=3, flip=on): 141 empty frames, retried on 132 (93.6% of them), found 115 hands (3 via mirror), 20.0 ms per try, 18.7 ms amortised |

The two columns are the same frames in the same process, so the difference is the retry and nothing else. This is the only hand-coverage figure in the project measured frame by frame through the live path. It is not the same quantity as the extractor's recall in the training report, which is offline **image** recall over all 122k files and can afford five detections per image because it runs once instead of thirty times a second. Neither is a webcam measurement; only the self-test (`python src/live.py --selftest`) is.

NOTE: only 0/95 clips were long enough for the 0.5s hold-to-commit timer to fire (10 frames = 0.5s of a 50ms tick, and the letter only has to be held long enough after it first clears the gate), so most `answer` cells below are the letter the app is still holding rather than one it committed. The held letter still exercises every stage of the pipeline; for a committed-letter count re-run with `--per 21`.

## live-path answer on `danrasband` frames — 21/24 letters reproduced (in-sample)

| letter | frames | hand found | tracking only | via retry | answer | correct | mean top1 | % below gate | % uncertain (gap) |
|---|---|---|---|---|---|---|---|---|---|
| A | 10 | 10 | 10 | 0 | A | yes | 91% | 10% | 0% |
| B | 10 | 10 | 9 | 1 | B | yes | 80% | 20% | 20% |
| C | 10 | 10 | 9 | 1 | C | yes | 80% | 30% | 0% |
| D | 10 | 10 | 10 | 0 | D | yes | 89% | 10% | 10% |
| E | 10 | 10 | 9 | 1 | E | yes | 82% | 20% | 20% |
| F | 10 | 10 | 9 | 1 | F | yes | 96% | 0% | 0% |
| G | 10 | 10 | 9 | 1 | G | yes | 77% | 30% | 10% |
| H | 10 | 10 | 10 | 0 | H | yes | 95% | 0% | 0% |
| I | 10 | 9 | 7 | 2 | I | yes | 88% | 11% | 11% |
| K | 10 | 10 | 9 | 1 | K | yes | 89% | 10% | 0% |
| L | 10 | 10 | 9 | 1 | L | yes | 95% | 0% | 0% |
| M | 10 | 9 | 6 | 3 | M | yes | 83% | 22% | 11% |
| N | 10 | 10 | 6 | 4 | M | no | 81% | 20% | 20% |
| O | 10 | 10 | 6 | 4 | O | yes | 96% | 0% | 0% |
| P | 10 | 10 | 8 | 2 | P | yes | 91% | 10% | 10% |
| Q | 10 | 10 | 8 | 2 | Q | yes | 88% | 0% | 0% |
| R | 10 | 10 | 9 | 1 | R | yes | 89% | 20% | 0% |
| S | 10 | 10 | 9 | 1 | S | yes | 87% | 10% | 0% |
| T | 10 | 10 | 8 | 2 | nan | no | 62% | 60% | 30% |
| U | 10 | 10 | 9 | 1 | nan | no | 78% | 40% | 0% |
| V | 10 | 10 | 9 | 1 | V | yes | 82% | 20% | 0% |
| W | 10 | 10 | 9 | 1 | W | yes | 89% | 10% | 10% |
| X | 10 | 10 | 6 | 4 | X | yes | 92% | 10% | 0% |
| Y | 10 | 10 | 7 | 3 | Y | yes | 91% | 10% | 0% |

## live-path answer on `grassknoted` frames — 24/24 letters reproduced (in-sample)

| letter | frames | hand found | tracking only | via retry | answer | correct | mean top1 | % below gate | % uncertain (gap) |
|---|---|---|---|---|---|---|---|---|---|
| A | 10 | 10 | 7 | 3 | A | yes | 76% | 30% | 30% |
| B | 10 | 10 | 7 | 3 | B | yes | 98% | 0% | 0% |
| C | 10 | 9 | 6 | 3 | C | yes | 89% | 11% | 0% |
| D | 10 | 8 | 7 | 1 | D | yes | 98% | 0% | 0% |
| E | 10 | 10 | 7 | 3 | E | yes | 90% | 0% | 0% |
| F | 10 | 10 | 10 | 0 | F | yes | 90% | 10% | 10% |
| G | 10 | 9 | 6 | 3 | G | yes | 93% | 11% | 0% |
| H | 10 | 10 | 7 | 3 | H | yes | 94% | 10% | 10% |
| I | 10 | 9 | 7 | 2 | I | yes | 81% | 22% | 0% |
| K | 10 | 10 | 7 | 3 | K | yes | 82% | 20% | 20% |
| L | 10 | 10 | 10 | 0 | L | yes | 83% | 30% | 0% |
| M | 10 | 9 | 5 | 4 | M | yes | 72% | 33% | 22% |
| N | 10 | 9 | 5 | 4 | N | yes | 88% | 11% | 11% |
| O | 10 | 8 | 4 | 4 | O | yes | 89% | 12% | 0% |
| P | 10 | 8 | 6 | 2 | P | yes | 98% | 0% | 0% |
| Q | 10 | 10 | 7 | 3 | Q | yes | 83% | 30% | 10% |
| R | 10 | 10 | 9 | 1 | R | yes | 91% | 20% | 0% |
| S | 10 | 10 | 6 | 4 | S | yes | 97% | 0% | 0% |
| T | 10 | 10 | 8 | 2 | T | yes | 82% | 20% | 10% |
| U | 10 | 10 | 6 | 4 | U | yes | 93% | 0% | 0% |
| V | 10 | 10 | 5 | 5 | V | yes | 97% | 0% | 0% |
| W | 10 | 10 | 6 | 4 | W | yes | 97% | 0% | 0% |
| X | 10 | 9 | 6 | 3 | X | yes | 89% | 11% | 11% |
| Y | 10 | 10 | 6 | 4 | Y | yes | 89% | 20% | 0% |

## live-path answer on `massey` frames — 24/24 letters reproduced (in-sample)

| letter | frames | hand found | tracking only | via retry | answer | correct | mean top1 | % below gate | % uncertain (gap) |
|---|---|---|---|---|---|---|---|---|---|
| A | 10 | 8 | 6 | 2 | A | yes | 77% | 25% | 12% |
| B | 10 | 10 | 10 | 0 | B | yes | 97% | 0% | 0% |
| C | 10 | 10 | 8 | 2 | C | yes | 93% | 10% | 0% |
| D | 10 | 10 | 10 | 0 | D | yes | 77% | 30% | 20% |
| E | 10 | 10 | 10 | 0 | E | yes | 92% | 10% | 0% |
| F | 10 | 10 | 10 | 0 | F | yes | 96% | 0% | 0% |
| G | 10 | 10 | 10 | 0 | G | yes | 87% | 10% | 10% |
| H | 10 | 10 | 10 | 0 | H | yes | 92% | 10% | 0% |
| I | 10 | 10 | 10 | 0 | I | yes | 95% | 0% | 0% |
| K | 10 | 10 | 10 | 0 | K | yes | 74% | 40% | 10% |
| L | 10 | 10 | 10 | 0 | L | yes | 97% | 0% | 0% |
| M | 10 | 10 | 10 | 0 | M | yes | 91% | 10% | 0% |
| N | 10 | 10 | 10 | 0 | N | yes | 98% | 0% | 0% |
| O | 10 | 10 | 9 | 1 | O | yes | 81% | 30% | 0% |
| P | 10 | 10 | 9 | 1 | P | yes | 77% | 30% | 20% |
| Q | 10 | 10 | 10 | 0 | Q | yes | 77% | 30% | 20% |
| R | 10 | 10 | 9 | 1 | R | yes | 95% | 0% | 0% |
| S | 10 | 9 | 9 | 0 | S | yes | 73% | 33% | 11% |
| T | 10 | 10 | 10 | 0 | T | yes | 94% | 0% | 0% |
| U | 10 | 10 | 10 | 0 | U | yes | 93% | 10% | 0% |
| V | 10 | 10 | 9 | 1 | V | yes | 81% | 20% | 10% |
| W | 10 | 10 | 10 | 0 | W | yes | 97% | 0% | 0% |
| X | 10 | 10 | 10 | 0 | X | yes | 79% | 30% | 0% |
| Y | 10 | 10 | 10 | 0 | Y | yes | 97% | 0% | 0% |

## live-path answer on `rushil` frames — 21/23 letters reproduced (in-sample)

| letter | frames | hand found | tracking only | via retry | answer | correct | mean top1 | % below gate | % uncertain (gap) |
|---|---|---|---|---|---|---|---|---|---|
| A | 10 | 10 | 10 | 0 | A | yes | 82% | 20% | 0% |
| B | 10 | 10 | 10 | 0 | B | yes | 98% | 0% | 0% |
| C | 10 | 10 | 10 | 0 | C | yes | 95% | 0% | 0% |
| D | 10 | 10 | 10 | 0 | D | yes | 95% | 0% | 0% |
| E | 10 | 10 | 10 | 0 | E | yes | 98% | 0% | 0% |
| F | 10 | 10 | 10 | 0 | F | yes | 91% | 0% | 0% |
| G | 10 | 10 | 10 | 0 | nan | no | 54% | 100% | 20% |
| I | 10 | 10 | 10 | 0 | I | yes | 82% | 20% | 0% |
| K | 10 | 10 | 10 | 0 | K | yes | 96% | 0% | 0% |
| L | 10 | 10 | 10 | 0 | L | yes | 97% | 0% | 0% |
| M | 10 | 10 | 10 | 0 | M | yes | 86% | 20% | 0% |
| N | 10 | 10 | 10 | 0 | N | yes | 84% | 20% | 0% |
| O | 10 | 10 | 10 | 0 | O | yes | 92% | 0% | 0% |
| P | 10 | 1 | 0 | 1 | nan | no | 31% | 100% | 0% |
| Q | 10 | 10 | 10 | 0 | Q | yes | 94% | 0% | 0% |
| R | 10 | 10 | 10 | 0 | R | yes | 83% | 20% | 20% |
| S | 10 | 10 | 10 | 0 | S | yes | 87% | 20% | 10% |
| T | 10 | 10 | 10 | 0 | T | yes | 91% | 0% | 0% |
| U | 10 | 10 | 10 | 0 | U | yes | 92% | 0% | 0% |
| V | 10 | 10 | 10 | 0 | V | yes | 95% | 0% | 0% |
| W | 10 | 10 | 10 | 0 | W | yes | 98% | 0% | 0% |
| X | 10 | 10 | 10 | 0 | X | yes | 73% | 50% | 0% |
| Y | 10 | 10 | 10 | 0 | Y | yes | 98% | 0% | 0% |

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
| V | V | 4 |
| W | W | 4 |
| X | X | 4 |
| Y | Y | 4 |
| G | G | 3 |
| H | H | 3 |
| P | P | 3 |
| N | N | 3 |
| T | T | 3 |
| U | U | 3 |
| M | N | 1 |
| nan | T | 1 |
| nan | U | 1 |
| nan | G | 1 |
| nan | P | 1 |

A block that commits *nothing* is the uncertain state working as designed: the gate or the top1-top2 margin refused to vote, which is better than a wrong letter on the screen. A block that commits the wrong letter is a real error and should be read against the profile report's per-letter table.

