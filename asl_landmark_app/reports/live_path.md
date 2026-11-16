# Live-path A-Z test (no camera)

Backend: `onnx` (`asl_all.onnx`), 71 features from `xy+tip_dists+curl+thumb`. Frames are real public dataset photographs, read BGR->RGB exactly as the webcam path reads them, and fed at 20 FPS through `live.frame_call` and `live.Smoother`. Nothing here uses a camera and nothing is saved to disk.

**This is a wiring test, not an accuracy test.** The model deployed by `live.py` and `app.py` is the final one, trained on every dataset in the cache, so every frame below is one it has already seen. A high score here proves the live path is connected correctly - colour order, video-mode timestamps, the shared `normalize.features()`, the gate, the smoother, the exported weights. It says nothing about transfer accuracy, which is the leave-one-dataset-out number in the profile's report. A *low* score here would be a bug, and that is the whole point.

Answered correctly per block: **70/72** (97.2%); 19/72 blocks committed inside the clip.

| measure | video tracking only | + retry (what ships) |
|---|---|---|
| loop speed, median per block (incl. decoding each frame from disk, which a webcam does not cost) | 99.3 frames/s | **82.4 frames/s** (PASS the 20 FPS target) |
| frames with a hand | 715/864 (82.8%) | 844/864 (97.7%) |
| usable detections/s | 82.9 | 82.4 |
| retry | – | recovery(conf=0.3, after=1, every=3, flip=on): 149 empty frames, retried on 145 (97.3% of them), found 129 hands (5 via mirror), 18.0 ms per try, 17.5 ms amortised |

The two columns are the same frames in the same process, so the difference is the retry and nothing else. This is the only hand-coverage figure in the project measured frame by frame through the live path. It is not the same quantity as the extractor's recall in the training report, which is offline **image** recall over all 122k files and can afford five detections per image because it runs once instead of thirty times a second. Neither is a webcam measurement; only the self-test (`python src/live.py --selftest`) is.

NOTE: only 19/72 clips were long enough for the 0.5s hold-to-commit timer to fire (12 frames = 0.6s of a 50ms tick, and the letter only has to be held long enough after it first clears the gate), so most `answer` cells below are the letter the app is still holding rather than one it committed. The held letter still exercises every stage of the pipeline; for a committed-letter count re-run with `--per 24`.

## live-path answer on `danrasband` frames — 22/24 letters reproduced (in-sample)

| letter | frames | hand found | tracking only | via retry | answer | correct | mean top1 | % below gate | % uncertain (gap) |
|---|---|---|---|---|---|---|---|---|---|
| A | 12 | 12 | 9 | 3 | A | yes | 89% | 17% | 0% |
| B | 12 | 12 | 8 | 4 | B | yes | 97% | 0% | 0% |
| C | 12 | 12 | 11 | 1 | C | yes | 80% | 25% | 0% |
| D | 12 | 12 | 12 | 0 | D | yes | 96% | 0% | 0% |
| E | 12 | 12 | 11 | 1 | E | yes | 95% | 8% | 0% |
| F | 12 | 12 | 11 | 1 | F | yes | 94% | 8% | 0% |
| G | 12 | 12 | 9 | 3 | G | yes | 92% | 8% | 0% |
| H | 12 | 12 | 12 | 0 | H | yes | 97% | 0% | 0% |
| I | 12 | 12 | 7 | 5 | I | yes | 97% | 0% | 0% |
| K | 12 | 12 | 11 | 1 | K | yes | 90% | 8% | 8% |
| L | 12 | 12 | 12 | 0 | L | yes | 89% | 8% | 8% |
| M | 12 | 11 | 7 | 4 | M | yes | 78% | 18% | 9% |
| N | 12 | 12 | 9 | 3 | M | no | 71% | 58% | 42% |
| O | 12 | 9 | 6 | 3 | O | yes | 84% | 22% | 0% |
| P | 12 | 12 | 11 | 1 | P | yes | 97% | 0% | 0% |
| Q | 12 | 11 | 8 | 3 | Q | yes | 87% | 9% | 0% |
| R | 12 | 12 | 10 | 2 | R | yes | 93% | 8% | 8% |
| S | 12 | 11 | 9 | 2 | S | yes | 78% | 18% | 9% |
| T | 12 | 12 | 10 | 2 | nan | no | 64% | 67% | 0% |
| U | 12 | 12 | 11 | 1 | U | yes | 72% | 42% | 8% |
| V | 12 | 12 | 11 | 1 | V | yes | 85% | 8% | 8% |
| W | 12 | 12 | 11 | 1 | W | yes | 95% | 0% | 0% |
| X | 12 | 12 | 9 | 3 | X | yes | 81% | 17% | 17% |
| Y | 12 | 12 | 10 | 2 | Y | yes | 92% | 8% | 0% |

## live-path answer on `grassknoted` frames — 24/24 letters reproduced (in-sample)

| letter | frames | hand found | tracking only | via retry | answer | correct | mean top1 | % below gate | % uncertain (gap) |
|---|---|---|---|---|---|---|---|---|---|
| A | 12 | 12 | 9 | 3 | A | yes | 85% | 17% | 8% |
| B | 12 | 9 | 7 | 2 | B | yes | 82% | 22% | 11% |
| C | 12 | 12 | 6 | 6 | C | yes | 95% | 8% | 0% |
| D | 12 | 12 | 12 | 0 | D | yes | 92% | 0% | 0% |
| E | 12 | 12 | 9 | 3 | E | yes | 84% | 25% | 0% |
| F | 12 | 12 | 12 | 0 | F | yes | 76% | 33% | 17% |
| G | 12 | 9 | 6 | 3 | G | yes | 96% | 0% | 0% |
| H | 12 | 12 | 9 | 3 | H | yes | 82% | 25% | 17% |
| I | 12 | 12 | 9 | 3 | I | yes | 94% | 0% | 0% |
| K | 12 | 12 | 10 | 2 | K | yes | 94% | 8% | 0% |
| L | 12 | 12 | 12 | 0 | L | yes | 78% | 33% | 8% |
| M | 12 | 11 | 6 | 5 | M | yes | 94% | 0% | 0% |
| N | 12 | 11 | 6 | 5 | N | yes | 90% | 0% | 0% |
| O | 12 | 12 | 8 | 4 | O | yes | 95% | 0% | 0% |
| P | 12 | 12 | 9 | 3 | P | yes | 92% | 8% | 0% |
| Q | 12 | 12 | 8 | 4 | Q | yes | 91% | 8% | 8% |
| R | 12 | 12 | 11 | 1 | R | yes | 86% | 17% | 0% |
| S | 12 | 10 | 7 | 3 | S | yes | 95% | 0% | 0% |
| T | 12 | 11 | 7 | 4 | T | yes | 92% | 9% | 0% |
| U | 12 | 12 | 7 | 5 | U | yes | 87% | 25% | 0% |
| V | 12 | 12 | 9 | 3 | V | yes | 78% | 33% | 17% |
| W | 12 | 12 | 8 | 4 | W | yes | 90% | 8% | 0% |
| X | 12 | 12 | 8 | 4 | X | yes | 77% | 33% | 17% |
| Y | 12 | 12 | 6 | 6 | Y | yes | 97% | 0% | 0% |

## live-path answer on `massey` frames — 24/24 letters reproduced (in-sample)

| letter | frames | hand found | tracking only | via retry | answer | correct | mean top1 | % below gate | % uncertain (gap) |
|---|---|---|---|---|---|---|---|---|---|
| A | 12 | 12 | 12 | 0 | A | yes | 76% | 33% | 25% |
| B | 12 | 12 | 12 | 0 | B | yes | 98% | 0% | 0% |
| C | 12 | 11 | 10 | 1 | C | yes | 86% | 18% | 9% |
| D | 12 | 12 | 11 | 1 | D | yes | 88% | 17% | 0% |
| E | 12 | 12 | 12 | 0 | E | yes | 91% | 8% | 0% |
| F | 12 | 12 | 12 | 0 | F | yes | 96% | 0% | 0% |
| G | 12 | 12 | 12 | 0 | G | yes | 89% | 8% | 8% |
| H | 12 | 12 | 12 | 0 | H | yes | 90% | 8% | 0% |
| I | 12 | 12 | 12 | 0 | I | yes | 89% | 8% | 8% |
| K | 12 | 12 | 12 | 0 | K | yes | 96% | 0% | 0% |
| L | 12 | 12 | 12 | 0 | L | yes | 93% | 8% | 0% |
| M | 12 | 12 | 12 | 0 | M | yes | 87% | 17% | 0% |
| N | 12 | 12 | 12 | 0 | N | yes | 88% | 17% | 8% |
| O | 12 | 10 | 8 | 2 | O | yes | 70% | 40% | 10% |
| P | 12 | 12 | 12 | 0 | P | yes | 76% | 25% | 17% |
| Q | 12 | 12 | 10 | 2 | Q | yes | 78% | 33% | 17% |
| R | 12 | 12 | 12 | 0 | R | yes | 90% | 17% | 8% |
| S | 12 | 12 | 12 | 0 | S | yes | 78% | 25% | 25% |
| T | 12 | 12 | 12 | 0 | T | yes | 91% | 8% | 0% |
| U | 12 | 12 | 12 | 0 | U | yes | 93% | 8% | 8% |
| V | 12 | 12 | 12 | 0 | V | yes | 91% | 8% | 8% |
| W | 12 | 12 | 12 | 0 | W | yes | 92% | 8% | 0% |
| X | 12 | 12 | 12 | 0 | X | yes | 91% | 0% | 0% |
| Y | 12 | 12 | 12 | 0 | Y | yes | 97% | 0% | 0% |

## Where the letters went, including the wrong ones

| committed | expected | count |
|---|---|---|
| A | A | 3 |
| B | B | 3 |
| C | C | 3 |
| D | D | 3 |
| E | E | 3 |
| F | F | 3 |
| G | G | 3 |
| H | H | 3 |
| I | I | 3 |
| K | K | 3 |
| L | L | 3 |
| M | M | 3 |
| O | O | 3 |
| P | P | 3 |
| Q | Q | 3 |
| R | R | 3 |
| S | S | 3 |
| U | U | 3 |
| V | V | 3 |
| W | W | 3 |
| X | X | 3 |
| Y | Y | 3 |
| N | N | 2 |
| T | T | 2 |
| M | N | 1 |
| nan | T | 1 |

A block that commits *nothing* is the uncertain state working as designed: the gate or the top1-top2 margin refused to vote, which is better than a wrong letter on the screen. A block that commits the wrong letter is a real error and should be read against the profile report's per-letter table.

