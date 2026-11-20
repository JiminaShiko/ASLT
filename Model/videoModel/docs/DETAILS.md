videoModel - long reference
===========================

README.md is the guide: what to run and in what order. This file is the
reference: the data contract, the feature contract, every threshold, what
each report column means, what each verification mode proves, and how to
change things.

The reported accuracy in this reference is for the historical 20-word model.
The expanded candidate has not been trained yet.

**Current-work note (2026-11-19):** most of this long reference describes the
original 20-word model. The active vocabulary now has 94 signs. Use the current
candidate workflow in `README.md`; the default `run_all.py` output scope writes
to the shipped-model directory and is not the command for this expansion.


1. WHAT THIS MODULE IS AND IS NOT
---------------------------------
Isolated-word ASL recognition from a webcam, built on ASL Citizen.

```
it is        landmarks, not pixels; one shared feature function for training
             and live; signer-disjoint evaluation; an importable API; a local
             OpenCV demo window; reports that name their own sample counts
it is not    a sentence translator (signing order is preserved, English is
             not reordered, and there is no copula in ASL)
             a fingerspelling reader
             a model fitted to any user's webcam (never done, never allowed)
             a web app - the Streamlit webcam app belongs to the alphabet
             project (../asl_landmark_app) and this module does not touch it
```

Self-contained: every path is derived from `src/config.ROOT`, which is the
folder holding this file. Nothing imports from a sibling project.


2. DATA CONTRACT
----------------
```
dataset   ASL_Citizen/ (a ~70 GB manual download; never move, rename,
          re-download or delete it, and never write inside it)
            videos/       one flat folder of *.mp4 isolated word clips
            splits/       train.csv, val.csv, test.csv (official)
            use.txt       licence terms, read by check_results.py and quoted
                          into models/meta.json
naming    <snowflake-id>-<GLOSS>.mp4, and some glosses contain a space, so
          "6508465822424006-THANK YOU.mp4" is normal. Never split a filename
          on "-"; the gloss is everything after the first hyphen.
content   ~83,400 clips, ~2,731 distinct glosses, 52 participants
splits    signer-disjoint by construction: 35 train / 6 val / 11 test
          participants. A participant appears in exactly one split, so no
          number in this project can be an "unseen hand" lie.
```

Gloss resolution (`src/vocab.py`)
```
requested word -> one or more exact gloss labels -> clips -> signers
- matching is exact against the gloss column, case-folded; a gloss that is
  not in the dataset makes the word MISSING, and status.py says so.
- variants are merged only when they are the same sign. The two are listed
  with their own rows so the decision is visible:
      eat    EAT1 + EAT2         merged   (the two ASL Citizen entries are
      drink  DRINK1 + DRINK2     merged    the same sign, different ids)
      thanks THANKYOU            alone    THANKSGIVING is NOT merged
      home   HOME                alone    HOMEWORK is NOT merged
      no     NO                  alone    NOON1/NOON2 are 'noon'
      food   (none)              MISSING  the dataset has no FOOD gloss at
                                          all, and EAT/MEAT are different
                                          signs, so nothing to merge into
- deliberately excluded glosses are recorded in vocab.json under
  excluded_glosses_deliberately and stay excluded; tests/test_logic.py
  asserts none of them crept back in.
```

Extra classes, both capped and both deterministic (a seed makes the sample
reproducible; the sample is recorded in the cache manifest):
```
REST    idle head/tail frames of the very same clips, trimmed by the signer's
        own shoulder test. min 4 frames, max 96, at most two slices per clip
        (one head, one tail). caps 600 train / 150 val / 300 test.
OTHER   a capped random sample of clips whose gloss is not in the vocabulary.
        per_gloss_max 2 so one popular sign cannot become the class. caps
        600 / 150 / 300.
```

Why they exist: without REST the demo would type a word the moment the camera
opened, and without OTHER the model would force every unknown sign into one
of our 20 words. Both are reported separately, never folded into "accuracy".


3. THE PACKED LANDMARK FORMAT
-----------------------------
`cache/landmarks/<same-name-as-the-video>.npy`, one array per clip:
`[T, 136, 3]` float16.

```
rows   0..26    pose upper body (MediaPipe pose landmarks 0-26; legs cropped)
       27..93   face subset - the sorted union of the lip, nose, chin/jaw and
                brow/forehead mesh indices (67 rows)
       94..114  left hand, 21 points, wrist first
       115..135 right hand, 21 points, wrist first
value  normalised image x, y and depth z; NaN where that group was never
       detected in that frame. No interpolation, no padding, no resizing.
```

The face subset is deliberately small. What a word needs from the face is
where the mouth is, how open and how wide it is and where the brows sit -
which is also what survives a mirror. The full 468-point mesh would be 7x the
cache for features that do not transfer through a flip.

`cache/manifest.csv`, one row per processed clip:
```
video, split, gloss, word, participant, frames, fps, stride, hand_rate,
hand_rate_seg, sign_start, sign_end, rest_frames, ms, restart_ms, bytes,
landmark_version, status, error
```
`status` is `ok` or `failed`; failures also get a line in
`cache/extract_errors.csv`. The two hand rates are NOT the same measurement
and confusing them has already misled this project once:

```
hand_rate      fraction of every frame of the video with at least one hand
               detected - including the idle head and tail the recorder
               leaves there. Mean 0.44. It says how the clip was filmed.
hand_rate_seg  the same fraction inside sign_start:sign_end only, the window
               the model is actually shown. Mean 0.90, and no core word is
               under 0.69. This is the one to judge a bad word by.
```

`python -m src.extract --rates` recomputes `hand_rate_seg` for every cached
clip straight from the `.npy` files - no MediaPipe, no videos, under a
second - and prints the per-word table. `restart_ms` is the cost of restarting
the MediaPipe graph, which is why extraction batches clips per process instead
of per frame.

Versioning: `LANDMARK_VERSION` bumps when this layout changes, `FEATURE_VERSION`
when the feature vector changes. The manifest records the landmark version, so
an old cache cannot be silently read as if it were new; `features.sequence()`
raises with the re-extract command in the message instead of guessing.
`models/meta.json` records the feature version and width the weights were
trained at, and the live path compares that to the code on every start: a
mismatch is reported as "uncertain" with a retrain reason rather than crashing
inside torch or, worse, reading garbage as a word.


4. FEATURES (src/features.py)
-----------------------------
One function, `sequence(arr, num_frames, augment, rng, blocks)`, used by
extraction-time training, by the offline verifiers and by every live frame.
`[T, 136, 3] -> [64, 200]`.

```
step             what happens
gap fill         an interior run of missing hand landmarks that is no longer
                 than GAP_FILL_FRAMES (5) source frames is linearly bridged
                 before anything else, so a two-frame detector flicker is not
                 recorded as two absences. 76% of all absence runs in the
                 cache are that short; the long ones are the hand genuinely
                 leaving the frame and are left alone. The presence flags and
                 the hand mask come from the UN-filled rows, so a bridged
                 frame is borrowed geometry, never a claimed sighting - and
                 the live path fills gaps the same way, in the same call.
resample         NaN-aware linear resampling to 64 frames. A clip of 20 or
                 200 frames becomes the same shape without stretching NaN.
centre           subtract the shoulder midpoint
scale            divide by shoulder width (so camera distance does not matter)
y direction      flipped so up is up (MediaPipe's y grows downwards)
missing group    its columns become 0 and its presence flag says 0
output           always finite; an all-NaN clip gives a valid all-zero-ish row
```

Block layout (widths are what the tests assert, so a change here fails loud):
```
block          width  contents
pose            30    15 upper-body points x,y: nose, eyes, ears, mouth
                      corners, shoulders, elbows, wrists, hips
hands           84    both hands, 21 points, x,y
hand_shape      18    9 per hand: palm size, finger curls, spreads,
                      thumb-index relation, pinky
face            12    mouth centre, width, openness, corner lift relative to
                      centre, nose, chin, brow line, brow width
hand_to_body    10    5 per hand: wrist-nose, palm-chin, palm-chest,
                      nearest fingertip-lips, palm above shoulder line
mask             4    left hand, right hand, pose, face present
velocity        12    wrist/palm/mouth speeds, signed y and x motion, wrist
                      acceleration
segment         30    the whole-clip read-out, per hand (14 each) plus 2
                      shared: distance to mouth and to chest at the start,
                      middle and end of the sign, maximum excursion, travel,
                      distance still gained after the closest mouth approach,
                      wrist height at that approach, the fraction of frames
                      the hand was seen, then wrist-to-wrist at the end and
                      the fraction of frames with BOTH hands visible. These
                      are constants per clip, in shoulder widths with y up,
                      and they are what GOOD-vs-THANKS (same hand shape,
                      different place and different ending) can be read from.
total          200
```

Why hand_shape and hand_to_body exist: the raw hand coordinates alone make
many signs look like a blob near the face; the shape descriptors and the
relation to the body are what separate, say, HELP from PLEASE at a glance.

Augmentation (`augment=True`, train split only) is listed completely in
`features.AUGMENT_DOC`, and `tests/test_logic.py` asserts every knob in
`config.AUG` appears in that text:
```
landmark level   mirror p=0.50 (swap hand blocks + swap anatomical pose pairs
                             + negate x - this is the handedness handling, so
                             the model never needs to know which hand is
                             dominant)
                 scale +-10%, rotate +-8 deg, translate 0.06 shoulder widths,
                 jitter sigma 0.015, sub-window at 0.8-1.2x speed
feature level    frame dropout 5% (repeat previous frame), velocity scale
                 U(0.8, 1.2)
```
Nothing here is a photometric augmentation: there is no pixel path, so colour,
brightness and background have no effect by construction.

The live path uses `features.RingBuffer` and `from_live_buffer()`, which is
literally `sequence()` on the same rows - the test asserts the buffer path and
the offline path give bit-identical output for the same input.


5. MODEL AND TRAINING (src/model.py, src/train.py)
--------------------------------------------------
```
architecture  conv1d x2 (64 then 96 channels, kernel 5, GroupNorm(8), GELU,
              MaxPool1d(2)) -> bidirectional GRU x2, hidden 96 -> dropout 0.3
              -> Linear(2*96, 22) -> logits. Softmax is applied by predict(),
              so the exported graph and the checkpoint end in probabilities.
              Pooling over time is a MAX, not a mean: a sign is identified by
              the frame where it passes through its target shape, and averaging
              would dilute exactly that frame.
input         [B, 64, 200] float32 (a bare [64, 200] row is accepted and gets
              a batch axis)
params        378,230 at 200 dims (1.5 MB on disk); the shipped 170-dim run
              was 368,630
standardise   NaN-aware per-feature mean/std over the TRAIN split, measured in
              one pass before epoch 1 and saved as buffers inside the module,
              so live and training cannot drift apart and the ONNX graph
              carries it too
loss          cross-entropy, label smoothing 0.05, inverse-frequency class
              weights normalised to mean 1 (a class with no samples gets the
              median count, never infinity)
sampling      class-balanced with replacement (dataset.sampler_weights),
              order and augmentation drawn from
              np.random.default_rng(SEED * 1_000_003 + epoch), so a rerun
              reproduces the run and every epoch still sees a different order
optimiser     AdamW, lr 1e-3, weight decay 1e-4, gradient clip 1.0,
              OneCycleLR (max_lr 1e-3, pct_start 0.15) stepped per batch,
              batch 64, 60 epochs, early stopping patience 8 on VAL top-1
resume        models/checkpoints/train_state.pt holds model, optimiser, epoch,
              best val, bad-epoch counter, seed, epochs, arch and the sample
              fingerprint. It is only trusted when seed, epochs, fingerprint
              and arch all match the current run; otherwise the log says
              "checkpoint is from a different run" and starts fresh. Because
              OneCycleLR counts batches, the resume path puts the scheduler
              clock back to start_epoch * steps_per_epoch - 1 and restores the
              learning rate from it.
outputs       models/words_core20.pt      weights + arch + mean/std + meta
              models/meta.json            the same meta as a plain file
              models/checkpoints/         resumable state (train_state.pt)
              logs/train.log
```

The single-artifact rule: one weights file, one ONNX graph, plain names, no
`_v2`/`_final`/`_best` side copies. A rerun replaces the file; a smoke run
writes into `models/checkpoints/smoke/` and is labelled `smoke: true` in its
own meta so it can never be mistaken for the release.

What a run writes into `models/meta.json`:
```
model, created, labels, num_classes, vocab_batches, feature_dim, num_frames,
landmark_version, feature_version, arch, params, train_samples, train_signers,
train_videos, best_val_top1, epochs_ran, seed, smoke, sample_fingerprint,
eval{split: {...}}, test_top1, test_n_signers, licence, note
```
`sample_fingerprint` is a hash of the exact `(video, kind, start, end)` list,
so a report can be tied to the precise data it came from and a changed cache
shows up as a changed fingerprint.


6. EVALUATION RULES AND THE REPORTS
-----------------------------------
Non-negotiable, and encoded in the code rather than in a comment:
```
- only the official signer-disjoint splits are ever scored
- val decides epochs, lr and refusal margins; test is scored once at the end
- every number is reported with the videos and distinct signers behind it
- REST and OTHER are shown separately, including the failure direction
  (REST typed as a word, OTHER typed as a word, a word typed as REST/OTHER)
- accuracy is reported both with and without the background classes
- a smoke run is announced as meaningless
```

Files in `reports/`, per split (`test` has no suffix, others get `_val` /
`_train`):
```
words_summary.json        the compact file check_results.py reads: per split
                          top1_all, top5_all, macro_top1_words, n_words_scored,
                          background counts, top1_words_only, epochs_ran
per_word[_split].csv      label, n_samples, videos, signers, top1, top5,
                          mean_p1, wrong
confusion_words[_split].csv  the full 22 x 22 matrix
confused_pairs[_split].csv   pair, swapped, a_as_b, b_as_a, n_a, n_b,
                             swap_rate - word classes only
predictions[_split].csv      every sample: video, gloss, label, participant,
                             kind, start, end, pred, p1, margin, top3, correct
```
`predictions*.csv` is what makes a claim checkable: you can always answer
"which signer, which clip, what probability".

`check_results.py` reads only the compact files plus the last 30 lines of
`logs/train.log`, prints a pasteable summary and writes `REPORT_WORDS.md`. It
exits non-zero while the files it needs do not exist, and its verdict section
names the blockers instead of the pretty numbers: an unfinished run, a smoke
run, a val-vs-test gap, REST being typed as a word, words never recognised,
extraction errors, and words with a low hand detection rate.

The pairs to expect confusion on (from the vocabulary itself, not from a
hunch): PLEASE/SORRY, GOOD/THANKS (thanks has a two-hand dominant-hand shape
that closes toward the chest like please), EAT/DRINK (merged variants of the
same handshape moved at different heights).


7. LIVE DECISIONS (src/predictor.py, src/live.py)
-------------------------------------------------
Segmentation, per frame:
```
hand above the shoulder line?  a detected wrist (pose) or any detected hand
                               point at or above shoulder_mid - margin 0.02
                               - the SAME test src/extract.py used to trim the
                               dataset clips, so the two paths cannot disagree
Segmenter(hold=3, rest=5)      opens a segment after 3 consecutive signing
                               frames, closes it after 5 consecutive rest
                               frames, so one noisy frame neither starts nor
                               ends a word. It returns SIGNING / REST / END,
                               and END is reported once.
decision frames                only the frames whose hands were up. The 5 rest
                               frames that closed the segment never enter the
                               feature vector, because training windows were
                               trimmed the same way.
rolling buffer                 BUFFER_SECONDS 2.5 s x fps, memory only
stride                         LIVE_STRIDE (1 = read every frame);
                               reading_due(frame_i, stride) keeps the schedule
                               honest when you set it to 2
```

Commit rule, in one pure function `commitable(Reading)` that both front-ends
use, so a threshold can never mean two things in two places:
```
p1 >= CONFIDENCE_MIN 0.85
and top1 - top2 >= MARGIN_MIN 0.15
and (top1, top2) not covered by a bigger PAIR_MARGINS override
and top1 is neither REST nor OTHER
otherwise: state UNCERTAIN, display says "uncertain", and reason says which
bar it missed
```
The shipped gate is 0.85 because that is where the measured trade-off sits
(75% of real words still typed, 96% of those typed correctly, OTHER slips
12%, REST 2%). Those four numbers are `typed_share`, `typed_precision`,
`other_slip` and `rest_slip` - computed by `gate_metrics()` in `src/train.py`
from `reports/predictions.csv`, recorded per seed in `reports/seed_runs.csv`,
and printed by `check_results.py`. The gate itself is a flag in the live demo
(`--conf`), not a code change.

Margin arithmetic worth knowing before you tune it: for two probabilities the
gap cannot be smaller than 2*p1 - 1, so at p1 >= 0.85 the gap is at least
0.70 and `MARGIN_MIN 0.15` can never bite. The margin (and `PAIR_MARGINS`)
only becomes a real lever once `--conf` is below about 0.58; that is what
`--margin` is for, and `tests/test_logic.py` asserts both cases rather than
assuming them.
`PAIR_MARGINS` keys are accepted as `("please", "sorry")` or `"please|sorry"`,
are case-folded, and anything that is not exactly two different words raises
`ValueError` at read time - a mistyped key would otherwise silently disable a
refusal margin, which is the worst kind of quiet bug in this file. Fill it in
from `reports/confused_pairs.csv` after the first real training run.

Phrase building (`src/phrases.py` + `phrases.json`, editable without code):
```
keeps the last max_buffer_words (8) committed words
tries the longest rule first, left to right, replaces the matched run with the
rule's text; unmatched words are typed as themselves, space separated
word_forms maps a single word (me -> I, thanks -> thank you)
REST and OTHER are never committed and never break a phrase
grammar note: ASL has no copula. ME SAD means "I am sad". The expansion in
phrases.json is a written-English convenience and you may delete it; signing
order is preserved and English word order is not reordered.
`python -m src.phrases` prints the rules, marks the ones today's 20 words
cannot fire, and shows example sentences.
```

The window (`src/live.py`): mirrored video, skeleton overlay drawn from
exactly the rows `features.py` reads (upper body, both hands, the face
subset), plus a HUD with state, live fps, the sentence so far, the top-3
boxes with probabilities, and the refusal reason when nothing was committed.
Keys: `q` or ESC quit, `r` reset. It prints "nothing is saved" in the HUD so
the guarantee is visible while it holds.

`WordPredictor.update()` return keys:
```
state            REST | SIGNING | UNCERTAIN
top3             [(word, p), ...] sorted, at most 3
committed_word   the word this segment just added, or ""
text             the sentence after the phrase rules
p1, margin       rounded, as the HUD shows them
reason           why a reading was refused ("" if committed)
buffer_frames, decision_frames, segment_frames, hand_in_frame, segment_cut
```
Input is either a `uint8 [H, W, 3]` picture (MediaPipe runs on it, in memory,
with a monotonic timestamp clamp) or a landmark row `[136, 3]`. Anything else
raises `ValueError`. `close()` releases the camera graph.


8. ONNX
-------
`python -m src.export_onnx` writes `models/words_core20.onnx`: opset 17,
dynamic batch, wrapped in a `_Probs` module so the graph returns
probabilities, and the module is put in `eval()` before export (torch.onnx
leaves it in train mode, and dropout then makes the graph disagree with the
checkpoint - that was a real bug and the reason the parity step exists).
`python -m src.train --parity` runs the same real clips through torch and
through onnxruntime and prints the biggest difference; the shipped checkpoint
measures 9.5e-07 with 64/64 identical top-1 (the smoke checkpoint was 1.5e-08
on 9 clips). Parity is checked on cached
clips, never on random noise, because noise does not exercise the
hand-absent and mask paths.


9. PRIVACY AND DATASET TERMS
----------------------------
```
- no user photos or videos are used for training, calibration or tuning. The
  only training data is ASL Citizen.
- the live path holds frames in local variables. src/live.py and
  src/predictor.py contain no cv2.VideoWriter, no imwrite and no np.save;
  `--render-test` proves it by snapshotting the folder tree (names, sizes and
  mtimes) before and after streaming real uint8 frames through update() and
  the drawing code, and reports files_touched: [] and nothing_saved: true.
- `--selftest` prints its table to the terminal only.
- ASL Citizen is Microsoft Research data for non-commercial research use;
  the terms are in ASL_Citizen/use.txt, and models/meta.json carries the
  licence line so a weights file cannot travel without it. Cite Desai et
  al., arXiv:2304.05934, 2023 if you use the data.
- MediaPipe Holistic is Apache-2.0 (Google). holistic_landmarker.task
  (13,683,609 bytes) is fetched once into models/ on first use; point
  ASL_WORDS_HOLISTIC_MODEL at a local copy to avoid the download.
- external corpora: nothing has been downloaded. The measured feasibility
  read-out for the two candidates - per-word coverage, size, licence, the
  conversion work and the cost of each step - is docs/EXTERNAL_DATA.md.
  PopSign (CC BY 4.0) would be a second training source, so the "only
  training data is ASL Citizen" bullet above stops being true the day it is
  added, and models/meta.json would have to carry both attributions.
- the Kaggle ISLR landmark corpus may not be handed to anyone who has not
  accepted its competition rules (rules section 7.B forbids redistributing
  the Competition Data), so it can never be committed to this repo; each
  person would download their own copy with their own Kaggle account.
```


10. TIMINGS (Windows 11, Python 3.13.15, CPU only, 16 logical CPUs)
---------------------------------------------------------------------------
Pre-run estimates first, then what the real run actually did, so the two can
be compared. All of it was measured on this machine; the "how" column says
where each number comes from.
```
what                        number          how proven
extraction per video        1.70/2.12/2.21s  3 probe clips, single process
   (estimated)              ~2.0 s mean      (cache/manifest.csv ms column)
graph restart               169-419 ms       restart_ms column, same rows
feature build               8.5 ms/sample    smoke run log line
train step + backward       0.73 ms/sample   600 random samples, batch 32
eval forward                0.13 ms/sample   same tensors, no grad
live update() median        24.58 ms         --benchmark on cached clips
                            = 40.7 fps       target is >= 15
  of which landmarks        23.71 ms         MediaPipe itself
  of which features          1.53 ms         features.sequence
  of which model             2.24 ms         net.predict

THE REAL RUN, 2026-11-16
whole work list             1723 videos      vocab.videos_to_process():
                            920/234/569      920 train, 234 val, 569 test
extraction, 4 workers       1471 s = 24.5    "done:" line, extract_*.log
                            min, 0 errors
extraction per frame        38.55 ms         same line (94.5 frames/s across
                                             the 4 processes)
whole cache on disk         109 MB           du over cache/landmarks, 64.5 KB
                                             per clip mean
train, 1520 samples         5.01 s per epoch train.log, 58 epochs then early
                            = 300 s total    stop at patience 8
onnx parity                 9.5e-07          --parity on the real weights,
                                             64/64 same top-1
```
The estimate said 15-45 minutes for extraction at 4 workers and 20-40 for
training; it took 24.5 and 5.0 minutes. `run_all.py` start to finish on this
machine is about 30 minutes, and the log prints its own per-epoch time from
epoch 1, which is always the number to trust.


11. VERIFICATION BATTERY - WHAT EACH MODE ACTUALLY PROVES
---------------------------------------------------------
```
python -m tests.test_logic        129 checks, no camera and no dataset
                                  writes. One check is allowed to fail, and it
                                  is a feature: "the weights were built by
                                  this version of features.py" compares
                                  meta.json's feature version and width to the
                                  code, so an old checkpoint cannot be quietly
                                  read by new features. Until the 200-dim
                                  retrain runs it reports
                                  127 passed / 1 failed / 1 skipped (the skip
                                  is the parity + ONNX contract that needs
                                  matching weights). After
                                  python -m src.train --seeds 3 it is 129
                                  passed, 0 failed, 0 skipped.
                                  Sections: config, vocab,
                                  features, landmarks, dataset, model,
                                  predictor, phrases, live, trained.
                                  The 'trained' section SKIPS itself with the
                                  reason until models/words_core20.pt exists,
                                  then it runs parity + the ONNX contract.
                                  It proves: the 200 features add up and are
                                  finite, translation/scale invariance, the
                                  mirror hand-swap, seeded augmentation,
                                  live-buffer == offline, block ablation
                                  widths, a bad block name raising, the packed
                                  layout arithmetic, the gloss resolution
                                  against the real dataset, OTHER sampling
                                  determinism, no duplicate work list row,
                                  REST/OTHER caps, REST slices never
                                  overlapping the sign window, the fingerprint
                                  being order-independent, class weights
                                  finite and mean 1, save/load round-trip, and
                                  that every live threshold behaves.

python -m src.live --logic-test   the live-path rules with a scripted network:
                                  53 checks, no camera, no weights file
                                  (confidence floor inclusive at the shipped
                                  0.85 and a 0.80 reading refused, the margin
                                  shown to be inert above p1 = 0.58 and the
                                  lever below it, pair overrides in both key
                                  shapes and both orders,
                                  5 malformed-key fail-loud checks, REST/OTHER
                                  never commitable, Segmenter traces, phrase
                                  rules and the 8-word cap, an end-to-end
                                  scripted commit producing exactly one word
                                  with decision_frames == 12, an OTHER-only
                                  stream committing nothing, an unsure stream
                                  still saying why, two segments giving
                                  "hello hello", the framing hint switching on
                                  a buffer with almost no hand in it and off
                                  again, hand_share() being the same
                                  measurement as the manifest's rate, the
                                  screen word vs "uncertain", weights of the
                                  wrong width refusing to read, the API key
                                  contract,
                                  ValueError on bad input, the stride schedule)

python -m src.live --render-test  real dataset frames pushed through update(),
                                  overlay() and hud(); prints files_touched,
                                  unexpected_writes and nothing_saved

python -m src.live --benchmark    CPU fps on cached clips vs TARGET_LIVE_FPS

python -m src.live --selftest     a guided drill ("show me HELLO now"), prints
                                  a table of what the recogniser said and what
                                  you pressed. Its last line says this is not
                                  a model score, and it writes nothing.

python status.py                  read-only: env, dataset, vocabulary, cache,
                                  model, reports, and the one NEXT command.
                                  --quiet prints only that command; --json
                                  gives the whole thing as one blob.
python check_results.py           the compact summary, --write for
                                  REPORT_WORDS.md
```


12. COMMAND REFERENCE
---------------------
```
python run_all.py [--workers 4] [--splits train,val,test] [--epochs 60]
                  [--only extract|train|onnx|report ...] [--from STEP]
                  [--dry-run] [--status] [--fresh] [--limit N] [--blocks ...]

python -m src.extract [--splits train,val,test] [--workers N] [--limit N]
                      [--force] [--min-conf 0.30] [--no-restart] [--log FILE]

python -m src.train [--epochs 60] [--batch 64] [--limit N] [--seed N]
                    [--fresh] [--blocks pose,hands] [--eval-splits val,test]
                    [--smoke] [--parity] [--log FILE]

python -m src.export_onnx [--weights FILE.pt] [--out FILE.onnx]

python -m src.live [--camera 0] [--fps 30] [--stride 1] [--buffer 2.5]
                   [--conf 0.6] [--weights FILE] [--phrases FILE]
                   [--no-mirror] [--headless] [--seconds 8] [--frames 60]
                   [--videos 2] [--json] [--log FILE]
                   one of: --selftest [words...] --benchmark --render-test
                           --logic-test --demo

python -m src.phrases                 rules, inactive rules, example sentences
python -m src.dataset                 sample counts per split from the cache
python -m src.features                the block layout as json
python -m src.model                   arch, param count, conv/gru shapes
python -m src.vocab                   the word -> gloss table with counts
python status.py [--json] [--quiet] [--next]
python check_results.py [--json] [--write] [--out FILE]
```
Run every one of those from inside the folder (the README shows the `cd`
line for each block) - they resolve their paths from `src/config.ROOT`, but
`python -m src.x` needs the working directory to be `videoModel`.


13. ENVIRONMENT PINS
--------------------
`requirements.txt` pins the set that was resolved, installed and smoke-tested
together here. Two things worth knowing:
```
- mediapipe>=1.0 removed the legacy mp.solutions API. This code uses the Tasks
  API only (HolisticLandmarker, RunningMode.VIDEO, detect_for_video), which is
  the supported equivalent and the only one that exists in 1.2.2.
- the usual numpy<2 warning in this stack does not apply to this machine:
  Python 3.13 has no cp313 wheels for numpy 1.26, and mediapipe 1.2.2 /
  opencv 5.0.0.93 / torch 2.14.1 require numpy>=2. So the pins are numpy
  2.5.3, pandas 3.0.6, pyarrow 25.0.1, onnx 1.23.2, onnxruntime 1.31.1,
  torch 2.14.1 (CPU), scikit-learn 1.9.1, tqdm 4.70.1, certifi 2026.7.22.
```
Windows console encoding bites this project (glosses contain non-ASCII), so
set `PYTHONIOENCODING=utf-8`; `run_all.py` and `status.py` do it for their own
subprocesses. Third-party startup chatter is silenced in `status.py` and
`config.py` (`TF_ENABLE_ONEDNN_OPTS=0`, `TF_CPP_MIN_LOG_LEVEL=3`,
`GLOG_minloglevel=2`) so a report stays readable.


14. CHANGING THINGS
-------------------
```
add a word      vocab.json: give it its exact gloss label(s), then
                python -m src.vocab to see the counts, then
                python -m src.extract (resumable, only the new clips) and
                python -m src.train. Check the table first: a word with no
                gloss in the dataset is MISSING and will not become a class.
The current `vocab.json` enables batches 1 through 4. It defines 95 active
words (including external-only chicken) plus REST and OTHER. Generic FOOD has
no verified ASL Citizen gloss, so individual food signs are included instead;
do not relabel EAT or MEAT as FOOD. Use the candidate workflow in `README.md`
to extract and train; candidate weights are isolated from the shipped model.
change caps     vocab.json: extra_classes.rest / .other (cap_train, cap_val,
                cap_test, min_frames, max_frames, per_gloss_max, seed).
                Caps keep a background class from eating the vocabulary.
change a rule   phrases.json. Restart the demo; no code change.
change a        config.py: CONFIDENCE_MIN, MARGIN_MIN, PAIR_MARGINS,
threshold       BUFFER_SECONDS, LIVE_STRIDE. For one run only, without
                editing anything: python -m src.live --conf 0.70 --margin 0.30.
                Then python -m tests.test_logic - the battery asserts the
                ranges the brief asked for, so a bad edit is caught, not
                shipped.
change the      config.py: FRAMING_HAND_RATE_MIN (0.35) and FRAMING_MIN_FRAMES
framing hint    (12); the text itself is FRAMING_HINT. The live window prints
                the hint under the skeleton, and python -m src.live
                --selftest puts the same hint in each drill caption, so you
                can tell "the hand was never in frame" from "the model got it
                wrong".
regularise      python -m src.train --reg none|small|reg|smooth|aug+|all
                (presets in train.REG_PRESETS; none is the measured baseline).
                Choose by VALIDATION only - the test split is scored once at
                the end of the run and never steers anything.
judge a change  python -m src.train --seeds 3 --reg reg --tag reg3
                One row per seed lands in reports/seed_runs.csv, which is the
                artifact; reports/seed_runs.md renders it as mean +/- spread
                (the half-range, not a std, because there are only three),
                baseline-vs-new table included. Decisions read the val_
                columns, and the seed promoted into models/ is the one with
                the best plain val top-1 over all 22 classes - REST/OTHER
                included, because those are the mistakes the live gate pays
                for. (The earlier words-only column has ~84 val rows and
                ignores them; on 2026-11-17 it tied two seeds at 0.9167 and
                promoted the weaker one, 4.6 test points below the seed it
                left on disk.) Adopt a change only when the gain is larger
                than the baseline's own spread; a win inside the noise is a
                coin flip. --promote none keeps the sweep's best seed out of
                models/.
ablate features python -m src.train --blocks pose,hands,mask
                (widths are asserted in the tests, so a typo in a block name
                raises instead of quietly shrinking the vector)
change the      bump LANDMARK_VERSION and re-extract --force. Never edit
packed layout   the layout in place: the manifest would then describe arrays
                that no longer match it.
```


15. TROUBLESHOOTING
-------------------
```
"dataset missing / put the ASL Citizen download at ..."
    status.py prints the exact path it expects. The dataset stays where you
    downloaded it; if it lives elsewhere, change DATASET_DIR in src/config.py
    rather than moving 70 GB.

"nothing extracted yet" or a large 'pending' count in status.py
    extraction is resumable, that is all it means. Re-run
    python -m src.extract --splits train,val,test --workers 4

feature layout drift / "this cache was written by a different landmark version"
    the cache and the code disagree. Re-extract with --force. Do not hand-edit
    the .npy files.

the demo opens and closes the camera on its own
    on this stack that has been a WebRTC/STUN timeout in the other project,
    not here: this demo is a plain OpenCV window, so the usual causes are a
    browser or Teams holding the device, or --camera pointing at the wrong
    index. Try python -m src.live --camera 1.

a word is never recognised, or always becomes another one
    python -m src.extract --rates and python status.py: judge the word by
    hand_rate_seg (the signing window), never by clip-wide hand_rate. Measured
    over the whole cache the window rate averages 0.90 and no core word is
    under 0.69, so detection is almost never the reason - good is at 0.94 and
    still gets 50%, thanks 0.90 and 33%. Look at reports/per_word_test.csv,
    reports/confused_pairs_test.csv and the video count next. Under 15
    training videos, or a runner-up that is a genuinely different sign with
    the same hand shape, is a data-and-features problem, not camera framing
    and not more epochs; status.py already flags the thin words.

REST keeps being typed as a word
    check_results.py names it as a blocker with the rate. Either the REST caps
    are too small for the vocabulary size or the segmenter thresholds are
    wrong; both are one-line config changes and the tests cover them.

it is slower than 15 fps
    python -m src.live --benchmark. If landmarks_only is the bottleneck (it
    usually is), set --stride 2. The cost is real: landmarks, features and the
    decision then happen every other frame, so a sign is sampled at 15
    positions per second instead of 30 and a segment opens and closes half a
    second later. Re-run --selftest and --benchmark after changing it.

a step fails inside run_all.py
    it prints the exact command it ran and why it failed. Fix the cause and
    run run_all.py again; finished steps skip themselves.
```


16. WHAT IS COMMITTED, WHAT IS NOT
---------------------------------
```
in git         the code, vocab.json, phrases.json, requirements.txt,
               README.md, docs/DETAILS.md, .gitignore, and once trained
               models/words_core20.pt + models/meta.json +
               models/words_core20.onnx (the release artefacts, ~3 MB, so
               someone who pulls the repo does not have to train)
never in git   ASL_Citizen/ (70 GB, third-party licence), cache/ (generated),
               logs/, models/checkpoints/ (resumable state), __pycache__,
               datasets/ bulk downloads, anything from a webcam
               (which does not exist, by design)
```
A bare clone was verified on this stack: the models are present so no training
is needed to try the demo, `python -m tests.test_logic` runs with no cache at
all (it skips the sections that need data, with the reason printed), and the
scripts that read the cache need `ASL_Citizen/` in place first.
