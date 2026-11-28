ASL WORD RECOGNIZER - videoModel
================================

Isolated ASL words from a webcam: MediaPipe Holistic landmarks (not pixels),
trained for a bounded communication vocabulary using signer-disjoint ASL Citizen
data and a small, optional PopSign subset.

You record nothing, save nothing and train on nothing you did not download.
The webcam frames are converted to numbers, used, and dropped in the same
function call.


ABOUT
-----
Name    : videoModel word recognizer (40 daily-communication words +
          REST + OTHER; 95 words are defined in vocab.json, the retained
          daily set is the `enabled_words` list there)
Data    : ASL Citizen (Microsoft Research), ~83,400 isolated word videos,
          52 signers, already on disk at `ASL_Citizen/`
Approach: landmarks, not pixels - one shared 200-number feature vector per
          clip, used identically for training and for the live demo
Model   : conv1d x2 -> bidirectional GRU x2 -> Linear, 382,090 parameters,
          CPU only. Standardisation (train-split mean/std) and the max-over-time
          pooling live inside the module, so the checkpoint, the ONNX graph and
          the webcam path cannot drift apart.
Result  : the shipped checkpoint is the daily-40 build (40 words + REST +
          OTHER = 42 classes, ASL Citizen plus PopSign clips). It is an
          ensemble of three checkpoints trained with different regularization
          (none / dropout 0.45 + weight decay / heavier augmentation); the
          typed number is the mean of their probabilities, and the ONNX graph
          exports that same average. Held-out, signer-disjoint: ASL Citizen
          84.8% top-1 on words / 85.5% over all classes, PopSign signers 83.8%
          - each member alone scores 82.8-83.9% on the same clips. The same
          corpus at 56 words scored 81.4% / 68.3%, so the smaller vocabulary is
          the accuracy, not only the scope.
          Full tables: REPORT_WORDS.md.
          Offline clips only - the live webcam path is verified for feature
          parity, not measured for accuracy.


CURRENT CANDIDATE WORKFLOW
--------------------------
Run the downloader first; it asks before downloading. Then extract the current
vocabulary and train into an isolated candidate directory:

```powershell
cd "D:\sign language project\ASTL project\ASLT"
python -m pip install -r requirements.txt
cd "D:\sign language project\ASTL project\ASLT\Model\videoModel"
python download_popsign_targets.py
python -m src.extract --splits train,val,test --workers 4
python -m src.train --candidate communication_v2 --seeds 3 --eval-splits val,test,ps_test
python check_results.py --candidate communication_v2 --write
```

The downloader defaults to 60 training and 20 each validation/test clips for
bye, please, sleep, and chicken. The other new signs come from the existing
ASL Citizen split files and are extracted from the local dataset. Existing
landmark cache entries are reused. The training outputs stay under
`candidates/communication_v2/`; the shipped model is not overwritten. Do not
use `run_all.py` for this candidate, since its default output scope is the
shipped model directory.


WHAT IT DOES
------------
```
1  extract   ASL_Citizen/videos/*.mp4  ->  cache/landmarks/*.npy
             one [T, 136, 3] float16 array per clip: 27 upper-body pose
             points, 67 face points, 2 x 21 hand points. NaN where MediaPipe
             saw nothing. Nothing is resized, cropped or re-encoded.

2  features  [T, 136, 3]  ->  [64, 200]
             resampled to 64 frames, centred on the shoulders, scaled by
             shoulder width, y flipped so up is up. Interior landmark gaps of
             up to 5 source frames are bridged before this (a hand that
             flickers for two frames is not two absences), then the vector is
             pose 30 + hands 84 + hand shape 18 + face 12 + hand-to-body 10 +
             presence 4 + velocity 12 + segment read-out 30.

3  train     [64, 200]  ->  42 classes
             40 daily-communication words, plus REST (idle head/tail of the
             same clips) and OTHER (a capped sample of signs outside the
             vocabulary). `python -m src.vocab --status` prints the exact set.

4  live      webcam frame -> landmarks -> features -> decision -> a word
             a segment opens when a hand rises above the shoulder line and
             closes when it drops; the word is only typed when the reading is
             confident and clearly ahead of the runner-up, otherwise the
             screen says "uncertain" and types nothing.
```


CHECK WHAT IS DONE WITHOUT RUNNING ANYTHING
-------------------------------------------
```bat
cd "D:\sign language project\ASTL project\ASLT\Model\videoModel"
python status.py
```
Read-only: it opens the split CSVs, the manifest, `meta.json` and the reports,
never a video. The last line is the one command to run next.
`python status.py --quiet` prints only that line.


THE FOUR STEPS SEPARATELY
-------------------------
Use these if you want to run one stage, or re-run one after changing it.

```bat
cd "D:\sign language project\ASTL project\ASLT\Model\videoModel"

python -m src.extract --splits train,val,test --workers 4
python -m src.train --epochs 60
python -m src.export_onnx
python -m src.train --parity
python check_results.py --write
```

More useful switches:

```
python -m src.extract --limit 20        first 20 pending videos only
python -m src.extract --force           re-extract even what is cached
python -m src.extract --rates           per-word hand rate: whole clip vs the
                                        signing window only (reads the cache,
                                        no MediaPipe, under a second)
python -m src.train --smoke             a few clips, 1 epoch, proves the code
python -m src.train --blocks pose,hands ablation: drop feature blocks
python -m src.train --fresh             ignore the saved checkpoint
python -m src.train --seeds 3           run 3 seeds, report mean +/- spread
python -m src.train --reg reg           regularization preset: none (the
                                        baseline), small, reg, smooth, aug+
python -m src.train --gate 0.85         what "typed" means when the gate
                                        numbers are reported
python -m src.train --promote none      keep the best seed out of models/
python -m src.train --compare           print the baseline-vs-new table only
python -m src.live --conf 0.70          lower (or raise) the commit gate
python -m src.live --margin 0.30        required top1-top2 gap; only bites
                                        below p1 = 0.58, see the note in
                                        src/predictor.py
python run_all.py --dry-run             print what would run, change nothing
python run_all.py --from train          start at a step and keep going
```

Every number the seed sweep produces is appended to
`reports/seed_runs.csv` and rendered as a table in `reports/seed_runs.md`,
so a change is judged against the spread of the run it replaced, never
against one lucky seed. Decisions are made on the validation columns; the
test split is scored once at the end. Adopt a change only if it beats the
baseline spread.


TRY IT LIVE
-----------
Needs `models/words_core20.pt` (step 3 above) and a webcam.

```bat
cd "D:\sign language project\ASTL project\ASLT\Model\videoModel"
python -m src.live
```

A window opens with your camera and a skeleton drawn on top: hands, upper
body and a face subset. The bar at the bottom shows the state (REST, SIGNING
or UNCERTAIN), the live frame rate, and the last words it committed. Press
`r` to reset the buffer, `q` to quit. It saves nothing - no frame, no clip,
no photo - and writes no files.

Without a camera, or to test the pipeline itself:

```bat
cd "D:\sign language project\ASTL project\ASLT\Model\videoModel"
python -m src.live --selftest           guided drill: show it each word, prints a table
python -m src.live --benchmark          CPU speed on cached clips, target >= 15 fps
python -m src.live --render-test        real dataset frames through the live path
python -m src.live --logic-test         the decision rules, no camera, no weights
python -m src.live --demo --videos 3    offline: what the segmenter did on a clip
python -m tests.test_logic              the whole offline battery (129 checks)
```

`--selftest` is a drill, not a score; its table says so on the last line.


WHAT TO EXPECT FROM THE DEMO
----------------------------
These are the shipped model's readings on held-out offline clips, not live
webcam accuracy. Measured at the shipped gate of `p1 >= 0.85`
(`CONFIDENCE_MIN` in `src/config.py`) on 731 ASL Citizen test word clips
(11 unseen signers) and 840 PopSign test clips (10 unseen signers):

```
types about 3 of every 5        the other two-fifths are refused, not
real words                      guessed - the screen says "uncertain" and
                                nothing is typed
about 94% of what it types      this is the number to quote, not the 68.3%
is right                        raw top-1: the model is often unsure, and
                                the demo is allowed to stay quiet
may skip a word you signed      that is the ~42% above. A skipped word costs
clearly                         a re-sign; a wrong word costs trust
turns a sign that is not in     measured on the OTHER clips of the same
the vocabulary into a wrong     split. There is no "I don't know" class to
word about 12% of the time      fall back to, so a non-word still has to
                                clear the same gate - and sometimes does
types a word while you are      the REST clips: 2%. A hand that moves near
doing nothing at all about      the body can genuinely look like a sign, and
2% of the time                  no threshold fixes that
```

Lower the gate and both good and bad numbers move together: `--conf 0.70`
types more words and is right less often. Raising it makes it quieter. The
margin (`--margin`) is not the second lever people expect - at `p1 >= 0.85`
the top-1/top-2 gap is forced to at least 0.70 by arithmetic, so the margin
only does anything once you have lowered `--conf` below about 0.58.

Seven pairs are hedged instead of gated (`HEDGE_PAIRS` in `src/config.py`):
when the top two are one of `good/thanks`, `when/hurt`, `now/today`,
`doctor/help`, `please/my`, `sorry/me`, `wait/want`, the demo types neither and
shows the hint. The model swaps those pairs even when very confident - the
swapped margins overlap the correct ones, so no threshold separates them.
Measured on the cached held-out readings, the six extra pairs take test word
precision from 93.9% to 96.1% (ps_test 96.8% to 97.9%) while typing 5.3 points
fewer word clips. Empty the dict with a one-line edit to get the old behaviour.
The numbers in `REPORT_WORDS.md` were measured with the training-time policy,
which knew only `good/thanks`.


TUNING THE WEAK WORDS
---------------------
Measured on the 58-class run, the weak words are weak for two different
reasons, and each needs a different fix.

*Low hand detection, not the classifier.* `thanks` 25%, `hungry` 33%,
`doctor` 46%, `when` 45-85%, `now` 50-67% on held-out signers, and those same
words sit at a 0.40-0.44 mean hand-detection rate against 0.52 for `please`
(`python -m src.extract --rates` prints this per word). No threshold or loss
change repairs a clip whose hands MediaPipe never saw.

*Too little data for six signers.* `me`, `my`, `you`, `your`, `need`, `come`,
`understand`, `family`, `name`, `school`, `sorry`, `bread`, `medicine` have
13-16 ASL Citizen training videos each and no PopSign folder at all. The
official signer-disjoint split is what caps them; adding val/test videos would
break the evaluation, not improve it.

What is already switched on for the next training run:

- `config.WEAK_WORD_BOOST` gives the eight worst classes 1.5-2x sampler weight
  on top of equal class mass (empty it to compare).
- Pair hedges above, which cost nothing at runtime.

What to run (a 1-seed 100-epoch retrain is roughly 20 minutes on this CPU;
the PopSign top-up below is a network download, so start it first and leave
it):

```powershell
cd "D:\sign language project\ASTL project\ASLT\Model\videoModel"
python download_popsign_targets.py --words thanks happy please hungry thirsty doctor when now wait --clips-per-word 150
python -m src.extract --splits train,val,test --workers 4
python -m src.train --candidate weakv2 --seeds 3 --epochs 100 --eval-splits val,test,ps_test
python check_results.py --candidate weakv2 --write
```

Judge it by the 3-seed VAL mean, not one run: the seed spread is +-3-5 points,
so a single-seed gain of 2 points is noise. `--promote best` copies the
VAL-winning seed into `models/` when you accept the result.

Levers that were measured and rejected: lowering the gate to 0.70 raises typed
share from 53% to 65% but sends 48% of REST/OTHER clips into a word (6.6% at
0.85), so it makes the demo confidently wrong about silence; per-word
confidence floors gain 0.4 points of precision for 2.5 points of coverage on a
87-clip val split, which is too small to fit honestly; and dropping train clips
with hand-detection under 0.15 removes only 1.9% of the cache, because the
badly-framed clips cluster in the 0.15-0.40 band instead.

If the hand is barely in frame the demo says so instead of guessing badly:
when a hand is detected in under 35% of the last ~1.5 s of the live buffer, a
hint appears on the screen ("keep both hands and your face in frame, chest
up"). That is framing, not accuracy - the recorded data has no such problem:
inside the signing window MediaPipe sees a hand in 90% of frames on average,
and no core word is under 69%.


USING IT FROM PYTHON
--------------------
```python
from src.predictor import WordPredictor

pred = WordPredictor()                      # models/words_core20.pt

for frame in camera_frames:                 # BGR uint8 [H, W, 3], from cv2
    r = pred.update(frame)
    print(r["state"], r["top3"], r["committed_word"], r["text"])

pred.close()
```

`update()` returns:

```
state            REST | SIGNING | UNCERTAIN
display          the word to put on screen, or "uncertain" - never a guess
top3             [(word, p), (word, p), (word, p)] sorted
committed_word   the word this segment just added, or ""
text             the sentence built so far, after the phrase rules
p1, margin       the two numbers the commit rule looks at
reason           why a reading was refused ("", if it was committed)
hand_rate, hint  the framing check: share of the live buffer with a hand in
                 it, and the on-screen hint when that share is low
buffer_frames, decision_frames, segment_frames, hand_in_frame, segment_cut
```

It also accepts a landmark row of shape `[136, 3]` instead of a picture, which
is how `--benchmark`, the tests and any offline batch use it. Passing anything
else raises `ValueError` rather than quietly doing nothing.

Words become phrases through `phrases.json`, which you can edit without
touching code: `ME + NAME` -> "my name", `no + good` -> "not good", `me` ->
"I". `python -m src.phrases` prints the rules that can fire with today's 20
words and the ones that are waiting for batch 2.


TIME AND DISK
-------------
Measured on this machine (Windows 11, Python 3.13, CPU only, 16 threads):

```
extraction     38.55 ms per frame; 1720 videos in 1471 s with 4 workers
               = 24.5 minutes, 0 errors, 94.5 frames/s effective
               (single-process probe clips ran 1.70-2.21 s per video)
cache size     109 MB for the whole work list (920 train, 234 val, 569 test),
               64 KB per clip on average
train          1520 train samples / 33 signers, 5.01 s per epoch,
               58 epochs in 300 s = 5 minutes (early stop at patience 8)
features       8.5 ms per feature build, re-spent on every epoch
live           median 24.6 ms per update() on a real clip = 40.7 fps
               (23.7 ms of that is MediaPipe itself; the model is 2.2 ms)
onnx parity    64/64 identical top-1 on held-out clips, largest probability
               difference 9.5e-07
weights        1.5 MB; the Holistic model file is 13.7 MB
whole run      about 30 minutes of CPU on this machine, start to finish
```

`logs/` keeps one timestamped log per extraction run, plus `logs/train.log`,
which is what `check_results.py` reads the tail of.


STOP AND RESUME
---------------
Ctrl-C at any point. Then run the same command again.

```
extract  skips every video already recorded in cache/manifest.csv
train    resumes from models/checkpoints/train_state.pt at the next epoch
         (--fresh to start over)
onnx     re-exports only when the weights are newer than the graph
report   re-runs in a second
```

To throw the work away: delete `cache/`, `logs/` and `models/checkpoints/`.
Never delete or move `ASL_Citizen/` (it is a ~70 GB manual download) and never
touch the videos, split CSVs or `use.txt` inside it.


WHAT THE NUMBERS MEAN
---------------------
The rules the reports are written under, in one place:

```
evaluation   official signer-disjoint splits only (35 / 6 / 11 signers).
             A test signer never appears in training. Never quote a
             random-split number as this model's accuracy.
tuning       val decides epochs, learning rate and refusal margins. The test
             split is scored once, at the end, and never steers anything.
background   REST and OTHER are real classes with their own rows in the
             reports: how often REST was typed as a word, how often a word
             was typed as REST/OTHER, and accuracy with and without them.
provenance   every reported number names the signers and videos behind it
             (reports/per_word*.csv, reports/predictions*.csv).
smoke        a --smoke run is labelled "smoke" in its own meta and is not a
             result; check_results.py says so out loud rather than printing
             its 0.0%.
```

Privacy and dataset terms are in `docs/DETAILS.md`; the live demo never writes
webcam frames. The PopSign video subset is downloaded into this folder's
ignored `datasets/external/` directory and is excluded from Git.


THE VOCABULARY
--------------
95 words across four batches are *defined* in `vocab.json`, plus REST and OTHER.
Batches 1-3 are enabled and `enabled_words` narrows them to the **40-word daily
communication set**, so the shipped model in `models/` covers 40 words + REST +
OTHER = 42 labels. Every other word - and the unavailable `food` - stays defined
but disabled, and its signs fall into OTHER. An empty `enabled_words` list uses
every enabled batch again. `src/train.py` refuses
to overwrite those weights with a different label set unless
`--allow-vocab-change` is passed. Enable more batches in a candidate first,
then compare before promoting:

    python -m src.train --candidate vocab95 --epochs 60
    python check_results.py --candidate vocab95

`me` is one visual class that the phrase layer displays as “I”.

```
eat  family  friend  good  hello  help  name  sad  school  drink
thanks  love  bye  sorry  yes  no  happy  please  bad  home
me  my  you  your  what  where  who  want  need  like  go  come
stop  more  water  understand  now  today  work
wait  finish  sleep  bathroom  hungry  thirsty  when  why  how
apple  bread  chicken  tomorrow  sick  hurt  doctor  medicine
meat  vegetable  milk  coffee  tea  fruit  fish  egg  potato  carrot
tomato  orange  park  hospital  restaurant  library  hotel  church  bank  office
police  nurse  teacher  mother  father  child  man
car  boat  bicycle  airplane  bus  train  motorcycle
phone  money  book  key
```

```
food         unavailable in ASL Citizen; EAT is not relabelled as FOOD
chicken      PopSign only; HEN is not relabelled as CHICKEN
school       ASL Citizen only; no unverified PopSign substitute is used
```

The new signs include pronouns and possessives, questions, requests, daily
needs, common food, basic health, places, people, transportation, and everyday
objects. The newest batch adds 38 signs: meat, vegetable, milk, coffee, tea,
fruit, fish, egg, potato, carrot, tomato, orange, park, hospital, restaurant,
library, hotel, church, bank, office, police, nurse, teacher, mother, father,
child, man, car, boat, bicycle, airplane, bus, train, motorcycle, phone, money,
book, and key. `I` and `me` deliberately share the `me` visual class. These
remain isolated-word predictions, not continuous ASL sentence translation.

The local training split has 12–18 examples for most new individual glosses;
the grouped variant classes have more. These are labels available to train,
not claims about their eventual accuracy. `POLICEMAN` is presented as `police`;
there is no verified `STATION` gloss, so “police station” is not a single class.

To add the focused PopSign clips, run these commands from PowerShell. The
downloader prints the selected clips and estimated total before prompting;
the default is 60 training clips and 20 each from validation and test for
`bye`, `please`, `sleep`, and `chicken` (about 400 clips). Data is saved under
`Model/videoModel/datasets/external/popsign_v2_1/` and is never added to Git.

```powershell
cd "D:\sign language project\ASTL project\ASLT\Model\videoModel"
python download_popsign_targets.py
python -m src.extract --splits train,val,test --workers 4
python -m src.train --candidate communication_v2 --seeds 3 --eval-splits val,test,ps_test
python check_results.py --candidate communication_v2 --write
```

Extraction reuses cached ASL Citizen landmarks and processes only missing
videos. Candidate weights and reports stay under `candidates/communication_v2/`;
the shipped model is not overwritten. The training command is intentionally
left for the project owner to run.


LICENCE AND CITATION
--------------------
ASL Citizen is Microsoft Research data released for non-commercial research;
the terms are in `ASL_Citizen/use.txt`, which ships with the download. Read
them before publishing anything built on this. If you cite the data, cite
Desai et al., "ASL Citizen: A Community-Sourced Dataset for Advancing
Isolated Sign Language Recognition", arXiv:2304.05934, 2023.

MediaPipe Holistic is Apache-2.0 (Google). The `holistic_landmarker.task` file
(13.7 MB) is fetched once into `models/` on first use.

One linguistics note worth knowing before you judge the output: ASL has no
copula, so there is no sign for "am"/"is". Signing ME SAD means "I am sad".
This code does not invent an "am" sign; the English expansion lives in
`phrases.json` and you can delete that rule.


FILES IN THIS FOLDER
--------------------
```
run_all.py             the one command (extract, train, onnx, report)
status.py              what is done, what is missing, what to run next
check_results.py       compact pasteable summary + writes REPORT_WORDS.md
README.md              this guide
docs/DETAILS.md        the long reference: contracts, thresholds, reports,
                       troubleshooting, privacy, licence text
docs/EXTERNAL_DATA.md  is 15 training clips per word the ceiling? what two
                       external corpora would cost, and what they would fix
phrases.json           your phrase rules (editable, no code change needed)
vocab.json             words, glosses, batches, REST/OTHER caps
src/config.py          every path and constant in one place
src/extract.py         video -> packed landmarks, resumable
src/landmarks.py       the MediaPipe Tasks wrapper + geometry helpers
src/features.py        the ONE shared feature function (train and live)
src/vocab.py           word -> gloss resolution, splits, caps, work list
src/dataset.py         sample building (sign windows, REST slices, OTHER)
src/model.py           the network, save/load, standardisation
src/train.py           signer-disjoint training and all the reporting
src/export_onnx.py     ONNX export + parity
src/predictor.py       WordPredictor: the importable live API
src/live.py            the webcam demo, --selftest, --benchmark, --logic-test
src/phrases.py         word buffer and phrase rewriting
tests/test_logic.py    the whole no-data battery in one command
```

Generated when you run it:
```
cache/landmarks/*.npy        one per processed clip
cache/manifest.csv           one row per clip: status, fps, window, hand rate
                             over the whole clip and over the signing window
cache/extract_errors.csv     why any clip failed
models/words_core20.pt       the primary weights (reg config)
models/words_core20_e2.pt    ensemble member, 'none' config
models/words_core20_e3.pt    ensemble member, 'aug+' config
models/words_core20.onnx     the exported graph (all three, averaged)
models/meta.json             labels, dims, splits, seed, licence, fingerprint
reports/*.csv *.json         per word, confusions, predictions
REPORT_WORDS.md              the human-readable result
logs/                          one log per run
```
