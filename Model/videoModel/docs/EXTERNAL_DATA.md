# External data - feasibility (STAGE D)

Written 2026-11-16. **Nothing was downloaded for this document.** Every number below
comes from either (a) the public HTTP directory listings of the dataset host, which
are HTML index pages, not data, or (b) this project's own cache and logs, which were
measured on this machine. Where a number could not be established without a download
it is labelled UNMEASURED.

**Current implementation note (2026-11-19):** the project now uses the bounded
communication vocabulary in `vocab.json` and the PopSign v2.1 downloader is
implemented. Its default subset is 60 train plus 20 validation and 20 test clips
for each of `bye`, `please`, `sleep`, and `chicken`; the files stay under
`Model/videoModel/datasets/external/`. This subset is for training and a separate
PopSign held-out evaluation. `school` remains Citizen-only because no verified
PopSign school folder was established. The earlier options and questions below
are historical research notes, not pending decisions.

The question: ASL Citizen is exhausted for our 20 words (15 training clips per
word), so 58.3% signer-disjoint top-1 is a data ceiling, not a model ceiling. What
would it cost to add a second corpus, and would it actually move the number?

---

## 1. What we have now (measured from `cache/manifest.csv`)

| | value |
|---|---|
| clips in the landmark cache | 1,723 (673 vocabulary + 1,050 OTHER) |
| participants (signers) | 49 |
| vocabulary | 20 words + OTHER |
| training clips per word | 13-30, mean **15** |
| total clips per word (all splits) | 29-63, most words 30-31 |
| signers per word | 26-32 across all splits, 12-17 in train |
| video format | 640x480, ~29.4 fps, mean 78 frames/clip |
| extraction cost | **38.3 ms/frame, 2.98 s/clip** single process |
| parallel scaling | 1,720 clips in 1,471 s on 4 workers = 3.5x speedup |
| cache size | 113.8 MB, ~64 KB/clip |
| feature vector | v3, 200 numbers per clip |

## 2. Candidate A - PopSign v2.1 (Georgia Tech signdata)

Source: `https://signdata.cc.gatech.edu/data/popsign_v2_1/game/<split>/<sign>/`
Dataset page: `https://signdata.cc.gatech.edu/view/datasets/popsign_v2_1/index.html`

Verified from the live listings on 2026-11-16:

* **563 sign folders** in `game/train/` (one per gloss). The dataset page states
  licence **CC BY 4.0**, "External - Open Access", corpus **1.1 TB**, **47 signers**,
  and three categories - `game` (the intended variant), `variant`, `unrecognizable` -
  but on the file host only `popsign_v2_1/game/{train,val,test}/` exists, and
  `popsign_v1_0/` has `game/` plus `non-game/`. So the extra categories are a v1.0
  thing; for v2.1 we would only ever see `game`.
* **The v2.1 splits are signer-disjoint, measured not assumed.** For three words the
  signer-code sets of train / val / test are pairwise disjoint (eat 39/4/6 signers,
  0 overlaps, 49 distinct; hello 35/6/6, 0 overlaps, 47 distinct; bad 18/4/5, 0
  overlaps, 27 distinct). Note the 49 for `eat` exceeds the 47 the page claims, so
  that page figure looks inherited from v1.0 - treat the per-word counts in the
  table below as the real numbers.

* Files are served as **individual `.mp4` per-clip objects**, one folder entry per
  recording, named `<signer>-<gloss>-<date time>-<take>.mp4` - the 4-character first
  field is a **usable signer id**. A HEAD on one clip returns `200 video/mp4`,
  `Content-Length: 5040809`, `Accept-Ranges: bytes` (resumable).
* **v2.1 has no `.tar` archives.** `.../game/train/good.tar` returns 404. The older
  **v1.0** release does still serve real tarballs - `popsign_v1_0/game/train/hello.tar`
  answers `200`, `Content-Type: application/x-tar`, `Last-Modified: 2024-06-06` -
  and that is the layout the official guide and this repo's
  `download_popsign_targets.py` were written for, so `--dataset popsign_v1_0` works
  with the existing tool while `popsign_v2_1` needs the per-clip parser described
  below. v1.0 has only 250 signs and, per its own published gloss list, lacks
  `eat`, `friend`, `good`, `help`, `love`, `family`, `name`, `school`, `sorry`.
  (`popsign_v1_0/game/train/eat.tar` is a 404 - a useful independent confirmation
  that the Kaggle ISLR corpus, which uses the same 250-sign vocabulary, also lacks
  `eat`.)

### Coverage of our 20 words (counts read from the listings)

| our word | PopSign label | train clips | train signers | train GB | val | test |
|---|---|---|---|---|---|---|
| eat | eat | 367 | 39 | 1.68 | 29 | 54 |
| friend | friend | 399 | 39 | 1.90 | 29 | 55 |
| good | good | 386 | 39 | 1.75 | 24 | 51 |
| hello | hello | 639 | 35 | 3.41 | 98 | 112 |
| help | help | 397 | 39 | 1.85 | 29 | 48 |
| sad | sad | 635 | 35 | 3.33 | 82 | 111 |
| drink | drink | 600 | 33 | 3.08 | 103 | 116 |
| thanks | thankyou | 652 | 35 | 3.36 | 101 | 114 |
| bye | bye | 485 | 32 | 2.27 | 94 | 101 |
| yes | yes | 637 | 35 | 3.30 | 95 | 107 |
| no | no | 649 | 35 | 3.16 | 103 | 116 |
| happy | happy | 649 | 35 | 3.58 | 105 | 111 |
| please | please | 660 | 35 | 3.82 | 102 | 115 |
| bad | bad | 312 | 18 | 1.29 | 63 | 82 |
| home | home | 636 | 35 | 3.27 | 102 | 118 |
| love | love | 357 | 39 | 1.64 | 21 | 44 |
| **family / name / school / sorry** | - | **not in the 563-sign list** | | | | |

Totals for the 16 covered words: **8,460 train clips / 42.7 GB**, 1,180 val clips /
6.1 GB, 1,455 test clips / 9.3 GB - **58.1 GB** for all three splits.

So on the words that exist, PopSign train gives **25-44x more clips per word**
(312-660 vs 13-16) and roughly **35-39 signers per word** instead of 12-17, all of
them people who have never appeared in ASL Citizen.

### Conversion work - what has to change in this repo

PopSign ships video, so it goes through the pipeline we already trust; there is no
new landmark format to reverse-engineer. Five small edits:

1. `src/config.py` - add `EXTERNAL_DIR = ROOT/"datasets"/"external"` and make
   `video_path()` honour a `source` prefix (`ps:game/train/good/122a-....mp4`)
   instead of always joining onto `VIDEOS_DIR`. ASL Citizen stays read-only.
2. `src/vocab.py` - an external job table that walks the downloaded folders and
   derives `gloss`, `word`, `participant` (the 4-char signer code) and `split` from
   the path/filename; map `thankyou` -> `thanks`. `videos_to_process()` concatenates
   it with the ASL Citizen table.
3. `src/extract.py` - unchanged in substance. `sign_start/sign_end` come from
   `landmarks.signing_window(arr)` (ASL Citizen's official window metadata does not
   exist here); `rest_frames = 0` because a PopSign clip is already one isolated
   sign. Namespace signer ids (`ac:...` vs `ps:...`) - PopSign codes like `122a`
   could otherwise collide with ASL Citizen participant ids.
4. `src/train.py` / `src/dataset.py` - the group-disjoint split must treat `source`
   as the outer key: **PopSign signers never mix into the ASL Citizen val/test**,
   and vice versa. Report three held-out groups (see "how we'd know" below).
5. `download_popsign_targets.py` - add a per-clip branch beside the existing `.tar`
   branch: list the `game/<split>/<sign>/` folder, take the `.mp4` rows, and use the
   `Content-Length` / `Accept-Ranges` headers the server already sends so a stopped
   run resumes, printing the total size before it asks. The current `.tar`-only
   parser is correct for v1.0 and would find nothing in v2.1.
   ~80-120 lines, and it keeps the existing "type DOWNLOAD" gate.


Nothing in `src/landmarks.py`, `src/features.py`, `src/normalize.py`, `src/live.py`
or `app.py` changes - live inference sees the same 200-number vector.

### Cost, with the measured basis

| step | full 16-word train set | sensible first subset |
|---|---|---|
| download | 42.7 GB, 8,460 files | 60 clips/word = 960 clips, ~4.8 GB |
| download time | UNMEASURED (no bulk route tested; per-file GETs) | same unknown, ~5 min at 15 MB/s |
| extraction | 8,460 clips x ~90 frames x est. 60-120 ms = 13-25 h CPU -> **4-7 h wall** at 4 workers | 960 clips -> **30-60 min wall** |
| landmark cache | ~0.55 GB extra | ~60 MB extra |
| feature build | +1.3 ms/clip -> ~+11 s/epoch | ~+1 s/epoch |
| 3-seed retrain | epoch cost grows ~6-7x, est. **1.5-2.5 h** total | est. **30-45 min** |
| disk on D: | 42.7 GB video + cache; 321 GB free | 4.8 GB |

The extraction line is the only estimate with a caveat: our measured 38.3 ms/frame
is for 640x480. PopSign records on a Pixel 4a at **1944x2592**, and
`landmarks.frames_of()` does not downscale. MediaPipe's landmark networks run at a
fixed internal input size, so the penalty should be resize+tracking overhead
(~1.5-3x), not 8x - but **measure it on the first 50 clips before committing to
the 8,460-clip run** (`python -m src.extract --limit 50` will print ms/frame).

### Expected benefit, honestly

* What is certain: 15 train clips/word cannot learn a word; 300-600 can, in
  principle. PopSign's own published benchmark on 250 classes over its
  signer-disjoint split is ~82% val / 84% test, which is evidence that isolated
  recognition at this data volume transfers across unseen hands **inside that
  corpus**.
* What is not certain, and is the whole risk: **corpus transfer**. Our alphabet
  project measured the same thing the hard way - a T trained on MSL-style data
  scored ~30-43% on its own corpus and 1.4% cross-corpus. PopSign participants are
  largely hearing parents learning from the app, filmed chest-up on a phone, and
  the corpus explicitly separates `game` (correct variant) from `variant` and
  `unrecognizable` - so expect a real but uneven style shift versus ASL Citizen.
* Practical expectation: pooled unseen-signer top-1 on **PopSign** test clips
  ~65-85%, on **ASL Citizen** test clips ~62-72% (today 58.3%). Gated precision
  should improve because more styles at high confidence means fewer confidently
  wrong answers; the demo's "types 3 of 4 words" promise would move up mostly by
  raising recall at the same precision.
* What it does **not** fix: the 4 absent words (family, name, school, sorry stay at
  15 clips); the OTHER class (PopSign has no rest clips - only ASL Citizen's
  mislabelled `OTHER` bucket does, and we already showed it is unlearnable); and
  the fact that `good`/`thanks` differ by only ~0.3 mean standard deviations in the
  features we have - more signers narrows that, it does not create a new cue.

### How we would know, before keeping it

Three held-out groups, printed separately in `REPORT_WORDS.md` and never pooled:

1. ASL Citizen test signers (comparable to today's 58.3% - the live-app number).
2. PopSign val+test signers (a second, independent style).
3. A "both sources, one signer out" check so a claim about the live webcam stays
   tied to group 1.

Adopt the data only if group 1 improves by more than the 3-seed spread (2.3 points,
measured) - the same rule as Stage B4.

## 3. Candidate B - Kaggle "Google ISLR" landmark corpus (`asl-signs`, 2023)

Sources: [competition data page](https://www.kaggle.com/competitions/asl-signs/data),
[description card](https://www.kaggle.com/competitions/asl-signs/overview/description),
[rules section 7](https://www.kaggle.com/competitions/asl-signs/rules),
[leaderboard](https://www.kaggle.com/competitions/asl-signs/leaderboard),
[1st-place write-up](https://www.kaggle.com/competitions/asl-signs/writeups),
and the ISLR section of [the PopSign NeurIPS 2023 Datasets & Benchmarks paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/00dada608b8db212ea7d9d92b24c68de-Paper-Datasets_and_Benchmarks.pdf).
Everything in this section was read off public pages by a research agent; nothing was
downloaded, and the two items that need a Kaggle login are labelled as such.

| fact | value | how well established |
|---|---|---|
| on-disk size | **56.43 GB**, ~94.5k files (Data Explorer total) | CONFIRMED on the data page |
| layout | `train_landmark_files/[participant_id]/[sequence_id].parquet`, plus `train.csv` and `sign_to_prediction_index_map.json` | CONFIRMED - the old TFRecord layout is gone from the page |
| landmark source | MediaPipe Holistic **0.9.0.1** (we run 1.2.2) | CONFIRMED from the card |
| signers | **21 Deaf signers**, `participant_id` present on every clip | CONFIRMED from the card |
| vocabulary | **250 signs** | CONFIRMED from the card |
| frames per clip | "not trimmed": 0.5 s buffer before the button press and after release; one EDA thread reports the effective fps varies a lot | PARTLY - community report |
| raw video | **not included** in this release; the videos live in PopSign v1.0 | CONFIRMED |
| our 20 words | **11 of 20 present** - hello, sad, drink, bye, yes, no, happy, please, bad, home, thankyou(=thanks). Absent: eat, family, friend, good, help, name, school, love, sorry | INFERRED, high confidence: the gloss list on the [PopSign v1.0 page](https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/index.html) is the same 250-sign vocabulary and matches the Kaggle folder names |
| per-word clip counts | UNVERIFIED - the smallest file that reveals them (`train.csv`) is inside the login-gated competition data |
| access | Kaggle sign-in **and** accepting the competition rules | CONFIRMED ("To see this data you need to agree to the competition rules") |
| licence | CC-BY 4.0, "for any purpose, whether commercial or non-commercial" | CONFIRMED, rules section 7 |
| redistribution | rules 7.B: **"You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition."** | CONFIRMED - so it cannot go in the group repo, and each teammate must download their own copy |
| stated non-use | landmarks are de-identified and "should not be used to identify or re-identify an individual"; the card says the data is not appropriate for ASL-to-English translation or natural-language interfaces | CONFIRMED |
| what unseen-signer accuracy looks like there | 1st place: participant-split 5-fold CV **0.80**, private leaderboard 0.88 (metric is accuracy over 250 classes); best public notebook 0.73 | CONFIRMED from the write-up and leaderboard |

Landmark schema, as reported by the page and by public derivative datasets: long
format, one row per (`frame`, `type`, `landmark_index`) with `x`, `y`, `z`, where
`type` is one of `face` (468 points), `pose` (33), `left_hand` (21), `right_hand`
(21) - 543 landmarks per frame in total, which is why the winning notebooks talk
about "selecting 66 landmarks out of 543". The page itself warns that "the MediaPipe
model is not fully trained to predict depth so you may wish to ignore the z values"
and that "not all of the frames necessarily had visible hands".

**Conversion to our layout is the cheap part.** A 543-point-per-frame record maps
onto our packed `[T, 136, 3]` array by index selection only: pose rows 0-26 are our
`POSE_UPPER`, the 468-point mesh is subset by our `FACE_ORDER` (67 of them), and the
two 21-point hand blocks go in as they are. No MediaPipe, no video decoding -
parsing a parquet is milliseconds, not the 3 s/clip our extractor costs. Two real
caveats: (a) those landmarks came out of MediaPipe **0.9.0.1** while training and the
live app both use **1.2.2**, and we have measured that the detector version changes
hand-presence rates, so the hand-mask block and the gap-filling will see a different
absence distribution; (b) `z` is unreliable and our normaliser divides by a
shoulder-width computed in `x, y`, so `z` must be rescaled or dropped, which changes
the 3 depth numbers per landmark in the pose/face blocks.

## 4. A versus B, and the decision I need

The two candidates are the same families of recordings viewed from two sides:
PopSign **v2.1** (signdata host) has **16 of our 20 words**, ~35-39 signers per word
and about **8,460 training clips**, but it is video, so it costs a MediaPipe run of
roughly 4-7 hours of wall time and 42.7 GB. The Kaggle **ISLR** release has the
landmarks already computed - minutes of parsing - but only **11 of our 20 words**,
**21 signers** in total, 56.4 GB of parquet, and a download that requires a personal
Kaggle account plus rule acceptance, with redistribution outside the competition
forbidden. Neither has `family`, `name`, `school` or `sorry`; only PopSign v2.1 has
`eat`, `friend`, `good`, `help` and `love`.

The words we are weakest on are exactly the ones only PopSign v2.1 can supply: the
Stage A read-out showed `good` vs `thanks` separated by a mean |d| of only 0.31
standard deviations over 15 training clips.

Recommended, in order:

1. **PopSign v2.1 subset** - 60 clips per word, chosen to maximise signer spread,
   ~4.8 GB, ~30-60 min extraction, one 3-seed retrain, evaluated on the three
   held-out groups. Decision rule: keep it only if the ASL Citizen unseen-signer
   number moves by more than the measured 2.3-point seed spread.
2. If that clears the bar, pay for the full 16-word train set (42.7 GB, 4-7 h).
3. Treat Kaggle ISLR as an optional add-on for the 11 shared words later, only if
   we want a no-MediaPipe corpus and can each download it personally under 7.B.
   It is also the one place with genuinely de-identified landmarks, if that ever
   matters for the write-up.

What I need from you: which of 1, 2, 3 to run, and whether you are willing to make a
Kaggle account + accept the rules (only for option 3). Nothing has been downloaded
and nothing will be until you say so.

