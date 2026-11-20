ASL ALPHABET - LIVE WEBCAM RECOGNISER (landmarks + MLP)
=======================================================

A webcam recogniser for the 24 static ASL letters (A-Z without J and Z).

```
    webcam frame -> MediaPipe Hands (21 landmarks) -> normalise -> small MLP
                 -> 4-of-7 voting window -> letter on screen
```

Everything is trained from public datasets. You never record, upload or add a
single image of your own, and no tool here saves your camera frames.

J and Z are movement, not shape, so they are produced by a separate trajectory
rule in front of the same model (see "J AND Z" below).


THE NUMBERS
===========
Measured, not estimated. "Cross-dataset" = leave-one-dataset-out, so every test
image was scored by a model that never saw that dataset.

```
  training data        : 122,359 landmark vectors from 3 public datasets
  features             : 71 numbers per hand (xy + tip distances + curl + thumb)
  model                : MLP 128-64, 24 classes, CPU only
  cross-dataset pooled : 67.0 %  [66.7-67.2]      <- the honest headline
      per holdout      : danrasband 94.4 % (712 imgs, small), grassknoted 79.5 %,
                         massey 53.8 %
  unseen hands,        : 85.7 % over 35,893 images (massey A+B -> C+D+E).
  one imaging style    : this is the closest dataset-side proxy for a webcam.
                         It is still not a webcam measurement.
  weakest letters      : N 0.4 %, M 3.2 %, T 16.8 %, C 36.8 %, D 40.1 %, P 40.0 %
                         In-sample the same model scores 90-96 % on them, so
                         this is transfer, not capacity.
  live frame coverage  : 98.3 % of frames get a hand, ~85 frames/s through the
                         webcam code path with no camera attached
  real webcam speed    : ~20 fps ceiling - MediaPipe VIDEO costs 45-52 ms a
                         frame at any resolution
  no live accuracy     : claimed anywhere. Only dataset measurements.
```


FILES IN THIS FOLDER
====================
```
  app.py                  browser UI (Streamlit) - live webcam, photo, results
  run_all.py              download -> extract -> train -> report -> export
  requirements.txt        pinned packages
  README.md               this file
  REPORT.md               the round-by-round story: what was tried, what failed
  docs/DETAILS.md         the long reference: every table, flag and fix
  data/datasets.json      which datasets, sizes, licences, per-letter counts
  src/                    19 scripts (live.py, train.py, normalize.py, ...)
  models/                 the trained weights (.pt + .onnx), in git; MediaPipe's
                          own 7.5 MB hand model is downloaded on first run
  reports/                the written evidence (.md + small .csv + PNGs)
  cache/                  GENERATED - landmark store, 76 MB, not in git
```

All other paths in this file are relative to this folder. The commands below
always start with the absolute `cd`, so you can paste them from anywhere.


WHAT YOU NEED
=============
```
  Python 3.11 or newer (tested on 3.13.15), CPU only, no GPU.
  A webcam, for the app.
  About 3 GB free for the public datasets (2.92 GB measured, see DATASETS USED).
  A Kaggle account only if a dataset is missing from your local Kaggle cache.
```
Datasets download into this folder's ignored `datasets/kagglehub/` directory.
This keeps training independent of a machine-global cache and keeps raw data
out of Git. The three currently available datasets are reused from that folder
on later runs. Extracted landmarks and generated caches also stay under this
folder in `cache/`.

Kaggle credentials, only if a download is needed: kaggle.com/settings ->
"Create New Token" -> save the file as `C:\Users\<you>\.kaggle\kaggle.json`.


HOW TO RUN
==========
0. Open a terminal in this folder. Everything below assumes you did:

```
     cd "D:\sign language project\ASTL project\ASLT\Model\asl_landmark_app"
```

   On another machine, `cd` into wherever you put asl_landmark_app.

1. Install the packages:

```
     python -m pip install -r requirements.txt
```

2. The whole pipeline from scratch - download, extract landmarks, train,
   report, export ONNX. Extraction is the long stage (~20 min on this machine
   with 12 workers):

```
     python run_all.py
```

   You can skip it: models/ holds the trained weights, and `python
   src/download_data.py` is what fetches MediaPipe's hand model into
   models/hand_landmarker.task. Once that file exists the app runs straight
   away.

3. The browser app:

```
     streamlit run app.py
```

   opens http://localhost:8501 -> "Live webcam" tab -> press START -> allow the
   camera in the browser prompt.

4. Or the console app (its own window, same model and same smoother):

```
     python src/live.py --profile all
```

   SPACE commits the letter you are holding, BACKSPACE deletes it, C clears the
   line, Q quits.


GUIDE - THE BROWSER APP
=======================
Left sidebar, in the order they appear (these are the widget labels):

```
  Profile                 "all" ships. "standard" is the experiment that removed
                          the odd signing styles and did not help - REPORT.md.
  Runtime                 torch or onnx. onnx is ~6x faster on the classifier
                          and gives the same answers (parity measured at
                          1.91e-06 max logit delta).
  Minimum confidence      below these two, the frame casts no vote.
   to vote /
   Minimum top1-top2
   gap to vote
  Raised gaps for the      P/Q, A/S, M/N, G/H need a 0.50 gap instead of 0.10 -
   pairs this model        the data says a close second letter is wrong.
   cannot separate
  Vote window (frames)    how many recent frames the majority is counted over.
  Hold before typing      how long a letter must be stable before it is typed.
   (seconds)
  J / Z motion rule       the trajectory rule on/off.
  Draw the 21 landmarks   overlay on the picture. Off saves ~1 ms a frame,
                          nothing more.
  Mirror the webcam        display only. No feature ever sees the mirror.
   (selfie view)
  Work every Nth frame    2 by default: one processed frame costs ~50 ms, so
                          every frame is 26 ms and fits. Set 1 if J or Z stops
                          firing.
  Show the 71 feature     the exact numbers the model is scored on.
   values
  Use a public STUN       OFF, and it should stay off. Only for opening the app
   server                 from another network. See "IF SOMETHING GOES WRONG".
```

The four tabs:

```
  Live webcam        the streaming panel + the read-out + Commit / Backspace /
                     Clear + "Calibrate J / Z" (3 J and 3 Z of your own signing;
                     nothing is written - it only loosens the motion thresholds
                     in memory and prints the numbers it chose)
  Upload a photo     one still image, same pipeline, top-3 probabilities
  Camera snapshot    one frame from the camera, scored, kept in memory only
  Model & results    the dataset tables, per-letter scores, feature list
```

The read-out tells you what the pipeline is doing, not just the answer: the
letter, its confidence, the runner-up, and why a letter was refused ("A/S needs
50" means the pair margin blocked it).


GUIDE - CHOOSING A PROFILE FOR YOUR OWN HAND
============================================
```
     cd "D:\sign language project\ASTL project\ASLT\asl_landmark_app"
     python src/live.py --selftest --profile all,standard
```

A guided drill: it shows a letter, counts down, records 3 s of your signing in
memory, and finishes with a per-profile table (hit rate, most common wrong
letter, how often no hand was found). Use the profile that wins for you. This
does not save or train on anything you sign.


J AND Z
=======
The model has 24 labels and no J or Z: one still frame of J is an I, and one
still frame of Z is an X. So a rule watches the movement in front of the model -
a J must hook sideways and come back, a Z must zig-zag - and the frame's own
reading is the precondition that lets that watch open.

If J or Z never fires, press "Calibrate J / Z" in the browser, or:

```
     cd "D:\sign language project\ASTL project\ASLT\asl_landmark_app"
     python src/live.py --profile all --calibrate-jz --motion-debug
```

Watch the debug line: it names the criterion that is binding and the number it
saw. On real movement that is usually the reversal count (SEG). If they fire
when you did not sign them, the same line tells you which threshold let it
through - tighten it with `--jz SEG=0.20` rather than touching the model.


IF SOMETHING GOES WRONG
=======================
```
  Camera opens, lags about a second, closes itself, START comes back, error on
  a white page
      -> fixed. "Use a public STUN server" must stay OFF: the app used to ask
         Google's STUN server for an address it already had, your network never
         answered, and the connection gave up at exactly 10 seconds. Turn the
         checkbox on only if the browser is on a different network from this
         server. Details: REPORT.md round 8.
  Picture stays black / "connecting"
      -> the browser needs its own camera permission (address-bar camera icon
         -> Allow -> reload) and no other app may hold the camera (Teams, Zoom,
         another tab). Windows Settings > Privacy & security > Camera > allow
         desktop apps.
  Read-out says "The webcam is not running" but the picture is there
      -> the panel measures its own frame heartbeat now, so this line means the
         browser component stopped sending, not the model. Keep the tab in the
         foreground, Ctrl+Shift+R, then START.
  A letter you were signing goes quiet for a second (P/Q, A/S, M/N, G/H)
      -> deliberate: those four pairs need a 0.50 gap instead of 0.10, because
         the data says a close second letter is a wrong letter.
  N, M or T never types right
      -> real and measured (N 0.4 %, M 3.2 %, T 16.8 % cross-dataset, 90-96 %
         in-sample). It is a data problem, not a bug. docs/DETAILS.md explains
         the diagnosis and what was tried.
  Too slow / choppy
      -> Runtime = onnx, overlay off, raise "Work every Nth frame" to 3 or 4.
  Port 8501 already in use
      -> another copy is running. Close it, or run
         streamlit run app.py --server.port 8502
  "landmarks.parquet was extracted with cache_version=1"
      -> deliberate guard: the extractor changed what it stores. Re-run
         python src/extract_landmarks.py --force
```


PRIVACY - WHAT THIS PROJECT DOES NOT DO
=======================================
No tool here records, saves or trains on your webcam images. live.py and app.py
read frames, draw on them in RAM and throw them away; nothing is written to
disk, uploaded, or added to any dataset. The training data is 100 % public
datasets already on disk. To check it yourself:

```
     cd "D:\sign language project\ASTL project\ASLT\asl_landmark_app"
     grep -rn -E 'imwrite|VideoWriter|np\.savez|\.csv\.to_csv' app.py src/live.py
     python src/live.py --logic-test
```

The first prints nothing (verified). The second is 87 checks over the decision
logic and never opens a camera. The only files the live path can write are
reports, and only when a script is run with an explicit `--out`.


VERIFY WITHOUT A CAMERA
=======================
```
     cd "D:\sign language project\ASTL project\ASLT\asl_landmark_app"

     python src/live.py --logic-test
         -> 87 checks on the voting, the pair margins and the J/Z rule. PASS/FAIL.

     python src/verify_livepath.py --profile all --backend onnx
         -> the whole live path on real dataset photographs at ~20 FPS, no
            camera. Currently 70/72 letter blocks answered correctly; writes
            reports/live_path.md. A wiring test, not an accuracy test.

     python src/verify_jz_trigger.py --per 300
         -> how often each J/Z gate opens per letter, and what fires. Writes
            reports/jz_trigger.md.

     python src/insample_check.py --profile all
         -> per-letter accuracy of the shipped model on its own training data:
            the transfer-vs-capacity evidence.
```


DATASETS USED
=============
```
  grassknoted/asl-alphabet  78,000 images   1.00 GB  GPL 2.0 (as stated on the
                              Kaggle page) - 26 real photographs per letter plus
                              3,000 synthetic renders per letter
  massey                    65,774 images   1.91 GB  licence not stated on the
                              Kaggle page - the only multi-signer set here, and
                              the hardest
  danrasband                   780 images   0.01 GB  CC0 - public domain
                              2.92 GB in total, measured from data/datasets.json

  MSL dynamic (Zenodo 14689869) - 622 short J/K/N/Q/X/Z clips, used only to
  measure whether a movement classifier could reach J/Z. It does not transfer to
  ASL and is not part of the shipped model (REPORT.md round 6).

  Deliberately NOT used: Sign Language MNIST (28x28 grayscale - no landmarks to
  extract).
```


WHAT IS IN GIT
==============
The `.gitignore` in this folder keeps the reviewable content small. Committed:
the code, requirements.txt, data/datasets.json, the two shipped profiles (.pt +
.onnx + meta.json) and the reports. Measured with `git ls-files -o
--exclude-standard` after that .gitignore: **52 files, 1.5 MB**.

Ignored, because `python run_all.py` regenerates them:

```
  cache/                    76 MB  landmark store
  reports/oof_*.csv         23 MB  per-frame predictions
  reports/samples/          contact sheets
  reports/preds_*.csv
  models/hand_landmarker.task   7.5 MB  MediaPipe download
  datasets/, data/raw/      downloaded images and archives
  *.zip *.7z *.tar.gz *.mp4 *.mov *.avi
  __pycache__/, .venv/, logs, *.tmp, *.bak
  .env, kaggle.json, credentials.json, .streamlit/secrets.toml
```


LIMITATIONS (read before trusting a number)
===========================================
```
  - 67.0 % is the honest cross-dataset figure; 85.7 % is unseen hands in a
    matching style, and a webcam is somewhere between the two. Neither is a
    live measurement, and no live measurement is claimed anywhere.
  - N and M are near 0 % cross-dataset because the datasets disagree about what
    they look like; more features will not fix it, more data will.
  - J and Z depend on your movement being big enough for the thresholds. They
    are calibrated from synthetic strokes until you run the calibration yourself.
  - ~20 fps is the ceiling on this machine's CPU, whatever the settings.
```


MORE DETAIL
===========
```
  REPORT.md        why each decision was made, round 1 to 8, including the two
                   ideas that failed and the reporting bugs that were fixed.
  docs/DETAILS.md  the full reference this README summarises: every measured
                   table, the stage-by-stage pipeline, all script flags, the
                   normalisation maths, the complete troubleshooting table, and
                   the reproducibility notes (seeds, parity, commands).
```
