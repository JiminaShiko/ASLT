# Video model: compact seed and incremental updates

The video workflow stores a compact replay/evaluation set so a later training run can
use earlier words after the original raw videos have been removed. This is still a
video model: each seed item is the existing preprocessed 16-frame, 112×112 RGB
`uint8` tensor, compressed as `.npz`. Seed storage therefore grows with the number of
retained clips; check the reported byte total before deleting raw data.

Run commands from this folder. Install dependencies with `python -m pip install -r
requirements.txt`; prepare source data using the root README workflow, then run
`python -m src.preprocess`. Training/evaluation artifacts are written to
`datasets/processed/training_artifacts/`, and standalone checkpoints are written to
`models/`. Prediction does not require the training dataset or processed arrays:

```powershell
python -m src.predict_video "path\to\video.mp4" --model cnn3d
```

For a full-length WLASL source video, pass `--wlasl-id <video-id>` so preprocessing
uses the annotated signing interval.

## Initial preparation

1. Prepare and preprocess the dataset as described in the root project README.
2. Create a seed from the processed manifest:

   ```powershell
   cd videoModel
   python -m src.seed_dataset --per-class-train 20 --per-class-val 5 --per-class-test 5
   ```

   This creates `datasets/processed/seed/manifest.csv`, `metadata.json`, and
   compressed clips beneath `datasets/processed/seed/{train,val,test}/`. It samples
   deterministically and round-robins across signer IDs. Increase or reduce the
   three quotas to fit available storage. Train seeds are replay examples; validation
   and test seeds are held-out anchors, not training samples.

3. Train the initial model normally:

   ```powershell
   python -m src.train_model --models cnn_gru,cnn3d
   ```

## Add words or examples later

1. Add new glosses to `TARGET_WORDS` in `src/config.py` and preprocess the new or
   expanded video collection so `datasets/processed/manifest.csv` lists it.
2. Rebuild/refresh the seed before removing source data. The seed builder merges the
   prior seed manifest with currently available processed examples and carries
   forward old seed clips that are absent from the raw manifest. It refreshes to the
   requested per-class, per-split quotas.
3. Run incremental training:

   ```powershell
   python -m src.train_model --models cnn_gru,cnn3d
   ```

   The prior model's class order is retained and newly requested labels are appended.
   Existing class names must remain in the requested vocabulary. For a deliberate
   clean retrain, pass `--fresh` (this does not delete the old checkpoint before a
   successful new model is written).

## Download more examples for the existing 20 words

The repository includes `download_popsign_targets.py`, which retrieves only exact
matching PopSign sign archives instead of downloading the full dataset. From the
`videoModel` folder, run:

```powershell
python download_popsign_targets.py
```

The default downloads the official `game/train` archives only and writes extracted
videos beneath `datasets/external/popsign_v2_1/`. It shows matched and unmatched
labels before downloading and asks you to type `DOWNLOAD`. To get all official
splits for evaluation as well, run:

```powershell
python download_popsign_targets.py --splits train val test
```

The downloader uses the official [PopSign v2.1 dataset](https://signdata.cc.gatech.edu/view/datasets/popsign_v2_1/index.html)
and its [per-sign download API](https://signdata.cc.gatech.edu/view/guides/downloading_popsign/index.html).
PopSign is isolated ASL video, but its labels may denote a particular sign variant;
review the sign and label mapping before merging it into an existing gloss class. The
downloaded clips are not automatically added to the current processed manifest or
model training pipeline yet. See `download_manifest.csv` for extracted file paths.

## What an incremental run does

- Merges the raw processed manifest and seed manifest by `(dataset, video_id, split)`;
  seed copies win deduplication so training still works if original processed clips
  are gone. Validation and test seed anchors remain in their respective splits.
- Initializes same-architecture layers from the previous checkpoint. Existing output
  rows are copied by gloss name, while new output rows keep their fresh initialization.
- Uses class-weighted cross entropy over current train data. On retained seed examples
  of old classes, it adds temperature-scaled KL distillation from the previous model.
  New-class examples are not forced to imitate old-class outputs.
- Evaluates the candidate on old-word validation anchors. The candidate is promoted
  to `models/<architecture>.pt` only when every old class has validation support and
  old-word macro recall drops by no more than `--old-class-recall-drop` (default 0.05).
  A rejected candidate leaves the previous standalone model in place. The candidate's
  training artifacts and metrics are written to the selected output directory only
  after acceptance.
- Saves class-order, parent vocabulary, seed, and distillation settings in checkpoint
  metadata. Prediction reads the saved class list, so output labels continue to match
  their classifier indices.

Useful controls:

```powershell
python -m src.train_model --help
python -m src.seed_dataset --help
```

`--distillation-weight` controls replay regularization (default `0.5`),
`--distillation-temperature` controls soft targets (default `2.0`), and
`--old-class-recall-drop` sets the validation acceptance limit. To allow deliberate
vocabulary change, train as a fresh model and explicitly manage the old checkpoint.

## Standalone Streamlit testing

The video model can be tested independently of the website or PHP application. From
the repository root in PowerShell, run:

```powershell
python -m pip install -r ".\videoModel\requirements.txt"
python -m streamlit run ".\videoModel\streamlit_app.py"
```

Or change into the model folder first:

```powershell
Set-Location "D:\Metro\AI Project\ASLT\videoModel"
python -m pip install -r ".\requirements.txt"
python -m streamlit run ".\streamlit_app.py"
```

Quote paths that contain spaces. Use `-r` when installing a requirements file;
without it, pip treats the file path as a package requirement.

The app loads the selected `cnn_gru` or `cnn3d` checkpoint from `models/` and offers:

- **Test a video clip:** upload a video and view its top word predictions. Optionally
  provide a WLASL video ID to use the annotated signing interval for a full source clip.
- **Live webcam:** start the camera to get predictions from a rolling window of 16
  frames, refreshed every 8 incoming frames.

The webcam must be permitted by the browser; camera access generally requires
`localhost` or an HTTPS connection. Prediction uses the saved model checkpoints and
does not require the training dataset. For command-line testing, the
`python -m src.predict_video` command above remains available.

Run the model's automated tests from `videoModel` with
`python -m unittest discover -s tests`.

## Storage and data handling

`seed/metadata.json` records the frame format, class list, selected quotas, record
count, and compressed byte total. The CSV records dataset/video ID, gloss, signer,
split, source path, processed path, preprocessing dimensions, and SHA-256 of each
stored clip. Keep the seed directory and its manifest together. Do not delete seed
validation/test clips if you expect the automated old-word acceptance gate to run.
This workflow reduces the need to retain raw videos but does not guarantee license
rights to redistribute seed clips; follow the source datasets' terms.

## Current constraints

- Seed selection covers signer diversity, but it does not yet score visual diversity
  or model uncertainty. Per-class quotas are a storage/coverage tradeoff.
- The class vocabulary remains defined in `TARGET_WORDS`; new labels must be added
  there before preprocessing/training. Gloss aliases should be added in
  `WORD_ALIASES` where appropriate.
- If old validation anchors are unavailable, incremental promotion fails closed. Keep
  at least one representative validation clip per existing class in the seed.
- Existing `cnn_gru` and `cnn3d` checkpoints are incrementally compatible when their
  architecture and class list are intact. A checkpoint from a different architecture
  cannot initialize the selected model.
