# Video Model Handoff

Run commands from this folder.

```powershell
python -m pip install -r requirements.txt
python -m src.preprocess
python -m src.train_model
python -m unittest discover -v -s tests
```

Training and preprocessing require the datasets. After training, standalone
checkpoints are saved in `models/`. Prediction does not read the dataset or
processed arrays; it needs the code, a checkpoint in `models/`, and the video
being tested.

Test a short clip without dataset files:

```powershell
python -m src.predict_video "path\to\video.mp4" --model cnn3d
```

Full-length WLASL source videos need their metadata to locate the signing interval; use `--wlasl-id 70173` for video ID 70173. For training, keep WLASL clips named with their video IDs and preserve the original metadata splits.

Metrics, plots, and the evaluation PDF are written to `datasets/processed/training_artifacts/`; standalone checkpoints are in `models/`.

Current data: 662 processed clips (346 train, 92 validation, 224 test); 200 records failed preprocessing. Current test accuracy is 4.02% for CNN-GRU and 7.59% for 3D CNN. These are experimental, low-accuracy results; the test split has no `eat` examples.

Read and comply with the dataset terms. Do not redistribute source or derived datasets; confirm with the instructor before sharing trained weights.
