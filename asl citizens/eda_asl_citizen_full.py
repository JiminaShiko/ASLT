#ASL Citizen - Exploratory Data Analysis (EDA)
#--------------------------------------------------
# 1. Import Libraries
#--------------------------------------------------
import json
import os
import shutil
import sys
import time
import urllib.request
import warnings
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Hide warning messages to keep the output clean
warnings.filterwarnings("ignore")

# Use a clean grid style for all charts
sns.set_theme(style = "whitegrid")

#--------------------------------------------------
# 2. Folder Paths and Settings
#--------------------------------------------------
# >>> CHANGE THIS to a folder on your D: drive that has ~70 GB free <<<
BIG_DATA_DIR = r"D:\ASL_Citizen_workspace"
os.makedirs(BIG_DATA_DIR, exist_ok = True)

# ZIP + extracted data (all on D:)
ZIP_PATH  = os.path.join(BIG_DATA_DIR, "ASL_Citizen.zip")
DATA_DIR  = os.path.join(BIG_DATA_DIR, "ASL_Citizen")

# Charts go next to the script (project folder)
CHART_DIR = "charts_asl_citizen"
os.makedirs(CHART_DIR, exist_ok = True)

# Delete the ZIP + extracted folder after the EDA finishes
CLEANUP     = True

# Safety pre-flight on D:
MIN_FREE_GB = 70

# Official download URL from Microsoft's ASL Citizen page
ZIP_URL = ("https://download.microsoft.com/download/b/8/8/"
           "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip")

# Stores every result - saved to results_asl_citizen.json at the end
R = {}

#--------------------------------------------------
# 3. Pre-flight Disk Check on D:
#--------------------------------------------------
drive   = os.path.splitdrive(BIG_DATA_DIR)[0] + os.sep
free_gb = shutil.disk_usage(drive).free / 1e9

print(f"Big-data folder : {BIG_DATA_DIR}")
print(f"Drive           : {drive}")
print(f"Free on {drive}   : {free_gb:.1f} GB")
print(f"Required        : ~{MIN_FREE_GB} GB")

if free_gb < MIN_FREE_GB:
    print(f"\nNot enough free space on {drive}. Free at least {MIN_FREE_GB} GB.")
    sys.exit(1)

#--------------------------------------------------
# 4. Resumable Download (~30 GB)
#--------------------------------------------------
def download_resumable(url, dest, chunk_mb = 8):
    """Download with HTTP Range resume - safe to rerun after interruption."""
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    req = urllib.request.Request(url)
    if existing > 0:
        req.add_header("Range", f"bytes={existing}-")
        print(f"Resuming from {existing / 1e9:.2f} GB........")

    with urllib.request.urlopen(req) as r:
        total = existing + int(r.headers.get("Content-Length", 0))
        mode  = "ab" if existing > 0 else "wb"
        with open(dest, mode) as f:
            done  = existing
            chunk = chunk_mb * 1024 * 1024
            last  = 0.0
            while True:
                data = r.read(chunk)
                if not data:
                    break
                f.write(data)
                done += len(data)
                now = time.time()
                if now - last > 2:
                    pct = done / total * 100 if total else 0
                    print(f"\r  {done/1e9:6.2f} / {total/1e9:.2f} GB "
                          f"({pct:5.1f}%)", end = "", flush = True)
                    last = now
        print()

print("Downloading ASL Citizen ZIP (~30 GB) to D:........")
if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) < 25_000_000_000:
    download_resumable(ZIP_URL, ZIP_PATH)
    print("Download finished.")
else:
    print("ZIP already present, skipping download.")

#--------------------------------------------------
# 5. Extract, then delete the ZIP
#--------------------------------------------------
if not os.path.isdir(DATA_DIR):
    print("\nExtracting ZIP on D: (~30 GB)........")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(BIG_DATA_DIR)

    # Some releases wrap everything in a single folder with a different name
    candidates = [d for d in os.listdir(BIG_DATA_DIR)
                  if d.lower().startswith("asl_citizen") and d != "ASL_Citizen"]
    if not os.path.isdir(DATA_DIR) and candidates:
        shutil.move(os.path.join(BIG_DATA_DIR, candidates[0]), DATA_DIR)

    print("Extracted to:", DATA_DIR)

    if CLEANUP and os.path.exists(ZIP_PATH):
        print("Deleting ZIP (frees 30 GB on D:)........")
        os.remove(ZIP_PATH)

#--------------------------------------------------
# 6. Find CSVs and Videos (recursive)
#--------------------------------------------------
csv_files   = [os.path.join(r, f) for r, _, fs in os.walk(DATA_DIR)
               for f in fs if f.lower().endswith(".csv")]
video_files = [os.path.join(r, f) for r, _, fs in os.walk(DATA_DIR)
               for f in fs if f.lower().endswith(".mp4")]

print(f"\nCSVs found  : {len(csv_files)}")
print(f"Videos found: {len(video_files)}")

if not csv_files:
    print("\nNo CSVs found under", DATA_DIR)
    print("Actual layout of", BIG_DATA_DIR, ":")
    for root, subs, files in os.walk(BIG_DATA_DIR):
        depth = root.replace(BIG_DATA_DIR, "").count(os.sep)
        if depth > 3:
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or BIG_DATA_DIR}/  "
              f"({len(subs)} dirs, {len(files)} files)")
    sys.exit(1)

#--------------------------------------------------
# 7. Load the Metadata
#--------------------------------------------------
def standardize(df):
    """Rename likely column variants to video_id / gloss / signer_id."""
    cols = {c.lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    ren = {}
    for key, target in [
        ("videoid", "video_id"), ("videofile", "video_id"),
        ("filename", "video_id"), ("video", "video_id"),
        ("gloss", "gloss"), ("label", "gloss"), ("word", "gloss"),
        ("signerid", "signer_id"), ("participantid", "signer_id"),
        ("signer", "signer_id"), ("participant", "signer_id"),
    ]:
        if key in cols and target not in ren.values():
            ren[cols[key]] = target
    return df.rename(columns = ren)

frames = []
for p in csv_files:
    name  = os.path.basename(p).lower()
    split = ("val" if "val" in name else
             "train" if "train" in name else
             "test" if "test" in name else "unknown")
    d = standardize(pd.read_csv(p))
    d["split"] = split
    frames.append(d)

meta = pd.concat(frames, ignore_index = True)
if "video_id" in meta.columns:
    meta = meta.drop_duplicates(subset = "video_id").reset_index(drop = True)

#--------------------------------------------------
# 8. Task 1 - Load and Inspect
#--------------------------------------------------
print("\nFirst five rows:")
print(meta.head())

print("\nShape (rows, columns):")
print(meta.shape)

print("\nInfo:")
meta.info()

print("\nDescribe:")
print(meta.describe(include = "all").T.head(10))

R["n_metadata_rows"]  = int(meta.shape[0])
R["n_videos_on_disk"] = int(len(video_files))
R["columns"]          = list(meta.columns)

#--------------------------------------------------
# 9. Task 2 - Features and Target
#--------------------------------------------------
print("""
TASK 2 - FEATURES AND TARGET
  Input features : raw RGB video frames + temporal information
  Target variable: gloss (word label, ~2,700 classes)
  Key columns    : video_id (file key), gloss (target),
                   signer_id (used for signer-independent splits),
                   split (official train / val / test)
  Relevance      : word-level units are what the translator outputs;
                   webcam recordings in ASL Citizen match the deployment
                   conditions of the real-time translator.
""")

#--------------------------------------------------
# 10. Task 3 - Data Quality
#--------------------------------------------------
R["missing_total"] = int(meta.isna().sum().sum())
R["dup_rows"]      = int(meta.duplicated().sum())
R["dup_ids"]       = int(meta["video_id"].duplicated().sum()) if "video_id" in meta else -1

vid_index = {os.path.splitext(os.path.basename(p))[0]: p for p in video_files}
if "video_id" in meta.columns:
    meta["path"]  = meta["video_id"].astype(str).map(vid_index)
    meta["found"] = meta["path"].notna()
else:
    meta["found"] = False
R["rows_without_video"] = int((~meta["found"]).sum())

print("\nDATA QUALITY")
print("Missing values      :", R["missing_total"])
print("Duplicate rows      :", R["dup_rows"])
print("Duplicate video_ids :", R["dup_ids"])
print("Rows with no .mp4   :", R["rows_without_video"])

#--------------------------------------------------
# 11. Task 4 - Descriptive Statistics
#--------------------------------------------------
if "gloss" in meta.columns:
    per_class = meta["gloss"].value_counts()
    R["n_classes"] = int(meta["gloss"].nunique())
    R["videos_per_class_mean"]   = float(round(per_class.mean(), 1))
    R["videos_per_class_median"] = float(per_class.median())
    R["videos_per_class_min"]    = int(per_class.min())
    R["videos_per_class_max"]    = int(per_class.max())
    R["classes_with_one_video"]  = int((per_class == 1).sum())
    print(f"\nVideos per class : mean={R['videos_per_class_mean']} | "
          f"median={R['videos_per_class_median']} | "
          f"min={R['videos_per_class_min']} | max={R['videos_per_class_max']}")

if "signer_id" in meta.columns:
    per_signer = meta["signer_id"].value_counts()
    R["n_signers"] = int(meta["signer_id"].nunique())
    R["videos_per_signer_min"] = int(per_signer.min())
    R["videos_per_signer_max"] = int(per_signer.max())
    print(f"Signers : {R['n_signers']} | per-signer min/max: "
          f"{R['videos_per_signer_min']}/{R['videos_per_signer_max']}")

R["split_counts"] = {str(k): int(v) for k, v in meta["split"].value_counts().items()}
print("Splits:", R["split_counts"])

#--------------------------------------------------
# 12. Probe ALL videos (header-only reads)
#--------------------------------------------------
def probe_header(path):
    """Read only the video header - fast even for 84,000 videos."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return {
        "fps": fps, "n_frames": n, "width": w, "height": h,
        "duration_sec": (n / fps) if fps and fps > 0 else np.nan,
        "size_mb": os.path.getsize(path) / 1e6,
        "resolution": f"{int(w)}x{int(h)}",
    }

print(f"\nProbing ALL {len(video_files)} videos (header reads only)........")
t0 = time.time()
rows, broken = [], 0
for i, p in enumerate(video_files, 1):
    info = probe_header(p)
    if info is None:
        broken += 1
    else:
        rows.append(info)
    if i % 1000 == 0 or i == len(video_files):
        elapsed = time.time() - t0
        print(f"\r  {i}/{len(video_files)}  ({elapsed:.0f}s)",
              end = "", flush = True)
print()

deep = pd.DataFrame(rows)

R["probe_total"]           = int(len(deep))
R["probe_broken"]          = int(broken)
R["probe_fps_median"]      = float(deep["fps"].median())
R["probe_fps_min"]         = float(deep["fps"].min())
R["probe_fps_max"]         = float(deep["fps"].max())
R["probe_duration_median"] = float(deep["duration_sec"].median())
R["probe_duration_mean"]   = float(deep["duration_sec"].mean())
R["probe_duration_min"]    = float(deep["duration_sec"].min())
R["probe_duration_max"]    = float(deep["duration_sec"].max())
R["probe_duration_p95"]    = float(deep["duration_sec"].quantile(0.95))
R["probe_size_median_mb"]  = float(round(deep["size_mb"].median(), 2))
R["probe_resolutions"]     = {str(k): int(v)
                             for k, v in deep["resolution"].value_counts().items()}
R["probe_short_clips"]     = int((deep["duration_sec"] < 0.3).sum())
R["probe_long_clips"]      = int((deep["duration_sec"] > 10).sum())

print("Broken videos       :", R["probe_broken"])
print("fps median          :", R["probe_fps_median"])
print("Duration median/min/max (s):",
      round(R["probe_duration_median"], 2), "/",
      round(R["probe_duration_min"], 2), "/",
      round(R["probe_duration_max"], 2))
print("Clips < 0.3 s       :", R["probe_short_clips"])
print("Clips > 10 s        :", R["probe_long_clips"])
print("Top resolutions     :", dict(list(R["probe_resolutions"].items())[:5]))

#--------------------------------------------------
# 13. Task 5 - EDA Visualizations
#--------------------------------------------------
def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, name), dpi = 150)
    plt.show()

# Chart 1 - Top-25 most frequent classes
if "gloss" in meta.columns:
    plt.figure(figsize = (14, 4.5))
    per_class.head(25).plot(kind = "bar", color = "steelblue")
    plt.title("Top-25 Most Frequent Signs")
    plt.xlabel("Gloss (word)")
    plt.ylabel("Number of videos")
    save("01_top25_classes.png")

# Chart 2 - Videos per signer
if "signer_id" in meta.columns:
    plt.figure(figsize = (14, 4))
    per_signer.sort_values().plot(kind = "bar", color = "teal")
    plt.title("Number of Videos per Signer")
    plt.xlabel("Signer ID")
    plt.ylabel("Number of videos")
    save("02_videos_per_signer.png")

# Chart 3 - Class long tail
if "gloss" in meta.columns:
    plt.figure(figsize = (8, 4.5))
    plt.plot(range(len(per_class)), per_class.values, color = "darkorange")
    plt.yscale("log")
    plt.title(f"Videos per Sign (all {len(per_class)} classes)")
    plt.xlabel("Class rank")
    plt.ylabel("Number of videos (log scale)")
    save("03_class_longtail.png")

# Chart 4 - Duration histogram
plt.figure(figsize = (8, 4.5))
sns.histplot(deep["duration_sec"].dropna(), bins = 50, color = "slateblue")
plt.axvline(R["probe_duration_median"], color = "red", linestyle = "--",
            label = f"median = {R['probe_duration_median']:.2f} s")
plt.title(f"Clip Durations (all {len(deep):,} videos)")
plt.xlabel("Duration (seconds)")
plt.ylabel("Number of videos")
plt.legend()
save("04_duration_histogram.png")

# Chart 5 - Resolution distribution
if len(R["probe_resolutions"]) > 0:
    res = pd.Series(R["probe_resolutions"]).sort_values(ascending = False).head(8)
    plt.figure(figsize = (10, 4.5))
    res.plot(kind = "bar", color = "steelblue")
    plt.title("Video Resolutions (all videos)")
    plt.xlabel("Resolution")
    plt.ylabel("Number of videos")
    plt.xticks(rotation = 45)
    save("05_resolution_counts.png")

# Chart 6 - File size histogram
plt.figure(figsize = (8, 4.5))
sns.histplot(deep["size_mb"].dropna(), bins = 50, color = "slateblue")
plt.axvline(R["probe_size_median_mb"], color = "red", linestyle = "--",
            label = f"median = {R['probe_size_median_mb']:.1f} MB")
plt.title("File Sizes (all videos)")
plt.xlabel("File size (MB)")
plt.ylabel("Number of videos")
plt.legend()
save("06_filesize_histogram.png")

# Chart 7 - Split distribution
plt.figure(figsize = (6, 4.5))
meta["split"].value_counts().plot(kind = "bar",
                                 color = ["steelblue", "darkorange", "seagreen"])
plt.title("Videos per Split")
plt.xlabel("Split")
plt.ylabel("Number of videos")
save("07_split_distribution.png")

#--------------------------------------------------
# 14. Save Results
#--------------------------------------------------
with open("results_asl_citizen.json", "w") as f:
    json.dump(R, f, indent = 2, default = str)

print("\nEDA finished. Charts in", CHART_DIR + "/")
print("Results in results_asl_citizen.json")

#--------------------------------------------------
# 15. Cleanup (deletes ZIP + extracted folder)
#--------------------------------------------------
if CLEANUP:
    print("\nCleaning up D: (frees ~30 GB)........")
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR, ignore_errors = True)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    try:
        os.rmdir(BIG_DATA_DIR)
    except OSError:
        pass
    print("Done. Only charts + results_asl_citizen.json remain.")
else:
    print("Data kept in:", BIG_DATA_DIR)
    print("Set CLEANUP = True to auto-delete next run.")