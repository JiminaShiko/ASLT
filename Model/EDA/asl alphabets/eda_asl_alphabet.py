#ASL Alphabet - Exploratory Data Analysis (EDA)
#--------------------------------------------------
# 1. Import Libraries
#--------------------------------------------------
import json
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

warnings.filterwarnings("ignore")
sns.set_theme(style = "whitegrid")

#--------------------------------------------------
# 2. Folder Paths and Settings
#--------------------------------------------------
CHART_DIR = "charts_asl_alphabet"
os.makedirs(CHART_DIR, exist_ok = True)

CLEANUP  = True
N_PIXEL  = 500
N_HEADER = 500

R = {}

#--------------------------------------------------
# 3. Download the Dataset (Automatic, ~1 GB)
#--------------------------------------------------
print("Downloading ASL Alphabet from Kaggle (~1 GB, temporary)........")
import kagglehub
cache_root = kagglehub.dataset_download("grassknoted/asl-alphabet")

# Find the asl_alphabet_train folder anywhere under the cache
TRAIN_DIR = None
for root, subs, _ in os.walk(cache_root):
    if "asl_alphabet_train" in subs:
        TRAIN_DIR = os.path.join(root, "asl_alphabet_train")
        break
if TRAIN_DIR is None and "asl_alphabet_train" in os.listdir(cache_root):
    TRAIN_DIR = os.path.join(cache_root, "asl_alphabet_train")
if TRAIN_DIR is None:
    print("Could not find asl_alphabet_train under", cache_root)
    print("Top-level entries:", os.listdir(cache_root))
    raise SystemExit

print("Images at:", TRAIN_DIR)

#--------------------------------------------------
# 4. Task 1 - Load and Inspect the Dataset
#--------------------------------------------------
print("Scanning for images (recursive)........")

# The 29 expected class folder names
EXPECTED_CLASSES = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {"space", "del", "nothing"}

rows = []
for root, subs, files in os.walk(TRAIN_DIR):
    label = os.path.basename(root)
    if label not in EXPECTED_CLASSES:
        continue
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append({"path": os.path.join(root, f), "label": label})

if not rows:
    print("ERROR: still no images found under", TRAIN_DIR)
    print("Folder structure (depth 3):")
    for root, subs, files in os.walk(TRAIN_DIR):
        depth = root.replace(TRAIN_DIR, "").count(os.sep)
        if depth > 3:
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or TRAIN_DIR}/  "
              f"({len(subs)} dirs, {len(files)} files)")
    raise SystemExit

df = pd.DataFrame(rows)
df["size_kb"] = df["path"].map(lambda p: os.path.getsize(p) / 1024)

found = sorted(df["label"].unique())
print(f"\nFound {len(df):,} images across {len(found)} classes")
missing = EXPECTED_CLASSES - set(found)
if missing:
    print("WARNING - missing classes:", sorted(missing))

print("\nFirst five rows:")
print(df.head())

print("\nShape (rows, columns):")
print(df.shape)

print("\nInfo:")
df.info()

print("\nDescribe (file_size_kb):")
print(df["size_kb"].describe().round(2))

R["n_images"]     = int(len(df))
R["n_classes"]    = int(df["label"].nunique())
R["label_values"] = found

#--------------------------------------------------
# 5. Task 2 - Features and Target Variable
#--------------------------------------------------
print("""
TASK 2 - FEATURES AND TARGET
  Input features : RGB pixel arrays of size 200 x 200 x 3
  Target variable: 'label' - 29 classes (A-Z, space, del, nothing)
  Key columns    : path (file location), label (target), size_kb
  Relevance      : letters are the fingerspelling input the real-time
                   translator needs; 'space', 'del' and 'nothing' are
                   useful control classes for live prediction.
""")

#--------------------------------------------------
# 6. Task 3 - Check Data Quality
#--------------------------------------------------
R["missing_labels"] = int(df["label"].isna().sum())
R["missing_files"]  = int((~df["path"].map(os.path.exists)).sum())

sample_h = df.sample(min(N_HEADER, len(df)), random_state = 0)
sizes, corrupt = [], 0
for p in sample_h["path"]:
    try:
        with Image.open(p) as im:
            sizes.append(im.size)
    except Exception:
        corrupt += 1

R["corrupt_sample"] = corrupt
R["sizes_sample"]   = {f"{w}x{h}": int(c)
                      for (w, h), c in pd.Series(sizes).value_counts().items()}

counts = df["label"].value_counts()
R["class_min"]  = int(counts.min())
R["class_max"]  = int(counts.max())
R["class_mean"] = float(round(counts.mean(), 1))

print("\nDATA QUALITY")
print("Missing labels  :", R["missing_labels"])
print("Missing files   :", R["missing_files"])
print("Corrupt (sample):", corrupt)
print("Image sizes     :", R["sizes_sample"])
print("Per class (min/mean/max):", R["class_min"], R["class_mean"], R["class_max"])

#--------------------------------------------------
# 7. Task 4 - Descriptive Statistics
#--------------------------------------------------
bright = []
for p in df.sample(min(N_PIXEL, len(df)), random_state = 1)["path"]:
    try:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype = np.float32)
        bright.append(arr.mean())
    except Exception:
        pass

R["brightness_mean"] = float(np.mean(bright))
R["brightness_std"]  = float(np.std(bright))
R["brightness_min"]  = float(np.min(bright))
R["brightness_max"]  = float(np.max(bright))

print("\nBRIGHTNESS (0-255)")
print(f"  mean  = {R['brightness_mean']:.1f}")
print(f"  std   = {R['brightness_std']:.1f}")
print(f"  range = {R['brightness_min']:.1f} to {R['brightness_max']:.1f}")

print("\nPer-class counts:\n", counts.sort_index())

#--------------------------------------------------
# 8. Task 5 - EDA Visualizations
#--------------------------------------------------
plt.figure(figsize = (14, 4.5))
counts.sort_index().plot(kind = "bar", color = "steelblue")
plt.axhline(counts.mean(), color = "red", linestyle = "--",
            label = f"mean = {counts.mean():.0f}")
plt.title("Number of Images per Class")
plt.xlabel("Class label")
plt.ylabel("Number of images")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "01_images_per_class.png"), dpi = 150)
plt.show()

labels = sorted(df["label"].unique())
fig, axes = plt.subplots(5, 6, figsize = (10, 8))
axes = axes.ravel()
for ax, lab in zip(axes, labels):
    p = df[df["label"] == lab].iloc[0]["path"]
    ax.imshow(Image.open(p).convert("RGB"))
    ax.set_title(lab)
    ax.axis("off")
for ax in axes[len(labels):]:
    ax.axis("off")
fig.suptitle("One Sample Image per Class (ASL Alphabet)")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "02_sample_grid.png"), dpi = 150)
plt.show()

plt.figure(figsize = (8, 4.5))
sns.histplot(bright, bins = 30, kde = True, color = "slateblue")
plt.axvline(R["brightness_mean"], color = "red", linestyle = "--",
            label = f"mean = {R['brightness_mean']:.1f}")
plt.title("Distribution of Mean Pixel Brightness (sampled images)")
plt.xlabel("Mean brightness (0-255)")
plt.ylabel("Number of images")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "03_brightness_histogram.png"), dpi = 150)
plt.show()

plt.figure(figsize = (14, 4.5))
sns.boxplot(data = df, x = "label", y = "size_kb",
            color = "#90cdf4", fliersize = 2)
plt.title("Image File Size per Class")
plt.xlabel("Class label")
plt.ylabel("File size (KB)")
plt.xticks(rotation = 90)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "04_filesize_by_class.png"), dpi = 150)
plt.show()

#--------------------------------------------------
# 9. Save Results and Cleanup
#--------------------------------------------------
with open("results_asl_alphabet.json", "w") as f:
    json.dump(R, f, indent = 2, default = str)

print("\nEDA finished. Charts in", CHART_DIR + "/")

if CLEANUP:
    print("Freeing ~1 GB by deleting the downloaded images........")
    shutil.rmtree(cache_root, ignore_errors = True)
    print("Done. Only charts + results_asl_alphabet.json remain.")
else:
    print("Images kept in:", cache_root)