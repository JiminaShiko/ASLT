# ============================================================
# WLASL - Exploratory Data Analysis (EDA)
# ============================================================
# Dataset: Word-Level American Sign Language (WLASL)
# Metadata file: WLASL_v0.3.json
# ============================================================

import os
import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 140)

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
JSON_PATH = "WLASL_v0.3.json"
CHART_DIR = "charts"
TABLE_DIR = "tables"

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

results = {}

print("=" * 65)
print("WLASL - WORD-LEVEL AMERICAN SIGN LANGUAGE EDA")
print("=" * 65)

# ------------------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------------------
print("\n1. LOADING DATASET")
print("-" * 65)

if not os.path.exists(JSON_PATH):
    print(f"ERROR: {JSON_PATH} not found.")
    print("Please place WLASL_v0.3.json in the same folder as this script.")
    raise SystemExit

with open(JSON_PATH, "r") as f:
    data = json.load(f)

print(f"File loaded successfully: {JSON_PATH}")
print(f"Type of data: {type(data)}")
print(f"Total number of words (glosses): {len(data)}")

results["total_words"] = len(data)

# ------------------------------------------------------------
# 3. Convert to DataFrame for easier analysis
# ------------------------------------------------------------
print("\n2. CONVERTING TO DATAFRAME")
print("-" * 65)

rows = []
for entry in data:
    gloss = entry.get("gloss", "unknown")
    instances = entry.get("instances", [])
    for inst in instances:
        rows.append({
            "gloss": gloss,
            "video_id": inst.get("video_id"),
            "signer_id": inst.get("signer_id"),
            "split": inst.get("split"),
            "source": inst.get("source"),
            "fps": inst.get("fps"),
            "frame_start": inst.get("frame_start"),
            "frame_end": inst.get("frame_end"),
            "url": inst.get("url"),
            "instance_id": inst.get("instance_id"),
            "variation_id": inst.get("variation_id")
        })

df = pd.DataFrame(rows)

print(f"Total video instances: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()
print(df.head())

results["total_videos"] = len(df)
results["columns"] = list(df.columns)

# ------------------------------------------------------------
# 4. Basic Inspection (Task 1)
# ------------------------------------------------------------
print("\n3. BASIC INSPECTION")
print("-" * 65)

print("\n--- Shape ---")
print(f"Rows (videos): {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Descriptive statistics (numerical) ---")
print(df.describe())

print("\n--- First 5 rows ---")
print(df.head())

# ------------------------------------------------------------
# 5. Features and Target (Task 2)
# ------------------------------------------------------------
print("\n4. FEATURES AND TARGET VARIABLE")
print("-" * 65)

print("Target variable: 'gloss' (the ASL word / sign)")
print(f"Number of unique words (classes): {df['gloss'].nunique()}")
print()
print("Important columns:")
print("  - gloss         : The ASL word (target label)")
print("  - video_id      : Unique ID of the video")
print("  - signer_id     : ID of the person who signed")
print("  - split         : train / val / test split")
print("  - source        : Where the video came from")
print("  - fps           : Frames per second")
print("  - frame_start/end : Start and end frame of the sign")

results["num_classes"] = int(df["gloss"].nunique())

# ------------------------------------------------------------
# 6. Data Quality Checks (Task 3)
# ------------------------------------------------------------
print("\n5. DATA QUALITY CHECKS")
print("-" * 65)

missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing)

print(f"\nTotal missing values: {missing.sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Check split values
print("\nSplit distribution:")
print(df["split"].value_counts(dropna=False))

results["missing_values"] = missing.to_dict()
results["duplicate_rows"] = int(df.duplicated().sum())

print("\nConclusion: Metadata is mostly clean. Some missing values may exist in optional fields.")

# ------------------------------------------------------------
# 7. Descriptive Statistics (Task 4)
# ------------------------------------------------------------
print("\n6. CLASS DISTRIBUTION (Top words)")
print("-" * 65)

word_counts = df["gloss"].value_counts()
print(f"Most frequent word: '{word_counts.index[0]}' with {word_counts.iloc[0]} videos")
print(f"Least frequent word (in top data): '{word_counts.index[-1]}' with {word_counts.iloc[-1]} videos")
print(f"Average videos per word: {word_counts.mean():.1f}")
print(f"Median videos per word: {word_counts.median():.1f}")

# Top 20 words
top20 = word_counts.head(20).reset_index()
top20.columns = ["gloss", "video_count"]
print("\nTop 20 most frequent words:")
print(top20.to_string(index=False))

top20.to_csv(os.path.join(TABLE_DIR, "top20_words.csv"), index=False)
word_counts.reset_index().rename(columns={"index": "gloss", "gloss": "video_count"}).to_csv(
    os.path.join(TABLE_DIR, "all_word_counts.csv"), index=False
)

results["avg_videos_per_word"] = float(word_counts.mean())
results["median_videos_per_word"] = float(word_counts.median())
results["top_10_words"] = word_counts.head(10).to_dict()

# Signer statistics
print("\n--- Signer Statistics ---")
print(f"Number of unique signers: {df['signer_id'].nunique()}")
print(df["signer_id"].value_counts().head(10))

# ------------------------------------------------------------
# 8. Visualizations (Task 5)
# ------------------------------------------------------------
print("\n7. CREATING CHARTS")
print("-" * 65)

# Chart 1: Top 20 words
plt.figure(figsize=(12, 6))
sns.barplot(x="video_count", y="gloss", data=top20, palette="viridis")
plt.title("Top 20 Most Frequent ASL Words in WLASL", fontsize=14)
plt.xlabel("Number of Videos")
plt.ylabel("Word (Gloss)")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "01_top20_words.png"), dpi=120)
plt.close()
print("Saved: 01_top20_words.png")

# Chart 2: Distribution of videos per word
plt.figure(figsize=(10, 5))
sns.histplot(word_counts.values, bins=30, kde=True, color="steelblue")
plt.title("Distribution of Number of Videos per Word", fontsize=14)
plt.xlabel("Number of Videos per Word")
plt.ylabel("Number of Words")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "02_videos_per_word_distribution.png"), dpi=120)
plt.close()
print("Saved: 02_videos_per_word_distribution.png")

# Chart 3: Split distribution
plt.figure(figsize=(8, 5))
split_counts = df["split"].value_counts()
sns.barplot(x=split_counts.index.astype(str), y=split_counts.values, palette="Set2")
plt.title("Train / Val / Test Split Distribution", fontsize=14)
plt.xlabel("Split")
plt.ylabel("Number of Videos")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "03_split_distribution.png"), dpi=120)
plt.close()
print("Saved: 03_split_distribution.png")

# Chart 4: Top 15 signers
top_signers = df["signer_id"].value_counts().head(15)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_signers.index.astype(str), y=top_signers.values, palette="magma")
plt.title("Top 15 Signers by Number of Videos", fontsize=14)
plt.xlabel("Signer ID")
plt.ylabel("Number of Videos")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "04_top_signers.png"), dpi=120)
plt.close()
print("Saved: 04_top_signers.png")

# Chart 5: Boxplot of videos per word
plt.figure(figsize=(8, 5))
sns.boxplot(x=word_counts.values, color="lightblue")
plt.title("Boxplot: Videos per Word", fontsize=14)
plt.xlabel("Number of Videos")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "05_boxplot_videos_per_word.png"), dpi=120)
plt.close()
print("Saved: 05_boxplot_videos_per_word.png")

print("\nAll charts saved in the 'charts/' folder.")

# ------------------------------------------------------------
# 9. Key Findings (Task 6)
# ------------------------------------------------------------
print("\n8. KEY FINDINGS FROM EDA")
print("-" * 65)

findings = [
    "1. WLASL contains 2,000 unique ASL words (glosses) and approximately 21,083 video instances.",
    "2. The number of videos per word varies significantly (some words have many examples, others have few).",
    "3. The dataset provides train/val/test splits, which is useful for model evaluation.",
    "4. Multiple signers contributed, which increases diversity but also adds variation in signing style.",
    "5. For this student project, it is recommended to select only 20–25 high-frequency / common words to keep the scope realistic.",
    "6. The metadata (JSON) is clean and sufficient for EDA. Actual video files can be downloaded later only for the selected words."
]

for f in findings:
    print(f)

results["findings"] = findings

# Suggested words for the project (common + reasonably frequent)
suggested = ["hello", "thanks", "yes", "no", "please", "help", "love", "sorry",
             "good", "bad", "eat", "drink", "home", "school", "friend", "family",
             "happy", "sad", "bye", "name", "mother", "father", "book", "water", "go"]

available_suggested = [w for w in suggested if w in word_counts.index]
print("\nSuggested common words available in WLASL:")
print(available_suggested)

results["suggested_words_available"] = available_suggested

# Save results
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 65)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 65)
print("Outputs created:")
print("  - charts/          (5 visualization files)")
print("  - tables/          (top20_words.csv, all_word_counts.csv)")
print("  - results.json     (key statistics and findings)")
print("=" * 65)
