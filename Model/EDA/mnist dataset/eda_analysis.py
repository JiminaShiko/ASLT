#Sign Language MNIST - Exploratory Data Analysis (EDA)
#--------------------------------------------------
# 1. Import Libraries
#--------------------------------------------------
import json
import os
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Hide warning messages to keep the output clean
warnings.filterwarnings("ignore")

# Use a clean grid style for all charts
sns.set_theme(style = "whitegrid")

# Show up to 12 columns when printing a DataFrame
pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 140)

#--------------------------------------------------
# 2. Folder Paths
#--------------------------------------------------
DATA_DIR  = "dataset"       # Folder that contains the two CSV files
CHART_DIR = "charts"        # Folder where the charts are saved
TABLE_DIR = "tables"        # Folder where the summary tables are saved

# Create the output folders if they do not exist yet
os.makedirs(CHART_DIR, exist_ok = True)
os.makedirs(TABLE_DIR, exist_ok = True)

# Dictionary that stores every result - saved to results.json at the end
R = {}

# Label 0-24 -> ASL letter (label 9 = J never appears, 10 = K, ..., 24 = Y)
def letter(label):
    return chr(65 + int(label))

#--------------------------------------------------
# 3. Download the Dataset (Automatic)
#--------------------------------------------------
TRAIN_CSV = os.path.join(DATA_DIR, "sign_mnist_train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "sign_mnist_test.csv")

# Download only if the two CSV files are not in the dataset folder yet
if not (os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV)):
    print("Dataset not found. Downloading Sign Language MNIST from Kaggle........")

    os.makedirs(DATA_DIR, exist_ok = True)

    try:
        # kagglehub downloads the public Kaggle dataset into its cache folder
        import kagglehub
        download_path = kagglehub.dataset_download("datamunge/sign-language-mnist")

        # Search the downloaded folder and copy the two CSV files into dataset/
        for folder, subfolders, files in os.walk(download_path):
            for file_name in files:
                if file_name in ["sign_mnist_train.csv", "sign_mnist_test.csv"]:
                    shutil.copy(os.path.join(folder, file_name), DATA_DIR)

        print("Download finished.")

    except Exception as error:
        print("Automatic download failed:", error)
        print("Please download the dataset manually from:")
        print("https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
        print("and put sign_mnist_train.csv and sign_mnist_test.csv in the dataset folder.")
        sys.exit(1)

#--------------------------------------------------
# 3b. Load the Dataset
#--------------------------------------------------
print("Loading Sign Language MNIST........")

train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

#--------------------------------------------------
# 4. Task 1 - Inspect the Dataset
#--------------------------------------------------
print("\nFirst five rows:")
print(train.head())

print("\nShape (rows, columns):")
print("Train:", train.shape)   # (27455, 785)
print("Test :", test.shape)    # (7172, 785)

print("\nInfo:")
train.info()

description = train.describe().T
description.to_csv(os.path.join(TABLE_DIR, "describe_train.csv"))

print("\nDescribe (label and the first 5 pixels):")
print(description.head(6).round(2))

R["n_train"]     = int(train.shape[0])
R["n_test"]      = int(test.shape[0])
R["n_cols"]      = int(train.shape[1])
R["dtype_counts"] = {str(k): int(v) for k, v in train.dtypes.value_counts().items()}
R["memory_mb"]   = round(float(train.memory_usage(deep = True).sum() / 1e6), 1)

#--------------------------------------------------
# 5. Task 2 - Features and Target Variable
#--------------------------------------------------
feature_cols = [c for c in train.columns if c != "label"]

X      = train[feature_cols].to_numpy(dtype = "float32")
y      = train["label"].to_numpy()
X_test = test[feature_cols].to_numpy(dtype = "float32")
y_test = test["label"].to_numpy()

R["n_features"] = len(feature_cols)

print("""
TASK 2 - FEATURES AND TARGET
  Input features : 784 pixel columns (28 x 28 grayscale image, 0-255)
  Target variable: 'label' - ASL letter class (0-24, but 9 = J is unused)
  Relevance      : letters are the fingerspelling input for the
                   real-time translator; this is the small, fast dataset
                   that will train the base letter classifier.
""")

#--------------------------------------------------
# 6. Task 3 - Check Missing Values
#--------------------------------------------------
R["missing_train"] = int(train.isna().sum().sum())
R["missing_test"]  = int(test.isna().sum().sum())

#--------------------------------------------------
# 7. Task 3 - Check Duplicate Records
#--------------------------------------------------
R["dup_train"] = int(train.duplicated().sum())
R["dup_test"]  = int(test.duplicated().sum())

hash_train = pd.util.hash_pandas_object(train, index = False)
hash_test  = pd.util.hash_pandas_object(test,  index = False)

# Count test images that also appear in the training set (data leakage)
R["test_rows_in_train"] = int(hash_test.isin(set(hash_train)).sum())

#--------------------------------------------------
# 8. Task 3 - Check Data Types and Column Names
#--------------------------------------------------
expected_cols = ["label"] + ["pixel" + str(i) for i in range(1, 785)]

R["columns_as_expected"] = bool(
    list(train.columns) == expected_cols and list(test.columns) == expected_cols
)
R["all_int"] = bool(all(np.issubdtype(t, np.integer) for t in train.dtypes))

#--------------------------------------------------
# 9. Task 3 - Check Unusual or Invalid Values
#--------------------------------------------------
R["pix_min"]           = float(X.min())
R["pix_max"]           = float(X.max())
R["pix_out_of_range"]  = int(((X < 0) | (X > 255)).sum())

labels_present = sorted(int(v) for v in np.unique(y))
R["labels_present"]  = labels_present
R["label_min"]       = labels_present[0]
R["label_max"]       = labels_present[-1]
R["n_classes"]       = len(labels_present)

# Label ids that are missing between the min and max (should be 9 = J)
R["unused_label_ids"] = [i for i in range(R["label_max"] + 1)
                        if i not in labels_present]

# Completely black images
R["blank_images"] = int((X.sum(axis = 1) == 0).sum())

# Pixel columns that never change
R["constant_pixel_cols"] = int((X.std(axis = 0) == 0).sum())

# Brightness outliers (IQR rule)
img_mean = X.mean(axis = 1)
q1, q3   = np.percentile(img_mean, [25, 75])
iqr      = q3 - q1
outliers = (img_mean < q1 - 1.5 * iqr) | (img_mean > q3 + 1.5 * iqr)
R["brightness_outliers"] = int(outliers.sum())

print("\nDATA QUALITY")
for key in ["missing_train", "missing_test", "dup_train", "dup_test",
            "test_rows_in_train", "pix_min", "pix_max", "pix_out_of_range",
            "unused_label_ids", "blank_images", "constant_pixel_cols",
            "brightness_outliers"]:
    print(f"  {key:22s}: {R[key]}")

#--------------------------------------------------
# 10. Task 4 - Pixel Statistics
#--------------------------------------------------
all_pixels = X.ravel()
mean       = float(all_pixels.mean())
std        = float(all_pixels.std())

R["pixel_mean"]   = mean
R["pixel_std"]    = std
R["pixel_median"] = float(np.median(all_pixels))
R["pixel_skew"]   = float(((all_pixels - mean) ** 3).mean() / std ** 3)

R["pct_zero_pixels"]   = float((all_pixels == 0).mean() * 100)
R["pct_bright_pixels"] = float((all_pixels >= 250).mean() * 100)

R["test_pixel_mean"] = float(X_test.mean())

R["img_mean_avg"] = float(img_mean.mean())
R["img_mean_std"] = float(img_mean.std())
R["img_mean_min"] = float(img_mean.min())
R["img_mean_max"] = float(img_mean.max())

# Mean pixel value of the centre 14x14 block and of the outer 4-pixel frame
mean_grid = X.mean(axis = 0).reshape(28, 28)
R["center_mean"] = float(mean_grid[7:21, 7:21].mean())
R["border_mean"] = float(np.concatenate([
    mean_grid[:4].ravel(),  mean_grid[-4:].ravel(),
    mean_grid[:, :4].ravel(), mean_grid[:, -4:].ravel()
]).mean())

#--------------------------------------------------
# 11. Task 4 - Class (Target) Summary
#--------------------------------------------------
df_classes = pd.DataFrame({
    "label":    y,
    "letter":   [letter(v) for v in y],
    "img_mean": img_mean,
})

class_table = df_classes.groupby(["label", "letter"])["img_mean"].agg(
    count           = "size",
    mean_brightness = "mean",
    std_brightness  = "std",
).reset_index()

class_table["percent"] = class_table["count"] / len(y) * 100
class_table.round(2).to_csv(os.path.join(TABLE_DIR, "class_summary.csv"), index = False)

print("\nClass summary:")
print(class_table.round(2))

counts     = class_table.set_index("letter")["count"]
brightness = class_table.set_index("letter")["mean_brightness"]

R["class_max_letter"]       = counts.idxmax()
R["class_max_count"]        = int(counts.max())
R["class_min_letter"]       = counts.idxmin()
R["class_min_count"]        = int(counts.min())
R["imbalance_ratio"]        = float(counts.max() / counts.min())
R["bright_max_letter"]      = brightness.idxmax()
R["bright_max"]             = float(brightness.max())
R["bright_min_letter"]      = brightness.idxmin()
R["bright_min"]             = float(brightness.min())
R["avg_within_class_std"]   = float(class_table["std_brightness"].mean())

#--------------------------------------------------
# 12. Task 4 - Compare Train and Test Classes
#--------------------------------------------------
share_train = pd.Series(y).value_counts(normalize = True).sort_index() * 100
share_test  = pd.Series(y_test).value_counts(normalize = True).sort_index() * 100
difference  = (share_test.reindex(share_train.index, fill_value = 0) - share_train).abs()

R["max_prop_diff_pp"]     = float(difference.max())
R["max_prop_diff_letter"] = letter(difference.idxmax())

#--------------------------------------------------
# 13. Average Image of Each Class and Similar Letters
#--------------------------------------------------
class_means = np.stack([X[y == c].mean(axis = 0) for c in labels_present])

distance = np.sqrt(((class_means[:, None, :] - class_means[None, :, :]) ** 2)
                   .sum(axis = -1))

rows, cols   = np.triu_indices(len(labels_present), k = 1)
pair_distances = distance[rows, cols]
closest        = np.argsort(pair_distances)[:5]

R["closest_pairs"] = [
    [letter(labels_present[rows[i]]), letter(labels_present[cols[i]]),
     float(pair_distances[i])]
    for i in closest
]
R["median_pair_dist"] = float(np.median(pair_distances))

#--------------------------------------------------
# 14. Pixel Correlation and PCA
#--------------------------------------------------
rng    = np.random.default_rng(42)
sample = X[rng.choice(len(X), size = min(5000, len(X)), replace = False)]

corr = np.nan_to_num(np.corrcoef(sample, rowvar = False))

index  = np.arange(784)
right  = index[index % 28 != 27]
bottom = index[index < 756]
neighbour_corr = np.concatenate([corr[right, right + 1], corr[bottom, bottom + 28]])

R["corr_neighbor"] = float(neighbour_corr.mean())
R["corr_avg_abs"]  = float((np.abs(corr).sum() - 784) / (784 * 783))

centered        = sample - sample.mean(axis = 0)
singular_values = np.linalg.svd(centered, compute_uv = False)
cumulative      = np.cumsum(singular_values ** 2) / np.sum(singular_values ** 2)

R["pca_90"]    = int(np.argmax(cumulative >= 0.90) + 1)
R["pca_95"]    = int(np.argmax(cumulative >= 0.95) + 1)
R["pca_first"] = float(cumulative[0] * 100)

#--------------------------------------------------
# 15. Chart 1 - Bar Chart: Class Distribution
#--------------------------------------------------
letters_order = [letter(c) for c in labels_present]

chart_df = pd.concat([
    pd.DataFrame({"letter": [letter(v) for v in y],      "set": "Train"}),
    pd.DataFrame({"letter": [letter(v) for v in y_test], "set": "Test"}),
])

plt.figure(figsize = (11, 4.5))
sns.countplot(data = chart_df, x = "letter", hue = "set",
              order = letters_order, palette = ["#2b6cb0", "#ed8936"])
plt.title("Class Distribution of ASL Letters (Train vs Test)")
plt.xlabel("ASL letter (target class)")
plt.ylabel("Number of images")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "01_class_distribution.png"), dpi = 150)
plt.show()

#--------------------------------------------------
# 16. Chart 2 - Histogram: Pixel Intensity
#--------------------------------------------------
random_rows = np.random.default_rng(1).choice(len(X), size = min(3000, len(X)),
                                              replace = False)

plt.figure(figsize = (8, 4.5))
plt.hist(X[random_rows].ravel(), bins = 64, color = "#2b6cb0")
plt.yscale("log")
plt.title("Distribution of Pixel Intensities (3,000 random training images)")
plt.xlabel("Pixel intensity (0 = black, 255 = white)")
plt.ylabel("Pixel count (log scale)")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "02_pixel_intensity_histogram.png"), dpi = 150)
plt.show()

#--------------------------------------------------
# 17. Chart 3 - Example Image of Each Letter
#--------------------------------------------------
fig, axes = plt.subplots(4, 6, figsize = (9, 6.5))
axes = axes.ravel()
for i in range(len(axes)):
    axes[i].axis("off")
for i, c in enumerate(labels_present):
    first_image = X[np.where(y == c)[0][0]]
    axes[i].imshow(first_image.reshape(28, 28), cmap = "gray", vmin = 0, vmax = 255)
    axes[i].set_title(letter(c))
fig.suptitle("Example Training Image for Each ASL Letter (28x28 grayscale)")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "03_sample_images.png"), dpi = 150)
plt.show()

#--------------------------------------------------
# 18. Chart 4 - Average Image of Each Letter
#--------------------------------------------------
fig, axes = plt.subplots(4, 6, figsize = (9, 6.5))
axes = axes.ravel()
for i in range(len(axes)):
    axes[i].axis("off")
for i, c in enumerate(labels_present):
    axes[i].imshow(class_means[i].reshape(28, 28), cmap = "viridis")
    axes[i].set_title(letter(c))
fig.suptitle("Average Image per ASL Letter")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "04_class_mean_images.png"), dpi = 150)
plt.show()

#--------------------------------------------------
# 19. Chart 5 - Box Plot: Brightness by Letter
#--------------------------------------------------
plt.figure(figsize = (11, 4.5))
sns.boxplot(data = df_classes, x = "letter", y = "img_mean",
            order = letters_order, color = "#90cdf4", fliersize = 2)
plt.title("Mean Pixel Brightness per Image, by ASL Letter")
plt.xlabel("ASL letter (target class)")
plt.ylabel("Mean pixel intensity of image (0-255)")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "05_brightness_boxplot.png"), dpi = 150)
plt.show()

#--------------------------------------------------
# 20. Chart 6 - Mean and Standard Deviation of Each Pixel
#--------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize = (9, 4.2))

picture = axes[0].imshow(X.mean(axis = 0).reshape(28, 28), cmap = "magma")
axes[0].set_title("Mean intensity per pixel")
fig.colorbar(picture, ax = axes[0], fraction = 0.046)

picture = axes[1].imshow(X.std(axis = 0).reshape(28, 28), cmap = "magma")
axes[1].set_title("Std. deviation per pixel")
fig.colorbar(picture, ax = axes[1], fraction = 0.046)

for ax in axes:
    ax.set_xlabel("Pixel column (0-27)")
    ax.set_ylabel("Pixel row (0-27)")

fig.suptitle("Where the Information Is: Pixel-wise Mean and Variation")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "06_pixel_mean_std_heatmap.png"), dpi = 150)
plt.show()

#--------------------------------------------------
# 21. Chart 7 - Correlation Heatmap
#--------------------------------------------------
plt.figure(figsize = (7, 6))
sns.heatmap(corr, cmap = "coolwarm", center = 0, vmin = -1, vmax = 1,
            xticklabels = False, yticklabels = False,
            cbar_kws = {"label": "Pearson correlation"})
plt.title("Correlation Between Pixel Positions (pixel1 ... pixel784)")
plt.xlabel("Pixel index (1-784, row by row)")
plt.ylabel("Pixel index (1-784, row by row)")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "07_pixel_correlation_heatmap.png"), dpi = 150)
plt.show()

#--------------------------------------------------
# 22. Chart 8 - PCA Explained Variance
#--------------------------------------------------
plt.figure(figsize = (7.5, 4.5))
plt.plot(np.arange(1, len(cumulative) + 1), cumulative * 100, color = "#2b6cb0")
plt.axhline(90, color = "gray", linestyle = "--", linewidth = 1)
plt.axvline(R["pca_90"], color = "gray", linestyle = "--", linewidth = 1)
plt.xlim(0, 200)
plt.title("Cumulative Variance Explained by Principal Components")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance (%)")
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "08_pca_explained_variance.png"), dpi = 150)
plt.show()

#--------------------------------------------------
# 23. Save the Results
#--------------------------------------------------
with open("results.json", "w") as f:
    json.dump(R, f, indent = 2, default = str)

print("\nEDA finished. Charts in charts/, tables in tables/.")
print("Results in results.json")