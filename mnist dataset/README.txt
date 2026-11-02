SIGN LANGUAGE MNIST - ASSIGNMENT 2 (EDA)
===========================================
Exploratory Data Analysis of the Sign Language MNIST dataset.

DATASET
=======
Name    : Sign Language MNIST
Source  : https://www.kaggle.com/datasets/datamunge/sign-language-mnist
Content : 27,455 train + 7,172 test images, 28x28 grayscale,
          24 classes (A-Z except J and Z - those need motion).
Size    : ~30 MB
License : Follow the Kaggle dataset page.

FILES IN THIS FOLDER
====================
eda_analysis.py     EDA script (Tasks 1-6 of the assignment)
requirements.txt    Python packages needed
Dataset_Source.txt  Dataset download link and license
README.txt          This file

OUTPUT (created when you run the script)
========================================
charts/             8 charts (PNG)
tables/             3 summary tables (CSV)
results.json        All numerical results

HOW TO RUN
==========
1. Install packages:
     pip install -r requirements.txt

2. Run the EDA:
     python eda_analysis.py

   The script downloads the dataset automatically (~30 MB) through
   kagglehub into the dataset/ folder. If the CSVs are already there,
   the download step is skipped.

3. When finished, charts/ contains:
     01_class_distribution.png
     02_pixel_intensity_histogram.png
     03_sample_images.png
     04_class_mean_images.png
     05_brightness_boxplot.png
     06_pixel_mean_std_heatmap.png
     07_pixel_correlation_heatmap.png
     08_pca_explained_variance.png

   And tables/ contains:
     describe_train.csv
     class_summary.csv

TASK COVERAGE
=============
  Task 1 - Load and inspect          :  yes (head, shape, info, describe)
  Task 2 - Features and target       :  yes (printed summary)
  Task 3 - Data quality              :  yes (missing, duplicates,
                                             out-of-range, blank images,
                                             constant pixels, brightness
                                             outliers, J label unused)
  Task 4 - Descriptive statistics    :  yes (pixel stats, class summary,
                                             train/test comparison)
  Task 5 - Visualizations            :  yes (8 charts)
  Task 6 - Interpretations           :  see EDA_Report_SignLanguageMNIST.pdf

MANUAL DOWNLOAD (if automatic fails)
====================================
1. Visit https://www.kaggle.com/datasets/datamunge/sign-language-mnist
2. Download sign_mnist_train.csv and sign_mnist_test.csv
3. Put them in a folder called dataset/ next to this script
4. Run: python eda_analysis.py

STORAGE
=======
Permanent footprint after running: ~35 MB (CSVs + charts + tables + JSON).
The CSVs are kept because they are small and needed for the next stage
(model training in Assignment 3+).