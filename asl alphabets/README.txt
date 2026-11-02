ASL ALPHABET - ASSIGNMENT 2 (EDA)
=====================================
Exploratory Data Analysis of the ASL Alphabet dataset.

DATASET
=======
Name    : ASL Alphabet (grassknoted)
Source  : https://www.kaggle.com/datasets/grassknoted/asl-alphabet
Content : ~87,000 colour images, 200x200 pixels, 29 classes
          (A-Z, space, del, nothing)
Size    : ~1 GB
License : Follow the Kaggle dataset page.

FILES IN THIS FOLDER
====================
eda_asl_alphabet.py       EDA script (Tasks 1-6 of the assignment)
requirements.txt          Python packages needed
Dataset_Source.txt        Dataset download link and license
README.txt                This file

OUTPUT (created when you run the script)
========================================
charts_asl_alphabet/      4 charts (PNG)
results_asl_alphabet.json All numerical results

HOW TO RUN
==========
1. Install packages:
     pip install -r requirements.txt

2. Run the EDA:
     python eda_asl_alphabet.py

   The script downloads the dataset automatically (~1 GB). After the
   EDA finishes, the dataset is deleted (CLEANUP = True in the script)
   so only the charts and the JSON stay on disk.

3. When finished, charts_asl_alphabet/ contains:
     01_images_per_class.png
     02_sample_grid.png
     03_brightness_histogram.png
     04_filesize_by_class.png

TASK COVERAGE
=============
  Task 1 - Load and inspect          :  yes
  Task 2 - Features and target       :  yes (printed summary)
  Task 3 - Data quality              :  yes (missing, corrupt, sizes)
  Task 4 - Descriptive statistics    :  yes (brightness, class counts)
  Task 5 - Visualizations            :  yes (4 charts)
  Task 6 - Interpretations           :  see EDA_Report_ASL_Alphabet.pdf

MANUAL DOWNLOAD (if automatic fails)
====================================
1. Visit https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Download the ZIP and extract it so that asl_alphabet_train/ exists
   next to eda_asl_alphabet.py
3. Open eda_asl_alphabet.py, comment out the kagglehub lines, and set:
     TRAIN_DIR = "asl_alphabet_train"
4. Run: python eda_asl_alphabet.py

STORAGE
=======
Permanent footprint after running: ~2 MB (charts + JSON).
The ~1 GB of images is deleted by the script itself.