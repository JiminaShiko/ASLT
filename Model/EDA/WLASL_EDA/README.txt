WLASL - Word-Level American Sign Language EDA
=============================================

This package performs Exploratory Data Analysis on the WLASL dataset
using the metadata file: WLASL_v0.3.json

DATASET
=======
- Source: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
- File used: WLASL_v0.3.json
- Total words (glosses): 2,000
- Total video instances: ~21,083

PROJECT FILES
=============
eda_analysis.py     - Main EDA script (Assignment 2)
README.txt          - This file
WLASL_v0.3.json     - Dataset metadata

HOW TO RUN
==========
1. Install packages (the merged requirements.txt is at the repository root):
   cd ../../..
   python -m pip install -r requirements.txt
   cd "Model/EDA/WLASL_EDA"

2. Run EDA:
   python eda_analysis.py

The script will create:
  - charts/          (visualizations)
  - tables/          (summary CSV files)
  - results.json     (key statistics and findings)

NOTES
=====
- This EDA only uses the JSON metadata (no video files needed).
- It is suitable for Assignment 2 (Dataset Exploration).
- For the actual project, we will later select only 20-25 common words.
