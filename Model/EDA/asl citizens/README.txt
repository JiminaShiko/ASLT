ASL CITIZEN - ASSIGNMENT 2 (EDA)
====================================
Exploratory Data Analysis of the ASL Citizen word-level video dataset.

DATASET
=======
Name    : ASL Citizen
Source  : https://www.microsoft.com/en-us/research/project/asl-citizen/
Content : ~84,000 isolated ASL word videos, ~2,700 glosses, 52 signers
Size    : ~30 GB ZIP
License : See the Dataset License page on the project site.
Paper   : Desai et al., "ASL Citizen: A Community-Sourced Dataset for
          Advancing Isolated Sign Language Recognition",
          arXiv:2304.05934, 2023. (must be cited if used)

FILES IN THIS FOLDER
====================
eda_asl_citizen_full.py   EDA script (Tasks 1-6 of the assignment)
Dataset_Source.txt        Dataset download link, paper, license
README.txt                This file

OUTPUT (created when you run the script)
========================================
charts_asl_citizen/       7 charts (PNG)
results_asl_citizen.json  All numerical results

IMPORTANT - BEFORE YOU RUN
==========================
The dataset is ~30 GB. The script stores it on the D: drive during the
download and extraction, then deletes it automatically at the end.

Open eda_asl_citizen_full.py and confirm this line:

    BIG_DATA_DIR = r"D:\ASL_Citizen_workspace"

Change the path if you want a different folder on D:.
The folder must have ~70 GB free (30 GB ZIP + 30 GB extracted + buffer).
The script performs a disk-space pre-flight check and refuses to run
if there is not enough space.

HOW TO RUN
==========
1. Install packages (the merged requirements.txt is at the repository root):
     cd ../../..
     python -m pip install -r requirements.txt
     cd "Model/EDA/asl citizens"

2. Run the EDA:
     python eda_asl_citizen_full.py

   Steps performed by the script:
     a. Checks free space on D:
     b. Downloads the ~30 GB ZIP from Microsoft (resumable)
     c. Extracts it on D:
     d. Deletes the ZIP (frees 30 GB)
     e. Reads the metadata CSVs (gloss, signer, split)
     f. Probes ALL ~84,000 videos for fps, duration, resolution
     g. Generates 7 charts
     h. Deletes the extracted folder (frees 30 GB)

3. When finished, charts_asl_citizen/ contains:
     01_top25_classes.png
     02_videos_per_signer.png
     03_class_longtail.png
     04_duration_histogram.png
     05_resolution_counts.png
     06_filesize_histogram.png
     07_split_distribution.png

RUNTIME
=======
Roughly 1.5 to 6 hours total depending on connection speed:
  Download 30 GB : 30 min (fast) to 4 h (slow)
  Extract        : 15 to 40 min
  Probe headers  : 30 to 60 min
  Charts + save  : under a minute

RESUME SUPPORT
==============
If the download is interrupted, just rerun the script. It sends an HTTP
Range header and continues from the exact byte where it stopped - it
does not restart from zero.

TASK COVERAGE
=============
  Task 1 - Load and inspect          :  yes
  Task 2 - Features and target       :  yes (printed summary)
  Task 3 - Data quality              :  yes (missing, duplicates,
                                             matched files, broken videos)
  Task 4 - Descriptive statistics    :  yes (class, signer, split, video
                                             properties over all videos)
  Task 5 - Visualizations            :  yes (7 charts)
  Task 6 - Interpretations           :  see EDA_Report_ASL_Citizen.pdf

MANUAL DOWNLOAD (if automatic fails)
====================================
1. Visit https://www.microsoft.com/en-us/research/project/asl-citizen/
2. Download the ZIP from the Microsoft Download Center.
3. Extract it so ASL_Citizen/ contains the CSVs and .mp4 files.
4. Open eda_asl_citizen_full.py and set:
     ZIP_PATH  = ""            (empty - no download)
     DATA_DIR  = "ASL_Citizen" (path to your extracted folder)
     CLEANUP   = False         (keep the data)
5. Run: python eda_asl_citizen_full.py

STORAGE
=======
Peak usage on D:    ~60 GB (ZIP + extracted, during extraction)
Permanent footprint : ~3 MB (charts + JSON) after the script finishes.