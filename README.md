# Python for Computational/Data Science Project 1


## DataSets

Phil Full Data: [https://drive.google.com/file/d/1XDPFrq7hWpuRYZrXpVWKhITfJSaz1mor/view?usp=sharing](https://drive.google.com/file/d/19pp2iGTpdYgNkf5parsd-_VGYE04UX6t/view?usp=sharing)

Phil Processed Data: https://drive.google.com/file/d/1Jmw_yF5x9ZhNPyR3kJweoEaY6bKgr51R/view?usp=sharing

Yelp Original Data: https://drive.google.com/drive/folders/1EbdIz1xcHyiarcGOSUk4pTke0hS8V1on?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

Yelp Processed Data: https://drive.google.com/drive/folders/1Q28XuCxloQZFX28V3Wf4j5AF9Mh0oVNz?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

Match Phil and Business: https://drive.google.com/file/d/1GCQDPpOqvTv1Qk6Wxh8VCgaVmgWdE-ix/view?usp=sharing




## Executing the Files

### Phase 1:
1. Create `data` Folder with imbedded folders `processed` & `raw`

2. Install Required Libraries

3. Download the Philadelphia Full Data
    - Download data from "Phil Full Data". 
    - Store in `raw` folder with the name `philly_311_2025`.

4. Download the Yelp Open Dataset (Business, Checkin, and Review JSON only)
    - Download all files under "Yelp Original Data".
    - Store files under `raw` folder with the original file names.
### Phase 2 (Data Cleaning):
1. Philadelphia 311 Cleaning:
    1. Terminal Command to clean data:
        - ```python clean_phi_311.py```
    - This will output the Phil Clean Data file in `processed` folder.
2. Yelp Cleaning Process:
    1. Intitial Clean:
        - ```python clean_yelp.py```
    2. Sentiment Analysis:
        - ```python yelp_text_cleaning.py```
    3. Phrase Processing/Complaint Filtering:
        - ```python yelp_heuristics.py```
    4. ML Category Training:
        - ```python yelp_training.py```
    5. Business Category Normalization:
        - ```business_normalization.py```

### Phase 3 (Complaint Matching)

#### Need all files from "Phil Processed Data" & "Yelp Processed Data" moved into `processed` subfolder

1. Data Matching:
    - ```python match_311_yelp.py```

### Phase 4 (Analysis & findings)
#### Make sure to have completed Phase 3 and that the file  `311_yelp_matches_full.csv` is in the processed folder.

1. Analysis & Insights:
    - ```python p4hotspots.py```
    - This will output statistics in terminal as well as graphs for the analysis.
