DataSets:

Phil Full Data: https://drive.google.com/file/d/19pp2iGTpdYgNkf5parsd-_VGYE04UX6t/view?usp=sharing

Phil Clean Data: https://drive.google.com/file/d/17CWmqAAFwgRh858azMmpOgti0rTy2XVk/view?usp=sharing

Yelp Original Data: https://drive.google.com/drive/folders/1EbdIz1xcHyiarcGOSUk4pTke0hS8V1on?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

yelp cleaned data: https://drive.google.com/drive/folders/1Q28XuCxloQZFX28V3Wf4j5AF9Mh0oVNz?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto


Match Phil and Business: https://drive.google.com/file/d/1GCQDPpOqvTv1Qk6Wxh8VCgaVmgWdE-ix/view?usp=sharing




Instructions on how to run each script and which file goes with each phase:

1. phase 1, data cleaning for 311 Dataset:

- Download the Phil Full Data from above. Put this file in data -> raw. Rename file to be philly_311_2025
- Go to your terminal and type: python clean_phi_311.py. You may need to import libraries if you have not yet imported them.
- This will output the Phil Clean Data file.

2. json files, ...

3. Phase 3 matching.

- Make sure you have both the Phil Clean Data files as well as the cleaned_business files in the processed folder.
- Run: python match_311_yelp.py. This will match the yelp and the 311.

4. Phase 4  Analysis & findings.
- Make sure have completed Phase 3 and file 311_yelp_matches_full is in the processed folder.
- Run: python p4hotspot.py.
- This will output data in terminal as well as graphs for the analysis.
