# Data folder

Put the downloaded datasets here exactly like this (unzip inside each folder):

data/raw/GasHisSDB/          <- GasHisSDB (https://gitee.com/neuhwm/GasHisSDB)
    160/Abnormal/*.png
    160/Normal/*.png
    (120/ and 80/ sub-folders can stay; the pipeline uses 160 by default)

data/raw/HMU-GC-HE-30K/      <- HMU-GC-HE-30K (figshare link in the Sci Data paper, DOI 10.1038/s41597-025-04489-9)
    ADI/ DEB/ LYM/ MUC/ MUS/ NOR/ STR/ TUM/   (8 class folders of 224x224 patches)

data/raw/demo_slides/        <- 1-3 whole-slide images (.svs) for testing the tool, e.g. from TCGA-STAD (GDC portal)

data/index/                  <- created automatically by the pipeline (CSV lists of images + splits). Do not edit by hand.

Raw data is never modified by the code.
