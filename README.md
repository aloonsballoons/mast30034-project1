# Congestion Pricing and NYC Drivers

MAST30034 Applied Data Science, Project 1.

On 5 January 2025 New York started charging a toll to enter Manhattan below 60th Street. This project measures how the toll changed yellow taxi and high volume for-hire vehicle (Uber/Lyft) trips and driver earnings, and turns the results into advice on where and when drivers should work.

Timeline: January 2023 to December 2025 (24 months before the toll, 12 months after).

## Repository layout

```
MAST30034-Project1/
├── main.ipynb              The whole pipeline, download to models. Run top to bottom.
├── scripts/                The functions main.ipynb imports, in pipeline order
│   ├── config.py           Shared settings: date range, folders, QUICK_RUN flag
│   ├── download.py         Downloads the TLC trip data and external datasets
│   ├── zones.py            Labels every taxi zone as cbd, ring or control (and a finer 5-level group)
│   ├── spark_io.py         Starts Spark, reads the monthly TLC files into one schema
│   ├── clean.py            Data checks, cleaning rules, new columns, summary table
│   ├── external.py         Weather, holidays, and the joins
│   ├── explore.py          Tables behind the exploration and maps (Section 5)
│   ├── plots.py            Plotting helpers
│   └── style.mplstyle      The shared style every figure uses
├── data/                   Created by the notebook on the first run, not committed
│   ├── raw/                Downloaded TLC and external data
│   └── curated/            Cleaned and summarised data
├── plots/                  Figures saved by the notebook
├── report/                 LaTeX report, built separately from the notebook
│   ├── main.tex
│   └── references.bib
└── requirements.txt        Pinned packages for the virtual environment
```

## Setup

### 1. Install the tools pip can't install

- **Python 3.12.** PySpark needs the same Python version in every process, and `shap` needs 3.12 or newer.
- **Java 17 or 21** for PySpark 4.2.

- **OpenMP for LightGBM (macOS only)**

  ```bash
  brew install libomp 
  ```
- **LaTeX** (only to build the report): a TeX distribution with `biblatex`, `biber`, `lipsum` and `mwe`: MacTeX on macOS, TeX Live on Linux, MiKTeX or TeX Live on Windows.

### 2. Create a virtual environment and install the packages

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the project

Start Jupyter from the repository root with the environment active, open `main.ipynb`, select `.venv` as the kernel, and run all cells:

```bash
jupyter lab main.ipynb
```

`scripts/config.py` sets `PYSPARK_PYTHON` to the notebook's Python, so Spark doesn't pick up another Python (such as Anaconda's) by mistake.

On an Apple silicon laptop with Java 21, Sections 1–4 (after the download) took about 35 minutes when every monthly summary part in `data/curated/summary_parts/` had to be built, and about 17 minutes when the saved parts were reused. Section 5 adds about 19 minutes, mostly the Spark passes over every Uber/Lyft trip for the zone medians and pickup waits, so a full run with saved parts takes about 37 minutes. A quick run takes about 5 minutes.

Spark runs in local mode with 2 GB of memory by default, which was enough for every step. To give it more, set `SPARK_DRIVER_MEMORY` (for example `export SPARK_DRIVER_MEMORY=6g`) before starting Jupyter. Spark's scratch files go to `spark-tmp/` in the repository root (not committed).

### Quick run

Processing all 36 months takes a while, on top of the first download. To check that the notebook runs end to end on a small slice of the data instead, set `QUICK_RUN` in the environment when starting Jupyter:

```bash
QUICK_RUN=1 jupyter lab main.ipynb
```

The notebook then uses only January and March of 2024 and 2025. `1`, `true` and `yes` are all accepted, in any case. The first cell prints which mode is on. Results in the report come from the full run.

### What the notebook does

| Section | What it does |
|---------|--------------|
| 1 | Download the TLC trip data and external datasets into `data/raw/`. Finished files are skipped on later runs. |
| 2 | External data: zone labels, the buffer ring, weather and holidays |
| 3 | Clean the taxi data with PySpark and build the summary table |
| 4 | Join weather and holidays onto the summary table. It comes after Section 3 because it needs the summary table. |
| 5 | Explore the data and make maps: distributions before and after cleaning, what each rule removes per trip group, weekly trips, maps of change by zone, change by time band and day, MTA speeds and Uber/Lyft pickup waits. Report figures are saved to `plots/` as PDF. |
| 6 | Fit the models: difference-in-differences and LightGBM |

Sections 1–4 write `data/curated/zone_labels.csv`, `trip_summary.parquet` (11.5M rows) and `model_table.parquet` (the summary table with weather and holidays joined on), plus one parquet file per service and month in `summary_parts/`. Deleting a file in `data/curated/` makes the next run rebuild it.

The external data is prepared before the taxi data is cleaned, because the cleaning step needs the zone labels. The report is built separately from `report/main.tex`.

## Code style

```bash
flake8 scripts/
nbqa flake8 main.ipynb
```
