# Congestion Pricing and NYC Drivers

MAST30034 Applied Data Science, Project 1.

On 5 January 2025 New York started charging a toll to enter Manhattan below 60th Street. This project measures how the toll changed high volume for-hire vehicle (Uber/Lyft) trips and driver earnings, and turns the results into advice on where and when drivers should work.

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
│   ├── models.py           Difference-in-differences and LightGBM models (Sections 6 and 7)
│   ├── plots.py            Plotting helpers
│   └── style.mplstyle      The shared style every figure uses
├── data/                   Created by the notebook on the first run, not committed
│   ├── raw/                Downloaded TLC and external data
│   └── curated/            Cleaned and summarised data
├── plots/                  Figures saved by the notebook
├── report/                 LaTeX report, built separately from the notebook
│   ├── main.tex
│   ├── references.bib
│   └── main.pdf            The built report (kept; the other build files are not)
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
| 5 | Explore the data and make maps: distributions before and after cleaning, what each rule removes per trip group, monthly trips relative to control, maps of change by zone and a scatter of the two changes together, change by time band and day, Uber/Lyft pickup waits, and what an engaged hour buys (trip speed, distance and pay, against the MTA's CBD speeds). The report's Figures 1 to 3 are saved here to `plots/` as PDF: `fig2_zone_map`, `fig3_zone_scatter` and `fig4_band_day`. `fig1_relative_trips` is an exploratory figure and stays in the notebook. |
| 6 | Fit the models: difference-in-differences (headline, event study, placebo, trend and buffer-band checks, time bands, and the headline in dollars an hour) and a LightGBM forecast of the toll period (validation against a 52-week baseline, a placebo read of the no-toll validation year, SHAP), then compare the two. The event study is saved as `fig5_event_study` and stays in the notebook; the model table is the report's Table 2. |
| 7 | Recommendations: read Section 6.7's forecast by pickup zone and time band, so a driver can compare a zone and shift against the year the toll never happened in. Saves `fig6_driver_heatmap`, the report's Figure 4. |

Sections 1–4 write `data/curated/zone_labels.csv`, `trip_summary.parquet` (5.9M rows) and `model_table.parquet` (the summary table with weather and holidays joined on), plus one parquet file per month in `summary_parts/`. Section 7 writes `driver_heatmap.csv`, the zone-by-time-band table the report's recommendations rest on. Deleting a file in `data/curated/` makes the next run rebuild it.

The external data is prepared before the taxi data is cleaned, because the cleaning step needs the zone labels. The report is built separately from `report/main.tex`.

## Code style

```bash
flake8 scripts/
nbqa flake8 main.ipynb
```
