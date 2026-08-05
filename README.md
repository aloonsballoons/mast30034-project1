# Congestion Pricing and NYC Drivers

MAST30034 Applied Data Science, Project 1.

On 5 January 2025 New York started charging a toll to enter Manhattan below 60th Street. This project measures how the toll changed yellow taxi and rideshare (Uber/Lyft) trips and driver earnings, and turns the results into advice on where and when drivers should work.

Timeline: January 2023 to December 2025 (24 months before the toll, 12 months after).

## Repository layout

| Folder / file | Contents |
|---------------|----------|
| `data/raw/` | Downloaded TLC and external data (not committed) |
| `data/curated/` | Cleaned and summarised data (not committed) |
| `main.ipynb` | The whole pipeline, from download to models. Run it top to bottom. |
| `scripts/` | Functions the notebook imports (downloading, cleaning, plotting, models) |
| `scripts/config.py` | Shared settings: date range, folders, `QUICK_RUN` flag |
| `scripts/style.mplstyle` | Shared figure style used by every figure |
| `plots/` | Figures saved by the notebook |
| `report/` | LaTeX report (`main.tex`, `references.bib`) |
| `external_datasets.md` | Sources and download notes for the external datasets |

## Setup

### 1. Install the tools pip can't install

- **Python 3.12.** PySpark needs the same Python version in every process, and `shap` needs 3.12 or newer.
- **Java 17 or 21** for PySpark 4.2. Check with `java -version`. If you have several Java versions, point PySpark at a supported one before running anything:

  ```bash
  export JAVA_HOME=$(/usr/libexec/java_home -v 21)   # macOS
  ```

  On macOS you can install it with `brew install openjdk@21`.
- **OpenMP for LightGBM (macOS only):**

  ```bash
  brew install libomp
  ```

- **LaTeX** (only to build the report): a TeX distribution with `biblatex`, `biber` and `lipsum`, for example MacTeX. With TinyTeX, run `tlmgr install lipsum biber`.

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

Processing all 36 months takes hours. To check that the notebook runs, set this in its first cell:

```python
config.QUICK_RUN = True
```

The notebook then uses only January and March of 2024 and 2025. Setting `QUICK_RUN=1` in the environment before starting Jupyter does the same. Results in the report come from the full run.

### What the notebook does

The notebook is still being built. Its sections follow `PROJECT_PLAN.md`, and run times will be added as each section is finished.

1. Download the TLC trip data and external datasets into `data/raw/` (Step 2)
2. Prepare the external data: CBD zone labels, buffer ring, subway, weather, holidays (Step 4)
3. Clean the taxi data with PySpark and build the summary table in `data/curated/` (Step 3)
4. Explore the data and make maps (Step 5)
5. Fit the models: difference-in-differences and LightGBM (Step 6)

Step 4 comes before Step 3 because the cleaning step needs the zone labels. The report is built separately from `report/main.tex`.

## Code style

```bash
flake8 scripts/
nbqa flake8 main.ipynb
```
