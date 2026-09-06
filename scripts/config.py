"""Shared settings for the pipeline in ``main.ipynb``.

The notebook sits in the repository root, so it imports this module with::

    from scripts import config

Modules in ``scripts/`` import it the same way.

To process only a few months, so the whole notebook can be checked in
minutes instead of hours, set ``QUICK_RUN=1`` in the environment before
starting Jupyter::

    QUICK_RUN=1 jupyter lab main.ipynb
"""

import os
import sys
from datetime import date
from pathlib import Path

# Make Spark's worker processes use the same Python as the notebook.
# Otherwise they may pick up another Python on the PATH (such as
# Anaconda's), and Spark stops with a PYTHON_VERSION_MISMATCH error.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

# ---------------------------------------------------------------------------
# Folders (built from this file's location, so they work from any directory)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
CURATED_DIR = ROOT_DIR / "data" / "curated"
PLOTS_DIR = ROOT_DIR / "plots"
STYLE_FILE = ROOT_DIR / "scripts" / "style.mplstyle"

for folder in (RAW_DIR, CURATED_DIR, PLOTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Timeline: 24 months before the toll and 12 months after
# ---------------------------------------------------------------------------
START_MONTH = (2023, 1)
END_MONTH = (2025, 12)

# The congestion toll started on 5 January 2025
TOLL_START_DATE = date(2025, 1, 5)

# Fake toll date for the placebo test on 2023-2024 data
PLACEBO_DATE = date(2024, 1, 5)

# TLC services used in the project, keyed by the prefix of their file names
SERVICES = ["yellow", "fhvhv"]

# ---------------------------------------------------------------------------
# Quick run
# ---------------------------------------------------------------------------
# The same two calendar months before and after the toll, so a quick run can
# still compare like with like
QUICK_RUN_MONTHS = [(2024, 1), (2024, 3), (2025, 1), (2025, 3)]


def quick_run():
    """Return whether the QUICK_RUN environment variable is on.

    It is read on every call rather than stored in this module, so the only
    way to turn a quick run on is to set QUICK_RUN=1 (or true/yes, in any
    case) in the environment before starting Jupyter.

    Returns:
        bool: True for a quick run, False (the default) for the full run.
    """
    return os.environ.get("QUICK_RUN", "0").lower() in {"1", "true", "yes"}


def all_months():
    """Return every (year, month) pair from START_MONTH to END_MONTH.

    Returns:
        list of tuple: (year, month) pairs in order, e.g. [(2023, 1), ...].
    """
    months = []
    year, month = START_MONTH
    while (year, month) <= END_MONTH:
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def months_to_process():
    """Return the months the pipeline should use.

    Returns:
        list of tuple: QUICK_RUN_MONTHS when QUICK_RUN is on in the
        environment, otherwise every month in the timeline.
    """
    return QUICK_RUN_MONTHS if quick_run() else all_months()
