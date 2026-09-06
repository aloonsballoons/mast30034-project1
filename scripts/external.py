"""Prepare the external datasets and join them onto the trip summary.

The zone labels (``cbd`` / ``ring`` / ``control``) are built in
``scripts/zones.py``, because the cleaning step in Step 3 already needs
them. This module covers the rest of the external data:

1. **Weather** (``load_weather``): the NOAA Central Park daily file, with
   missing snow values set to 0.
2. **Public holidays** (``HOLIDAYS``, ``add_holidays``): the US federal
   holidays of 2023-2025, plus the weekday they were observed on.
3. **The joins** (``join_external``): weather and holidays onto the
   summary table, recording rows x columns after each one.
"""

import pandas as pd

from scripts import config

WEATHER_FILE = config.RAW_DIR / "noaa" / "central_park_daily.csv"
# The summary table with the external data joined on: the table the models
# in Step 6 are fitted to
MODEL_TABLE_FILE = config.CURATED_DIR / "model_table.parquet"

# US federal holidays, 2023-2025 (US Office of Personnel Management). Where
# the holiday falls at a weekend, the weekday it was observed on is listed
# as well, because that is the day people don't go to work.
HOLIDAYS = [
    # 2023
    "2023-01-01", "2023-01-02", "2023-01-16", "2023-02-20", "2023-05-29",
    "2023-06-19", "2023-07-04", "2023-09-04", "2023-10-09", "2023-11-10",
    "2023-11-11", "2023-11-23", "2023-12-25",
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19",
    "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-11", "2024-11-28",
    "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-11", "2025-11-27",
    "2025-12-25",
]


# ---------------------------------------------------------------------------
# 1. Weather
# ---------------------------------------------------------------------------
def load_weather():
    """Load the NOAA Central Park daily weather in metric units.

    The file was requested in metric units, so rainfall and snow are in
    millimetres and temperatures in degrees Celsius; no conversion is
    needed. A missing snow depth (``SNWD``) means no snow was lying, so it
    becomes 0.

    Returns:
        pandas.DataFrame: One row per date with ``date``, ``rain_mm``,
        ``snow_mm``, ``snow_depth_mm``, ``temp_max_c`` and ``temp_min_c``.
    """
    weather = pd.read_csv(WEATHER_FILE, parse_dates=["DATE"])
    weather = weather.rename(columns={
        "DATE": "date", "PRCP": "rain_mm", "SNOW": "snow_mm",
        "SNWD": "snow_depth_mm", "TMAX": "temp_max_c", "TMIN": "temp_min_c"})
    for column in ["snow_mm", "snow_depth_mm"]:
        weather[column] = weather[column].fillna(0.0)
    weather = weather.drop(columns="STATION")
    assert weather["date"].is_unique
    assert weather.notna().all().all(), "weather still has missing values"
    return weather.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Public holidays
# ---------------------------------------------------------------------------
def add_holidays(table, date_column="date"):
    """Add an ``is_holiday`` flag from the federal holiday list.

    Args:
        table (pandas.DataFrame): Any table with a date column.
        date_column (str): Name of that column.

    Returns:
        pandas.DataFrame: ``table`` with a boolean ``is_holiday`` column.
    """
    holidays = pd.to_datetime(pd.Series(HOLIDAYS))
    table = table.copy()
    table["is_holiday"] = (pd.to_datetime(table[date_column])
                           .isin(set(holidays)))
    return table


# ---------------------------------------------------------------------------
# 3. Join everything onto the summary table
# ---------------------------------------------------------------------------
def join_external(summary, weather=None):
    """Join weather and holidays onto the summary table.

    Weather is citywide and daily, so it joins on the date alone.

    Args:
        summary (pandas.DataFrame): Output of ``clean.build_summary``.
        weather (pandas.DataFrame, optional): Output of ``load_weather``.

    The result is saved to ``data/curated/model_table.parquet``, which is
    the table the models in Step 6 are fitted to.

    Returns:
        tuple: (pandas.DataFrame, pandas.DataFrame) - the joined table and
        a table of rows x columns after each join.
    """
    weather = load_weather() if weather is None else weather

    shapes = [{"step": "summary table (Step 3)", "rows": len(summary),
               "columns": summary.shape[1]}]
    table = summary.copy()
    table["date"] = pd.to_datetime(table["date"])

    table = table.merge(weather, on="date", how="left")
    shapes.append({"step": "+ daily weather (Central Park)",
                   "rows": len(table), "columns": table.shape[1]})
    assert table["rain_mm"].notna().all(), "a date has no weather"

    table = add_holidays(table)
    shapes.append({"step": "+ public holiday flag", "rows": len(table),
                   "columns": table.shape[1]})

    assert len(table) == len(summary), "a join changed the number of rows"
    table.to_parquet(MODEL_TABLE_FILE, index=False)
    return table, pd.DataFrame(shapes)
