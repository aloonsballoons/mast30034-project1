"""Prepare the external datasets and join them onto the trip summary.

The zone labels (``cbd`` / ``ring`` / ``control``) are built in
``scripts/zones.py``, because the cleaning step in Step 3 already needs
them. This module covers the rest of the external data:

1. **Subway ridership** (``subway_by_zone``): each station complex is
   matched to the taxi zone it stands in, and ridership is totalled per
   zone, date and time band.
2. **Weather** (``load_weather``): the NOAA Central Park daily file, with
   missing snow values set to 0.
3. **Public holidays** (``HOLIDAYS``, ``add_holidays``): the US federal
   holidays of 2023-2025, plus the weekday they were observed on.
4. **The joins** (``join_external``): weather, holidays and subway
   ridership onto the summary table, recording rows x columns after each
   one.
"""

import geopandas as gpd
import pandas as pd

from scripts import config
from scripts.clean import TIME_BANDS, TIME_BAND_ORDER
from scripts.zones import load_zone_shapes

SUBWAY_DIR = config.RAW_DIR / "mta" / "subway"
WEATHER_FILE = config.RAW_DIR / "noaa" / "central_park_daily.csv"
SUBWAY_ZONE_FILE = config.CURATED_DIR / "subway_by_zone.parquet"
STATION_ZONE_FILE = config.CURATED_DIR / "subway_station_zones.csv"
# The summary table with the external data joined on: the table the models
# in Step 6 are fitted to
MODEL_TABLE_FILE = config.CURATED_DIR / "model_table.parquet"

# A station complex is matched to a zone it does not stand in only if it is
# within this distance of one (about 500 m), which covers stations whose
# coordinates fall just outside the zone polygons, such as those on piers.
MAX_STATION_DISTANCE_FT = 1640

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
# Time bands
# ---------------------------------------------------------------------------
def time_band(hours):
    """Return the time band of each pickup hour.

    The bands are the ones in ``clean.TIME_BANDS``, so the external data is
    grouped exactly like the trip data.

    Args:
        hours (pandas.Series): Hour of day (0-23).

    Returns:
        pandas.Categorical: One of ``clean.TIME_BAND_ORDER`` per row.
    """
    band = pd.Series(pd.NA, index=hours.index, dtype="object")
    for name, start, end in TIME_BANDS:
        if start > end:  # the band wraps past midnight
            inside = (hours >= start) | (hours < end)
        else:
            inside = (hours >= start) & (hours < end)
        band = band.mask(inside, name)
    assert band.notna().all(), "an hour fell outside every time band"
    return pd.Categorical(band, TIME_BAND_ORDER, ordered=True)


# ---------------------------------------------------------------------------
# 1. Subway ridership per taxi zone
# ---------------------------------------------------------------------------
def subway_files():
    """List the monthly subway files that were downloaded.

    Returns:
        list of pathlib.Path: The files, in month order.
    """
    return sorted(SUBWAY_DIR.glob("subway_hourly_*.parquet"))


def station_locations():
    """Give every station complex one location, its busiest coordinates.

    In 2023-2024 a complex can have more than one coordinate pair (one per
    platform, at most about 450 m apart), so the same complex and hour
    appears on several rows. The 2025 file gives each complex a single
    pair. Ridership is summed over the pairs and the busiest one is kept,
    so both datasets describe each complex in the same way.

    Returns:
        pandas.DataFrame: One row per complex with ``station_complex_id``,
        ``station_complex``, ``borough``, ``latitude``, ``longitude`` and
        ``riders`` (total ridership at the kept coordinates).
    """
    keys = ["station_complex_id", "station_complex", "borough",
            "latitude", "longitude"]
    parts = []
    for path in subway_files():
        month = pd.read_parquet(path, columns=keys + ["ridership"])
        parts.append(month.groupby(keys, as_index=False)["ridership"].sum())
    totals = (pd.concat(parts, ignore_index=True)
              .groupby(keys, as_index=False)["ridership"].sum()
              .rename(columns={"ridership": "riders"}))
    busiest = (totals.sort_values("riders", ascending=False)
               .drop_duplicates("station_complex_id")
               .sort_values("station_complex_id")
               .reset_index(drop=True))
    return busiest


def station_zones(stations=None, overwrite=False):
    """Match every station complex to the taxi zone it stands in.

    The coordinates are latitude and longitude (EPSG:4326) and the zone
    shapefile is in NY State Plane feet (EPSG:2263), so the points are
    reprojected before the join. Some zone polygons overlap along their
    edges, so a point can land in more than one; the nearest zone centre
    wins, and the number of such stations is reported in ``matched_by``.
    Stations whose point falls outside every zone (piers, and stations just
    over a zone edge) are matched to the nearest zone within
    ``MAX_STATION_DISTANCE_FT``.

    The table is saved to ``data/curated/subway_station_zones.csv`` and
    read back on the next run, because working out each complex's busiest
    coordinates means reading every monthly file.

    Args:
        stations (pandas.DataFrame, optional): Output of
            ``station_locations``. Read from the files if not given.
        overwrite (bool): Match the stations again even if the saved file
            exists (needed after changing the zone shapes).

    Returns:
        pandas.DataFrame: One row per complex with its columns plus
        ``LocationID``, ``zone``, ``borough_zone`` and ``matched_by``
        (``inside`` or ``nearest``).
    """
    if STATION_ZONE_FILE.exists() and not overwrite and stations is None:
        return pd.read_csv(STATION_ZONE_FILE, dtype={
            "station_complex_id": "string"})
    stations = station_locations() if stations is None else stations
    points = gpd.GeoDataFrame(
        stations.copy(),
        geometry=gpd.points_from_xy(stations["longitude"],
                                    stations["latitude"]),
        crs="EPSG:4326").to_crs(epsg=2263)

    shapes = load_zone_shapes().rename(columns={"borough": "borough_zone"})
    inside = points.sjoin(shapes, predicate="within", how="left")
    # A point on a shared edge can match two zones: keep the nearest centre
    centres = shapes.set_index("LocationID").geometry.centroid
    inside["centre_distance"] = [
        point.distance(centres[zone]) if pd.notna(zone) else float("inf")
        for point, zone in zip(inside.geometry, inside["LocationID"])]
    inside = (inside.sort_values("centre_distance")
              .drop_duplicates("station_complex_id")
              .sort_index())
    inside["matched_by"] = "inside"

    missing = inside["LocationID"].isna()
    if missing.any():
        nearest = points[missing.to_numpy()].sjoin_nearest(
            shapes, how="left", max_distance=MAX_STATION_DISTANCE_FT,
            distance_col="zone_distance_ft")
        nearest = nearest.drop_duplicates("station_complex_id")
        nearest["matched_by"] = "nearest"
        inside = pd.concat([inside[~missing], nearest])

    matched = inside.drop(columns=["geometry", "index_right",
                                   "centre_distance"], errors="ignore")
    matched = matched.sort_values("station_complex_id").reset_index(drop=True)
    assert matched["station_complex_id"].is_unique
    assert matched["LocationID"].notna().all(), "a station matched no zone"
    matched["LocationID"] = matched["LocationID"].astype("int64")
    matched.to_csv(STATION_ZONE_FILE, index=False)
    return matched


def subway_by_zone(overwrite=False):
    """Total subway ridership per taxi zone, date and time band.

    Each month is read on its own and grouped straight away, so the 11.2
    million hourly rows never sit in memory at once. Ridership is summed
    over the coordinate pairs of a complex first (see
    ``station_locations``), which makes 2023-2024 comparable with 2025.
    The result is saved to ``data/curated/subway_by_zone.parquet``.

    Args:
        overwrite (bool): Rebuild the table even if the file exists.

    Returns:
        pandas.DataFrame: ``LocationID``, ``date``, ``time_band``,
        ``subway_riders`` and ``subway_transfers``.
    """
    months = {path.stem[-7:] for path in subway_files()}
    if SUBWAY_ZONE_FILE.exists() and not overwrite:
        cached = pd.read_parquet(SUBWAY_ZONE_FILE)
        # A quick run downloads only four months of subway data, so the
        # saved table can cover fewer months than the files on disk. It is
        # rebuilt rather than reused, which would leave the missing months
        # joined to 0 riders without saying so.
        if months <= set(cached["date"].dt.strftime("%Y-%m")):
            return cached

    zones = station_zones().set_index("station_complex_id")["LocationID"]
    keys = ["LocationID", "date", "time_band"]
    parts = []
    for path in subway_files():
        month = pd.read_parquet(path, columns=[
            "station_complex_id", "transit_timestamp", "ridership",
            "transfers"])
        month["LocationID"] = month["station_complex_id"].map(zones)
        assert month["LocationID"].notna().all(), f"unknown station in {path}"
        month["date"] = month["transit_timestamp"].dt.normalize()
        month["time_band"] = time_band(month["transit_timestamp"].dt.hour)
        parts.append(month.groupby(keys, as_index=False, observed=True)
                     [["ridership", "transfers"]].sum())

    table = (pd.concat(parts, ignore_index=True)
             .groupby(keys, as_index=False, observed=True)
             [["ridership", "transfers"]].sum()
             .rename(columns={"ridership": "subway_riders",
                              "transfers": "subway_transfers"}))
    table["LocationID"] = table["LocationID"].astype("int64")
    table = table.sort_values(keys).reset_index(drop=True)
    table.to_parquet(SUBWAY_ZONE_FILE, index=False)
    return table


# ---------------------------------------------------------------------------
# 2. Weather
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
# 3. Public holidays
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
# 4. Join everything onto the summary table
# ---------------------------------------------------------------------------
def join_external(summary, subway=None, weather=None):
    """Join weather, holidays and subway ridership onto the summary table.

    The subway table is joined on the **pickup zone**, date and time band,
    so it says how busy the subway was where and when the trip started.
    Zones with no station get 0 riders. Weather is citywide and daily, so
    it joins on the date alone.

    Args:
        summary (pandas.DataFrame): Output of ``clean.build_summary``.
        subway (pandas.DataFrame, optional): Output of ``subway_by_zone``.
        weather (pandas.DataFrame, optional): Output of ``load_weather``.

    The result is saved to ``data/curated/model_table.parquet``, which is
    the table the models in Step 6 are fitted to.

    Returns:
        tuple: (pandas.DataFrame, pandas.DataFrame) - the joined table and
        a table of rows x columns after each join.
    """
    subway = subway_by_zone() if subway is None else subway
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

    table = table.merge(
        subway.rename(columns={"LocationID": "PULocationID"}),
        on=["PULocationID", "date", "time_band"], how="left")
    for column in ["subway_riders", "subway_transfers"]:
        table[column] = table[column].fillna(0.0)
    shapes.append({"step": "+ subway riders (pickup zone, date, band)",
                   "rows": len(table), "columns": table.shape[1]})

    assert len(table) == len(summary), "a join changed the number of rows"
    table.to_parquet(MODEL_TABLE_FILE, index=False)
    return table, pd.DataFrame(shapes)
