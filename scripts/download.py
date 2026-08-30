"""Download the TLC trip data and the external datasets into ``data/raw/``.

The notebook calls two functions::

    from scripts import download
    download.download_tlc(months)
    download.download_external(months)

Every download is skipped if its file already exists, so re-running the
notebook does not fetch the data again. Files are written to a ``.part`` file
first and renamed at the end, so a stopped download never leaves a file that
looks complete.

Sources and notes for each external dataset are in ``external_datasets.md``.
"""

import calendar
import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

from scripts import config

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
TLC_URL = ("https://d37ci6vzurychx.cloudfront.net/trip-data/"
           "{service}_tripdata_{year}-{month:02d}.parquet")
ZONE_LOOKUP_URL = ("https://d37ci6vzurychx.cloudfront.net/misc/"
                   "taxi_zone_lookup.csv")
ZONE_SHAPEFILE_URL = ("https://d37ci6vzurychx.cloudfront.net/misc/"
                      "taxi_zones.zip")

SOCRATA_URL = "https://data.ny.gov/resource/{dataset_id}.csv"
CBD_ZONES_ID = "yfdc-w5jh"
CBD_SPEEDS_ID = "6p29-6xqn"
# Subway hourly ridership is split into two datasets at the start of 2025
SUBWAY_IDS = {2023: "wujg-7c2s", 2024: "wujg-7c2s", 2025: "5wq4-mkjj"}

# NOAA daily summaries for Central Park, in metric units (mm and deg C)
WEATHER_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
WEATHER_PARAMS = {
    "dataset": "daily-summaries",
    "stations": "USW00094728",
    "dataTypes": "PRCP,SNOW,SNWD,TMAX,TMIN",
    "units": "metric",
    "format": "csv",
}

# data.ny.gov returns only 1,000 rows unless $limit is set
PAGE_SIZE = 50_000

# Parallel downloads. The TLC server and data.ny.gov both handle a few
# requests at a time without refusing them.
TLC_WORKERS = 4
SUBWAY_WORKERS = 4

RETRIES = 5
TIMEOUT = (30, 300)  # seconds to connect, seconds between bytes

# Optional data.ny.gov app token, which raises the request rate limit
SOCRATA_HEADERS = ({"X-App-Token": os.environ["SOCRATA_APP_TOKEN"]}
                   if os.environ.get("SOCRATA_APP_TOKEN") else {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _retry(action, label):
    """Run ``action()``, retrying with a growing wait if it fails.

    Args:
        action (callable): Function with no arguments that does one attempt.
        label (str): Name to print when retrying.

    Returns:
        The value ``action()`` returns.
    """
    for attempt in range(1, RETRIES + 1):
        try:
            return action()
        except (requests.RequestException, IOError) as error:
            if attempt == RETRIES:
                raise
            wait = 2 ** attempt
            print(f"  {label}: retry {attempt}/{RETRIES - 1} in {wait}s "
                  f"({error})")
            time.sleep(wait)


def _get(url, params=None, headers=None):
    """Send a GET request and read the whole body, retrying if it fails.

    Args:
        url (str): Address to request.
        params (dict, optional): Query parameters.
        headers (dict, optional): Request headers.

    Returns:
        requests.Response: The successful response.
    """
    def attempt():
        response = requests.get(url, params=params, headers=headers,
                                timeout=TIMEOUT)
        response.raise_for_status()
        return response
    return _retry(attempt, url)


def download_file(url, dest, params=None):
    """Download a file to ``dest``, skipping it if it already exists.

    The size is checked against the server's Content-Length when the server
    sends one. A dropped connection or short file is retried.

    Args:
        url (str): Address of the file.
        dest (Path): Where to save it.
        params (dict, optional): Query parameters.

    Returns:
        Path: ``dest``.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    def attempt():
        with requests.get(url, params=params, stream=True,
                          timeout=TIMEOUT) as response:
            response.raise_for_status()
            expected = response.headers.get("Content-Length")
            with open(part, "wb") as file:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    file.write(chunk)
        size = part.stat().st_size
        if expected is not None and size != int(expected):
            raise IOError(f"got {size:,} of {int(expected):,} bytes")
        part.rename(dest)
        return dest
    return _retry(attempt, dest.name)


def socrata_count(dataset_id, where=None, select=None, group=None):
    """Count the rows a data.ny.gov query returns, using the server.

    For a grouped query this counts the groups, by piping the grouped query
    into ``SELECT count(*)``.

    Args:
        dataset_id (str): data.ny.gov dataset ID, e.g. ``"yfdc-w5jh"``.
        where (str, optional): SoQL filter.
        select (str, optional): SoQL select list (needed with ``group``).
        group (str, optional): SoQL group-by list.

    Returns:
        int: Number of rows.
    """
    url = SOCRATA_URL.format(dataset_id=dataset_id).replace(".csv", ".json")
    if group:
        query = f"SELECT {select}"
        if where:
            query += f" WHERE {where}"
        query += f" GROUP BY {group} |> SELECT count(*) AS n"
        params = {"$query": query}
    else:
        params = {"$select": "count(*) AS n"}
        if where:
            params["$where"] = where
    return int(_get(url, params=params, headers=SOCRATA_HEADERS)
               .json()[0]["n"])


def socrata_download(dataset_id, order, select=None, where=None, group=None):
    """Download every row of a data.ny.gov query, one page at a time.

    The API returns only 1,000 rows by default, so this sets ``$limit``,
    sorts with ``$order`` so pages don't overlap or skip rows, and moves
    ``$offset`` on until a page comes back short. The row count is then
    checked against the server's own count.

    Args:
        dataset_id (str): data.ny.gov dataset ID.
        order (str): SoQL sort order. It must give every row a unique
            position, or pages can overlap.
        select (str, optional): SoQL select list (all columns if omitted).
        where (str, optional): SoQL filter.
        group (str, optional): SoQL group-by list.

    Returns:
        pandas.DataFrame: All rows, with every column read as text.
    """
    url = SOCRATA_URL.format(dataset_id=dataset_id)
    params = {"$order": order, "$limit": PAGE_SIZE}
    for key, value in (("$select", select), ("$where", where),
                       ("$group", group)):
        if value:
            params[key] = value

    pages = []
    offset = 0
    while True:
        params["$offset"] = offset
        text = _get(url, params=params, headers=SOCRATA_HEADERS).text
        page = pd.read_csv(io.StringIO(text), dtype=str,
                           keep_default_na=False, na_values=[""])
        pages.append(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    data = pd.concat(pages, ignore_index=True)
    expected = socrata_count(dataset_id, where=where, select=select,
                             group=group)
    if len(data) != expected:
        raise ValueError(f"{dataset_id}: downloaded {len(data):,} rows but "
                         f"the server has {expected:,}")
    return data


def _save_atomic(data, dest):
    """Save a DataFrame as CSV or parquet (by extension) via a .part file.

    Args:
        data (pandas.DataFrame): Table to save.
        dest (Path): Output file ending in ``.csv`` or ``.parquet``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if dest.suffix == ".parquet":
        data.to_parquet(part, index=False)
    else:
        data.to_csv(part, index=False)
    part.rename(dest)


# ---------------------------------------------------------------------------
# TLC trip data
# ---------------------------------------------------------------------------
def tlc_path(service, year, month):
    """Return the local path of one TLC monthly file.

    Args:
        service (str): ``"yellow"`` or ``"fhvhv"``.
        year (int): Year.
        month (int): Month (1-12).

    Returns:
        Path: e.g. ``data/raw/tlc/yellow/yellow_tripdata_2023-01.parquet``.
    """
    return (config.RAW_DIR / "tlc" / service
            / f"{service}_tripdata_{year}-{month:02d}.parquet")


def download_tlc(months, services=None):
    """Download the TLC monthly parquet files for each service and month.

    Args:
        months (list of tuple): (year, month) pairs, usually
            ``config.months_to_process()``.
        services (list of str, optional): Services to download. Defaults to
            ``config.SERVICES`` (yellow taxi and High Volume FHV).

    Returns:
        list of Path: The local files, in service then month order.
    """
    services = services or config.SERVICES
    jobs = [(service, year, month)
            for service in services for year, month in months]
    todo = [job for job in jobs if not tlc_path(*job).exists()]
    print(f"TLC: {len(jobs)} files, {len(jobs) - len(todo)} already "
          f"downloaded, {len(todo)} to download")

    def fetch(job):
        path = download_file(TLC_URL.format(service=job[0], year=job[1],
                                            month=job[2]), tlc_path(*job))
        print(f"  {path.name} ({path.stat().st_size / 1e6:,.0f} MB)")

    with ThreadPoolExecutor(TLC_WORKERS) as pool:
        list(pool.map(fetch, todo))
    return [tlc_path(*job) for job in jobs]


def split_large_row_groups(paths, rows_per_group=1_048_576,
                           max_group_mb=200):
    """Rewrite TLC files whose row groups are too big for Spark to split.

    HVFHV January to July 2023 each store the whole month (about 20 million
    rows, 820-935 MB uncompressed) in a single row group. Spark can't split
    a row group, so one task has to hold the whole file, and a few of those
    at once ran Spark out of memory. Later files use row groups of
    1,048,576 rows. This rewrites the large ones with the same row group
    size, the same schema and zstd compression, checks that the rows and
    schema match, and only then replaces the original.

    Args:
        paths (list of Path): TLC parquet files.
        rows_per_group (int): Rows per row group in the rewritten file.
        max_group_mb (float): Rewrite files with any row group larger than
            this (uncompressed).

    Returns:
        list of str: Names of the files that were rewritten.
    """
    rewritten = []
    for path in paths:
        meta = pq.read_metadata(path)
        largest = max(meta.row_group(i).total_byte_size
                      for i in range(meta.num_row_groups))
        if largest <= max_group_mb * 1e6:
            continue
        table = pq.read_table(path)
        part = path.with_name(path.name + ".part")
        pq.write_table(table, part, row_group_size=rows_per_group,
                       compression="zstd")
        new_meta = pq.read_metadata(part)
        if (new_meta.num_rows != meta.num_rows
                or not pq.read_schema(part).equals(table.schema)):
            part.unlink()
            raise ValueError(f"{path.name}: rewritten file doesn't match")
        del table
        part.replace(path)
        rewritten.append(path.name)
        print(f"  {path.name}: {meta.num_rows:,} rows, 1 row group -> "
              f"{new_meta.num_row_groups}")
    return rewritten


def parquet_shapes(paths):
    """Read rows x columns of parquet files from their metadata only.

    This is fast because it doesn't load the data.

    Args:
        paths (list of Path): Parquet files.

    Returns:
        pandas.DataFrame: One row per file with ``file``, ``rows``,
        ``columns`` and ``size_mb``.
    """
    records = []
    for path in paths:
        meta = pq.read_metadata(path)
        records.append({"file": Path(path).name, "rows": meta.num_rows,
                        "columns": meta.num_columns,
                        "size_mb": round(Path(path).stat().st_size / 1e6, 1)})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# External datasets
# ---------------------------------------------------------------------------
def download_taxi_zones():
    """Download the TLC taxi zone lookup table and zone shapefile.

    The shapefile parts are extracted from the zip (which has its own
    ``taxi_zones/`` folder inside) straight into ``data/raw/taxi_zones/``.

    Returns:
        dict: Paths of the lookup CSV and the ``.shp`` file.
    """
    folder = config.RAW_DIR / "taxi_zones"
    lookup = download_file(ZONE_LOOKUP_URL, folder / "taxi_zone_lookup.csv")
    archive = download_file(ZONE_SHAPEFILE_URL, folder / "taxi_zones.zip")
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.namelist():
            target = folder / Path(member).name
            if not member.endswith("/") and not target.exists():
                target.write_bytes(zipped.read(member))
    return {"lookup": lookup, "shapefile": folder / "taxi_zones.shp"}


def download_cbd_zones():
    """Download the MTA list of taxi zones inside the congestion zone.

    Returns:
        Path: ``data/raw/mta/cbd_taxi_zones.csv``, one row per zone with
        its polygon.
    """
    dest = config.RAW_DIR / "mta" / "cbd_taxi_zones.csv"
    if not dest.exists():
        data = socrata_download(CBD_ZONES_ID, order="taxi_zone")
        _save_atomic(data, dest)
    return dest


def download_cbd_speeds():
    """Download the MTA monthly taxi/FHV speeds in and around the zone.

    Returns:
        Path: ``data/raw/mta/cbd_taxi_fhv_speeds.csv``.
    """
    dest = config.RAW_DIR / "mta" / "cbd_taxi_fhv_speeds.csv"
    if not dest.exists():
        data = socrata_download(CBD_SPEEDS_ID, order="month, zone")
        _save_atomic(data, dest)
    return dest


def download_weather():
    """Download NOAA daily weather for Central Park over the timeline.

    Values are metric: precipitation, snowfall and snow depth in mm,
    temperatures in deg C.

    Returns:
        Path: ``data/raw/noaa/central_park_daily.csv``.
    """
    first = date(*config.START_MONTH, 1)
    last = date(*config.END_MONTH, calendar.monthrange(*config.END_MONTH)[1])
    params = dict(WEATHER_PARAMS, startDate=first.isoformat(),
                  endDate=last.isoformat())
    return download_file(WEATHER_URL,
                         config.RAW_DIR / "noaa" / "central_park_daily.csv",
                         params=params)


def subway_path(year, month):
    """Return the local path of one month of subway ridership.

    Args:
        year (int): Year.
        month (int): Month (1-12).

    Returns:
        Path: e.g. ``data/raw/mta/subway/subway_hourly_2023-01.parquet``.
    """
    return (config.RAW_DIR / "mta" / "subway"
            / f"subway_hourly_{year}-{month:02d}.parquet")


def _subway_day(day):
    """Download one day of subway ridership, grouped by station and hour.

    The server sums over payment method and fare class. Grouping a whole
    month at once takes minutes per request on data.ny.gov, while one day
    (about 10,000 rows) takes a few seconds.

    Args:
        day (date): The day to download.

    Returns:
        pandas.DataFrame: One row per station complex and hour.
    """
    start = f"{day.isoformat()}T00:00:00"
    end = f"{(day + timedelta(days=1)).isoformat()}T00:00:00"
    where = (f"transit_mode = 'subway' AND transit_timestamp >= '{start}' "
             f"AND transit_timestamp < '{end}'")
    keys = ("station_complex_id, station_complex, borough, latitude, "
            "longitude, transit_timestamp")
    select = (f"{keys}, sum(ridership) AS ridership, "
              f"sum(transfers) AS transfers")
    return socrata_download(SUBWAY_IDS[day.year], select=select, where=where,
                            group=keys, order=keys)


def download_subway(months):
    """Download hourly subway ridership by station, one file per month.

    Only ``transit_mode = 'subway'`` is kept (not the tram or Staten Island
    Railway). 2023-2024 and 2025 come from two different datasets with the
    same columns.

    Args:
        months (list of tuple): (year, month) pairs to download.

    Returns:
        list of Path: One parquet file per month.
    """
    paths = []
    for year, month in months:
        dest = subway_path(year, month)
        paths.append(dest)
        if dest.exists():
            continue
        days = [date(year, month, day) for day in
                range(1, calendar.monthrange(year, month)[1] + 1)]
        with ThreadPoolExecutor(SUBWAY_WORKERS) as pool:
            data = pd.concat(pool.map(_subway_day, days), ignore_index=True)

        data["transit_timestamp"] = pd.to_datetime(data["transit_timestamp"])
        for column in ("latitude", "longitude", "ridership", "transfers"):
            data[column] = data[column].astype(float)
        _save_atomic(data, dest)
        print(f"  {dest.name}: {len(data):,} rows")
    return paths


def external_shapes(paths):
    """Tabulate rows x columns of each downloaded external dataset.

    Subway months are added together into one row.

    Args:
        paths (dict): The output of ``download_external``.

    Returns:
        pandas.DataFrame: One row per dataset with ``dataset``, ``files``,
        ``rows`` and ``columns``.
    """
    import geopandas as gpd

    records = []
    for name in ("lookup", "cbd_zones", "cbd_speeds", "weather"):
        data = pd.read_csv(paths[name])
        records.append({"dataset": name, "files": 1, "rows": len(data),
                        "columns": data.shape[1]})
    zones = gpd.read_file(paths["shapefile"])
    records.append({"dataset": "shapefile", "files": 1, "rows": len(zones),
                    "columns": zones.shape[1]})
    subway = parquet_shapes(paths["subway"])
    records.append({"dataset": "subway", "files": len(subway),
                    "rows": subway["rows"].sum(),
                    "columns": subway["columns"].max()})
    return pd.DataFrame(records)


def download_external(months):
    """Download every external dataset listed in ``external_datasets.md``.

    Args:
        months (list of tuple): Months of subway ridership to download.
            The other datasets are small and always cover the full timeline.

    Returns:
        dict: Local paths, keyed by dataset name.
    """
    print("Taxi zones")
    paths = download_taxi_zones()
    print("MTA CBD taxi zones")
    paths["cbd_zones"] = download_cbd_zones()
    print("MTA CBD speeds")
    paths["cbd_speeds"] = download_cbd_speeds()
    print("NOAA weather")
    paths["weather"] = download_weather()
    print(f"MTA subway ridership: {len(months)} months")
    paths["subway"] = download_subway(months)
    return paths
