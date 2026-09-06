"""Tables behind the exploration and maps in Section 5 of ``main.ipynb``.

Two kinds of function:

1. Spark checks on trip-level data (``removals_by_group``,
   ``yellow_vendors``, ``histograms``, ``zone_medians``, ``pickup_wait``).
   Each makes one pass over the data it is given and returns a small pandas
   table.
2. pandas tables built from the summary table. They compare the toll
   period (5 January to 31 December 2025) with the same dates a year
   earlier (``PERIODS``), so both periods cover the same season and the
   four pre-toll days of January 2025 are left out.
"""

from datetime import date

import pandas as pd
from pyspark.sql import functions as F

from scripts import config
from scripts.clean import (COARSE_TRIP_GROUPS, TIME_BAND_ORDER, TIME_COLUMNS,
                           _money, rules)

# The toll period and the same dates a year earlier
PERIODS = {
    "before": (config.TOLL_START_DATE.replace(year=2024), date(2024, 12, 31)),
    "after": (config.TOLL_START_DATE, date(2025, 12, 31)),
}

# Zones with fewer pickups a day than this before the toll are greyed out on
# the maps, because their percentage changes rest on too few trips
MIN_DAILY_PICKUPS = 10

# Histogram ranges for the before/after cleaning plots: (start, stop, bin
# width). Each range reaches past its cut-off in clean.py (5 hours, $500),
# so the removed tail shows.
HIST_SPECS = {
    "trip_minutes": (0, 360, 3),
    "money": (-50, 550, 5),
    "money_per_hour": (-100, 400, 5),
}


def _period(column):
    """Name the period a date column falls in, or null outside both."""
    period = F
    for name, (start, end) in PERIODS.items():
        period = period.when(column.between(F.lit(start), F.lit(end)), name)
    return period


def add_period(table, column="date"):
    """Add a ``period`` column (``before``, ``after`` or missing).

    Args:
        table (pandas.DataFrame): Any table with a date column.
        column (str): Name of that column.

    Returns:
        pandas.DataFrame: ``table`` with ``period`` added.
    """
    table = table.copy()
    table["period"] = None
    for name, (start, end) in PERIODS.items():
        inside = table[column].between(pd.Timestamp(start),
                                       pd.Timestamp(end))
        table.loc[inside, "period"] = name
    return table


def _pct_change(before, after):
    """Percentage change from ``before`` to ``after`` (NaN if before is 0)."""
    return (after / before.where(before > 0) - 1) * 100


def _coarse_trip_group(pickup, dropoff):
    """Spark rule for the three trip groups, with ``unknown_zone`` for the
    trips that have an end in zone 264 or 265 and don't touch the CBD or
    ring."""
    return (F.when((pickup == "cbd") | (dropoff == "cbd"), "treated")
            .when((pickup == "ring") | (dropoff == "ring"), "ring")
            .when((pickup == "control") & (dropoff == "control"), "control")
            .otherwise("unknown_zone"))


def _join_groups(data, spark, labels, column="zone_group"):
    """Left-join a zone group onto both ends of every trip."""
    groups = F.broadcast(spark.createDataFrame(
        labels.loc[labels[column].notna(), ["LocationID", column]]
        .astype({"LocationID": "int32"})))
    return (data
            .join(groups.withColumnsRenamed({"LocationID": "PULocationID",
                                             column: "pickup_group"}),
                  "PULocationID", "left")
            .join(groups.withColumnsRenamed({"LocationID": "DOLocationID",
                                             column: "dropoff_group"}),
                  "DOLocationID", "left"))


# ---------------------------------------------------------------------------
# 1. Spark checks on trip-level data
# ---------------------------------------------------------------------------
def removals_by_group(data, service, spark, labels):
    """Rows removed by each cleaning rule, by year and trip group.

    Each raw row is counted once, under the first rule it fails (the order
    of ``clean.rules``), so the shares add up to the total removed. Trips
    with an end in zone 264 or 265 that don't touch the CBD or ring can't
    be given a group and are counted as ``unknown_zone``.

    Args:
        data (pyspark.sql.DataFrame): Raw trips with ``add_trip_time``.
        service (str): ``"yellow"`` or ``"fhvhv"``.
        spark (SparkSession): Active session.
        labels (pandas.DataFrame): Zone labels.

    Returns:
        pandas.DataFrame: ``year``, ``trip_group``, ``rule`` (``kept`` for
        rows that pass every rule) and ``rows``.
    """
    first_failed = F
    for name, _, condition in rules(service):
        first_failed = first_failed.when(
            ~F.coalesce(condition, F.lit(False)), name)
    first_failed = first_failed.otherwise("kept")
    data = _join_groups(data, spark, labels)
    return (data
            .groupBy(F.year("file_month").alias("year"),
                     _coarse_trip_group(F.col("pickup_group"),
                                        F.col("dropoff_group"))
                     .alias("trip_group"),
                     first_failed.alias("rule"))
            .agg(F.count("*").alias("rows"))
            .toPandas())


def removal_shares(table):
    """Share of each group-year's raw rows removed by each rule (%).

    Args:
        table (pandas.DataFrame): Output of ``removals_by_group``.

    Returns:
        pandas.DataFrame: Rows are rules (plus ``all rules``), columns are
        (trip group, year).
    """
    total = table.groupby(["trip_group", "year"])["rows"].sum()
    removed = table[table["rule"] != "kept"]
    shares = (removed.groupby(["rule", "trip_group", "year"])["rows"].sum()
              / total * 100).unstack(["trip_group", "year"]).fillna(0)
    shares.loc["all rules"] = shares.sum()
    groups = [g for g in COARSE_TRIP_GROUPS + ["unknown_zone"]
              if g in shares.columns.get_level_values(0)]
    return shares[groups]


def add_money_columns(data, service):
    """Add the columns ``HIST_SPECS`` bins.

    ``money`` is ``fare_amount`` (yellow) or ``driver_pay`` (HVFHV), and
    ``money_per_hour`` is money divided by trip hours. Both exist before
    and after cleaning, so the raw and cleaned histograms measure the same
    thing. (The earnings per engaged hour used later also adds the driver
    surcharges to yellow fares, which can only be worked out on clean
    rows.)

    Args:
        data (pyspark.sql.DataFrame): Trips with ``add_trip_time``.
        service (str): ``"yellow"`` or ``"fhvhv"``.

    Returns:
        pyspark.sql.DataFrame: ``data`` with ``trip_minutes``, ``money``
        and ``money_per_hour`` added.
    """
    money = F.col(_money(service))
    return (data
            .withColumn("trip_minutes", F.col("trip_seconds") / 60)
            .withColumn("money", money)
            .withColumn("money_per_hour",
                        F.try_divide(money, F.col("trip_hours"))))


def null_shares(nulls):
    """Share of raw rows that are null in each column, by year (%).

    Args:
        nulls (pandas.DataFrame): Output of ``clean.null_counts``.

    Returns:
        pandas.DataFrame: Rows are years, columns the columns that have any
        nulls.
    """
    year = pd.to_datetime(nulls["file_month"]).dt.year.rename("year")
    counts = (nulls.select_dtypes("number").drop(columns=["VendorID",
                                                          "payment_type"],
                                                 errors="ignore")
              .groupby(year).sum())
    shares = counts.drop(columns="rows").div(counts["rows"], axis=0) * 100
    return shares.loc[:, (shares > 0).any()]


def yellow_vendors(yellow, spark, labels):
    """Raw yellow rows by month, vendor and pickup zone group.

    Also counts the rows whose drop-off time is at or before the pickup
    time, which the cleaning removes.

    Args:
        yellow (pyspark.sql.DataFrame): Raw yellow trips with
            ``add_trip_time``.
        spark (SparkSession): Active session.
        labels (pandas.DataFrame): Zone labels.

    Returns:
        pandas.DataFrame: ``file_month``, ``VendorID``, ``pickup_group``
        (missing for zones 264 and 265), ``rows`` and ``zero_time``.
    """
    return (_join_groups(yellow, spark, labels)
            .groupBy("file_month", "VendorID", "pickup_group")
            .agg(F.count("*").alias("rows"),
                 F.sum((F.col("trip_seconds") <= 0).cast("int"))
                 .alias("zero_time"))
            .toPandas())


def vendor_table(vendors, vendor, year):
    """One vendor's rows by month, and where its trips start.

    Args:
        vendors (pandas.DataFrame): Output of ``yellow_vendors``.
        vendor (int): ``VendorID`` to look at.
        year (int): Year for the pickup group shares.

    Returns:
        tuple: (monthly ``rows``, ``zero_time`` and ``share_of_yellow``
        (%) for the vendor; pickup group shares (%) in ``year`` for the
        vendor against all yellow rows).
    """
    vendors = vendors.assign(file_month=pd.to_datetime(vendors["file_month"]))
    mine = vendors[vendors["VendorID"] == vendor]
    monthly = mine.groupby("file_month")[["rows", "zero_time"]].sum()
    all_rows = vendors.groupby("file_month")["rows"].sum()
    monthly["share_of_yellow"] = monthly["rows"] / all_rows * 100
    groups = pd.DataFrame({
        name: table[table["file_month"].dt.year == year]
        .groupby("pickup_group", dropna=False)["rows"].sum()
        for name, table in [(f"vendor_{vendor}", mine),
                            ("all_yellow", vendors)]})
    groups = groups / groups.sum() * 100
    return monthly, groups


def histograms(data, specs):
    """Binned counts of several columns in one pass over the data.

    Works like ``clean.histogram`` (values outside the range go into the
    end bins, nulls are skipped), but counts every column at once, so the
    715 million HVFHV rows are read only once.

    Args:
        data (pyspark.sql.DataFrame): Trips.
        specs (dict): Column name -> (start, stop, bin width).

    Returns:
        dict: Column name -> pandas.DataFrame with ``left`` edge and
        ``count`` for each bin.
    """
    entries = []
    for column, (start, stop, width) in specs.items():
        bins = int(round((stop - start) / width))
        index = F.floor((F.col(column) - start) / width)
        index = F.least(F.greatest(index, F.lit(0)), F.lit(bins - 1))
        entries.append(F.when(F.col(column).isNotNull(), F.struct(
            F.lit(column).alias("column"), index.cast("int").alias("bin"))))
    counts = (data.select(F.explode(F.array(*entries)).alias("entry"))
              .where(F.col("entry").isNotNull())
              .groupBy("entry.column", "entry.bin").count().toPandas())
    tables = {}
    for column, (start, stop, width) in specs.items():
        bins = int(round((stop - start) / width))
        found = counts[counts["column"] == column].set_index("bin")["count"]
        tables[column] = pd.DataFrame({
            "left": [start + i * width for i in range(bins)],
            "count": found.reindex(range(bins), fill_value=0).to_numpy()})
    return tables


def zone_medians(trips):
    """Median earnings per engaged hour by pickup zone, before and after.

    Medians are taken over every trip (``percentile_approx`` with accuracy
    1,000), not over the summary table's cell medians.

    Args:
        trips (pyspark.sql.DataFrame): Output of ``clean.add_columns``.

    Returns:
        pandas.DataFrame: ``PULocationID``, ``period``, ``trips`` and
        ``median_earnings_per_engaged_hour``.
    """
    period = _period(F.col("date"))
    return (trips.withColumn("period", period)
            .where(F.col("period").isNotNull())
            .groupBy("PULocationID", "period")
            .agg(F.count("*").alias("trips"),
                 F.percentile_approx("earnings_per_engaged_hour", 0.5, 1000)
                 .alias("median_earnings_per_engaged_hour"))
            .toPandas())


def pickup_wait(fhvhv):
    """HVFHV pickup wait (request to pickup) by month and trip group.

    ``on_scene_datetime`` would be closer to the driver's arrival, but the
    data dictionary says it is only recorded for accessible vehicles, so
    the wait runs from ``request_datetime`` to ``pickup_datetime``. Rows
    where the pickup is before the request are counted, then left out.

    Args:
        fhvhv (pyspark.sql.DataFrame): Output of ``clean.add_columns``
            (HVFHV).

    Returns:
        pandas.DataFrame: ``month``, ``trip_group_coarse``, ``trips``,
        ``negative`` (pickup before request), ``median_wait_min`` and
        ``p90_wait_min``.
    """
    pickup = TIME_COLUMNS["fhvhv"][0]
    wait = (F.col(pickup).cast("timestamp").cast("long")
            - F.col("request_datetime").cast("timestamp").cast("long")) / 60
    quantiles = F.percentile_approx(F.when(wait >= 0, wait),
                                    [0.5, 0.9], 1000)
    table = (fhvhv
             .groupBy(F.trunc("date", "month").alias("month"),
                      "trip_group_coarse")
             .agg(F.count("*").alias("trips"),
                  F.sum((wait < 0).cast("int")).alias("negative"),
                  quantiles.alias("quantiles"))
             .toPandas())
    table["median_wait_min"] = table["quantiles"].str[0]
    table["p90_wait_min"] = table["quantiles"].str[1]
    table["month"] = pd.to_datetime(table["month"])
    return (table.drop(columns="quantiles")
            .sort_values(["month", "trip_group_coarse"])
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# 2. pandas tables from the summary table
# ---------------------------------------------------------------------------
def relative_trips(summary):
    """Monthly treated and ring trips relative to control trips.

    Each month's treated (and ring) trips are divided by its control trips,
    then indexed so the service's 2024 average is 100. Seasons, holidays
    and citywide shocks hit every group, so they mostly cancel in the
    ratio, and what is left is whether the groups move apart. A fall after
    the toll means the group lost trips compared with the rest of the city.

    Args:
        summary (pandas.DataFrame): The summary table.

    Returns:
        pandas.DataFrame: ``service``, ``month`` (its first day),
        ``treated`` and ``ring`` (2024 average = 100).
    """
    monthly = (summary
               .groupby(["service", summary["date"].dt.to_period("M")
                         .rename("month"), "trip_group_coarse"])["trips"]
               .sum().unstack("trip_group_coarse"))
    ratios = pd.DataFrame({group: monthly[group] / monthly["control"]
                           for group in ["treated", "ring"]})
    in_2024 = ratios.index.get_level_values("month").year == 2024
    base = ratios[in_2024].groupby("service").mean()
    ratios = ratios / base.reindex(
        ratios.index.get_level_values("service")).to_numpy() * 100
    ratios = ratios.reset_index()
    ratios["month"] = ratios["month"].dt.to_timestamp()
    return ratios


def group_change(summary, by="trip_group"):
    """Trips before and after the toll, and the change, by group.

    Args:
        summary (pandas.DataFrame): The summary table.
        by (str or list): Grouping columns besides ``service``.

    Returns:
        pandas.DataFrame: ``before``, ``after`` and ``change_pct`` by
        service and ``by``.
    """
    by = [by] if isinstance(by, str) else list(by)
    table = add_period(summary[["service", "date", "trips"] + by])
    table = (table.dropna(subset=["period"])
             .groupby(["service"] + by + ["period"], observed=True)["trips"]
             .sum().unstack("period"))
    table["change_pct"] = _pct_change(table["before"], table["after"])
    return table[["before", "after", "change_pct"]]


def zone_change(summary, medians=None, min_daily=MIN_DAILY_PICKUPS):
    """Change in pickups (and earnings per engaged hour) by pickup zone.

    Args:
        summary (pandas.DataFrame): The summary table.
        medians (pandas.DataFrame, optional): Output of ``zone_medians``
            for the same service, to add the change in median earnings per
            engaged hour.
        min_daily (float): Pickups a day before the toll a zone needs for
            ``enough_trips``.

    Returns:
        pandas.DataFrame: One row per service and pickup zone with
        ``before``, ``after``, ``change_pct``, ``daily_before`` and
        ``enough_trips``, plus ``eph_before``, ``eph_after`` and
        ``eph_change_pct`` when ``medians`` is given.
    """
    table = add_period(summary[["service", "PULocationID", "date",
                                "trips"]])
    table = table.dropna(subset=["period"])
    days = table.groupby("period")["date"].nunique()
    zones = (table.groupby(["service", "PULocationID", "period"])["trips"]
             .sum().unstack("period").reset_index())
    zones["change_pct"] = _pct_change(zones["before"], zones["after"])
    zones["daily_before"] = zones["before"] / days["before"]
    zones["enough_trips"] = zones["daily_before"] >= min_daily
    if medians is not None:
        eph = (medians.pivot(index="PULocationID", columns="period",
                             values="median_earnings_per_engaged_hour")
               .rename(columns=lambda p: f"eph_{p}"))
        zones = zones.merge(eph, on="PULocationID", how="left")
        zones["eph_change_pct"] = _pct_change(zones["eph_before"],
                                              zones["eph_after"])
    return zones


def band_day_change(summary):
    """Change in trips by time band and day of week, for each trip group.

    Args:
        summary (pandas.DataFrame): The summary table.

    Returns:
        pandas.DataFrame: The output of ``group_change`` by
        ``trip_group_coarse``, ``time_band`` and ``day_of_week``.
    """
    return group_change(summary, ["trip_group_coarse", "time_band",
                                  "day_of_week"])


def relative_change(changes, service, group="treated", base="control"):
    """How much more ``group`` changed than ``base``, by band and day (%).

    The change is (1 + group change) / (1 + base change) - 1, so -5 means
    the group's trips ended up 5% lower than if they had changed the way
    ``base`` trips did. It is a raw version of the difference-in-differences
    estimate, with no weather or other controls.

    Args:
        changes (pandas.DataFrame): Output of ``band_day_change``.
        service (str): ``"yellow"`` or ``"fhvhv"``.
        group (str): Trip group to compare.
        base (str): Trip group to compare against.

    Returns:
        pandas.DataFrame: Rows are time bands, columns days of the week
        (1 = Monday), values in %.
    """
    ratio = changes.loc[service]
    ratio = (ratio.loc[group, "after"] / ratio.loc[group, "before"]
             / (ratio.loc[base, "after"] / ratio.loc[base, "before"]))
    table = ((ratio - 1) * 100).unstack("day_of_week")
    return table.reindex([b for b in TIME_BAND_ORDER if b in table.index])


def band_day_table(changes, service, group):
    """One trip group's change by time band and day of week (%).

    Args:
        changes (pandas.DataFrame): Output of ``band_day_change``.
        service (str): ``"yellow"`` or ``"fhvhv"``.
        group (str): Trip group.

    Returns:
        pandas.DataFrame: Rows are time bands, columns days of the week.
    """
    table = changes.loc[(service, group), "change_pct"].unstack("day_of_week")
    return table.reindex([b for b in TIME_BAND_ORDER if b in table.index])
