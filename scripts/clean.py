"""Check, clean and summarise the TLC trip data with PySpark.

The notebook runs this module in four parts:

1. Checks on the full raw data: nulls (``null_counts``), CBD fee values
   (``cbd_fee_values``, ``cbd_fee_vs_zone_rule``) and distributions
   (``distribution_quantiles``, ``histogram``).
2. Cleaning (``clean``): every removal rule is in ``rules``, with its reason.
   ``step_shapes`` counts rows x columns after each rule in one pass.
3. New columns (``add_columns``): trip time, earnings per engaged hour,
   zone and trip groups, time band and calendar columns.
4. The summary table (``build_summary``), saved to ``data/curated/``.
"""

import pandas as pd
from pyspark.sql import functions as F

from scripts import config, zones
from scripts.spark_io import COLUMNS, TIME_COLUMNS

# ---------------------------------------------------------------------------
# Business rules
# ---------------------------------------------------------------------------
# Zones with no location: 264 = Unknown, 265 = Outside of NYC. A trip with
# an end in one of them is kept only when its other end is in the CBD, so
# it is still a treated trip (the CBD fee applies to trips that start or
# end in the zone)
UNKNOWN_ZONES = [264, 265]
# The zone group those two zones get
UNKNOWN_GROUP = "unknown"

# Outlier cut-offs, chosen from the full distributions in the notebook
MIN_TRIP_MINUTES = 1
MAX_TRIP_MINUTES = 300
MAX_SPEED_MPH = 60
MAX_DISTANCE_MILES = 100
MAX_MONEY = 500  # driver_pay, in dollars

# Time bands, by pickup hour. They follow the demand pattern and the car
# toll's hours (peak weekdays 5am-9pm, off-peak overnight).
TIME_BANDS = [
    ("overnight", 21, 5),      # 21:00-04:59
    ("morning_peak", 5, 10),   # 05:00-09:59
    ("midday", 10, 16),        # 10:00-15:59
    ("evening", 16, 21),       # 16:00-20:59
]
TIME_BAND_ORDER = [band for band, _, _ in TIME_BANDS]

# Zone groups (``zone_group_fine`` in the zone labels), in the order that
# decides a trip's group: each trip takes the first group either end falls
# in. Both the Spark rule (``add_columns``) and the pandas rule
# (``build_summary``) follow this order.
ZONE_GROUPS = ["cbd", "ring_adjacent", "ring_across", "control_near",
               "control_far"]
# The trip group each zone group gives. Trips touching the CBD are treated
TRIP_GROUPS = {"cbd": "treated", "ring_adjacent": "ring_adjacent",
               "ring_across": "ring_across", "control_near": "control_near",
               "control_far": "control_far"}
# Back to three trip groups: treated, ring and control
TRIP_COARSE = {"treated": "treated", "ring_adjacent": "ring",
               "ring_across": "ring", "control_near": "control",
               "control_far": "control"}
COARSE_TRIP_GROUPS = ["treated", "ring", "control"]

# The HVFHV columns the rules below read, named once here
DISTANCE_COLUMN = "trip_miles"
VENDOR_COLUMN = "hvfhs_license_num"
MONEY_COLUMN = "driver_pay"  # driver pay after commission, excluding tips

SUMMARY_FILE = config.CURATED_DIR / "trip_summary.parquet"
PARTS_DIR = config.CURATED_DIR / "summary_parts"


def _zone_groups(labels):
    """``LocationID`` and ``zone_group`` (the fine group) of every zone,
    with ``UNKNOWN_GROUP`` for zones 264 and 265."""
    return (labels[["LocationID", "zone_group_fine"]]
            .rename(columns={"zone_group_fine": "zone_group"})
            .fillna({"zone_group": UNKNOWN_GROUP})
            .astype({"LocationID": "int32"}))


# ---------------------------------------------------------------------------
# 1. Checks on the raw data
# ---------------------------------------------------------------------------
def add_trip_time(data, service):
    """Add ``trip_seconds``, ``trip_hours`` and ``speed_mph``.

    The timestamps are New York wall-clock times. Converting them to real
    instants first means trips across a daylight saving change get the
    right length.

    Args:
        data (pyspark.sql.DataFrame): Trips.
        service (str): ``"fhvhv"``.

    Returns:
        pyspark.sql.DataFrame: ``data`` with the three columns added.
    """
    pickup, dropoff = TIME_COLUMNS[service]
    seconds = (F.col(dropoff).cast("timestamp").cast("long")
               - F.col(pickup).cast("timestamp").cast("long"))
    return (data
            .withColumn("trip_seconds", seconds)
            .withColumn("trip_hours", F.col("trip_seconds") / 3600)
            # Null instead of an error for zero-length trips
            .withColumn("speed_mph", F.try_divide(F.col(DISTANCE_COLUMN),
                                                  F.col("trip_hours"))))


def null_counts(data, service):
    """Count nulls in every kept column by month and vendor.

    Args:
        data (pyspark.sql.DataFrame): Raw trips from ``read_service``.
        service (str): ``"fhvhv"``.

    Returns:
        pandas.DataFrame: One row per group with ``rows`` and a null count
        per column.
    """
    keys = ["file_month", VENDOR_COLUMN]
    counted = [c for c in COLUMNS[service] if c not in keys]
    table = data.groupBy(keys).agg(
        F.count("*").alias("rows"),
        *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in counted])
    return table.orderBy(keys).toPandas()


def nulls_by_year(table, keys):
    """Add the monthly null counts from ``null_counts`` up by year.

    Args:
        table (pandas.DataFrame): Output of ``null_counts``.
        keys (list of str): Grouping columns to keep besides the year,
            e.g. ``["hvfhs_license_num"]``.

    Returns:
        pandas.DataFrame: ``rows`` and the null count of every column that
        has any nulls, plus ``null_pct`` (share of rows with a null in the
        column with the most nulls).
    """
    table = table.assign(year=pd.to_datetime(table["file_month"]).dt.year)
    totals = table.drop(columns="file_month").groupby(["year"] + keys).sum()
    totals = totals.loc[:, (totals != 0).any()]
    counts = totals.drop(columns="rows")
    if not counts.empty:
        totals["null_pct"] = (counts.max(axis=1) / totals["rows"] * 100
                              ).round(2)
    return totals


def hourly_counts(data, service):
    """Count trips by pickup hour, separately for weekdays and weekends.

    Args:
        data (pyspark.sql.DataFrame): Trips.
        service (str): ``"fhvhv"``.

    Returns:
        pandas.DataFrame: Rows are hours (0-23), columns ``weekday`` and
        ``weekend``, values are the share of that day type's trips in %.
    """
    pickup = TIME_COLUMNS[service][0]
    counts = data.groupBy(
        F.hour(pickup).alias("hour"),
        F.when(F.dayofweek(pickup).isin(1, 7), "weekend")
        .otherwise("weekday").alias("day"),
    ).count().toPandas()
    table = counts.pivot(index="hour", columns="day", values="count")
    return table / table.sum() * 100


def cbd_fee_values(data, service):
    """Count 2025 trips by ``cbd_congestion_fee`` value and pickup hour.

    Args:
        data (pyspark.sql.DataFrame): Raw trips.
        service (str): ``"fhvhv"``.

    Returns:
        pandas.DataFrame: One row per fee value with ``trips``,
        ``trips_pct`` (share of 2025 trips) and ``hours`` (how many of the
        24 pickup hours the value appears in).
    """
    pickup = TIME_COLUMNS[service][0]
    counts = (data.where(F.year("file_month") == 2025)
              .groupBy(F.round("cbd_congestion_fee", 2).alias("fee"),
                       F.hour(pickup).alias("hour"))
              .count().toPandas())
    table = counts.groupby("fee").agg(trips=("count", "sum"),
                                      hours=("hour", "nunique"))
    table.insert(1, "trips_pct",
                 (table["trips"] / table["trips"].sum() * 100).round(3))
    return table


def cbd_fee_vs_zone_rule(data, labels):
    """Compare the trips charged the CBD fee with the trips the zone rule
    calls treated.

    The fee is charged on trips that start, end *or pass through* the zone,
    but the records hold no route, so the rule can only read the two ends.
    A trip that only passes through is therefore counted as a control trip,
    which pulls the two groups together. The gap this returns is how large
    that is, and it goes in Limitations.

    Args:
        data (pyspark.sql.DataFrame): Raw trips. Only 2025 rows are used,
            since the fee column is null before the toll.
        labels (pandas.DataFrame): Zone labels, with ``in_cbd``.

    Returns:
        pandas.DataFrame: One row per 2025 month, with the number of
        ``charged`` and ``treated`` trips, ``treated_of_charged_pct`` (charged
        trips the zone rule marks treated) and ``charged_of_treated_pct``
        (treated trips that paid the fee).
    """
    cbd = labels.loc[labels["in_cbd"], "LocationID"].astype(int).tolist()
    charged = F.coalesce(F.col("cbd_congestion_fee"), F.lit(0.0)) > 0
    treated = (F.col("PULocationID").isin(cbd)
               | F.col("DOLocationID").isin(cbd))
    table = (data.where(F.year("file_month") == 2025)
             .groupBy(F.month("file_month").alias("month"))
             .agg(F.sum(charged.cast("int")).alias("charged"),
                  F.sum(treated.cast("int")).alias("treated"),
                  F.sum((charged & treated).cast("int")).alias("both"))
             .toPandas().set_index("month").sort_index())
    table["treated_of_charged_pct"] = table["both"] / table["charged"] * 100
    table["charged_of_treated_pct"] = table["both"] / table["treated"] * 100
    return table.drop(columns="both")


def distribution_quantiles(data, service):
    """Quantiles of trip length, distance, speed and money.

    Only rows with a positive trip time and distance count, since speed
    needs both.

    Each variable is first counted into fine bins with ``histogram`` (a
    plain group-by count, which needs little memory), and the quantiles are
    read off the cumulative counts, so they are accurate to one bin width.
    ``percentile_approx`` on the 715 million HVFHV rows ran for over 20
    minutes and filled the laptop's memory. Values past the last bin show
    as the bin's upper edge with a ``+`` (e.g. ``"360+"``).

    Args:
        data (pyspark.sql.DataFrame): Trips with ``add_trip_time`` columns.
        service (str): ``"fhvhv"``. Unused, so that every step in this
            module takes the same pair.

    Returns:
        pandas.DataFrame: Rows are quantiles, columns are variables.
    """
    # column: (start, stop, bin width)
    bins = {
        "trip_minutes": (0, 360, 0.1),
        DISTANCE_COLUMN: (0, 100, 0.05),
        "speed_mph": (0, 100, 0.1),
        MONEY_COLUMN: (-50, 500, 0.25),
    }
    data = (data.where((F.col("trip_seconds") > 0)
                       & (F.col(DISTANCE_COLUMN) > 0))
            .withColumn("trip_minutes", F.col("trip_seconds") / 60))
    probs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    table = pd.DataFrame(index=pd.Index(probs, name="quantile"))
    for column, (start, stop, width) in bins.items():
        counts = histogram(data, column, start, stop, width)
        share = counts["count"].cumsum() / counts["count"].sum()
        values = []
        for prob in probs:
            first = int((share >= prob).to_numpy().argmax())
            if first == len(counts) - 1:
                values.append(f"{stop}+")
            elif first == 0:
                values.append(f"<{start + width}")
            else:
                values.append(round(counts["left"].iloc[first] + width, 2))
        table[column] = values
    return table


def histogram(data, column, start, stop, width):
    """Count every row into equal-width bins, so plots use the full data.

    Values below ``start`` go into the first bin and values at or above
    ``stop`` into the last bin, so outliers stay visible. Nulls are skipped.

    Args:
        data (pyspark.sql.DataFrame): Trips.
        column (str): Column to bin.
        start (float): Left edge of the first bin.
        stop (float): Right edge of the last bin.
        width (float): Bin width.

    Returns:
        pandas.DataFrame: ``left`` edge and ``count`` for each bin.
    """
    bins = int(round((stop - start) / width))
    index = F.floor((F.col(column) - start) / width)
    index = F.least(F.greatest(index, F.lit(0)), F.lit(bins - 1))
    counts = (data.where(F.col(column).isNotNull())
              .groupBy(index.alias("bin")).count().toPandas())
    table = pd.DataFrame({"left": [start + i * width for i in range(bins)]})
    table["count"] = (counts.set_index("bin")["count"]
                      .reindex(range(bins), fill_value=0).values)
    return table


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
def rules(service):
    """List the removal rules for one service, in the order they apply.

    Each rule is (name, reason, keep), where ``keep`` is true for rows that
    stay. The data needs the ``add_trip_time`` columns.

    Args:
        service (str): ``"fhvhv"``.

    Returns:
        list of tuple: (str, str, pyspark.sql.Column).
    """
    pickup, dropoff = TIME_COLUMNS[service]
    distance, money = DISTANCE_COLUMN, MONEY_COLUMN
    in_file_month = (F.trunc(F.to_date(pickup), "month")
                     == F.col("file_month"))
    labels = zones.load_zone_labels()
    cbd = labels.loc[labels["in_cbd"], "LocationID"].tolist()
    pickup_unknown = F.col("PULocationID").isin(UNKNOWN_ZONES)
    dropoff_unknown = F.col("DOLocationID").isin(UNKNOWN_ZONES)
    common_start = [
        ("pickup_in_file_month",
         "Pickup date outside the month of the file (wrong dates)",
         in_file_month),
        ("dropoff_after_pickup",
         "Drop-off time before or equal to pickup time",
         F.col("trip_seconds") > 0),
    ]
    # There is no distance rule: the raw data has no negative or null
    # distances, and zero-distance trips are kept, with unknown earnings
    # (add_columns), because most are paid trips the meter didn't measure
    common_end = [
        ("trip_length",
         f"Trip shorter than {MIN_TRIP_MINUTES} minute or longer than "
         f"{MAX_TRIP_MINUTES // 60} hours",
         F.col("trip_seconds").between(MIN_TRIP_MINUTES * 60,
                                       MAX_TRIP_MINUTES * 60)),
        ("distance_cap", f"Distance over {MAX_DISTANCE_MILES} miles",
         F.col(distance) <= MAX_DISTANCE_MILES),
        ("speed_cap", f"Average speed over {MAX_SPEED_MPH} mph (impossible "
                      "in NYC traffic)",
         F.col("speed_mph") <= MAX_SPEED_MPH),
        ("money_cap", f"{money} over ${MAX_MONEY}",
         F.col(money) <= MAX_MONEY),
        ("known_zones", "Pickup or drop-off in zone 264 (Unknown) or 265 "
                        "(Outside of NYC), and the other end is not in the "
                        "CBD, so the trip group is unknown",
         ~(pickup_unknown | dropoff_unknown)
         | (pickup_unknown & F.col("DOLocationID").isin(cbd))
         | (dropoff_unknown & F.col("PULocationID").isin(cbd))),
    ]
    middle = [
        ("positive_driver_pay", "Zero or negative driver pay",
         F.col("driver_pay") > 0),
    ]
    return common_start + middle + common_end


def clean(data, service):
    """Apply every rule in ``rules`` to the trips.

    Args:
        data (pyspark.sql.DataFrame): Raw trips with ``add_trip_time``.
        service (str): ``"fhvhv"``.

    Returns:
        pyspark.sql.DataFrame: Trips that pass every rule.
    """
    keep = F.lit(True)
    for _, _, condition in rules(service):
        keep = keep & F.coalesce(condition, F.lit(False))
    return data.where(keep)


def step_shapes(data, service, raw_columns):
    """Count rows x columns after each cleaning step, in one pass.

    Args:
        data (pyspark.sql.DataFrame): Raw trips with ``add_trip_time``
            (already cut down to the kept columns).
        service (str): ``"fhvhv"``.
        raw_columns (str): Column count of the raw files, e.g. ``"19-20"``.

    Returns:
        pandas.DataFrame: One row per step with ``step``, ``reason``,
        ``rows``, ``columns``, ``removed`` and ``removed_pct`` (share of the
        raw rows).
    """
    kept_columns = len(COLUMNS[service])
    steps = rules(service)
    keep = F.lit(True)
    counts = []
    for name, _, condition in steps:
        keep = keep & F.coalesce(condition, F.lit(False))
        counts.append(F.sum(keep.cast("int")).alias(name))
    row = data.agg(F.count("*").alias("all"), *counts).first()

    records = [
        {"step": "raw", "reason": "Raw files", "rows": row["all"],
         "columns": raw_columns},
        {"step": "keep_columns", "reason": "Keep only the columns we need",
         "rows": row["all"], "columns": str(kept_columns)},
    ]
    for name, reason, _ in steps:
        records.append({"step": name, "reason": reason, "rows": row[name],
                        "columns": str(kept_columns)})
    table = pd.DataFrame(records)
    table["removed"] = -table["rows"].diff().fillna(0).astype(int)
    table["removed_pct"] = (table["removed"] / row["all"] * 100).round(3)
    return table


# ---------------------------------------------------------------------------
# 3. New columns
# ---------------------------------------------------------------------------
def time_band(hour):
    """Return the time band name for a pickup hour column.

    Args:
        hour (pyspark.sql.Column): Hour of day (0-23).

    Returns:
        pyspark.sql.Column: One of ``TIME_BAND_ORDER``.
    """
    band = F
    for name, start, end in TIME_BANDS:
        if start > end:  # the band wraps past midnight
            inside = (hour >= start) | (hour < end)
        else:
            inside = (hour >= start) & (hour < end)
        band = band.when(inside, name)
    return band


def add_columns(data, service, spark, labels):
    """Add the analysis columns to cleaned trips.

    Added: ``trip_hours`` (from ``add_trip_time``), ``earnings``
    (``driver_pay``), ``earnings_per_engaged_hour``,
    ``pickup_group`` and ``dropoff_group`` (one of ``ZONE_GROUPS``, or
    ``UNKNOWN_GROUP`` for zones 264 and 265),
    ``trip_group`` (the first of ``ZONE_GROUPS`` either end falls in, named
    by ``TRIP_GROUPS``), ``trip_group_coarse`` (``treated``, ``ring`` or
    ``control``), ``after_toll``, ``date``, ``hour``, ``time_band`` and
    ``day_of_week`` (1 = Monday).

    Earnings per engaged hour is left null for shared HVFHV rides
    (``shared_match_flag`` = Y), because two trips share the same driving
    time, and for trips with unknown earnings. Earnings are unknown for
    zero-distance trips.

    Args:
        data (pyspark.sql.DataFrame): Output of ``clean``.
        service (str): ``"fhvhv"``.
        spark (SparkSession): Active session, to load the zone labels.
        labels (pandas.DataFrame): Output of ``zones.load_zone_labels``.

    Returns:
        pyspark.sql.DataFrame: ``data`` with the new columns.
    """
    pickup = TIME_COLUMNS[service][0]
    # A zero distance means the meter didn't measure the trip, so its fare
    # or pay can't be trusted either. The trip still counts
    measured = F.col(DISTANCE_COLUMN) > 0
    data = data.withColumn("earnings", F.when(measured, F.col("driver_pay")))
    shared = F.col("shared_match_flag") == "Y"

    data = data.withColumn(
        "earnings_per_engaged_hour",
        F.when(~shared, F.try_divide(F.col("earnings"), F.col("trip_hours"))))

    # Zones 264 and 265 get UNKNOWN_GROUP. clean only keeps them on trips
    # whose other end is in the CBD, so every trip below is still treated,
    # ring or control
    groups = F.broadcast(spark.createDataFrame(_zone_groups(labels)))
    data = (data
            .join(groups.withColumnsRenamed({"LocationID": "PULocationID",
                                             "zone_group": "pickup_group"}),
                  "PULocationID")
            .join(groups.withColumnsRenamed({"LocationID": "DOLocationID",
                                             "zone_group": "dropoff_group"}),
                  "DOLocationID"))
    trip_group = F
    for group in ZONE_GROUPS:
        either = ((F.col("pickup_group") == group)
                  | (F.col("dropoff_group") == group))
        trip_group = trip_group.when(either, TRIP_GROUPS[group])
    coarse = F.create_map(*[F.lit(v) for pair in TRIP_COARSE.items()
                            for v in pair])
    data = (data.withColumn("trip_group", trip_group)
            .withColumn("trip_group_coarse", coarse[F.col("trip_group")]))

    toll_start = config.TOLL_START_DATE.isoformat()
    return (data
            .withColumn("date", F.to_date(pickup))
            .withColumn("hour", F.hour(pickup))
            .withColumn("time_band", time_band(F.col("hour")))
            .withColumn("day_of_week",
                        (F.dayofweek(pickup) + 5) % 7 + 1)
            .withColumn("after_toll", F.col("date") >= F.lit(toll_start)
                        .cast("date")))


def earnings_coverage(trips):
    """Share of cleaned trips with known earnings per engaged hour.

    Args:
        trips (pyspark.sql.DataFrame): Output of ``add_columns``.

    Returns:
        pandas.DataFrame: Rows are years, columns coarse trip groups, values
        the share of trips (%) whose ``earnings_per_engaged_hour`` is not
        null.
    """
    table = trips.groupBy(F.year("date").alias("year"),
                          "trip_group_coarse").agg(
        (F.count("earnings_per_engaged_hour") / F.count("*") * 100)
        .alias("known_pct")).toPandas()
    return table.pivot(index="year", columns="trip_group_coarse",
                       values="known_pct")[COARSE_TRIP_GROUPS]


# ---------------------------------------------------------------------------
# 4. Summary table
# ---------------------------------------------------------------------------
def summarise(data, service):
    """Group trips into service x pickup zone x drop-off group x date x band.

    The median uses ``percentile_approx`` with accuracy 1,000 (within 0.1%
    of the true rank), computed on every trip in the group.

    Args:
        data (pyspark.sql.DataFrame): Output of ``add_columns``.
        service (str): ``"fhvhv"``.

    Returns:
        pyspark.sql.DataFrame: One row per group that has trips.
    """
    return (data
            .groupBy("PULocationID", "dropoff_group", "date", "time_band")
            .agg(F.count("*").alias("trips"),
                 F.count("earnings").alias("trips_with_earnings"),
                 F.sum("earnings").alias("total_earnings"),
                 F.percentile_approx("earnings_per_engaged_hour", 0.5, 1000)
                 .alias("median_earnings_per_engaged_hour"))
            .withColumn("service", F.lit(service)))


def summarise_month(spark, service, year, month, labels, overwrite=False):
    """Clean and summarise one month of one service, saving the result.

    Months are independent (every summary row has one date), so doing one
    month at a time keeps Spark's memory use small. Each month is saved to
    ``data/curated/summary_parts/`` and reused on the next run unless
    ``overwrite`` is true.

    Args:
        spark (SparkSession): Active session.
        service (str): ``"fhvhv"``.
        year (int): Year.
        month (int): Month (1-12).
        labels (pandas.DataFrame): Zone labels.
        overwrite (bool): Rebuild the month even if it was saved before.

    Returns:
        pandas.DataFrame: The month's summary rows (groups with trips only).
    """
    from scripts.spark_io import read_month

    path = PARTS_DIR / f"{service}_{year}-{month:02d}.parquet"
    if path.exists() and not overwrite:
        return pd.read_parquet(path)
    trips = add_trip_time(read_month(spark, service, year, month), service)
    trips = add_columns(clean(trips, service), service, spark, labels)
    table = summarise(trips, service).toPandas()
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


def build_summary(spark, labels, months, services=("fhvhv",),
                  overwrite=False):
    """Build the summary table, with zero rows for groups with no trips.

    Every service x pickup zone x drop-off group x date x time band gets a
    row, so days with no trips count as 0 instead of being missing. The
    cells are pickup zones 1-263 x the five ``ZONE_GROUPS``, plus the CBD
    zones x ``UNKNOWN_GROUP`` and zones 264 and 265 x ``cbd``, the only
    cells with an unknown end that cleaning keeps.
    The table is saved to ``data/curated/trip_summary.parquet``.

    Args:
        spark (SparkSession): Active session.
        labels (pandas.DataFrame): Zone labels.
        months (list of tuple): Months to include.
        services (tuple of str): Services to include.
        overwrite (bool): Rebuild months that were saved before (needed
            after changing a cleaning rule).

    Returns:
        pandas.DataFrame: The full summary table.
    """
    parts = []
    for service in services:
        for year, month in months:
            parts.append(summarise_month(spark, service, year, month, labels,
                                         overwrite))
        print(f"{service}: {len(months)} months summarised")
    found = pd.concat(parts, ignore_index=True)
    found["date"] = pd.to_datetime(found["date"]).dt.date

    # Every pickup zone x drop-off group a cleaned trip can have. An
    # unknown end is only kept opposite the CBD (the known_zones rule)
    cells = (_zone_groups(labels)
             .rename(columns={"LocationID": "PULocationID",
                              "zone_group": "pickup_group"})
             .merge(pd.DataFrame({"dropoff_group":
                                  ZONE_GROUPS + [UNKNOWN_GROUP]}),
                    how="cross"))
    ends = cells[["pickup_group", "dropoff_group"]]
    known = (ends != UNKNOWN_GROUP).all(axis=1)
    with_cbd = (ends == "cbd").any(axis=1)
    cells = cells[known | with_cbd]

    dates = pd.concat([pd.Series(pd.date_range(
        f"{year}-{month:02d}-01", periods=pd.Period(
            f"{year}-{month:02d}").days_in_month)) for year, month in months])
    grid = (pd.DataFrame({"service": list(services)})
            .merge(cells, how="cross")
            .merge(pd.DataFrame({"date": dates.dt.date}), how="cross")
            .merge(pd.DataFrame({"time_band": TIME_BAND_ORDER}), how="cross"))

    keys = ["service", "PULocationID", "dropoff_group", "date", "time_band"]
    table = grid.merge(found, on=keys, how="left")
    assert table["trips"].sum() == found["trips"].sum(), \
        "some trips fall outside the grid"
    for column in ["trips", "trips_with_earnings", "total_earnings"]:
        table[column] = table[column].fillna(0)
    table[["trips", "trips_with_earnings"]] = (
        table[["trips", "trips_with_earnings"]].astype("int64"))

    # Same rule as add_columns: the first of ZONE_GROUPS either end is in.
    # Going through them in reverse lets the earlier groups overwrite
    for group in reversed(ZONE_GROUPS):
        either = ((table["pickup_group"] == group)
                  | (table["dropoff_group"] == group))
        table.loc[either, "trip_group"] = TRIP_GROUPS[group]
    assert table["trip_group"].notna().all(), "a row got no trip group"
    table["trip_group_coarse"] = table["trip_group"].map(TRIP_COARSE)
    table["date"] = pd.to_datetime(table["date"])
    table["after_toll"] = table["date"] >= pd.Timestamp(config.TOLL_START_DATE)
    table["day_of_week"] = table["date"].dt.dayofweek + 1
    table["time_band"] = pd.Categorical(table["time_band"],
                                        TIME_BAND_ORDER, ordered=True)
    table = table.sort_values(keys).reset_index(drop=True)
    table.to_parquet(SUMMARY_FILE, index=False)
    return table
