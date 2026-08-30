"""Check, clean and summarise the TLC trip data with PySpark.

The notebook runs this module in four parts:

1. Checks on the full raw data: nulls (``null_counts``), yellow Flex Fare
   trips (``flex_fare_check``), yellow ``extra`` values (``extra_values``)
   and distributions (``distribution_quantiles``, ``histogram``).
2. Cleaning (``clean``): every removal rule is in ``RULES``, with its reason.
   ``step_shapes`` counts rows x columns after each rule in one pass.
3. New columns (``add_columns``): trip time, revenue, earnings per engaged
   hour, zone and trip groups, time band and calendar columns.
4. The summary table (``build_summary``), saved to ``data/curated/``.
"""

import pandas as pd
from pyspark.sql import functions as F

from scripts import config
from scripts.spark_io import COLUMNS, TIME_COLUMNS

# ---------------------------------------------------------------------------
# Business rules
# ---------------------------------------------------------------------------
# Yellow payment types (data dictionary, 18 March 2025)
FLEX_FARE = 0
NO_CHARGE, DISPUTE, VOIDED = 3, 4, 6
UNKNOWN_RATECODE = 99

# Zones with no location: 264 = Unknown, 265 = Outside of NYC
UNKNOWN_ZONES = [264, 265]

# The surcharges a yellow driver keeps (TLC taxi fare page): $1 overnight
# (8pm-6am), $2.50 weekday rush hour (4-8pm) and $5 for LaGuardia trips,
# alone or combined
DRIVER_EXTRAS = [0.0, 1.0, 2.5, 5.0, 6.0, 7.5]

# Vendor 1 (Creative Mobile Technologies) adds the congestion surcharge,
# airport fee and usually the CBD fee into `extra` as well as recording
# them in their own columns
VENDOR_WITH_FEES_IN_EXTRA = 1

# Outlier cut-offs, chosen from the full distributions in the notebook
MIN_TRIP_MINUTES = 1
MAX_TRIP_MINUTES = 300
MAX_SPEED_MPH = 60
MAX_DISTANCE_MILES = 100
MAX_MONEY = 500  # fare_amount (yellow) or driver_pay (HVFHV), in dollars

# Time bands, by pickup hour. They follow the demand pattern and the car
# toll's hours (peak weekdays 5am-9pm, off-peak overnight).
TIME_BANDS = [
    ("overnight", 21, 5),      # 21:00-04:59
    ("morning_peak", 5, 10),   # 05:00-09:59
    ("midday", 10, 16),        # 10:00-15:59
    ("evening", 16, 21),       # 16:00-20:59
]
TIME_BAND_ORDER = [band for band, _, _ in TIME_BANDS]
ZONE_GROUPS = ["cbd", "ring", "control"]

SUMMARY_FILE = config.CURATED_DIR / "trip_summary.parquet"
PARTS_DIR = config.CURATED_DIR / "summary_parts"


def _distance(service):
    return "trip_distance" if service == "yellow" else "trip_miles"


def _vendor(service):
    return "VendorID" if service == "yellow" else "hvfhs_license_num"


def _money(service):
    return "fare_amount" if service == "yellow" else "driver_pay"


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
        service (str): ``"yellow"`` or ``"fhvhv"``.

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
            .withColumn("speed_mph", F.try_divide(F.col(_distance(service)),
                                                  F.col("trip_hours"))))


def null_counts(data, service):
    """Count nulls in every kept column by month, vendor (and payment type).

    Args:
        data (pyspark.sql.DataFrame): Raw trips from ``read_service``.
        service (str): ``"yellow"`` or ``"fhvhv"``.

    Returns:
        pandas.DataFrame: One row per group with ``rows`` and a null count
        per column.
    """
    keys = ["file_month", _vendor(service)]
    if service == "yellow":
        keys.append("payment_type")
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
            e.g. ``["VendorID", "payment_type"]``.

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


def flex_fare_check(yellow):
    """Check that the yellow null rows are exactly the Flex Fare trips.

    Args:
        yellow (pyspark.sql.DataFrame): Raw yellow trips.

    Returns:
        pandas.DataFrame: One row per month with ``rows``, ``flex_fare``
        (``payment_type`` 0), ``all_four_null`` (null ``passenger_count``,
        ``RatecodeID``, ``congestion_surcharge`` and ``airport_fee``),
        ``both`` and ``some_null`` (some but not all four null).
    """
    columns = ["passenger_count", "RatecodeID", "congestion_surcharge",
               "airport_fee"]
    nulls = sum(F.col(c).isNull().cast("int") for c in columns)
    flex = F.col("payment_type") == FLEX_FARE
    table = yellow.groupBy("file_month").agg(
        F.count("*").alias("rows"),
        F.sum(flex.cast("int")).alias("flex_fare"),
        F.sum((nulls == 4).cast("int")).alias("all_four_null"),
        F.sum((flex & (nulls == 4)).cast("int")).alias("both"),
        F.sum(((nulls > 0) & (nulls < 4)).cast("int")).alias("some_null"),
    ).orderBy("file_month").toPandas()
    table["flex_share"] = table["flex_fare"] / table["rows"]
    return table


def fare_signs(yellow):
    """Count zero and negative yellow fares by year, vendor and payment type.

    Args:
        yellow (pyspark.sql.DataFrame): Raw yellow trips.

    Returns:
        pandas.DataFrame: ``rows``, ``negative``, ``zero`` and
        ``negative_pct`` for each group.
    """
    table = yellow.groupBy(
        F.year("file_month").alias("year"), "VendorID", "payment_type"
    ).agg(
        F.count("*").alias("rows"),
        F.sum((F.col("fare_amount") < 0).cast("int")).alias("negative"),
        F.sum((F.col("fare_amount") == 0).cast("int")).alias("zero"),
    ).orderBy("year", "VendorID", "payment_type").toPandas()
    table["negative_pct"] = (table["negative"] / table["rows"] * 100).round(2)
    return table


def extra_values(yellow):
    """Count yellow trips by vendor, month, hour, ``extra`` and fee values.

    Args:
        yellow (pyspark.sql.DataFrame): Raw yellow trips.

    Returns:
        pandas.DataFrame: Trip counts for every combination seen, with
        ``weekend`` (Saturday or Sunday pickup).
    """
    pickup = TIME_COLUMNS["yellow"][0]
    return yellow.groupBy(
        "VendorID", "file_month",
        F.hour(pickup).alias("hour"),
        F.dayofweek(pickup).isin(1, 7).alias("weekend"),
        F.round("extra", 2).alias("extra"),
        F.round("congestion_surcharge", 2).alias("congestion_surcharge"),
        F.round("airport_fee", 2).alias("airport_fee"),
        F.round("cbd_congestion_fee", 2).alias("cbd_congestion_fee"),
    ).count().toPandas()


def driver_extra(data):
    """Return the part of yellow ``extra`` that goes to the driver.

    Vendor 1 puts the congestion surcharge, airport fee and usually the CBD
    fee into ``extra``, so those are subtracted: first all three, then
    without the CBD fee if that doesn't leave a driver surcharge. Other
    vendors' ``extra`` is used as it is. If the result isn't one of
    ``DRIVER_EXTRAS`` it is null, so revenue for that trip is unknown.

    Args:
        data (pyspark.sql.DataFrame): Yellow trips.

    Returns:
        pyspark.sql.Column: The driver's surcharges in dollars, or null.
    """
    fees = {c: F.coalesce(F.col(c), F.lit(0.0)) for c in
            ["congestion_surcharge", "airport_fee", "cbd_congestion_fee"]}
    extra = F.col("extra")
    all_fees = F.round(extra - fees["congestion_surcharge"]
                       - fees["airport_fee"] - fees["cbd_congestion_fee"], 2)
    no_cbd = F.round(extra - fees["congestion_surcharge"]
                     - fees["airport_fee"], 2)
    valid = DRIVER_EXTRAS
    return (F.when(F.col("VendorID") != VENDOR_WITH_FEES_IN_EXTRA,
                   F.when(F.round(extra, 2).isin(valid), extra))
            .when(all_fees.isin(valid), all_fees)
            .when(no_cbd.isin(valid), no_cbd))


def extra_rule_coverage(yellow):
    """Count how often ``driver_extra`` finds the driver's surcharges.

    Args:
        yellow (pyspark.sql.DataFrame): Yellow trips.

    Returns:
        pandas.DataFrame: One row per vendor and year with ``rows``,
        ``matched`` (a driver surcharge was found) and ``matched_share``.
    """
    table = yellow.groupBy("VendorID", F.year("file_month").alias("year")).agg(
        F.count("*").alias("rows"),
        F.count(driver_extra(yellow)).alias("matched"),
    ).orderBy("VendorID", "year").toPandas()
    table["matched_share"] = table["matched"] / table["rows"]
    return table


def driver_extra_by_hour(yellow):
    """Share of weekday yellow trips with each driver surcharge, by hour.

    If ``driver_extra`` works, $1 should appear overnight (8pm-6am) and
    $2.50 in the weekday rush hour (4-8pm).

    Args:
        yellow (pyspark.sql.DataFrame): Yellow trips.

    Returns:
        pandas.DataFrame: Rows are driver surcharge amounts, columns are
        pickup hours, values are the share of that hour's matched weekday
        trips in %.
    """
    pickup = TIME_COLUMNS["yellow"][0]
    counts = (yellow.where(~F.dayofweek(pickup).isin(1, 7))
              .withColumn("driver_extra", driver_extra(yellow))
              .where(F.col("driver_extra").isNotNull())
              .groupBy(F.hour(pickup).alias("hour"), "driver_extra")
              .count().toPandas())
    table = counts.pivot(index="driver_extra", columns="hour",
                         values="count").fillna(0)
    return (table / table.sum() * 100).round(1)


def hourly_counts(data, service):
    """Count trips by pickup hour, separately for weekdays and weekends.

    Args:
        data (pyspark.sql.DataFrame): Trips.
        service (str): ``"yellow"`` or ``"fhvhv"``.

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


def cbd_fee_by_hour(data, service):
    """Count 2025 trips by pickup hour and ``cbd_congestion_fee`` value.

    Args:
        data (pyspark.sql.DataFrame): Raw trips.
        service (str): ``"yellow"`` or ``"fhvhv"``.

    Returns:
        pandas.DataFrame: Rows are fee values seen on at least 0.1% of 2025
        trips, columns are pickup hours, values are trip counts.
    """
    pickup = TIME_COLUMNS[service][0]
    counts = (data.where(F.col("file_month") >= "2025-01-01")
              .groupBy(F.hour(pickup).alias("hour"),
                       F.round("cbd_congestion_fee", 2).alias("fee"))
              .count().toPandas())
    table = counts.pivot(index="fee", columns="hour",
                         values="count").fillna(0).astype("int64")
    return table[table.sum(axis=1) >= 0.001 * table.values.sum()]


def distribution_quantiles(data, service):
    """Quantiles of trip length, distance, speed and money on every row.

    Each variable is first counted into fine bins with ``histogram`` (a
    plain group-by count, which needs little memory), and the quantiles are
    read off the cumulative counts, so they are accurate to one bin width.
    ``percentile_approx`` on the 715 million HVFHV rows ran for over 20
    minutes and filled the laptop's memory. Values past the last bin show
    as the bin's upper edge with a ``+`` (e.g. ``"360+"``).

    Args:
        data (pyspark.sql.DataFrame): Trips with ``add_trip_time`` columns.
        service (str): ``"yellow"`` or ``"fhvhv"``.

    Returns:
        pandas.DataFrame: Rows are quantiles, columns are variables.
    """
    # column: (start, stop, bin width)
    bins = {
        "trip_minutes": (0, 360, 0.1),
        _distance(service): (0, 100, 0.05),
        "speed_mph": (0, 100, 0.1),
        _money(service): (-50, 500, 0.25),
    }
    data = data.withColumn("trip_minutes", F.col("trip_seconds") / 60)
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
        service (str): ``"yellow"`` or ``"fhvhv"``.

    Returns:
        list of tuple: (str, str, pyspark.sql.Column).
    """
    pickup, dropoff = TIME_COLUMNS[service]
    distance, money = _distance(service), _money(service)
    in_file_month = (F.trunc(F.to_date(pickup), "month")
                     == F.col("file_month"))
    common_start = [
        ("pickup_in_file_month",
         "Pickup date outside the month of the file (wrong dates)",
         in_file_month),
        ("dropoff_after_pickup",
         "Drop-off time before or equal to pickup time",
         F.col("trip_seconds") > 0),
    ]
    common_end = [
        ("positive_distance", "Zero or negative distance",
         F.col(distance) > 0),
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
                        "(Outside of NYC), which have no location",
         ~F.col("PULocationID").isin(UNKNOWN_ZONES)
         & ~F.col("DOLocationID").isin(UNKNOWN_ZONES)),
    ]
    if service == "yellow":
        flex = F.col("payment_type") == FLEX_FARE
        middle = [
            ("charged_trip",
             "payment_type 3 (no charge), 4 (dispute) or 6 (voided trip): "
             "the fare was not collected or the trip is in doubt",
             ~F.col("payment_type").isin(NO_CHARGE, DISPUTE, VOIDED)),
            ("positive_fare",
             "Zero or negative fare (reversal rows). Flex Fare trips are "
             "kept, because Vendor 2 recorded negative fares on real Flex "
             "Fare trips in 2025; their revenue is set to unknown instead",
             (F.col("fare_amount") > 0) | flex),
            ("known_ratecode", "RatecodeID 99 (null/unknown). Flex Fare "
                               "trips have a null RatecodeID and are kept",
             F.coalesce(F.col("RatecodeID"), F.lit(0))
             != UNKNOWN_RATECODE),
        ]
    else:
        middle = [
            ("positive_driver_pay", "Zero or negative driver pay",
             F.col("driver_pay") > 0),
        ]
    return common_start + middle + common_end


def clean(data, service):
    """Apply every rule in ``rules`` to the trips.

    Args:
        data (pyspark.sql.DataFrame): Raw trips with ``add_trip_time``.
        service (str): ``"yellow"`` or ``"fhvhv"``.

    Returns:
        pyspark.sql.DataFrame: Trips that pass every rule.
    """
    keep = F.lit(True)
    for _, _, condition in rules(service):
        keep = keep & F.coalesce(condition, F.lit(False))
    return data.where(keep)


def step_shapes(data, service, raw_rows, raw_columns):
    """Count rows x columns after each cleaning step, in one pass.

    Args:
        data (pyspark.sql.DataFrame): Raw trips with ``add_trip_time``
            (already cut down to the kept columns).
        service (str): ``"yellow"`` or ``"fhvhv"``.
        raw_rows (int): Rows in the raw files.
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
        {"step": "raw", "reason": "Raw files", "rows": raw_rows,
         "columns": raw_columns},
        {"step": "keep_columns", "reason": "Keep only the columns we need",
         "rows": row["all"], "columns": str(kept_columns)},
    ]
    for name, reason, _ in steps:
        records.append({"step": name, "reason": reason, "rows": row[name],
                        "columns": str(kept_columns)})
    table = pd.DataFrame(records)
    table["removed"] = -table["rows"].diff().fillna(0).astype(int)
    table["removed_pct"] = (table["removed"] / raw_rows * 100).round(3)
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

    Added: ``trip_hours`` (from ``add_trip_time``), ``revenue`` (yellow:
    ``fare_amount`` plus the driver's part of ``extra``; null when that
    can't be worked out), ``earnings`` (``revenue`` for yellow,
    ``driver_pay`` for HVFHV), ``earnings_per_engaged_hour``,
    ``pickup_group``, ``dropoff_group``, ``trip_group``, ``after_toll``,
    ``date``, ``hour``, ``time_band`` and ``day_of_week`` (1 = Monday).

    Earnings per engaged hour is left null for shared HVFHV rides
    (``shared_match_flag`` = Y), because two trips share the same driving
    time, and for yellow trips with unknown revenue.

    Args:
        data (pyspark.sql.DataFrame): Output of ``clean``.
        service (str): ``"yellow"`` or ``"fhvhv"``.
        spark (SparkSession): Active session, to load the zone labels.
        labels (pandas.DataFrame): Output of ``zones.load_zone_labels``.

    Returns:
        pyspark.sql.DataFrame: ``data`` with the new columns.
    """
    pickup = TIME_COLUMNS[service][0]
    if service == "yellow":
        extra = driver_extra(data)
        revenue = F.when((F.col("fare_amount") > 0) & extra.isNotNull(),
                         F.col("fare_amount") + extra)
        data = data.withColumn("revenue", revenue)
        data = data.withColumn("earnings", F.col("revenue"))
        shared = F.lit(False)
    else:
        data = data.withColumn("earnings", F.col("driver_pay"))
        shared = F.col("shared_match_flag") == "Y"

    data = data.withColumn(
        "earnings_per_engaged_hour",
        F.when(~shared, F.try_divide(F.col("earnings"), F.col("trip_hours"))))

    groups = spark.createDataFrame(
        labels.loc[labels["zone_group"].notna(), ["LocationID", "zone_group"]]
        .astype({"LocationID": "int32"}))
    groups = F.broadcast(groups)
    data = (data
            .join(groups.withColumnsRenamed({"LocationID": "PULocationID",
                                             "zone_group": "pickup_group"}),
                  "PULocationID")
            .join(groups.withColumnsRenamed({"LocationID": "DOLocationID",
                                             "zone_group": "dropoff_group"}),
                  "DOLocationID"))
    either = (lambda group: (F.col("pickup_group") == group)
              | (F.col("dropoff_group") == group))
    data = data.withColumn(
        "trip_group",
        F.when(either("cbd"), "treated").when(either("ring"), "ring")
        .otherwise("control"))

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
        pandas.DataFrame: Rows are years, columns trip groups, values the
        share of trips (%) whose ``earnings_per_engaged_hour`` is not null.
    """
    table = trips.groupBy(F.year("date").alias("year"), "trip_group").agg(
        (F.count("earnings_per_engaged_hour") / F.count("*") * 100)
        .alias("known_pct")).toPandas()
    return table.pivot(index="year", columns="trip_group",
                       values="known_pct")[["treated", "ring", "control"]]


def flex_fare_by_group(yellow):
    """Share of Flex Fare trips by year and trip group, after cleaning.

    If the share grows much faster in one group, Flex Fare's growth could
    bias the before/after comparison, which goes in Limitations.

    Args:
        yellow (pyspark.sql.DataFrame): Output of ``add_columns`` (yellow).

    Returns:
        pandas.DataFrame: Rows are years, columns trip groups, values the
        Flex Fare share of trips in %.
    """
    table = yellow.groupBy(F.year("date").alias("year"), "trip_group").agg(
        F.avg((F.col("payment_type") == FLEX_FARE).cast("int") * 100)
        .alias("flex_fare_pct")).toPandas()
    return table.pivot(index="year", columns="trip_group",
                       values="flex_fare_pct")[["treated", "ring", "control"]]


# ---------------------------------------------------------------------------
# 4. Summary table
# ---------------------------------------------------------------------------
def summarise(data, service):
    """Group trips into service x pickup zone x drop-off group x date x band.

    Medians use ``percentile_approx`` with accuracy 1,000 (within 0.1% of
    the true rank), computed on every trip in the group.

    Args:
        data (pyspark.sql.DataFrame): Output of ``add_columns``.
        service (str): ``"yellow"`` or ``"fhvhv"``.

    Returns:
        pyspark.sql.DataFrame: One row per group that has trips.
    """
    def median(column):
        return F.percentile_approx(column, 0.5, 1000)

    return (data
            .groupBy("PULocationID", "dropoff_group", "date", "time_band")
            .agg(F.count("*").alias("trips"),
                 F.count("earnings").alias("trips_with_earnings"),
                 F.sum("earnings").alias("total_earnings"),
                 median("earnings").alias("median_earnings"),
                 median("earnings_per_engaged_hour")
                 .alias("median_earnings_per_engaged_hour"),
                 F.sum("trip_hours").alias("engaged_hours"),
                 median("speed_mph").alias("median_speed_mph"))
            .withColumn("service", F.lit(service)))


def summarise_month(spark, service, year, month, labels, overwrite=False):
    """Clean and summarise one month of one service, saving the result.

    Months are independent (every summary row has one date), so doing one
    month at a time keeps Spark's memory use small. Each month is saved to
    ``data/curated/summary_parts/`` and reused on the next run unless
    ``overwrite`` is true.

    Args:
        spark (SparkSession): Active session.
        service (str): ``"yellow"`` or ``"fhvhv"``.
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


def build_summary(spark, labels, months, services=("yellow", "fhvhv"),
                  overwrite=False):
    """Build the summary table, with zero rows for groups with no trips.

    Every service x pickup zone (1-263) x drop-off group x date x time band
    gets a row, so days with no trips count as 0 instead of being missing.
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

    zones = labels.loc[labels["zone_group"].notna(),
                       ["LocationID", "zone_group"]]
    dates = pd.concat([pd.Series(pd.date_range(
        f"{year}-{month:02d}-01", periods=pd.Period(
            f"{year}-{month:02d}").days_in_month)) for year, month in months])
    grid = (pd.DataFrame({"service": list(services)})
            .merge(zones.rename(columns={"LocationID": "PULocationID",
                                         "zone_group": "pickup_group"}),
                   how="cross")
            .merge(pd.DataFrame({"dropoff_group": ZONE_GROUPS}), how="cross")
            .merge(pd.DataFrame({"date": dates.dt.date}), how="cross")
            .merge(pd.DataFrame({"time_band": TIME_BAND_ORDER}), how="cross"))

    keys = ["service", "PULocationID", "dropoff_group", "date", "time_band"]
    table = grid.merge(found, on=keys, how="left")
    for column in ["trips", "trips_with_earnings", "total_earnings",
                   "engaged_hours"]:
        table[column] = table[column].fillna(0)
    table[["trips", "trips_with_earnings"]] = (
        table[["trips", "trips_with_earnings"]].astype("int64"))

    either = (lambda group: (table["pickup_group"] == group)
              | (table["dropoff_group"] == group))
    table["trip_group"] = "control"
    table.loc[either("ring"), "trip_group"] = "ring"
    table.loc[either("cbd"), "trip_group"] = "treated"
    table["date"] = pd.to_datetime(table["date"])
    table["after_toll"] = table["date"] >= pd.Timestamp(config.TOLL_START_DATE)
    table["day_of_week"] = table["date"].dt.dayofweek + 1
    table["time_band"] = pd.Categorical(table["time_band"],
                                        TIME_BAND_ORDER, ordered=True)
    table = table.sort_values(keys).reset_index(drop=True)
    table.to_parquet(SUMMARY_FILE, index=False)
    return table
