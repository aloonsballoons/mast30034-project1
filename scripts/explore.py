"""Tables behind the exploration and maps in Section 5 of ``main.ipynb``.

Two kinds of function:

1. Spark checks on trip-level data (``removals_by_group``,
   ``histograms``, ``zone_medians``, ``pickup_wait``).
   Each makes one pass over the data it is given and returns a small pandas
   table.
2. pandas tables built from the summary table. They compare the toll
   period (5 January to 31 December 2025) with the same dates a year
   earlier (``PERIODS``), so both periods cover the same season and the
   four pre-toll days of January 2025 are left out. ``speed_change`` does
   the same for the MTA's monthly speeds, by month.
"""

from datetime import date

import pandas as pd
from pyspark.sql import functions as F

from scripts import config
from scripts.clean import (COARSE_TRIP_GROUPS, TIME_BAND_ORDER, TIME_COLUMNS,
                           UNKNOWN_ZONES, _money, rules)

# The toll period and the same dates a year earlier
PERIODS = {
    "before": (config.TOLL_START_DATE.replace(year=2024), date(2024, 12, 31)),
    "after": (config.TOLL_START_DATE, date(2025, 12, 31)),
}

# Zones with fewer pickups a day than this before the toll are greyed out on
# the maps, because their percentage changes rest on too few trips
MIN_DAILY_PICKUPS = 10

# The pickup groups ``engaged_hour_by_group`` reports, coarsened from the
# five zone groups of ``clean.ZONE_GROUPS``
PICKUP_GROUPS = ["cbd", "ring", "control"]

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


def _coarse_pickup_group(column):
    """Spark rule turning a fine pickup zone group into one of
    ``PICKUP_GROUPS``, or null for zones 264 and 265."""
    return (F.when(column == "cbd", "cbd")
            .when(column.startswith("ring"), "ring")
            .when(column.startswith("control"), "control"))


def _coarse_trip_group(pickup, dropoff):
    """Spark rule for the three trip groups, with ``unknown_zone`` for the
    trips that have an end in zone 264 or 265 and don't touch the CBD or
    ring."""
    return (F.when((pickup == "cbd") | (dropoff == "cbd"), "treated")
            .when((pickup == "ring") | (dropoff == "ring"), "ring")
            .when((pickup == "control") & (dropoff == "control"), "control")
            .otherwise("unknown_zone"))


def _join_groups(data, spark, labels):
    """Left-join the coarse zone group onto both ends of every trip."""
    groups = F.broadcast(spark.createDataFrame(
        labels.loc[labels["zone_group"].notna(), ["LocationID", "zone_group"]]
        .astype({"LocationID": "int32"})))
    return (data
            .join(groups.withColumnsRenamed({"LocationID": "PULocationID",
                                             "zone_group": "pickup_group"}),
                  "PULocationID", "left")
            .join(groups.withColumnsRenamed({"LocationID": "DOLocationID",
                                             "zone_group": "dropoff_group"}),
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
        service (str): ``"fhvhv"``.
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

    ``money`` is ``driver_pay``, and ``money_per_hour`` is money divided by
    trip hours. Both exist before and after cleaning, so the raw and
    cleaned histograms measure the same thing.

    Args:
        data (pyspark.sql.DataFrame): Trips with ``add_trip_time``.
        service (str): ``"fhvhv"``.

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
    counts = nulls.select_dtypes("number").groupby(year).sum()
    shares = counts.drop(columns="rows").div(counts["rows"], axis=0) * 100
    return shares.loc[:, (shares > 0).any()]


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


def engaged_hour_by_group(fhvhv):
    """What an engaged hour buys in each pickup group, before and after.

    An engaged hour is the time from pickup to drop-off, so earnings per
    engaged hour rise if the same hour covers more ground. This puts the
    parts of that on one row: the speed and distance of the trips
    themselves, the earnings they pay, and how much of the per-hour figure
    is missing. It tests the speed explanation on the same trips that
    produced the earnings change, rather than on an outside average.

    Medians are taken over every trip in the group (``percentile_approx``
    with accuracy 1,000). Trips picked up in zones 264 and 265 have no
    pickup group and are left out.

    Args:
        fhvhv (pyspark.sql.DataFrame): Output of ``clean.add_columns``
            (HVFHV).

    Returns:
        pandas.DataFrame: One row per ``pickup_group`` and ``period``, with
        ``trips``, ``shared_pct`` (trips sharing their driving time with
        another trip, which have no engaged hour of their own),
        ``eph_known_pct`` (trips whose earnings per engaged hour is known),
        ``median_speed_mph``, ``median_trip_miles``,
        ``median_trip_minutes`` and ``median_earnings_per_engaged_hour``.
    """
    # A null flag is not a shared ride, but ``clean.add_columns`` leaves
    # those trips without an engaged hour too, so eph_known_pct is what
    # the per-hour medians are actually computed on
    shared = F.coalesce((F.col("shared_match_flag") == "Y").cast("int"),
                        F.lit(0))
    table = (fhvhv
             .withColumn("pickup_group",
                         _coarse_pickup_group(F.col("pickup_group")))
             .withColumn("period", _period(F.col("date")))
             .where(F.col("pickup_group").isNotNull()
                    & F.col("period").isNotNull())
             .groupBy("pickup_group", "period")
             .agg(F.count("*").alias("trips"),
                  (F.avg(shared) * 100).alias("shared_pct"),
                  (F.count("earnings_per_engaged_hour") / F.count("*") * 100)
                  .alias("eph_known_pct"),
                  F.percentile_approx("speed_mph", 0.5, 1000)
                  .alias("median_speed_mph"),
                  F.percentile_approx("trip_miles", 0.5, 1000)
                  .alias("median_trip_miles"),
                  F.percentile_approx(F.col("trip_seconds") / 60, 0.5, 1000)
                  .alias("median_trip_minutes"),
                  F.percentile_approx("earnings_per_engaged_hour", 0.5, 1000)
                  .alias("median_earnings_per_engaged_hour"))
             .toPandas())
    order = pd.MultiIndex.from_product([PICKUP_GROUPS, list(PERIODS)],
                                       names=["pickup_group", "period"])
    return table.set_index(["pickup_group", "period"]).reindex(order)


def engaged_hour_change(table):
    """Each measure of ``engaged_hour_by_group`` before and after the toll.

    Args:
        table (pandas.DataFrame): Output of ``engaged_hour_by_group``.

    Returns:
        pandas.DataFrame: One row per measure and pickup group, with
        ``before``, ``after`` and ``change_pct``.
    """
    wide = table.unstack("period")
    rows = [pd.DataFrame({"measure": measure,
                          "pickup_group": wide.index,
                          "before": wide[(measure, "before")].to_numpy(),
                          "after": wide[(measure, "after")].to_numpy()})
            for measure in table.columns]
    changes = pd.concat(rows, ignore_index=True)
    changes["change_pct"] = _pct_change(changes["before"], changes["after"])
    return changes.set_index(["measure", "pickup_group"])


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


def monthly_ratio(relative, service, group="treated"):
    """One group's monthly ratio laid out as month of year by year.

    The ratio has a season in it even after dividing by control trips, so
    a month of the toll year is only worth reading against the same month
    of the years before it.

    Args:
        relative (pandas.DataFrame): Output of ``relative_trips``.
        service (str): ``"fhvhv"``.
        group (str): ``"treated"`` or ``"ring"``.

    Returns:
        pandas.DataFrame: Rows are months of the year (1-12), columns
        years, values the indexed ratio.
    """
    rows = relative[relative["service"] == service]
    month_of_year = rows["month"].dt.month.rename("month_of_year")
    return rows.pivot_table(index=month_of_year,
                            columns=rows["month"].dt.year.rename("year"),
                            values=group)


def monthly_group_trips(summary, service):
    """Trips a day by month and coarse group, and the year-on-year change.

    Args:
        summary (pandas.DataFrame): The summary table.
        service (str): ``"fhvhv"``.

    Returns:
        pandas.DataFrame: One row per month, with a column of trips a day
        for each coarse group and a ``_change_pct`` column giving the
        change from the same month a year earlier.
    """
    rows = summary[summary["service"] == service]
    months = rows["date"].dt.to_period("M").rename("month")
    table = (rows.groupby([months, "trip_group_coarse"])["trips"].sum()
             .unstack("trip_group_coarse")[COARSE_TRIP_GROUPS])
    table = table.div(rows.groupby(months)["date"].nunique(), axis=0)
    for group in COARSE_TRIP_GROUPS:
        table[f"{group}_change_pct"] = _pct_change(table[group].shift(12),
                                                   table[group])
    table.index = table.index.to_timestamp()
    return table


def zone_correlation(changes, labels):
    """Spearman correlation of a zone's two changes, overall and by group.

    Args:
        changes (pandas.DataFrame): One service's rows of ``zone_change``.
        labels (pandas.DataFrame): Zone labels, for ``zone_group``.

    Returns:
        pandas.DataFrame: ``zones`` and ``spearman`` for all the zones
        shown and for each zone group.
    """
    rows = (changes[changes["enough_trips"]]
            .join(labels.set_index("LocationID")["zone_group"],
                  on="PULocationID")
            .dropna(subset=["change_pct", "eph_change_pct"]))
    groups = {"all shown": rows}
    for group in ["cbd", "ring", "control"]:
        groups[group] = rows[rows["zone_group"] == group]
    return pd.DataFrame(
        [{"group": name,
          "zones": len(part),
          "spearman": part["change_pct"].corr(part["eph_change_pct"],
                                              method="spearman")}
         for name, part in groups.items()]).set_index("group")


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


def zone_change(summary, medians):
    """Change in pickups and earnings per engaged hour by pickup zone.

    Args:
        summary (pandas.DataFrame): The summary table.
        medians (pandas.DataFrame): Output of ``zone_medians`` for the same
            service.

    Returns:
        pandas.DataFrame: One row per service and pickup zone with
        ``before``, ``after``, ``change_pct``, ``daily_before``,
        ``enough_trips`` (at least ``MIN_DAILY_PICKUPS`` a day before the
        toll), ``eph_before``, ``eph_after`` and ``eph_change_pct``.
    """
    # Zones 264 and 265 have no shape and aren't one place, so they are
    # left out of the zone table
    table = summary.loc[~summary["PULocationID"].isin(UNKNOWN_ZONES),
                        ["service", "PULocationID", "date", "trips"]]
    table = add_period(table).dropna(subset=["period"])
    days = table.groupby("period")["date"].nunique()
    zones = (table.groupby(["service", "PULocationID", "period"])["trips"]
             .sum().unstack("period").reset_index())
    zones["change_pct"] = _pct_change(zones["before"], zones["after"])
    zones["daily_before"] = zones["before"] / days["before"]
    zones["enough_trips"] = zones["daily_before"] >= MIN_DAILY_PICKUPS
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
        service (str): ``"fhvhv"``.
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
        service (str): ``"fhvhv"``.
        group (str): Trip group.

    Returns:
        pandas.DataFrame: Rows are time bands, columns days of the week.
    """
    table = changes.loc[(service, group), "change_pct"].unstack("day_of_week")
    return table.reindex([b for b in TIME_BAND_ORDER if b in table.index])


def speed_change(path):
    """MTA average speed by area, 2025 against the same months of 2024.

    The MTA publishes one average speed a month for each area, so the
    comparison runs over the months of 2025 in the file (January includes
    the four days before the toll).

    Args:
        path (Path): The MTA CBD taxi/FHV speeds file.

    Returns:
        pandas.DataFrame: One row per ``zone`` (area), with the average
        ``before_mph`` and ``after_mph`` and ``change_pct``.
    """
    speeds = pd.read_csv(path, parse_dates=["month"])
    speeds["year"] = speeds["month"].dt.year
    months = speeds.loc[speeds["year"] == 2025, "month"].dt.month.unique()
    same = speeds[speeds["year"].isin([2024, 2025])
                  & speeds["month"].dt.month.isin(months)]
    table = (same.pivot_table(index="zone", columns="year",
                              values="zonal_speed", aggfunc="mean")
             .set_axis(["before_mph", "after_mph"], axis=1))
    table["change_pct"] = _pct_change(table["before_mph"],
                                      table["after_mph"])
    return table
