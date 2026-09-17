"""Start Spark and read the raw TLC files into one consistent schema.

The monthly files don't all share a schema. January 2023 stores the zone IDs
as int64 (int32 afterwards), and ``cbd_congestion_fee`` only exists from
2025. Reading every file at once would fail or mix types, so each month is
read and cast on its own, then the months are stacked.
"""

import os

from pyspark.sql import SparkSession, functions as F

from scripts import config
from scripts.download import tlc_path

# Columns kept from the service, with the type every month is cast to.
# Everything else in the raw files is dropped at the first step.
FHVHV_COLUMNS = {
    "hvfhs_license_num": "string",
    "request_datetime": "timestamp_ntz",
    "pickup_datetime": "timestamp_ntz",
    "dropoff_datetime": "timestamp_ntz",
    "PULocationID": "int",
    "DOLocationID": "int",
    "trip_miles": "double",
    "cbd_congestion_fee": "double",
    "driver_pay": "double",
    "shared_match_flag": "string",
}

COLUMNS = {"fhvhv": FHVHV_COLUMNS}

# The pickup and drop-off time columns of the service
TIME_COLUMNS = {
    "fhvhv": ("pickup_datetime", "dropoff_datetime"),
}


def get_spark(memory=None):
    """Start (or reuse) a local Spark session for the pipeline.

    Timestamps in the TLC files are New York local times with no time zone,
    so they are read as ``timestamp_ntz`` and the session time zone is set to
    New York to match.

    Args:
        memory (str, optional): Driver memory. In local mode the driver does
            all the work, so this is the memory Spark can use. Defaults to
            the ``SPARK_DRIVER_MEMORY`` environment variable, or ``"2g"``,
            which was enough for every step on a 16 GB laptop.

    Returns:
        pyspark.sql.SparkSession: The session.
    """
    memory = memory or os.environ.get("SPARK_DRIVER_MEMORY", "2g")
    if "JAVA_HOME" not in os.environ:
        print("JAVA_HOME is not set. If Spark fails to start, point it at "
              "Java 17 or 21 (see README.md).")
    return (SparkSession.builder
            .master("local[*]")
            .appName("congestion-pricing")
            .config("spark.driver.memory", memory)
            .config("spark.sql.session.timeZone", "America/New_York")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.sql.parquet.inferTimestampNTZ.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.local.dir", str(config.ROOT_DIR / "spark-tmp"))
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def read_month(spark, service, year, month):
    """Read one raw TLC file with the kept columns in the shared schema.

    Columns that don't exist in this month (``cbd_congestion_fee`` before
    2025) are added as nulls. ``file_month``, the first day of the file's
    month, is added so later steps know which file each row came from.

    Args:
        spark (SparkSession): Active session.
        service (str): ``"fhvhv"``.
        year (int): Year.
        month (int): Month (1-12).

    Returns:
        pyspark.sql.DataFrame: The month's rows.
    """
    raw = spark.read.parquet(str(tlc_path(service, year, month)))
    # Match column names without caring about case
    names = {name.lower(): name for name in raw.columns}
    columns = []
    for name, dtype in COLUMNS[service].items():
        if name.lower() in names:
            column = F.col(names[name.lower()]).cast(dtype)
        else:
            column = F.lit(None).cast(dtype)
        columns.append(column.alias(name))
    columns.append(F.lit(f"{year}-{month:02d}-01").cast("date")
                   .alias("file_month"))
    return raw.select(columns)


def read_service(spark, service, months):
    """Read and stack every month of one service.

    Args:
        spark (SparkSession): Active session.
        service (str): ``"fhvhv"``.
        months (list of tuple): (year, month) pairs.

    Returns:
        pyspark.sql.DataFrame: All months in the shared schema.
    """
    frames = [read_month(spark, service, year, month)
              for year, month in months]
    data = frames[0]
    for frame in frames[1:]:
        data = data.unionByName(frame)
    return data
