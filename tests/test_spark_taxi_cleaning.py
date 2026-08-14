# pylint: disable=redefined-outer-name
from collections.abc import Iterator
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from spark.session import get_spark_session
from spark.transformations.taxi_cleaning import apply_cleaning_rules, validate_schema

JAN_FILE = Path("data/sample/yellow_tripdata_2025-01.parquet")


@pytest.fixture(scope="module")
def spark_session() -> Iterator[SparkSession]:
    session = get_spark_session("test_clean_taxi", master="local[1]")
    yield session
    session.stop()


def _sample(session: SparkSession):
    return session.createDataFrame(
        [
            (1.5, 1, 12.5, 15.0, 1, "2025-01-01 10:00:00", "2025-01-01 10:20:00", 1),
            (0.0, 0, 10.0, 12.0, None, "2025-01-01 11:00:00", "2025-01-01 10:50:00", 4),
            (-1.0, None, 8.0, 10.0, 1, "2025-01-01 12:00:00", "2025-01-01 12:10:00", 1),
            (250.0, 12, 900.0, 920.0, 1, "2024-12-31 23:00:00", "2024-12-31 23:30:00", 1),
            (2.0, 1, -10.0, -12.0, 1, "2025-01-01 13:00:00", "2025-01-01 13:20:00", 4),
        ],
        schema=[
            "trip_distance",
            "passenger_count",
            "fare_amount",
            "total_amount",
            "RatecodeID",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "payment_type",
        ],
    )


def test_schema_requires_core_columns(spark_session: SparkSession) -> None:
    df = spark_session.createDataFrame([(1,)], ["trip_distance"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)


def test_negative_fare_is_flagged_not_removed(spark_session: SparkSession) -> None:
    cleaned, stats = apply_cleaning_rules(_sample(spark_session), JAN_FILE)
    assert stats["flagged"]["fare_amount:below_min"] == 1
    assert "fare_amount:below_min" not in stats["removed"]
    assert cleaned.filter(cleaned.fare_amount < 0).count() == 1


def test_dropoff_before_pickup_and_zero_distance_removed(spark_session: SparkSession) -> None:
    cleaned, stats = apply_cleaning_rules(_sample(spark_session), JAN_FILE)
    assert stats["removed"]["dropoff_before_pickup"] == 1
    assert stats["removed"]["trip_distance:below_min"] == 2
    assert cleaned.count() == 3


def test_passenger_count_invalid_set_to_null(spark_session: SparkSession) -> None:
    cleaned, stats = apply_cleaning_rules(_sample(spark_session), JAN_FILE)
    assert stats["set_null"]["passenger_count:below_min"] == 1
    assert stats["set_null"]["passenger_count:above_max"] == 1
    remaining = cleaned.filter(cleaned.trip_distance == 250.0).collect()[0]
    assert remaining.passenger_count is None


def test_outside_month_flagged(spark_session: SparkSession) -> None:
    cleaned, stats = apply_cleaning_rules(_sample(spark_session), JAN_FILE)
    assert stats["flagged"]["pickup_outside_file_month"] == 1
    flagged = cleaned.filter(cleaned.dq_flags.contains("pickup_outside_file_month"))
    assert flagged.count() == 1
