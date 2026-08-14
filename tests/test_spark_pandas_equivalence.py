"""Cross-engine check: Spark cleaning must match pandas cleaning on the sample month."""

# pylint: disable=redefined-outer-name
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from data_quality import DEFAULT_TRIPS
from data_quality.clean import apply_rules
from data_quality.profile import load_trips
from data_quality.rules import DROPOFF_COL, PICKUP_COL
from spark.transformations.taxi_cleaning import apply_cleaning_rules

RAW_PATH = Path(DEFAULT_TRIPS)

pytestmark = pytest.mark.skipif(
    not RAW_PATH.exists(),
    reason=f"Missing sample file: {RAW_PATH}",
)


def _flag_counts(series: pd.Series) -> dict[str, int]:
    return series.fillna("__none__").astype(str).value_counts().sort_index().to_dict()


def test_spark_pandas_cleaning_equivalence(spark_session: SparkSession) -> None:
    pandas_in = load_trips(RAW_PATH)
    pandas_out, pandas_stats = apply_rules(pandas_in, RAW_PATH)

    spark_in = spark_session.read.parquet(str(RAW_PATH))
    spark_out, spark_stats = apply_cleaning_rules(spark_in, RAW_PATH)

    assert spark_stats["input_rows"] == pandas_stats["input_rows"]
    assert spark_stats["output_rows"] == pandas_stats["output_rows"]
    assert spark_stats["rows_removed"] == pandas_stats["rows_removed"]
    assert spark_stats["output_rows"] == pandas_stats["input_rows"] - pandas_stats["rows_removed"]

    pandas_cols = set(pandas_out.columns)
    spark_cols = set(spark_out.columns)
    assert pandas_cols == spark_cols

    assert spark_stats["removed"].get("trip_distance:below_min") == pandas_stats["removed"].get(
        "trip_distance:below_min"
    )
    assert spark_stats["removed"].get("dropoff_before_pickup") == pandas_stats["removed"].get(
        "dropoff_before_pickup"
    )
    assert spark_stats["removed"].get("exact_duplicate_rows", 0) == pandas_stats["removed"].get(
        "exact_duplicate_rows", 0
    )
    assert spark_stats["set_null"].get("passenger_count:below_min") == pandas_stats["set_null"].get(
        "passenger_count:below_min"
    )
    assert spark_stats["flagged"].get("fare_amount:below_min") == pandas_stats["flagged"].get(
        "fare_amount:below_min"
    )

    pandas_neg_fare = int((pandas_out["fare_amount"] < 0).sum())
    spark_neg_fare = spark_out.filter(F.col("fare_amount") < 0).count()
    assert spark_neg_fare == pandas_neg_fare
    assert spark_neg_fare > 0

    pandas_passenger_nulls = int(pandas_out["passenger_count"].isna().sum())
    spark_passenger_nulls = spark_out.filter(F.col("passenger_count").isNull()).count()
    assert spark_passenger_nulls == pandas_passenger_nulls
    assert spark_out.filter(F.col("passenger_count") <= 0).count() == 0

    spark_inverted = spark_out.filter(
        F.to_timestamp(F.col(DROPOFF_COL)) < F.to_timestamp(F.col(PICKUP_COL))
    ).count()
    pandas_inverted = int(
        (
            pd.to_datetime(pandas_out[DROPOFF_COL], errors="coerce")
            < pd.to_datetime(pandas_out[PICKUP_COL], errors="coerce")
        ).sum()
    )
    assert spark_inverted == 0
    assert pandas_inverted == 0

    spark_flag_rows = (
        spark_out.groupBy("dq_flags")
        .count()
        .toPandas()
        .set_index("dq_flags")["count"]
        .sort_index()
    )
    spark_flag_dict = {
        ("__none__" if pd.isna(idx) else str(idx)): int(val) for idx, val in spark_flag_rows.items()
    }
    assert spark_flag_dict == _flag_counts(pandas_out["dq_flags"])
