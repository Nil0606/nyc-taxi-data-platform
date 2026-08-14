from pathlib import Path

import pandas as pd

from data_quality.clean import apply_rules
from data_quality.profile import evaluate_column, evaluate_cross_field
from data_quality.rules import QUALITY_RULES

JAN_FILE = Path("data/sample/yellow_tripdata_2025-01.parquet")


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_distance": [1.5, 0.0, -1.0, 250.0, 2.0],
            "passenger_count": [1, 0, None, 12, 1],
            "fare_amount": [12.5, 10.0, 8.0, 900.0, -10.0],
            "total_amount": [15.0, 12.0, 10.0, 920.0, -12.0],
            "RatecodeID": [1, None, 1, 1, 1],
            "Airport_fee": [None, 1.75, 0.0, 0.0, 0.0],
            "tpep_pickup_datetime": [
                "2025-01-01 10:00:00",
                "2025-01-01 11:00:00",
                "2025-01-01 12:00:00",
                "2024-12-31 23:00:00",
                "2025-01-01 13:00:00",
            ],
            "tpep_dropoff_datetime": [
                "2025-01-01 10:20:00",
                "2025-01-01 10:50:00",
                "2025-01-01 12:10:00",
                "2024-12-31 23:30:00",
                "2025-01-01 13:20:00",
            ],
            "payment_type": [1, 4, 1, 1, 4],
        }
    )


def test_trip_distance_zero_is_invalid() -> None:
    result = evaluate_column(_sample(), "trip_distance", QUALITY_RULES["trip_distance"])
    assert result["below_min"] == 2
    assert result["above_max"] == 1


def test_passenger_count_zero_and_over_max() -> None:
    result = evaluate_column(_sample(), "passenger_count", QUALITY_RULES["passenger_count"])
    assert result["below_min"] == 1
    assert result["above_max"] == 1
    assert result["missing"] == 1


def test_negative_fare_is_flagged_not_removed() -> None:
    df = _sample()
    result = evaluate_column(df, "fare_amount", QUALITY_RULES["fare_amount"])
    assert result["below_min"] == 1
    cleaned, stats = apply_rules(df, JAN_FILE)
    assert stats["flagged"]["fare_amount:below_min"] == 1
    assert "fare_amount:below_min" not in stats["removed"]
    assert (cleaned["fare_amount"] < 0).sum() == 1


def test_dropoff_before_pickup_and_zero_distance_removed() -> None:
    df = _sample()
    cleaned, stats = apply_rules(df, JAN_FILE)
    assert stats["removed"]["dropoff_before_pickup"] == 1
    assert stats["removed"]["trip_distance:below_min"] == 2
    # rows 0, 3, 4 remain (1 and 2 removed for distance; row 1 also inverted)
    assert len(cleaned) == 3


def test_cross_field_outside_month() -> None:
    df = _sample()
    cross = evaluate_cross_field(df, JAN_FILE)
    assert cross["pickup_outside_file_month"]["invalid"] == 1
    assert cross["dropoff_before_pickup"]["invalid"] == 1
