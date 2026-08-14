"""Spark job: clean TLC yellow taxi Parquet using QUALITY_RULES."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import SparkSession

from data_quality import DEFAULT_TRIPS, REPO_ROOT
from spark.session import get_spark_session
from spark.transformations.taxi_cleaning import apply_cleaning_rules

DEFAULT_SPARK_CLEANED = (
    REPO_ROOT / "data" / "sample" / "yellow_tripdata_2025-01.spark_cleaned"
)


def build_spark(app_name: str = "clean_taxi_data") -> SparkSession:
    return get_spark_session(app_name, master="local[*]")


def run(
    input_path: str,
    output_path: str,
    spark_session: SparkSession | None = None,
) -> dict:
    own_session = spark_session is None
    session = spark_session if spark_session is not None else build_spark()
    try:
        df = session.read.parquet(input_path)
        cleaned, stats = apply_cleaning_rules(df, input_path)
        (
            cleaned.write.mode("overwrite")
            .option("compression", "snappy")
            .parquet(output_path)
        )
        stats["output_path"] = output_path
        return stats
    finally:
        if own_session:
            session.stop()


def print_stats(stats: dict) -> None:
    print("SPARK CLEANING SUMMARY")
    print("======================")
    print(f"Input rows:    {stats['input_rows']:,}")
    print(f"Rows removed:  {stats['rows_removed']:,}")
    print(f"Output rows:   {stats['output_rows']:,}")
    print(f"Rows flagged:  {stats['rows_flagged']:,}")
    print("\nRemoved by rule:")
    for name, count in stats["removed"].items():
        print(f"  {name}: {count:,}")
    print("\nSet to NULL:")
    for name, count in stats["set_null"].items():
        print(f"  {name}: {count:,}")
    print("\nFlagged (kept):")
    for name, count in stats["flagged"].items():
        print(f"  {name}: {count:,}")
    if stats.get("duplicate_elapsed_seconds") is not None:
        print(
            f"\nDuplicate drop: {stats['removed'].get('exact_duplicate_rows', 0):,} rows "
            f"in {stats['duplicate_elapsed_seconds']}s"
        )
    if stats.get("output_path"):
        print(f"\nWrote {stats['output_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean TLC trip Parquet with PySpark using the same rules "
            "as data_quality.clean."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_TRIPS),
        help="Input Parquet file or directory.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SPARK_CLEANED),
        help="Output Parquet directory.",
    )
    args = parser.parse_args()
    stats = run(str(Path(args.input)), str(Path(args.output)))
    print_stats(stats)


if __name__ == "__main__":
    main()
