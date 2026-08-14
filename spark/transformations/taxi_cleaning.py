"""Spark transformations that apply QUALITY_RULES from data_quality.rules."""

from __future__ import annotations

import time
from pathlib import Path

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from data_quality.profile import file_month
from data_quality.rules import (
    CROSS_FIELD_RULES,
    DROPOFF_COL,
    PICKUP_COL,
    QUALITY_RULES,
)

REQUIRED_COLUMNS = (
    PICKUP_COL,
    DROPOFF_COL,
    "trip_distance",
    "passenger_count",
    "fare_amount",
    "total_amount",
)


def validate_schema(df: DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _below_min(column: str, spec: dict) -> Column:
    if "min" not in spec:
        return F.lit(False)
    col = F.col(column)
    if spec.get("min_inclusive", True):
        return col.isNotNull() & (col < spec["min"])
    return col.isNotNull() & (col <= spec["min"])


def _above_max(column: str, spec: dict) -> Column:
    if "max" not in spec:
        return F.lit(False)
    col = F.col(column)
    if spec.get("max_inclusive", True):
        return col.isNotNull() & (col > spec["max"])
    return col.isNotNull() & (col >= spec["max"])


def _not_allowed(column: str, spec: dict) -> Column:
    allowed = spec.get("allowed")
    if not allowed:
        return F.lit(False)
    col = F.col(column)
    in_domain = col.isin(*list(allowed))
    return col.isNotNull() & (in_domain == F.lit(False))


def drop_exact_duplicates(df: DataFrame) -> tuple[DataFrame, dict]:
    """Remove exact duplicate rows with Spark's hash aggregate, not a full-width window.

    Applied as a separate step so it can be timed independently of quality-rule
    filters. ``dropDuplicates`` keeps one copy of each identical row.
    """
    started = time.perf_counter()
    before = df.count()
    deduped = df.dropDuplicates()
    after = deduped.count()
    return deduped, {
        "input_rows": before,
        "output_rows": after,
        "removed": before - after,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _quality_masks(
    df: DataFrame, source_path: str | Path
) -> tuple[dict[str, tuple[str, Column]], dict[str, Column], Column, list[Column]]:
    """Build mask expressions against ``df`` without adding intermediate columns."""
    stat_masks: dict[str, tuple[str, Column]] = {}
    set_null_by_column: dict[str, Column] = {}
    drop = F.lit(False)
    flag_parts: list[Column] = []

    for column, spec in QUALITY_RULES.items():
        if column not in df.columns:
            continue
        checks = (
            (_below_min(column, spec), spec.get("on_below_min"), f"{column}:below_min"),
            (_above_max(column, spec), spec.get("on_above_max"), f"{column}:above_max"),
            (_not_allowed(column, spec), spec.get("on_violation"), f"{column}:domain"),
        )
        set_null_mask = F.lit(False)
        applied_set_null = False
        for mask, action, label in checks:
            action = action or spec.get("on_violation")
            if action is None:
                continue
            if action == "remove":
                drop = drop | mask
                stat_masks[label] = ("removed", mask)
            elif action == "set_null":
                applied_set_null = True
                set_null_mask = set_null_mask | mask
                stat_masks[label] = ("set_null", mask)
            elif action == "flag":
                flag_parts.append(F.when(mask, F.lit(label)))
                stat_masks[label] = ("flagged", mask)
        if applied_set_null:
            set_null_by_column[column] = set_null_mask

    if PICKUP_COL in df.columns and DROPOFF_COL in df.columns:
        pickup = F.to_timestamp(F.col(PICKUP_COL))
        dropoff = F.to_timestamp(F.col(DROPOFF_COL))
        inverted = dropoff.isNotNull() & pickup.isNotNull() & (dropoff < pickup)
        if CROSS_FIELD_RULES["dropoff_before_pickup"]["on_violation"] == "remove":
            drop = drop | inverted
            stat_masks["dropoff_before_pickup"] = ("removed", inverted)

        month = file_month(Path(source_path))
        if month and CROSS_FIELD_RULES["pickup_outside_file_month"]["on_violation"] == "flag":
            year, mon = month
            outside = pickup.isNotNull() & (
                (F.year(pickup) != year) | (F.month(pickup) != mon)
            )
            flag_parts.append(F.when(outside, F.lit("pickup_outside_file_month")))
            stat_masks["pickup_outside_file_month"] = ("flagged", outside)

    return stat_masks, set_null_by_column, drop, flag_parts


def apply_cleaning_rules(df: DataFrame, source_path: str | Path) -> tuple[DataFrame, dict]:
    """Apply the same TLC cleaning actions as data_quality.clean.apply_rules."""
    validate_schema(df)
    stat_masks, set_null_by_column, drop, flag_parts = _quality_masks(df, source_path)

    agg_exprs = [F.count(F.lit(1)).alias("input_rows")]
    alias_for: dict[str, str] = {}
    for label, (_kind, mask) in stat_masks.items():
        alias = "stat_" + label.replace(":", "_")
        alias_for[label] = alias
        agg_exprs.append(F.sum(mask.cast("int")).alias(alias))
    counts = df.agg(*agg_exprs).collect()[0].asDict()

    stats: dict = {
        "input_rows": int(counts["input_rows"]),
        "removed": {},
        "set_null": {},
        "flagged": {},
    }
    for label, (kind, _mask) in stat_masks.items():
        n = int(counts.get(alias_for[label]) or 0)
        if n:
            stats[kind][label] = n

    if flag_parts:
        flags = F.nullif(F.concat_ws(";", *flag_parts), F.lit(""))
    else:
        flags = F.lit(None).cast("string")

    projections: list[Column] = []
    for column in df.columns:
        if column in set_null_by_column:
            projections.append(
                F.when(set_null_by_column[column], F.lit(None))
                .otherwise(F.col(column))
                .alias(column)
            )
        else:
            projections.append(F.col(column))
    projections.append(flags.alias("dq_flags"))
    projections.append(drop.alias("_drop"))

    cleaned = df.select(*projections).filter(F.col("_drop") == F.lit(False)).drop("_drop")

    if CROSS_FIELD_RULES["exact_duplicate_rows"]["on_violation"] == "remove":
        cleaned, dup_stats = drop_exact_duplicates(cleaned)
        stats["removed"]["exact_duplicate_rows"] = dup_stats["removed"]
        stats["duplicate_elapsed_seconds"] = dup_stats["elapsed_seconds"]

    stats["output_rows"] = cleaned.count()
    stats["rows_removed"] = stats["input_rows"] - stats["output_rows"]
    stats["rows_flagged"] = cleaned.filter(F.col("dq_flags").isNotNull()).count()
    return cleaned, stats
