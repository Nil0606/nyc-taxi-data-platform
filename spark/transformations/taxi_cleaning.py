"""Spark transformations that apply QUALITY_RULES from data_quality.rules."""

from __future__ import annotations

import re
from pathlib import Path

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

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


def file_month(path: str | Path) -> tuple[int, int] | None:
    match = re.search(r"(20\d{2})-(\d{2})", Path(path).name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


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


def apply_cleaning_rules(df: DataFrame, source_path: str | Path) -> tuple[DataFrame, dict]:
    """Apply the same TLC cleaning actions as data_quality.clean.apply_rules."""
    validate_schema(df)
    work = df
    drop = F.lit(False)
    flag_parts: list[Column] = []
    stat_masks: dict[str, tuple[str, Column]] = {}

    for column, spec in QUALITY_RULES.items():
        if column not in work.columns:
            continue
        below = _below_min(column, spec)
        above = _above_max(column, spec)
        domain = _not_allowed(column, spec)
        prefix = f"_m_{column}_"
        work = (
            work.withColumn(prefix + "below", below)
            .withColumn(prefix + "above", above)
            .withColumn(prefix + "domain", domain)
        )
        checks = (
            (F.col(prefix + "below"), spec.get("on_below_min"), f"{column}:below_min"),
            (F.col(prefix + "above"), spec.get("on_above_max"), f"{column}:above_max"),
            (F.col(prefix + "domain"), spec.get("on_violation"), f"{column}:domain"),
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
            work = work.withColumn(
                column,
                F.when(set_null_mask, F.lit(None)).otherwise(F.col(column)),
            )

    if PICKUP_COL in work.columns and DROPOFF_COL in work.columns:
        pickup = F.to_timestamp(F.col(PICKUP_COL))
        dropoff = F.to_timestamp(F.col(DROPOFF_COL))
        inverted = dropoff.isNotNull() & pickup.isNotNull() & (dropoff < pickup)
        if CROSS_FIELD_RULES["dropoff_before_pickup"]["on_violation"] == "remove":
            drop = drop | inverted
            stat_masks["dropoff_before_pickup"] = ("removed", inverted)

        month = file_month(source_path)
        if month and CROSS_FIELD_RULES["pickup_outside_file_month"]["on_violation"] == "flag":
            year, mon = month
            outside = pickup.isNotNull() & (
                (F.year(pickup) != year) | (F.month(pickup) != mon)
            )
            flag_parts.append(F.when(outside, F.lit("pickup_outside_file_month")))
            stat_masks["pickup_outside_file_month"] = ("flagged", outside)

    helper_cols = {"_drop", "_dup", "_row_id", "dq_flags"}
    data_cols = [
        c
        for c in work.columns
        if c not in helper_cols
        and not c.startswith("_m_")
        and not c.startswith("_stat_")
    ]
    work = work.withColumn("_row_id", F.monotonically_increasing_id())
    dup_window = Window.partitionBy(*data_cols).orderBy("_row_id")
    work = work.withColumn("_dup", F.row_number().over(dup_window) > 1)
    if CROSS_FIELD_RULES["exact_duplicate_rows"]["on_violation"] == "remove":
        drop = drop | F.col("_dup")
        stat_masks["exact_duplicate_rows"] = ("removed", F.col("_dup"))

    work = work.withColumn("_drop", drop)
    if flag_parts:
        flags = F.nullif(F.concat_ws(";", *flag_parts), F.lit(""))
    else:
        flags = F.lit(None).cast("string")
    work = work.withColumn("dq_flags", flags)

    stat_col_names: dict[str, str] = {}
    for label, (_kind, mask) in stat_masks.items():
        col_name = "_stat_" + label.replace(":", "_")
        work = work.withColumn(col_name, mask.cast("int"))
        stat_col_names[label] = col_name

    agg_exprs = [F.count(F.lit(1)).alias("input_rows")]
    for label, col_name in stat_col_names.items():
        agg_exprs.append(F.sum(col_name).alias(col_name))
    counts = work.agg(*agg_exprs).collect()[0].asDict()

    stats: dict = {
        "input_rows": int(counts["input_rows"]),
        "removed": {},
        "set_null": {},
        "flagged": {},
    }
    for label, (kind, _mask) in stat_masks.items():
        n = int(counts.get(stat_col_names[label]) or 0)
        if n:
            stats[kind][label] = n

    mask_cols = [c for c in work.columns if c.startswith("_m_")]
    cleaned = work.filter(F.col("_drop") == F.lit(False)).drop(
        "_drop",
        "_dup",
        "_row_id",
        *stat_col_names.values(),
        *mask_cols,
    )
    stats["output_rows"] = cleaned.count()
    stats["rows_removed"] = stats["input_rows"] - stats["output_rows"]
    stats["rows_flagged"] = cleaned.filter(F.col("dq_flags").isNotNull()).count()
    return cleaned, stats
