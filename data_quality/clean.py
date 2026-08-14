"""Apply QUALITY_RULES cleaning actions. Does not use Spark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_quality.profile import (
    _above_max,
    _below_min,
    _not_allowed,
    file_month,
    load_trips,
)
from data_quality.rules import (
    CROSS_FIELD_RULES,
    DROPOFF_COL,
    PICKUP_COL,
    QUALITY_RULES,
)


def _flag_column(flags: pd.Series, mask: pd.Series, label: str) -> pd.Series:
    if not mask.any():
        return flags
    addition = pd.Series(label, index=flags.index).where(mask, "")
    both = flags.ne("") & addition.ne("")
    flags = flags.where(~both, flags + ";" + addition)
    flags = flags.where(~(flags.eq("") & addition.ne("")), addition)
    return flags


def apply_rules(df: pd.DataFrame, source_path: Path) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    flags = pd.Series("", index=work.index, dtype="object")
    stats = {
        "input_rows": len(work),
        "removed": {},
        "set_null": {},
        "flagged": {},
    }

    drop_mask = pd.Series(False, index=work.index)

    for column, spec in QUALITY_RULES.items():
        if column not in work.columns:
            continue
        series = work[column]
        below = _below_min(series, spec)
        above = _above_max(series, spec)
        domain = _not_allowed(series, spec)

        for mask, action_key, label in (
            (below, "on_below_min", f"{column}:below_min"),
            (above, "on_above_max", f"{column}:above_max"),
            (domain, "on_violation", f"{column}:domain"),
        ):
            action = spec.get(action_key) or spec.get("on_violation")
            if action is None or not mask.any():
                continue
            if action == "remove":
                drop_mask |= mask
                stats["removed"][label] = int(mask.sum())
            elif action == "set_null":
                stats["set_null"][label] = int(mask.sum())
                work.loc[mask, column] = pd.NA
            elif action == "flag":
                flags = _flag_column(flags, mask, label)
                stats["flagged"][label] = int(mask.sum())

    pickup = pd.to_datetime(work[PICKUP_COL], errors="coerce") if PICKUP_COL in work.columns else None
    dropoff = (
        pd.to_datetime(work[DROPOFF_COL], errors="coerce") if DROPOFF_COL in work.columns else None
    )

    if pickup is not None and dropoff is not None:
        inverted = dropoff.notna() & pickup.notna() & (dropoff < pickup)
        action = CROSS_FIELD_RULES["dropoff_before_pickup"]["on_violation"]
        if action == "remove":
            stats["removed"]["dropoff_before_pickup"] = int(inverted.sum())
            drop_mask |= inverted

        month = file_month(source_path)
        if month:
            year, mon = month
            outside = pickup.notna() & ((pickup.dt.year != year) | (pickup.dt.month != mon))
            action = CROSS_FIELD_RULES["pickup_outside_file_month"]["on_violation"]
            if action == "flag":
                flags = _flag_column(flags, outside, "pickup_outside_file_month")
                stats["flagged"]["pickup_outside_file_month"] = int(outside.sum())

    dupes = work.duplicated()
    if CROSS_FIELD_RULES["exact_duplicate_rows"]["on_violation"] == "remove":
        stats["removed"]["exact_duplicate_rows"] = int(dupes.sum())
        drop_mask |= dupes

    cleaned = work.loc[~drop_mask].copy()
    cleaned_flags = flags.loc[~drop_mask]
    cleaned["dq_flags"] = cleaned_flags.replace("", pd.NA)
    stats["output_rows"] = len(cleaned)
    stats["rows_removed"] = int(drop_mask.sum())
    stats["rows_flagged"] = int((cleaned["dq_flags"].notna()).sum())
    return cleaned, stats


def write_cleaned(cleaned: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        cleaned.to_csv(out_path, index=False)
        return
    cleaned.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean TLC trip data using quality rules.")
    parser.add_argument(
        "--path",
        default="data/sample/yellow_tripdata_2025-01.parquet",
        help="Input CSV or Parquet.",
    )
    parser.add_argument(
        "--out",
        default="data/sample/yellow_tripdata_2025-01.cleaned.parquet",
        help="Cleaned output path (parquet or csv).",
    )
    args = parser.parse_args()
    source = Path(args.path)
    df = load_trips(source)
    cleaned, stats = apply_rules(df, source)
    out_path = Path(args.out)
    write_cleaned(cleaned, out_path)
    print("CLEANING SUMMARY")
    print("================")
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
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
