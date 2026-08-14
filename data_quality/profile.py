"""Profile a TLC trip file against QUALITY_RULES and write a text report."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_quality import DEFAULT_TRIPS, REPORTS_DIR, REPO_ROOT
from data_quality.rules import (
    CLEANING_POLICY,
    CROSS_FIELD_RULES,
    DROPOFF_COL,
    PAYMENT_TYPE_COL,
    PICKUP_COL,
    QUALITY_RULES,
    REVERSAL_PAYMENT_TYPES,
)


def load_trips(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def file_month(path: Path) -> tuple[int, int] | None:
    match = re.search(r"(20\d{2})-(\d{2})", path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _below_min(series: pd.Series, spec: dict) -> pd.Series:
    if "min" not in spec:
        return pd.Series(False, index=series.index)
    if spec.get("min_inclusive", True):
        return series.notna() & (series < spec["min"])
    return series.notna() & (series <= spec["min"])


def _above_max(series: pd.Series, spec: dict) -> pd.Series:
    if "max" not in spec:
        return pd.Series(False, index=series.index)
    if spec.get("max_inclusive", True):
        return series.notna() & (series > spec["max"])
    return series.notna() & (series >= spec["max"])


def _not_allowed(series: pd.Series, spec: dict) -> pd.Series:
    allowed = spec.get("allowed")
    if not allowed:
        return pd.Series(False, index=series.index)
    return series.notna() & ~series.isin(allowed)


def evaluate_column(df: pd.DataFrame, column: str, spec: dict) -> dict:
    n = len(df)
    if column not in df.columns:
        return {"column": column, "present": False, "total": n}

    series = df[column]
    missing = int(series.isna().sum())
    below = int(_below_min(series, spec).sum())
    above = int(_above_max(series, spec).sum())
    domain = int(_not_allowed(series, spec).sum())
    invalid = below + above + domain
    observed = n - missing
    valid = observed - invalid
    return {
        "column": column,
        "present": True,
        "total": n,
        "missing": missing,
        "invalid": invalid,
        "valid": valid,
        "below_min": below,
        "above_max": above,
        "domain_invalid": domain,
        "on_below_min": spec.get("on_below_min"),
        "on_above_max": spec.get("on_above_max"),
        "on_violation": spec.get("on_violation"),
        "on_null": spec.get("on_null"),
        "description": spec.get("description", ""),
    }


def evaluate_cross_field(df: pd.DataFrame, path: Path) -> dict[str, dict]:
    n = len(df)
    results: dict[str, dict] = {}

    pickup = pd.to_datetime(df[PICKUP_COL], errors="coerce") if PICKUP_COL in df.columns else None
    dropoff = (
        pd.to_datetime(df[DROPOFF_COL], errors="coerce") if DROPOFF_COL in df.columns else None
    )

    if pickup is not None and dropoff is not None:
        invalid = int((dropoff.notna() & pickup.notna() & (dropoff < pickup)).sum())
        results["dropoff_before_pickup"] = {
            "invalid": invalid,
            "valid": n - invalid,
            "missing": int(pickup.isna().sum() + dropoff.isna().sum()),
            **CROSS_FIELD_RULES["dropoff_before_pickup"],
        }

        month = file_month(path)
        if month:
            year, mon = month
            outside = int(
                (
                    pickup.notna()
                    & ((pickup.dt.year != year) | (pickup.dt.month != mon))
                ).sum()
            )
            results["pickup_outside_file_month"] = {
                "invalid": outside,
                "valid": int(pickup.notna().sum()) - outside,
                "expected_month": f"{year:04d}-{mon:02d}",
                "min_pickup": str(pickup.min()) if pickup.notna().any() else None,
                "max_pickup": str(pickup.max()) if pickup.notna().any() else None,
                **CROSS_FIELD_RULES["pickup_outside_file_month"],
            }

    dupes = int(df.duplicated().sum())
    results["exact_duplicate_rows"] = {
        "invalid": dupes,
        "valid": n - dupes,
        **CROSS_FIELD_RULES["exact_duplicate_rows"],
    }
    return results


def negative_money_by_payment_type(df: pd.DataFrame) -> pd.DataFrame | None:
    if PAYMENT_TYPE_COL not in df.columns:
        return None
    fare_neg = df["fare_amount"] < 0 if "fare_amount" in df.columns else False
    total_neg = df["total_amount"] < 0 if "total_amount" in df.columns else False
    flagged = df.loc[fare_neg | total_neg]
    if flagged.empty:
        return None
    counts = (
        flagged[PAYMENT_TYPE_COL]
        .value_counts(dropna=False)
        .rename_axis("payment_type")
        .reset_index(name="rows")
    )
    counts["meaning"] = counts["payment_type"].map(
        lambda v: REVERSAL_PAYMENT_TYPES.get(int(v), "not_a_known_reversal")
        if pd.notna(v)
        else "missing"
    )
    return counts


def format_report(
    path: Path,
    df: pd.DataFrame,
    column_results: Iterable[dict],
    cross: dict[str, dict],
    money_breakdown: pd.DataFrame | None,
) -> str:
    try:
        display_path = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        display_path = path
    n = len(df)
    lines = [
        "DATA QUALITY REPORT",
        "===================",
        "",
        f"File: {display_path}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Total records: {n:,}",
        "",
    ]

    for result in column_results:
        if not result.get("present"):
            lines.append(f"{result['column']}")
            lines.append("    (column not in file)")
            lines.append("")
            continue
        lines.append(result["column"])
        lines.append(f"    Invalid: {result['invalid']:,}")
        lines.append(f"    Valid:   {result['valid']:,}")
        if result["missing"]:
            lines.append(f"    Missing: {result['missing']:,}")
        if result.get("below_min"):
            action = result.get("on_below_min") or result.get("on_violation")
            lines.append(f"      below min: {result['below_min']:,}  -> {action}")
        if result.get("above_max"):
            action = result.get("on_above_max") or "flag"
            lines.append(f"      above max: {result['above_max']:,}  -> {action}")
        if result.get("domain_invalid"):
            action = result.get("on_violation")
            lines.append(f"      not in domain: {result['domain_invalid']:,}  -> {action}")
        if result.get("on_null"):
            lines.append(f"      nulls: {result.get('on_null')}")
        lines.append("")

    pickup = cross.get("dropoff_before_pickup")
    if pickup:
        lines.append("pickup/dropoff")
        lines.append(f"    Invalid: {pickup['invalid']:,}")
        lines.append(f"    Valid:   {pickup['valid']:,}")
        lines.append(f"    Action:  {pickup['on_violation']}")
        lines.append("")

    outside = cross.get("pickup_outside_file_month")
    if outside:
        lines.append("pickup_outside_file_month")
        lines.append(f"    Invalid: {outside['invalid']:,}")
        lines.append(f"    Valid:   {outside['valid']:,}")
        lines.append(f"    Expected month: {outside.get('expected_month')}")
        lines.append(f"    Pickup min/max: {outside.get('min_pickup')} / {outside.get('max_pickup')}")
        lines.append(f"    Action:  {outside['on_violation']}")
        lines.append("")

    dupes = cross["exact_duplicate_rows"]
    lines.append("exact_duplicate_rows")
    lines.append(f"    Invalid: {dupes['invalid']:,}")
    lines.append(f"    Valid:   {dupes['valid']:,}")
    lines.append(f"    Action:  {dupes['on_violation']}")
    lines.append("")

    lines.append("CLEANING POLICY")
    lines.append("---------------")
    for row in CLEANING_POLICY:
        lines.append(f"{row['problem']:<32} {row['rule']}")
    lines.append("")

    lines.append("NEGATIVE MONEY INVESTIGATION")
    lines.append("----------------------------")
    lines.append(
        "Negative fare/total amounts are flagged, not removed. "
        "TLC payment_type 3=No charge, 4=Dispute, 6=Voided trip."
    )
    if money_breakdown is None or money_breakdown.empty:
        lines.append("No negative fare_amount or total_amount rows.")
    else:
        lines.append(money_breakdown.to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def run_profile(path: Path, report_path: Path | None = None) -> Path:
    df = load_trips(path)
    column_results = [evaluate_column(df, col, spec) for col, spec in QUALITY_RULES.items()]
    cross = evaluate_cross_field(df, path)
    money = negative_money_by_payment_type(df)
    text = format_report(path, df, column_results, cross, money)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if report_path is None:
        report_path = REPORTS_DIR / f"{path.stem}_dq_report.txt"
    report_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile TLC trip data against quality rules.")
    parser.add_argument(
        "--path",
        default=str(DEFAULT_TRIPS),
        help="CSV or Parquet trip file.",
    )
    parser.add_argument("--report", help="Optional report output path.")
    args = parser.parse_args()
    run_profile(Path(args.path), Path(args.report) if args.report else None)


if __name__ == "__main__":
    main()
