"""Pandas data-quality rules, profiling, and cleaning for TLC trip files."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "data_quality" / "reports"
DEFAULT_TRIPS = REPO_ROOT / "data" / "sample" / "yellow_tripdata_2025-01.parquet"
DEFAULT_CLEANED = REPO_ROOT / "data" / "sample" / "yellow_tripdata_2025-01.cleaned.parquet"
