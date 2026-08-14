"""Explicit TLC yellow-taxi quality rules and cleaning actions.

Actions
-------
remove     Drop the row from the cleaned dataset.
set_null   Keep the row; replace the offending value with NULL.
flag       Keep the row unchanged; record the issue for investigation.
keep_null  Missing values are allowed and stay NULL.

Negative fare / total amounts are flagged, not deleted. TLC payment_type
includes Dispute (4), Voided trip (6), and No charge (3), which can produce
legitimate reversals. Confirm those semantics before treating negatives as
errors.
"""

from __future__ import annotations

from typing import Any

# Canonical column names as they appear in TLC Parquet files.
PICKUP_COL = "tpep_pickup_datetime"
DROPOFF_COL = "tpep_dropoff_datetime"
PAYMENT_TYPE_COL = "payment_type"

# TLC payment_type codes that can explain negative money fields.
REVERSAL_PAYMENT_TYPES = {
    3: "no_charge",
    4: "dispute",
    6: "voided_trip",
}

RATECODE_VALID = {1, 2, 3, 4, 5, 6, 99}
STORE_AND_FWD_VALID = {"Y", "N"}

# Column-level numeric / domain rules.
QUALITY_RULES: dict[str, dict[str, Any]] = {
    "trip_distance": {
        "min": 0,
        "min_inclusive": False,  # <= 0 is invalid
        "max": 100,  # miles; investigate GPS / meter spikes above this
        "on_below_min": "remove",
        "on_above_max": "flag",
        "on_null": "keep_null",
        "description": "Elapsed trip distance in miles from the taximeter.",
    },
    "passenger_count": {
        "min": 1,
        "max": 9,
        "min_inclusive": True,
        "max_inclusive": True,
        "on_below_min": "set_null",
        "on_above_max": "set_null",
        "on_null": "keep_null",
        "description": "Number of passengers. 0 is not a valid occupancy; missing is allowed.",
    },
    "fare_amount": {
        "min": 0,
        "min_inclusive": True,  # < 0 is invalid for the check; 0 can be no-charge
        "max": 500,
        "on_below_min": "flag",
        "on_above_max": "flag",
        "on_null": "keep_null",
        "description": "Meter fare. Negatives may be disputes/voids/refunds — do not auto-delete.",
    },
    "total_amount": {
        "min": 0,
        "min_inclusive": True,
        "max": 500,
        "on_below_min": "flag",
        "on_above_max": "flag",
        "on_null": "keep_null",
        "description": "Total charged to passenger. Negatives treated like fare_amount.",
    },
    "RatecodeID": {
        "allowed": RATECODE_VALID,
        "on_violation": "flag",
        "on_null": "keep_null",
        "description": "1=standard 2=JFK 3=Newark 4=Nassau/Westchester 5=negotiated 6=group 99=unknown.",
    },
    "store_and_fwd_flag": {
        "allowed": STORE_AND_FWD_VALID,
        "on_violation": "flag",
        "on_null": "keep_null",
        "description": "Y if the trip record was stored and forwarded; N otherwise.",
    },
    "Airport_fee": {
        "on_null": "keep_null",
        "description": "LaGuardia/JFK pickup surcharge. Missing is allowed.",
    },
    "congestion_surcharge": {
        "on_null": "keep_null",
        "description": "NYS congestion surcharge. Missing is allowed.",
    },
}

CROSS_FIELD_RULES: dict[str, dict[str, Any]] = {
    "dropoff_before_pickup": {
        "on_violation": "remove",
        "description": "Dropoff timestamp is earlier than pickup.",
    },
    "pickup_outside_file_month": {
        "on_violation": "flag",
        "description": "Pickup timestamp is outside the month implied by the TLC filename.",
    },
    "exact_duplicate_rows": {
        "on_violation": "remove",
        "description": "Exact duplicate rows. None expected in TLC monthly files.",
    },
}

CLEANING_POLICY: list[dict[str, str]] = [
    {"problem": "trip_distance <= 0", "rule": "Remove"},
    {"problem": "trip_distance > 100 miles", "rule": "Flag (investigate outlier)"},
    {"problem": "fare_amount < 0", "rule": "Flag; keep row (possible reversal)"},
    {"problem": "total_amount < 0", "rule": "Flag; keep row (possible reversal)"},
    {"problem": "fare/total > 500", "rule": "Flag (investigate outlier)"},
    {"problem": "passenger_count <= 0", "rule": "Set to NULL"},
    {"problem": "passenger_count > 9", "rule": "Set to NULL"},
    {"problem": "dropoff < pickup", "rule": "Remove"},
    {"problem": "Missing passenger_count", "rule": "Keep as NULL"},
    {"problem": "Missing Airport_fee", "rule": "Keep as NULL"},
    {"problem": "Missing RatecodeID", "rule": "Keep as NULL"},
    {"problem": "Missing store_and_fwd_flag", "rule": "Keep as NULL"},
    {"problem": "Missing congestion_surcharge", "rule": "Keep as NULL"},
    {"problem": "Duplicate rows", "rule": "Remove if found"},
    {"problem": "Pickup outside file month", "rule": "Flag (keep row)"},
]
