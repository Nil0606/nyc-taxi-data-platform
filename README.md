# NYC Taxi Data Platform

Batch and streaming data platform for NYC Taxi & Limousine Commission (TLC) trip records: ingest, transform with Spark, enforce data quality, warehouse with SQL, and deploy on cloud infrastructure.

## Layout

```
├── data/sample/          # Small fixtures for local runs (not production data)
├── ingestion/            # Source extract & load into GCS landing zone
├── spark/                # Spark jobs (bronze → silver → gold)
├── data_quality/         # Checks, expectations, and quarantine logic
├── sql/                  # Warehouse models, views, and analytics queries
├── streaming/            # Near-real-time pipelines (e.g. Kafka / Spark Structured Streaming)
├── infrastructure/
│   └── terraform/        # Cloud resources (storage, compute, IAM, networking)
├── tests/                # Unit and integration tests
├── docker/               # Local stack (Spark, Kafka, Postgres, etc.)
├── requirements.txt
└── README.md
```

## Data source

[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — yellow, green, and FHV trip files (Parquet).

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Place a small sample file under `data/sample/` for development. Do not commit large trip files.

## Next steps

1. Implement ingestion into a GCS landing/bronze bucket.
2. Add Spark transforms and data-quality gates.
3. Model gold tables in `sql/`.
4. Provision GCP resources from `infrastructure/terraform/`.
