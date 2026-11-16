# Driving labels pipeline

![ci](https://github.com/ASP-01/driving-labels-pipeline/actions/workflows/ci.yml/badge.svg)

A PySpark and Elasticsearch pipeline that turns Open Images bounding-box labels for driving-scene classes (cars, people, traffic lights, signs, bikes, buses and so on) into a searchable index, with data-quality checks and a pandas analysis of label balance.

```
Open Images CSVs ──> PySpark job ──> Parquet ──> Elasticsearch ──> FastAPI search / Kibana
                     (quality checks,  boxes/ (by class)              
                      per-image         images/                      
                      features)              └──> pandas analysis (class balance, box sizes)
```

## Results (Open Images validation split)

| Metric | Value |
|---|---|
| Input box labels | 303,980 |
| Rows rejected by quality rules | 0 (clean source data) |
| Driving-scene boxes / images | 28,222 / 11,877 |
| Spark job runtime (MacBook Air, local mode) | 9.1 s |
| Docs indexed into Elasticsearch / failed | 11,877 / 0 |
| Class imbalance (person vs. stop sign) | 1,289x |
| Images with a car and 3+ boxes | 1,633 |

The imbalance is the main finding: a detector trained on this data sees about 1,300 people for every stop sign, so the rare classes would need sampling or targeted labeling before training.

## What it does

1. **PySpark job** (`spark_job.py`) reads the box annotations and class names, drops bad rows and counts each rejection rule (missing coordinates, out-of-range coordinates, zero-size boxes, duplicates, unknown labels), keeps driving classes, and writes two Parquet datasets:
   - `boxes/`: one row per valid box, partitioned by `class_name`
   - `images/`: one row per image with `classes`, `num_boxes`, `class_counts`, `mean_box_area`, `occluded_ratio` and more
2. **Elasticsearch loader** (`es_load.py`) creates a strict mapping, bulk-loads the image features with refresh turned off during the load, and records throughput.
3. **Search API** (`api.py`) runs FastAPI queries such as "images with both a car and a person, at least 5 boxes, mostly unoccluded, and at least 3 cars."
4. **pandas analysis** (`analysis.py`) produces class distribution, the imbalance ratio, box-size quantiles per class and boxes-per-image charts, with a markdown summary.
5. **Benchmark** (`bench.py`) measures p50 and p95 search latency over a fixed, seeded set of queries.

## Quickstart

Needs Python 3.10+, Java 17 (for Spark) and Docker.

```bash
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev]"
make test             # unit tests run without Elasticsearch

make data             # downloads the validation split CSVs (~25 MB)
make up               # starts Elasticsearch on :9200 and Kibana on :5601
make spark            # PySpark job -> data/processed/
make load             # bulk-load into Elasticsearch
make analyze          # pandas report -> data/processed/analysis/
make bench            # latency report
make api              # http://localhost:8000/docs
```

Example queries:

```bash
curl "localhost:8000/search?classes=car&classes=person&min_boxes=5"
curl "localhost:8000/search?classes=traffic_light&max_occluded_ratio=0.2&min_count=car:3"
curl "localhost:8000/stats"
```

Once the validation split works, run the train split (`./scripts/download_data.sh train`, about 2 GB) to see how the job behaves at roughly 15 million rows.

## Reports it writes

| File | What's in it |
|---|---|
| `data/processed/quality_report.json` | Input rows, rejections per rule, output counts, Spark runtime |
| `data/processed/load_report.json` | Docs indexed, failures, seconds, docs per second |
| `data/processed/bench_report.json` | Server and client p50/p95 latency |
| `data/processed/analysis/summary.md` | Class balance, imbalance ratio, box sizes |

## Design decisions

- **Parquet, partitioned by class.** Columnar files are much smaller than CSV and let pandas read only the columns it needs. Partitioning by class means a job that only cares about traffic lights reads one folder.
- **Quality rules counted one at a time.** Each rule removes its rows before the next rule runs, so the counts add up to the total rejected and you can see which problem is most common.
- **No Python UDFs.** Class names are normalized with built-in Spark functions (`regexp_replace`, `lower`), so rows never leave the JVM. A Python UDF version also broke on workers that couldn't import the package. There's a test that keeps the Spark logic and the Python `slug()` in sync.
- **Broadcast join for class names.** The class table has a few hundred rows, so it's broadcast to every executor instead of shuffling millions of box rows.
- **Strict Elasticsearch mapping.** `dynamic: strict` rejects unexpected fields instead of silently guessing their types. Class counts get explicit integer fields so range queries like `class_counts.car >= 3` work.
- **Refresh off during bulk load.** Refreshing makes new docs searchable but costs time. Turning it off for the load and back on afterwards is the standard way to speed up bulk indexing.
- **Filter context in queries.** All search conditions are filters, not scored matches, so Elasticsearch can cache them and skip relevance scoring.

## Tests and CI

`pytest` covers quality-rule counts, Parquet outputs and partitioning, per-image features, the analysis, document building, query building and the API. `tests/test_es_integration.py` loads and queries a real Elasticsearch. It skips locally when nothing is running, and runs in GitHub Actions against an Elasticsearch service container.

## Data

[Open Images](https://storage.googleapis.com/openimages/web/index.html) annotations. Check the dataset's license terms before redistributing anything derived from it. This repo never commits the data (`data/` is gitignored).
