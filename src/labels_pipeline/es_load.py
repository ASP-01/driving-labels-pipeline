"""Bulk-load per-image features from Parquet into Elasticsearch and time it.

Usage:
  python -m labels_pipeline.es_load            # reads data/processed/images
"""
import argparse
import json
import math
import time
from pathlib import Path

import pyarrow.dataset as ds
from elasticsearch import Elasticsearch, helpers

from labels_pipeline.config import DRIVING_CLASSES, ES_INDEX, ES_URL, OUT_DIR, slug

MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "image_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "classes": {"type": "keyword"},
            "num_boxes": {"type": "integer"},
            "num_classes": {"type": "integer"},
            "mean_box_area": {"type": "float"},
            "min_box_area": {"type": "float"},
            "occluded_ratio": {"type": "float"},
            "has_group_box": {"type": "boolean"},
            "class_counts": {
                "properties": {slug(c): {"type": "integer"} for c in DRIVING_CLASSES}
            },
        },
    },
}


def to_doc(row: dict, index: str = ES_INDEX) -> dict:
    """Turn one Parquet row into a bulk action. class_counts comes out of
    Parquet as a list of (key, value) pairs; ES wants an object."""
    counts = row.get("class_counts") or []
    if not isinstance(counts, dict):
        counts = {k: int(v) for k, v in counts}
    src = {k: row[k] for k in MAPPING["mappings"]["properties"] if k in row and k != "class_counts"}
    src["classes"] = list(src.get("classes") or [])
    for k, v in list(src.items()):
        if isinstance(v, float) and math.isnan(v):
            src[k] = None
    src["class_counts"] = counts
    return {"_index": index, "_id": row["image_id"], "_source": src}


def iter_docs(images_dir: Path, index: str, batch_size: int = 5000):
    dataset = ds.dataset(str(images_dir), format="parquet")
    for batch in dataset.to_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield to_doc(row, index)


def load(es: Elasticsearch, images_dir: Path, index: str = ES_INDEX, recreate: bool = True) -> dict:
    if recreate and es.indices.exists(index=index):
        es.indices.delete(index=index)
    if not es.indices.exists(index=index):
        es.indices.create(index=index, **MAPPING)

    start = time.perf_counter()
    ok, failed = 0, 0
    for success, _ in helpers.streaming_bulk(es, iter_docs(images_dir, index),
                                             chunk_size=2000, raise_on_error=False):
        ok += success
        failed += not success
    elapsed = time.perf_counter() - start

    es.indices.put_settings(index=index, settings={"refresh_interval": "1s"})
    es.indices.refresh(index=index)
    return {
        "indexed": ok,
        "failed": failed,
        "seconds": round(elapsed, 2),
        "docs_per_second": round(ok / elapsed, 1) if elapsed else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--images", default=str(OUT_DIR / "images"))
    p.add_argument("--index", default=ES_INDEX)
    a = p.parse_args()
    report = load(Elasticsearch(ES_URL, request_timeout=60), Path(a.images), a.index)
    (OUT_DIR / "load_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
