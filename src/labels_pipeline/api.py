"""Search API over the indexed images.

Run:  uvicorn labels_pipeline.api:app --reload
Try:  curl "localhost:8000/search?classes=car&classes=person&min_boxes=5"
"""
from functools import lru_cache
from typing import Optional

from elasticsearch import Elasticsearch
from fastapi import Depends, FastAPI, Query

from labels_pipeline.config import ES_INDEX, ES_URL

app = FastAPI(title="Driving labels search")


@lru_cache
def get_es() -> Elasticsearch:
    return Elasticsearch(ES_URL, request_timeout=30)


def build_query(classes: list[str], min_boxes: int, max_occluded_ratio: Optional[float],
                min_counts: dict[str, int]) -> dict:
    filters = [{"term": {"classes": c}} for c in classes]
    if min_boxes:
        filters.append({"range": {"num_boxes": {"gte": min_boxes}}})
    if max_occluded_ratio is not None:
        filters.append({"range": {"occluded_ratio": {"lte": max_occluded_ratio}}})
    for cls, n in min_counts.items():
        filters.append({"range": {f"class_counts.{cls}": {"gte": n}}})
    return {"bool": {"filter": filters}} if filters else {"match_all": {}}


def parse_min_counts(values: list[str]) -> dict[str, int]:
    """'car:3' -> {'car': 3}"""
    out = {}
    for v in values:
        cls, _, n = v.partition(":")
        if cls and n.isdigit():
            out[cls] = int(n)
    return out


@app.get("/search")
def search(classes: list[str] = Query(default=[]),
           min_boxes: int = 0,
           max_occluded_ratio: Optional[float] = None,
           min_count: list[str] = Query(default=[], description="e.g. car:3"),
           size: int = Query(default=20, le=100),
           es: Elasticsearch = Depends(get_es)):
    query = build_query(classes, min_boxes, max_occluded_ratio, parse_min_counts(min_count))
    res = es.search(index=ES_INDEX, query=query, size=size, track_total_hits=True)
    return {
        "took_ms": res["took"],
        "total": res["hits"]["total"]["value"],
        "results": [h["_source"] for h in res["hits"]["hits"]],
    }


@app.get("/stats")
def stats(es: Elasticsearch = Depends(get_es)):
    res = es.search(index=ES_INDEX, size=0, aggs={
        "classes": {"terms": {"field": "classes", "size": 50}},
        "boxes": {"stats": {"field": "num_boxes"}},
    })
    return {
        "images_per_class": {b["key"]: b["doc_count"] for b in res["aggregations"]["classes"]["buckets"]},
        "boxes_per_image": res["aggregations"]["boxes"],
    }
