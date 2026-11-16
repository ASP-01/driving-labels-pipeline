"""Measure search latency (p50/p95) against the index with a fixed set of queries.

Usage:
  python -m labels_pipeline.bench --runs 200
"""
import argparse
import json
import random
import statistics
import time

from elasticsearch import Elasticsearch

from labels_pipeline.api import build_query
from labels_pipeline.config import DRIVING_CLASSES, ES_INDEX, ES_URL, OUT_DIR, slug

CLASSES = [slug(c) for c in DRIVING_CLASSES]


def random_query(rng: random.Random) -> dict:
    classes = rng.sample(CLASSES, k=rng.choice([1, 2, 3]))
    return build_query(classes, min_boxes=rng.choice([0, 3, 5, 10]),
                       max_occluded_ratio=rng.choice([None, 0.5]), min_counts={})


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 2)


def run(es: Elasticsearch, runs: int = 200, seed: int = 7, index: str = ES_INDEX) -> dict:
    rng = random.Random(seed)
    server_ms, client_ms = [], []
    for _ in range(runs):
        q = random_query(rng)
        t = time.perf_counter()
        res = es.search(index=index, query=q, size=20)
        client_ms.append((time.perf_counter() - t) * 1000)
        server_ms.append(res["took"])
    return {
        "runs": runs,
        "server_p50_ms": percentile(server_ms, 0.50),
        "server_p95_ms": percentile(server_ms, 0.95),
        "client_p50_ms": percentile(client_ms, 0.50),
        "client_p95_ms": percentile(client_ms, 0.95),
        "server_mean_ms": round(statistics.mean(server_ms), 2),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=int, default=200)
    report = run(Elasticsearch(ES_URL), p.parse_args().runs)
    (OUT_DIR / "bench_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
