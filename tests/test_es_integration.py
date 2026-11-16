"""Runs against a real Elasticsearch (CI starts one as a service container).
Skipped automatically when nothing is listening at ES_URL."""
import pytest
from elasticsearch import Elasticsearch

from labels_pipeline import bench, es_load, spark_job
from labels_pipeline.config import ES_URL


@pytest.fixture(scope="module")
def es():
    client = Elasticsearch(ES_URL, request_timeout=10)
    try:
        client.info()
    except Exception:
        pytest.skip(f"no Elasticsearch at {ES_URL}")
    return client


def test_load_and_query(es, spark, raw_files, tmp_path):
    out = tmp_path / "out"
    spark_job.run(str(raw_files[0]), str(raw_files[1]), out, spark=spark)
    report = es_load.load(es, out / "images", index="test-driving-images")
    assert report["indexed"] == 2 and report["failed"] == 0

    res = es.search(index="test-driving-images",
                    query={"range": {"class_counts.car": {"gte": 2}}})
    assert [h["_id"] for h in res["hits"]["hits"]] == ["img1"]

    assert bench.run(es, runs=5, index="test-driving-images")["runs"] == 5
    es.indices.delete(index="test-driving-images")
