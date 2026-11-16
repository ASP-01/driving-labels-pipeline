from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from labels_pipeline import api, bench, es_load


def test_to_doc_converts_map_pairs_and_nan():
    row = {"image_id": "img1", "source": "xclick", "classes": ["car", "person"],
           "num_boxes": 3, "num_classes": 2, "mean_box_area": float("nan"),
           "min_box_area": 0.1, "occluded_ratio": 0.33, "has_group_box": False,
           "class_counts": [("car", 2), ("person", 1)]}
    doc = es_load.to_doc(row, "idx")
    assert doc["_id"] == "img1" and doc["_index"] == "idx"
    assert doc["_source"]["class_counts"] == {"car": 2, "person": 1}
    assert doc["_source"]["mean_box_area"] is None


def test_mapping_covers_every_driving_class():
    props = es_load.MAPPING["mappings"]["properties"]["class_counts"]["properties"]
    assert "traffic_light" in props and "car" in props


def test_build_query_filters():
    q = api.build_query(["car", "person"], 5, 0.5, {"car": 3})
    f = q["bool"]["filter"]
    assert {"term": {"classes": "car"}} in f
    assert {"range": {"num_boxes": {"gte": 5}}} in f
    assert {"range": {"occluded_ratio": {"lte": 0.5}}} in f
    assert {"range": {"class_counts.car": {"gte": 3}}} in f
    assert api.build_query([], 0, None, {}) == {"match_all": {}}


def test_parse_min_counts_ignores_bad_values():
    assert api.parse_min_counts(["car:3", "bus", "person:x"]) == {"car": 3}


def test_search_endpoint():
    fake = MagicMock()
    fake.search.return_value = {"took": 4, "hits": {"total": {"value": 1},
                                                     "hits": [{"_source": {"image_id": "img1"}}]}}
    api.app.dependency_overrides[api.get_es] = lambda: fake
    try:
        r = TestClient(api.app).get("/search", params={"classes": ["car"], "min_count": ["car:2"]})
    finally:
        api.app.dependency_overrides.clear()
    assert r.status_code == 200
    assert r.json() == {"took_ms": 4, "total": 1, "results": [{"image_id": "img1"}]}
    sent = fake.search.call_args.kwargs["query"]["bool"]["filter"]
    assert {"range": {"class_counts.car": {"gte": 2}}} in sent


def test_percentile():
    assert bench.percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert bench.percentile([10, 20], 0.95) == 19.5
