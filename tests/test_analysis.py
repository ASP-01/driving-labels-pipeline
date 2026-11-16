import json

import pandas as pd

from labels_pipeline import analysis, spark_job


def sample_boxes():
    return pd.DataFrame({
        "image_id": ["a", "a", "a", "b", "b", "c"],
        "class_name": ["car", "car", "person", "car", "bus", "car"],
        "box_area": [0.1, 0.2, 0.05, 0.3, 0.4, 0.15],
        "is_occluded": [0, 1, 0, 0, 1, 0],
    })


def test_class_distribution_and_imbalance():
    dist = analysis.class_distribution(sample_boxes())
    assert dist.index[0] == "car"
    assert dist.loc["car", "boxes"] == 4
    assert dist.loc["car", "images"] == 3
    assert analysis.imbalance_ratio(dist) == 4.0


def test_labels_per_image():
    per_image = analysis.labels_per_image(sample_boxes())
    assert per_image.to_dict() == {"a": 3, "b": 2, "c": 1}


def test_run_end_to_end(spark, raw_files, tmp_path):
    out = tmp_path / "out"
    spark_job.run(str(raw_files[0]), str(raw_files[1]), out, spark=spark)
    summary = analysis.run(out)
    assert summary["boxes"] == 5
    assert summary["most_common_class"] == "car"
    for f in ["class_distribution.csv", "box_size_quantiles.csv", "boxes_per_class.png",
              "boxes_per_image.png", "summary.md"]:
        assert (out / "analysis" / f).exists()
    assert json.loads((out / "analysis" / "summary.json").read_text())["images"] == 2
