import json

import pandas as pd

from labels_pipeline import spark_job


def test_quality_checks_count_each_rule(spark, raw_files):
    boxes, classes = spark_job.read_inputs(spark, str(raw_files[0]), str(raw_files[1]))
    clean, report = spark_job.quality_checks(boxes, classes)

    assert report["total_input_rows"] == 11
    assert report["missing_coords"] == 1
    assert report["coords_out_of_range"] == 1
    assert report["zero_or_negative_size"] == 1
    assert report["duplicate_rows"] == 1
    assert report["unknown_label"] == 1
    assert report["total_rejected"] == 5
    assert clean.count() == 6


def test_run_writes_outputs(spark, raw_files, tmp_path):
    out = tmp_path / "out"
    report = spark_job.run(str(raw_files[0]), str(raw_files[1]), out, spark=spark)

    assert report["driving_boxes"] == 5          # cat is dropped
    assert report["driving_images"] == 2
    assert json.loads((out / "quality_report.json").read_text())["total_rejected"] == 5

    # boxes are partitioned by class
    assert {p.name for p in (out / "boxes").iterdir() if p.is_dir()} == {
        "class_name=car", "class_name=person", "class_name=traffic_light", "class_name=bicycle"}

    images = pd.read_parquet(out / "images").set_index("image_id")
    img1 = images.loc["img1"]
    assert list(img1["classes"]) == ["car", "person"]
    assert img1["num_boxes"] == 3
    assert dict(img1["class_counts"]) == {"car": 2, "person": 1}
    assert round(img1["occluded_ratio"], 4) == 0.3333
    assert bool(images.loc["img2"]["has_group_box"]) is True


def test_spark_class_name_matches_python_slug(spark):
    from labels_pipeline.config import DRIVING_CLASSES, slug
    df = spark.createDataFrame([(c, "/m/x", "img", "src", 1.0, 0.1, 0.2, 0.1, 0.2, 0, 0, 0)
                                for c in DRIVING_CLASSES],
                               ["DisplayName", "LabelName", "ImageID", "Source", "Confidence",
                                "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf"])
    got = {r.class_name for r in spark_job.keep_driving_classes(df).collect()}
    assert got == {slug(c) for c in DRIVING_CLASSES}
