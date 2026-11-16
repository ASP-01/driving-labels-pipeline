"""PySpark job: Open Images box annotations -> cleaned boxes + per-image features.

Reads the bounding-box CSV and the class-descriptions CSV, keeps driving classes,
drops bad rows (with a count per rule), and writes two Parquet datasets:

  boxes/   one row per valid box, partitioned by class_name
  images/  one row per image with the features we search on

Usage:
  python -m labels_pipeline.spark_job \
      --boxes data/raw/validation-annotations-bbox.csv \
      --classes data/raw/oidv7-class-descriptions-boxable.csv
"""
import argparse
import json
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from labels_pipeline.config import DRIVING_CLASSES, OUT_DIR

BOX_COLS = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax",
            "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf"]


def spark_session(app: str = "driving-labels") -> SparkSession:
    return (SparkSession.builder.appName(app)
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.enabled", "false")
            .getOrCreate())


def read_inputs(spark: SparkSession, boxes_path: str, classes_path: str):
    boxes = spark.read.csv(boxes_path, header=True, inferSchema=False)
    for c in ["XMin", "XMax", "YMin", "YMax", "Confidence"]:
        boxes = boxes.withColumn(c, F.col(c).cast("double"))
    for c in ["IsOccluded", "IsTruncated", "IsGroupOf"]:
        boxes = boxes.withColumn(c, F.col(c).cast("int"))

    # The class file has a header in v7 (LabelName,DisplayName) and none in older
    # releases, so read it raw and drop a header row if there is one.
    classes = (spark.read.csv(classes_path, header=False)
               .toDF("LabelName", "DisplayName")
               .filter(F.col("LabelName") != "LabelName"))
    return boxes.select(*BOX_COLS), classes


def quality_checks(boxes: DataFrame, classes: DataFrame):
    """Return (clean_boxes, rejected_counts). Each rule is counted separately,
    in order, so the counts add up to the total rejected."""
    counts = {}
    total = boxes.count()

    def drop(df, name, bad_condition):
        bad = df.filter(bad_condition)
        counts[name] = bad.count()
        return df.filter(~bad_condition)

    df = drop(boxes, "missing_ids",
              F.col("ImageID").isNull() | F.col("LabelName").isNull())
    df = drop(df, "missing_coords",
              F.col("XMin").isNull() | F.col("XMax").isNull()
              | F.col("YMin").isNull() | F.col("YMax").isNull())
    out_of_range = F.lit(False)
    for c in ["XMin", "XMax", "YMin", "YMax"]:
        out_of_range = out_of_range | (F.col(c) < 0) | (F.col(c) > 1)
    df = drop(df, "coords_out_of_range", out_of_range)
    df = drop(df, "zero_or_negative_size",
              (F.col("XMax") <= F.col("XMin")) | (F.col("YMax") <= F.col("YMin")))

    before = df.count()
    df = df.dropDuplicates()
    counts["duplicate_rows"] = before - df.count()

    joined = df.join(F.broadcast(classes), "LabelName", "left")
    counts["unknown_label"] = joined.filter(F.col("DisplayName").isNull()).count()
    clean = joined.filter(F.col("DisplayName").isNotNull())

    counts["total_input_rows"] = total
    counts["total_rejected"] = sum(v for k, v in counts.items() if k != "total_input_rows")
    return clean, counts


def keep_driving_classes(clean: DataFrame) -> DataFrame:
    # Built-in functions instead of a Python UDF: same result as config.slug(),
    # but it stays in the JVM, so no rows are shipped to Python workers.
    class_name = F.regexp_replace(
        F.regexp_replace(F.lower("DisplayName"), "[^a-z0-9]+", "_"), "^_+|_+$", "")
    return (clean.filter(F.col("DisplayName").isin(DRIVING_CLASSES))
            .withColumn("class_name", class_name)
            .withColumn("box_area", (F.col("XMax") - F.col("XMin")) * (F.col("YMax") - F.col("YMin")))
            .select(F.col("ImageID").alias("image_id"), F.col("Source").alias("source"),
                    "class_name", "box_area", F.col("Confidence").alias("confidence"),
                    F.col("IsOccluded").alias("is_occluded"),
                    F.col("IsTruncated").alias("is_truncated"),
                    F.col("IsGroupOf").alias("is_group_of"),
                    F.col("XMin").alias("x_min"), F.col("XMax").alias("x_max"),
                    F.col("YMin").alias("y_min"), F.col("YMax").alias("y_max")))


def image_features(boxes: DataFrame) -> DataFrame:
    per_class = (boxes.groupBy("image_id", "class_name")
                 .agg(F.count("*").alias("n")))
    counts = (per_class.groupBy("image_id")
              .agg(F.map_from_entries(F.collect_list(F.struct("class_name", "n"))).alias("class_counts")))
    stats = (boxes.groupBy("image_id")
             .agg(F.first("source").alias("source"),
                  F.sort_array(F.collect_set("class_name")).alias("classes"),
                  F.count("*").alias("num_boxes"),
                  F.countDistinct("class_name").alias("num_classes"),
                  F.round(F.avg("box_area"), 6).alias("mean_box_area"),
                  F.round(F.min("box_area"), 6).alias("min_box_area"),
                  F.round(F.avg(F.coalesce("is_occluded", F.lit(0))), 4).alias("occluded_ratio"),
                  (F.max(F.coalesce("is_group_of", F.lit(0))) == 1).alias("has_group_box")))
    return stats.join(counts, "image_id")


def run(boxes_path: str, classes_path: str, out_dir: Path = OUT_DIR, spark=None) -> dict:
    start = time.perf_counter()
    spark = spark or spark_session()
    boxes, classes = read_inputs(spark, boxes_path, classes_path)
    clean, report = quality_checks(boxes, classes)
    driving = keep_driving_classes(clean).cache()
    images = image_features(driving)

    out_dir = Path(out_dir)
    driving.write.mode("overwrite").partitionBy("class_name").parquet(str(out_dir / "boxes"))
    images.write.mode("overwrite").parquet(str(out_dir / "images"))

    report.update({
        "driving_boxes": driving.count(),
        "driving_images": images.count(),
        "runtime_seconds": round(time.perf_counter() - start, 2),
    })
    (out_dir / "quality_report.json").write_text(json.dumps(report, indent=2))
    driving.unpersist()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--boxes", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--out", default=str(OUT_DIR))
    a = p.parse_args()
    print(json.dumps(run(a.boxes, a.classes, Path(a.out)), indent=2))


if __name__ == "__main__":
    main()
