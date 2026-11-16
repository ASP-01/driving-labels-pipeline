"""pandas analysis of the processed labels: class balance, box sizes, labels per image.

Writes CSVs, PNG charts and a short markdown summary to data/processed/analysis/.

Usage:
  python -m labels_pipeline.analysis
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from labels_pipeline.config import OUT_DIR  # noqa: E402


def load_boxes(boxes_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(boxes_dir, columns=["image_id", "class_name", "box_area", "is_occluded"])


def class_distribution(boxes: pd.DataFrame) -> pd.DataFrame:
    dist = (boxes.groupby("class_name", observed=True)
            .agg(boxes=("image_id", "size"), images=("image_id", "nunique"),
                 occluded_share=("is_occluded", "mean"))
            .sort_values("boxes", ascending=False))
    dist["share_of_boxes"] = (dist["boxes"] / dist["boxes"].sum()).round(4)
    dist["occluded_share"] = dist["occluded_share"].round(3)
    return dist


def imbalance_ratio(dist: pd.DataFrame) -> float:
    """Boxes in the most common class divided by boxes in the rarest one."""
    return round(dist["boxes"].max() / dist["boxes"].min(), 1)


def box_size_quantiles(boxes: pd.DataFrame) -> pd.DataFrame:
    q = (boxes.groupby("class_name", observed=True)["box_area"]
         .quantile([0.1, 0.5, 0.9]).unstack())
    q.columns = ["p10_area", "median_area", "p90_area"]
    return q.round(5)


def labels_per_image(boxes: pd.DataFrame) -> pd.Series:
    return boxes.groupby("image_id").size()


def run(processed: Path = OUT_DIR) -> dict:
    out = processed / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    boxes = load_boxes(processed / "boxes")

    dist = class_distribution(boxes)
    sizes = box_size_quantiles(boxes)
    per_image = labels_per_image(boxes)
    dist.to_csv(out / "class_distribution.csv")
    sizes.to_csv(out / "box_size_quantiles.csv")

    ax = dist["boxes"].plot.barh(figsize=(7, 4), title="Boxes per class")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out / "boxes_per_class.png", dpi=120)
    plt.close()

    per_image.clip(upper=30).plot.hist(bins=30, figsize=(7, 4),
                                       title="Boxes per image (capped at 30)")
    plt.tight_layout()
    plt.savefig(out / "boxes_per_image.png", dpi=120)
    plt.close()

    summary = {
        "boxes": int(len(boxes)),
        "images": int(per_image.size),
        "imbalance_ratio": imbalance_ratio(dist),
        "most_common_class": dist.index[0],
        "rarest_class": dist.index[-1],
        "median_boxes_per_image": float(per_image.median()),
        "p95_boxes_per_image": float(per_image.quantile(0.95)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(
        "# Label analysis\n\n"
        f"- {summary['boxes']:,} boxes across {summary['images']:,} images\n"
        f"- Most common class: {summary['most_common_class']}; rarest: {summary['rarest_class']}\n"
        f"- Imbalance ratio (most / rarest): {summary['imbalance_ratio']}x\n"
        f"- Boxes per image: median {summary['median_boxes_per_image']:.0f}, "
        f"p95 {summary['p95_boxes_per_image']:.0f}\n\n"
        "## Class distribution\n\n" + dist.to_markdown() + "\n\n"
        "## Box size quantiles (fraction of image area)\n\n" + sizes.to_markdown() + "\n"
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", default=str(OUT_DIR))
    print(json.dumps(run(Path(p.parse_args().processed)), indent=2))


if __name__ == "__main__":
    main()
