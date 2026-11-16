import pytest
from pyspark.sql import SparkSession

CLASSES_CSV = """LabelName,DisplayName
/m/0k4j,Car
/m/01g317,Person
/m/015qff,Traffic light
/m/0199g,Bicycle
/m/01yrx,Cat
"""

HEADER = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside"
ROWS = [
    # img1: 2 cars + 1 person, all valid
    "img1,xclick,/m/0k4j,1,0.1,0.4,0.1,0.5,0,0,0,0,0",
    "img1,xclick,/m/0k4j,1,0.5,0.9,0.2,0.6,1,0,0,0,0",
    "img1,xclick,/m/01g317,1,0.0,0.2,0.3,0.9,0,0,0,0,0",
    # img2: traffic light + bicycle + a cat (valid but not a driving class)
    "img2,xclick,/m/015qff,1,0.45,0.5,0.0,0.1,0,1,0,0,0",
    "img2,xclick,/m/0199g,1,0.2,0.6,0.4,0.9,0,0,1,0,0",
    "img2,xclick,/m/01yrx,1,0.1,0.3,0.1,0.3,0,0,0,0,0",
    # bad rows, one per rule
    "img3,xclick,/m/0k4j,1,,0.4,0.1,0.5,0,0,0,0,0",          # missing coord
    "img3,xclick,/m/0k4j,1,0.1,1.4,0.1,0.5,0,0,0,0,0",       # out of range
    "img3,xclick,/m/0k4j,1,0.6,0.4,0.1,0.5,0,0,0,0,0",       # xmax < xmin
    "img1,xclick,/m/0k4j,1,0.1,0.4,0.1,0.5,0,0,0,0,0",       # duplicate of first row
    "img4,xclick,/m/unknown,1,0.1,0.4,0.1,0.5,0,0,0,0,0",    # unknown label
]


@pytest.fixture(scope="session")
def spark():
    s = (SparkSession.builder.master("local[2]").appName("tests")
         .config("spark.sql.shuffle.partitions", "2")
         .config("spark.ui.enabled", "false").getOrCreate())
    yield s
    s.stop()


@pytest.fixture
def raw_files(tmp_path):
    boxes = tmp_path / "boxes.csv"
    boxes.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n")
    classes = tmp_path / "classes.csv"
    classes.write_text(CLASSES_CSV)
    return boxes, classes
