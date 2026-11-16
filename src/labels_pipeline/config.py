"""Shared settings for the pipeline."""
import os
import re
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "driving-images")

# Open Images display names for classes that show up in driving scenes.
DRIVING_CLASSES = [
    "Car", "Person", "Traffic light", "Traffic sign", "Stop sign", "Bicycle",
    "Motorcycle", "Bus", "Truck", "Van", "Taxi", "Ambulance",
]


def slug(name: str) -> str:
    """'Traffic light' -> 'traffic_light' so class names work as ES field names."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
