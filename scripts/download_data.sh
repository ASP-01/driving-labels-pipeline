#!/usr/bin/env bash
# Downloads Open Images annotation CSVs (no images). Start with the validation
# split (~25 MB); switch to train (~2 GB) once everything works.
# Check https://storage.googleapis.com/openimages/web/download_v7.html for the
# current file names and license before you run this.
set -euo pipefail
SPLIT="${1:-validation}"
mkdir -p data/raw
cd data/raw

curl -fLO https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv
if [ "$SPLIT" = "train" ]; then
  curl -fLO https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv
else
  curl -fLO https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv
fi
ls -lh
