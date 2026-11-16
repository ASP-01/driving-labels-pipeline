BOXES ?= data/raw/validation-annotations-bbox.csv
CLASSES ?= data/raw/oidv7-class-descriptions-boxable.csv

.PHONY: install data up spark load analyze bench api test all

install:
	pip install -e ".[dev]"

data:
	./scripts/download_data.sh validation

up:
	docker compose up -d

spark:
	python -m labels_pipeline.spark_job --boxes $(BOXES) --classes $(CLASSES)

load:
	python -m labels_pipeline.es_load

analyze:
	python -m labels_pipeline.analysis

bench:
	python -m labels_pipeline.bench --runs 200

api:
	uvicorn labels_pipeline.api:app --reload

test:
	pytest -q

all: spark load analyze bench
