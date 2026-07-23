.PHONY: ingest features train-baseline test all

# Phase d'ingestion batch
ingest:
	python -m src.pipeline --step ingest

# Phase de feature engineering
features:
	python -m src.pipeline --step features

# Phase d'entraînement de la baseline Isolation Forest
train-baseline:
	python -m src.pipeline --step train-baseline

# Lance la suite de tests unitaires (15+ tests requis)
test:
	python -m pytest tests/ --import-mode=importlib

# Pipeline complet
all:
	python -m src.pipeline --step all
