.PHONY: help install dev test test-unit lint format clean pipeline figures validate

PYTHON  ?= python3
DATASET ?= data/datasets/ewat_v3
FEATURES?= data/features/v3
SEED    ?= 42

help:
	@echo "Cibles :"
	@echo "  make install    Installe le paquet en editable"
	@echo "  make dev        Installe le paquet + les dépendances de dev"
	@echo "  make test       Lance toute la suite de tests"
	@echo "  make test-unit  Tests unitaires seuls (rapide)"
	@echo "  make lint       ruff check src scripts tests experiments v5"
	@echo "  make format     ruff format + ruff check --fix sur src scripts tests"
	@echo "  make pipeline   Encodeur -> typage -> précurseurs -> alertes"
	@echo "  make figures    Régénère les figures et tables du rapport"
	@echo "  make validate   Portes qualité sur DATASET"
	@echo "  make clean      Supprime les caches et les fichiers générés"
	@echo ""
	@echo "Variables : DATASET=$(DATASET) FEATURES=$(FEATURES) SEED=$(SEED)"

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

test-unit:
	$(PYTHON) -m pytest tests/unit -q

lint:
	$(PYTHON) -m ruff check src scripts tests experiments v5

format:
	$(PYTHON) -m ruff format src scripts tests
	$(PYTHON) -m ruff check --fix src scripts tests

# Enchaîne les étapes 1 à 3 du pipeline sur un dataset déjà assemblé.
# Suppose data/datasets/ et data/features/ présents — voir HANDOVER.md § 5.
pipeline:
	$(PYTHON) scripts/run_pipeline.py \
		--dataset $(DATASET) --features-root $(FEATURES) \
		--output experiments/thesis_run --seed $(SEED)

figures:
	$(PYTHON) -m scripts.export_thesis_figures

validate:
	$(PYTHON) -m scripts.validate_dataset --dataset $(DATASET) --strict

# Ne supprime que du régénérable. Ne touche ni à data/, ni à experiments/,
# ni à mlruns/, ni à .venv/.
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist
	find . -type d -name '*.egg-info' -not -path './.venv/*' -prune -exec rm -rf {} +
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
	find . -name '*.pyc' -not -path './.venv/*' -delete
