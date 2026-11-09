SHELL := /bin/bash

PYTHON ?= python3
CONDA ?= conda
CFFCONVERT ?= cffconvert
FIGURE ?= all
DATA_DIR ?= data/panels/current
OUTPUT_DIR ?= outputs/figures/current

.PHONY: help install-figures create-training-env update-training-env \
	reproduce-figures validate-release validate validate-metadata \
	validate-python validate-files audit-paths test

help:
	@echo "Available targets:"
	@echo "  install-figures      Install dependencies used by analyses and figures"
	@echo "  create-training-env  Create the downstream-training environment"
	@echo "  update-training-env  Update an existing Conda environment"
	@echo "  reproduce-figures    Replay current quantitative panels; e.g. FIGURE=Fig5d"
	@echo "  validate-release     Check frozen strict401 data and scientific invariants"
	@echo "  test                 Run offline unit tests; no training or live retrieval"
	@echo "  validate             Validate metadata, Python syntax, and file sizes"
	@echo "  audit-paths          Report machine-specific absolute paths in code"

install-figures:
	$(PYTHON) -m pip install -r requirements-figures.txt

create-training-env:
	$(CONDA) env create -f environment-training.yml

update-training-env:
	$(CONDA) env update -f environment-training.yml --prune

reproduce-figures:
	@test -f scripts/reproduce_figures.py || { echo "Missing scripts/reproduce_figures.py"; exit 2; }
	@if [[ "$(FIGURE)" == "all" ]]; then selector="--all"; else selector="--figure $(FIGURE)"; fi; \
	$(PYTHON) scripts/reproduce_figures.py $$selector --data-dir "$(DATA_DIR)" --output-dir "$(OUTPUT_DIR)"

validate-release:
	@test -f scripts/validate_release.py || { echo "Missing scripts/validate_release.py"; exit 2; }
	$(PYTHON) scripts/validate_release.py --root .

test:
	$(PYTHON) -m unittest discover -s tests -v

validate-metadata:
	$(PYTHON) -c 'import importlib.util, pathlib; tomllib=__import__("tomllib" if importlib.util.find_spec("tomllib") else "tomli"); tomllib.loads(pathlib.Path("pyproject.toml").read_text())'
	$(PYTHON) -c 'import pathlib, yaml; [yaml.safe_load(pathlib.Path(p).read_text()) for p in ("environment-training.yml", ".github/workflows/validate.yml")]'
	$(PYTHON) -c 'from pathlib import Path; from packaging.requirements import Requirement; lines=[x.strip() for x in Path("requirements-figures.txt").read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]; [Requirement(x) for x in lines]'
	$(PYTHON) -c 'from pathlib import Path; import yaml; from packaging.requirements import Requirement; cfg=yaml.safe_load(Path("environment-training.yml").read_text()); pip=next(x["pip"] for x in cfg["dependencies"] if isinstance(x, dict) and "pip" in x); [Requirement(x) for x in pip if not x.startswith("-r ")]'
	$(CFFCONVERT) --validate

validate-python:
	$(PYTHON) -c 'import ast; from pathlib import Path; files=[p for d in ("scripts", "pipeline", "tests") for p in Path(d).rglob("*.py")]; [ast.parse(p.read_text(), filename=str(p)) for p in files]; print("Parsed", len(files), "Python files without executing them")'

validate-files:
	@bad="$$(find . -path ./.git -prune -o -type f -size +95M -print)"; \
	if [[ -n "$$bad" ]]; then \
		echo "Files larger than 95 MB must not be committed:"; \
		echo "$$bad"; \
		exit 1; \
	fi

audit-paths:
	@dirs=""; for d in scripts pipeline; do [[ -d "$$d" ]] && dirs="$$dirs $$d"; done; \
	if [[ -n "$$dirs" ]]; then \
		rg -n '(/home/[^/]+/|/Users/[^/]+/)' $$dirs -g '*.py' -g '*.sh' || true; \
	fi

validate: validate-metadata validate-python validate-files
