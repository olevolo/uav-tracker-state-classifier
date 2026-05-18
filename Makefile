# UAV Entropy-Guided Tracker — Makefile
# -----------------------------------------------------------------------------
# Single source of truth for local developer ergonomics. Every target here
# should Just Work on a clean macOS (Apple Silicon or Intel) or Ubuntu 22.04
# box after a shell restart. CI invokes these same targets to keep behavior
# identical between developer machines and GitHub runners.
# -----------------------------------------------------------------------------

SHELL := /bin/bash

# Tool discovery — fail loudly rather than silently using system Python.
PYENV := $(shell command -v pyenv 2>/dev/null)
UV := $(shell command -v uv 2>/dev/null)
PRE_COMMIT := .venv/bin/pre-commit
PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
MYPY := .venv/bin/mypy

PYTHON_VERSION := $(shell cat .python-version)

.PHONY: help
help:
	@echo "UAV Entropy-Guided Tracker — developer targets"
	@echo ""
	@echo "  setup            Bootstrap pyenv/uv, create .venv, install deps + hooks"
	@echo "  compile-deps     Regenerate requirements*.txt from pyproject.toml via uv"
	@echo "  lint             ruff check + ruff format --check"
	@echo "  fmt              ruff format (auto-fix style)"
	@echo "  typecheck        mypy src/"
	@echo "  test             pytest unit + property"
	@echo "  test-contract    pytest tests/contract"
	@echo "  test-integration pytest tests/integration"
	@echo "  smoke-eval       3-frame synthetic smoke (<5 s, no dataset needed)"
	@echo "  smoke-eval-uav123 3-sequence UAV123 OPE smoke (requires subset download)"
	@echo "  bench            pytest-benchmark"
	@echo "  demo             Phase 8 stub"
	@echo "  reproduce        Phase 7 — regenerate all tables (synthetic + uav123 if available)"
	@echo "  reproduce-fast   Phase 7 — synthetic only, limit=2 (<2 min)"
	@echo "  docker-cpu       Build CPU image from infra/docker/Dockerfile.cpu"
	@echo "  docker-gpu       Build GPU image from infra/docker/Dockerfile.gpu"
	@echo "  docker-jetson    Build Jetson image from infra/docker/Dockerfile.jetson"
	@echo "  shell-gpu        Drop into GPU container (requires nvidia-container-toolkit)"
	@echo "  list-plugins     uav-tracker list-plugins"
	@echo "  clean            Remove build artifacts + caches"

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
.PHONY: setup
setup:
ifndef PYENV
	$(error "pyenv not found. Install via: brew install pyenv (macOS) or https://github.com/pyenv/pyenv#installation")
endif
ifndef UV
	$(error "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh")
endif
	@echo "[setup] pyenv install $(PYTHON_VERSION) if missing…"
	@pyenv install --skip-existing $(PYTHON_VERSION)
	@echo "[setup] creating .venv with Python $(PYTHON_VERSION)…"
	@uv venv .venv --python $(PYTHON_VERSION)
	@echo "[setup] syncing deps from pyproject.toml…"
	@uv pip install --python .venv/bin/python -e ".[dev]"
	@echo "[setup] installing pre-commit hooks…"
	@$(PRE_COMMIT) install --install-hooks --hook-type pre-commit --hook-type commit-msg
	@echo "[setup] done. Activate with: source .venv/bin/activate"

.PHONY: compile-deps
compile-deps:
ifndef UV
	$(error "uv not found. See 'make setup' for install instructions.")
endif
	@uv pip compile pyproject.toml -o requirements.txt
	@uv pip compile pyproject.toml --extra dev -o requirements-dev.txt

# -----------------------------------------------------------------------------
# Quality gates
# -----------------------------------------------------------------------------
.PHONY: lint
lint:
	@$(RUFF) check src/ tests/ scripts/
	@$(RUFF) format --check src/ tests/ scripts/

.PHONY: fmt
fmt:
	@$(RUFF) format src/ tests/ scripts/
	@$(RUFF) check --fix src/ tests/ scripts/

.PHONY: typecheck
typecheck:
	@$(MYPY) src/

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
.PHONY: test
test:
	@$(PYTEST) tests/ -m "unit or property" -v

.PHONY: test-contract
test-contract:
	@$(PYTEST) tests/contract/ -v

.PHONY: test-integration
test-integration:
	@$(PYTEST) tests/integration/ -v

.PHONY: bench
bench:
	@$(PYTEST) tests/ -m "benchmark" --benchmark-only

# -----------------------------------------------------------------------------
# Phase-gated stubs (implementations land in their respective phases)
# -----------------------------------------------------------------------------

# UAV_DATA_ROOT — read from environment (default: ./data). Always quoted in
# shell expansions to guard against paths with spaces.
UAV_DATA_ROOT ?= $(PWD)/data

.PHONY: smoke-eval
smoke-eval:
ifndef PYTHON
	$(error Run 'make setup' first — no .venv found at $(PYTHON))
endif
	@test -f "$(PYTHON)" || (echo "[smoke-eval] ERROR: $(PYTHON) not found. Run 'make setup' first." && exit 1)
	@echo "[smoke-eval] running 3-frame synthetic smoke (CPU, <5 s) …"
	@"$(PYTHON)" -m uav_tracker.cli evaluate \
		--tracker kcf_kalman \
		--dataset synthetic \
		--limit 3
	@echo "[smoke-eval] done."

# smoke-eval-uav123: real UAV123 data path — requires subset pre-downloaded.
.PHONY: smoke-eval-uav123
smoke-eval-uav123:
ifndef PYTHON
	$(error Run 'make setup' first — no .venv found at $(PYTHON))
endif
	@test -f "$(PYTHON)" || (echo "[smoke-eval-uav123] ERROR: $(PYTHON) not found. Run 'make setup' first." && exit 1)
	@test -d "$(UAV_DATA_ROOT)/uav123/bike1" || \
		(echo "[smoke-eval-uav123] ERROR: $(UAV_DATA_ROOT)/uav123/bike1 not found." && \
		 echo "  Fetch the subset first:" && \
		 echo "    python scripts/download_datasets.py uav123 --subset" && \
		 echo "  (Populate real SHA256s in scripts/manifests/uav123_subset.sha256 first.)" && \
		 exit 1)
	@echo "[smoke-eval-uav123] running 3-sequence UAV123 smoke (CPU) …"
	@"$(PYTHON)" -m uav_tracker.cli evaluate \
		--tracker kcf_kalman \
		--dataset uav123 \
		--limit 3
	@echo "[smoke-eval-uav123] done."

.PHONY: demo
demo:
	@echo "[demo] TODO: implement in Phase 8 per PLAN §11"

# UAV_RESULTS_ROOT — read from environment (default: ./results).
UAV_RESULTS_ROOT ?= $(PWD)/results

# reproduce: full Table 2 sweep — synthetic (always) + uav123 (if data present).
.PHONY: reproduce
reproduce:
ifndef PYTHON
	$(error Run 'make setup' first — no .venv found at $(PYTHON))
endif
	@test -f "$(PYTHON)" || (echo "[reproduce] ERROR: $(PYTHON) not found. Run 'make setup' first." && exit 1)
	@echo "[reproduce] Phase 7: regenerating all paper tables…"
	@UAV_RESULTS_ROOT="$(UAV_RESULTS_ROOT)" "$(PYTHON)" scripts/run_benchmark.py \
		--sweep configs/experiments/paper_table2.yaml \
		--dataset synthetic \
		--out-dir "$(UAV_RESULTS_ROOT)"
	@if [ -d "$(UAV_DATA_ROOT)/uav123" ]; then \
		echo "[reproduce] UAV123 data found — running uav123 dataset…"; \
		UAV_RESULTS_ROOT="$(UAV_RESULTS_ROOT)" "$(PYTHON)" scripts/run_benchmark.py \
			--sweep configs/experiments/paper_table2.yaml \
			--dataset uav123 \
			--out-dir "$(UAV_RESULTS_ROOT)"; \
	else \
		echo "[reproduce] UAV123 data not found at $(UAV_DATA_ROOT)/uav123 — skipping real dataset."; \
	fi
	@echo "[reproduce] done. See results under $(UAV_RESULTS_ROOT)/"

# reproduce-fast: synthetic only, max 2 sequences per variant (<2 min).
.PHONY: reproduce-fast
reproduce-fast:
ifndef PYTHON
	$(error Run 'make setup' first — no .venv found at $(PYTHON))
endif
	@test -f "$(PYTHON)" || (echo "[reproduce-fast] ERROR: $(PYTHON) not found. Run 'make setup' first." && exit 1)
	@echo "[reproduce-fast] Phase 7 fast path: synthetic only, limit=2…"
	@UAV_RESULTS_ROOT="$(UAV_RESULTS_ROOT)" "$(PYTHON)" scripts/run_benchmark.py \
		--sweep configs/experiments/paper_table2.yaml \
		--dataset synthetic \
		--limit 2 \
		--out-dir "$(UAV_RESULTS_ROOT)"
	@echo "[reproduce-fast] done."

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------
.PHONY: docker-cpu
docker-cpu:
	@docker buildx build -f infra/docker/Dockerfile.cpu -t uav-tracker:cpu .

.PHONY: docker-gpu
docker-gpu:
	@docker buildx build -f infra/docker/Dockerfile.gpu -t uav-tracker:gpu .

.PHONY: docker-jetson
docker-jetson:
	@docker buildx build -f infra/docker/Dockerfile.jetson --platform linux/arm64 -t uav-tracker:jetson .

.PHONY: shell-gpu
shell-gpu:
	@docker run --rm -it --gpus all -v "$(PWD)":/app uav-tracker:gpu bash

# -----------------------------------------------------------------------------
# CLI passthroughs
# -----------------------------------------------------------------------------
.PHONY: list-plugins
list-plugins:
	@$(PYTHON) -m uav_tracker.cli list-plugins

# -----------------------------------------------------------------------------
# Housekeeping
# -----------------------------------------------------------------------------
.PHONY: clean
clean:
	@rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .hypothesis/
	@rm -rf outputs/ multirun/ htmlcov/ coverage.xml .coverage
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
