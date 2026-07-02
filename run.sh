#!/usr/bin/env bash
# Evaluation suite shortcuts. Needs PostgreSQL (pgvector) + LM Studio up.
# See docs/EVALUATION.md for the full methodology.

# Smoke test — 3 queries per layer
cd evaluation && uv run python run_all.py --quick

# Full run — all five layers + figures
cd evaluation && uv run python run_all.py

# (Re)build IR indexes before the retrieval layer
cd evaluation && uv run python run_all.py --build

# Skip specific layers
cd evaluation && uv run python run_all.py --skip prompting safety

# Only regenerate figures from existing result CSVs
cd evaluation && uv run python ../analysis/analyze.py
