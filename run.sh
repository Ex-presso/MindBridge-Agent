#!/usr/bin/env bash
set -euo pipefail

# Current release evaluation. See docs/EVALUATION.md for prerequisites and scope.
cd "$(dirname "$0")/evaluation"

# Frozen routing and safety suites.
uv run --project ../backend python run_all.py

# Live API, outbox, worker, Store, and cross-conversation memory acceptance.
uv run --project ../backend python eval_memory.py
