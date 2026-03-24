# Quick smoke test (3 queries, fast)
cd evaluation && uv run python run_all.py --quick

# Full evaluation (all 50 queries × all configs — takes a while)
cd evaluation && uv run python run_all.py

# Run only RAG eval or only prompting eval
cd evaluation && uv run python run_all.py --skip-prompting
cd evaluation && uv run python run_all.py --skip-rag

# Just regenerate figures from existing results
cd evaluation && uv run python run_all.py --skip-rag --skip-prompting

