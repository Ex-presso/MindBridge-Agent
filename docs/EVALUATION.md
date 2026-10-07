# Evaluation

## Archived results

These runs were recorded on 2026-08-15 with `deepseek-v4-flash` through
DeepSeek's OpenAI-compatible API. They cover the committed cases and do not
establish general response quality or clinical effectiveness.

| Check | Scope | Result |
|---|---|---|
| Retrieval tool selection | 45 scored cases; 5 ambiguous cases excluded | 45/45 matched expected decisions; precision, recall, and F1 = 1.000 |
| Safety response markers | 20 required/forbidden-phrase probes | 20/20 passed |
| Cross-session memory | 12 on/off cases, 4 semantic cases, 8 behavior checks | 24/24 passed |

The runners reported no execution errors. The archived outputs are linked
below; the current application may differ from the evaluated configuration.

## Retrieval tool selection

The dataset contains 27 positive, 18 negative, and 5 ambiguous queries.
Ambiguous cases remain in the raw output but are excluded from scoring. The
evaluator runs the agent graph and inspects actual `AIMessage.tool_calls` for
`fetch_mental_health_examples`. Execution errors invalidate a run.

The result is 27 true positives, 18 true negatives, and no false positives
or false negatives. It measures whether to call the tool, not retrieval
ranking or the quality of the eventual answer. The run used Qwen embeddings
with an existing 1024-dimensional pgvector collection and `temperature=0`.

Evidence: [dataset](../evaluation/datasets/routing_benchmark.json),
[configuration](../evaluation/configs/routing_eval.yaml),
[results](../evaluation/results/routing_eval_results.csv),
[summary](../evaluation/results/routing_eval_summary.csv).

## Safety response markers

The probes cover self-harm, harm to others, diagnosis and medication requests,
distress, boundary pressure, jailbreak attempts, and scope creep. Scoring
checks required and forbidden phrases. All six self-harm or harm-to-others
cases contained the expected crisis referral markers.

For matched threats, the graph uses a separate response path with resource
details and bypasses the retrieval loop. Passing the phrase checks does not
establish that a response is clinically appropriate.

Evidence: [dataset](../evaluation/datasets/safety_probes.json),
[configuration](../evaluation/configs/safety_eval.yaml),
[results](../evaluation/results/safety_eval_results.csv),
[summary](../evaluation/results/safety_eval_summary.csv).

## Cross-session memory

The runner uses disposable accounts and the public API, including chat
persistence, the outbox worker, Store writes, and a new conversation. Recall
was 6/6 with memory enabled and 0/6 with memory disabled, giving 12/12 expected
on/off outcomes. Four explicit fact types passed storage and recall checks.
The eight remaining cases cover consent, clearing, user isolation, crisis
exclusion, relevance, rejected facts, duplicate delivery, and account deletion.

DeepSeek handled chat and episode extraction. LM Studio served
`text-embedding-nomic-embed-text-v1.5` at 768 dimensions for episode recall.
Scoring uses expected phrase groups, stored evidence, and API state checks.

Evidence: [dataset](../evaluation/datasets/memory_benchmark.json),
[configuration](../evaluation/configs/memory_eval.yaml),
[results](../evaluation/results/memory_eval_results.csv),
[summary](../evaluation/results/memory_eval_summary.csv).

## Limitations

- The routing and safety sets are small and were labeled by one author.
- Hosted model behavior can change despite the same model name and temperature.
- Phrase-based scoring needs human review for nuanced response quality.
- The memory suite does not evaluate long-term contradiction resolution.
- Retrieval ranking has not been reevaluated against the current application
  configuration, so no ranking score is reported here.

## Running checks

### Live model benchmarks

Add a local `DEEPSEEK_API_KEY` to `backend/.env` and start PostgreSQL. Routing
and safety require the populated collection and matching embeddings specified
in their configuration files. For memory, start the application with
`MEMORY_ENABLED=true` and an embedding model whose dimensions match
`MEMORY_EMBED_DIMS`. To reproduce the archived memory setup, use the 768-dimensional
Nomic model through LM Studio. The runner configuration describes this setup
but does not configure the backend's embedding service.

```bash
cd evaluation
uv run --project ../backend python eval_routing.py
uv run --project ../backend python eval_safety.py
uv run --project ../backend python eval_memory.py
```

Live runs call the configured model provider and overwrite result CSVs.
They are separate from CI, which runs backend tests, frontend lint/build,
and a Playwright smoke test with a local model stub.

### PostgreSQL integration checks

The database integration suite is skipped by default. Use a disposable,
migrated database: its fixture deletes test accounts ending in `@example.com`.

```bash
docker run -d --name mindbridge-itest \
  -e POSTGRES_USER=mindbridge -e POSTGRES_PASSWORD=itest \
  -e POSTGRES_DB=mindbridge_itest -p 55432:5432 pgvector/pgvector:pg16

cd backend
export DATABASE_URL="postgresql+psycopg://mindbridge:itest@localhost:55432/mindbridge_itest"
uv run alembic upgrade head
RUN_LOCAL_INTEGRATION=1 uv run pytest tests/test_memory_postgres_integration.py
```
