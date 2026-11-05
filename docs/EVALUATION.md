# Evaluation

## Archived results

These runs were recorded on 2026-08-15 with `deepseek-v4-flash` through
DeepSeek's OpenAI-compatible API. They cover the committed cases and do not
establish general response quality or clinical effectiveness.

| Check | Scope | Result |
|---|---|---|
| Retrieval tool selection | 45 scored cases; 5 ambiguous cases excluded | 45/45 matched expected decisions; precision, recall, and F1 = 1.000 |
| Safety response markers | 20 required/forbidden-phrase probes | 20/20 passed |

The runners reported no execution errors. The archived outputs are linked
below. They predate the current retrieval index, so they describe the
evaluated configuration, not the current application.

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

## Limitations

- The routing and safety sets are small and were labeled by one author.
- Hosted model behavior can change despite the same model name and temperature.
- Phrase-based scoring needs human review for nuanced response quality.
- Memory has no reported result. The earlier memory benchmark tested an
  implementation that mem0 has replaced, and its runner was removed with it.
  The scenarios remain in `evaluation/datasets/memory_benchmark.json` for a
  new runner.
- Retrieval ranking has no reported score. The benchmark for the current
  index rewrites sampled client questions into search queries and scores the
  rank of the source example (`build_ir_benchmark.py`, `eval_retrieval.py`);
  it has not been run yet. Its scores are a lower bound: the dataset holds
  many near-identical concerns, and only the source example counts as a hit.

## Running checks

### Live model benchmarks

Add a local `DEEPSEEK_API_KEY` to `backend/.env`, start PostgreSQL, and build
the knowledge base with the embedding model named in the configuration files.

```bash
cd evaluation
uv run --project ../backend python eval_routing.py
uv run --project ../backend python eval_safety.py
uv run --project ../backend python build_ir_benchmark.py   # once, calls the model
uv run --project ../backend python eval_retrieval.py
```

Live runs call the configured model provider and overwrite result CSVs.
They are separate from CI, which runs backend tests, frontend lint/build,
and a Playwright smoke test with a local model stub.
