# Evaluation

## Release evidence

These results were generated on 2026-08-15 with `deepseek-v4-flash` through
DeepSeek's OpenAI-compatible API.

| Capability | Frozen scope | Result |
|---|---:|---:|
| Conditional RAG routing | 45 scored hand-labeled cases | Precision **1.000**, recall **1.000**, F1 **1.000** |
| Safety behavior | 20 required/forbidden-marker probes | **20/20 passed** |
| Episodic memory | 6 paired cross-session scenarios | Memory on **6/6**, memory off **0/6** |
| Semantic memory | 4 explicit fact kinds | Storage **4/4**, cross-session recall **4/4** |
| Memory hard gates | 8 privacy, safety, isolation, deletion, and idempotency cases | **8/8 passed** |

All runners completed without execution errors. The scores apply only to the
committed cases; they do not measure general mental-health quality.

## 1. Conditional RAG routing

The routing benchmark asks whether the LangGraph agent calls
`fetch_mental_health_examples` once for a substantive mental-health support
request and abstains for greetings, conversation closure, assistant-meta
questions, and unrelated factual requests.

### Design

- 50 hand-labeled queries: 27 positive, 18 negative, and 5 ambiguous.
- Ambiguous cases remain in the raw output but are excluded from the headline
  confusion matrix.
- The full agent graph runs with `AgentRunContext`; the evaluator inspects
  actual `AIMessage.tool_calls` and rejects any run containing an execution
  error.
- The RAG tool executes against PostgreSQL/pgvector. The configured Qwen
  embedding matches the existing 1024-dimensional evaluation collection.
- `temperature=0` is passed when the model is created.

### Result

| n | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 45 | 27 | 0 | 0 | 18 | **1.000** | **1.000** | **1.000** | **1.000** |

This metric measures the binary tool-routing decision only. It does not measure
retrieval ranking, response quality, or memory quality.

Evidence:

- `evaluation/results/routing_eval_results.csv`
- `evaluation/results/routing_eval_summary.csv`
- `evaluation/results/routing_eval_by_category.csv`

## 2. Safety probes

The safety suite covers self-harm, harm-to-others, diagnosis requests,
medication requests, minor distress, boundary pressure, jailbreak attempts,
and scope creep. Each probe defines acceptable and forbidden response markers.

### Result

| Category | n | Pass rate | Crisis referral | Professional referral |
|---|---:|---:|---:|---:|
| Self-harm | 4 | **100%** | **100%** | **100%** |
| Harm-to-others | 2 | **100%** | **100%** | **100%** |
| All probes | 20 | **100%** | 35% | 35% |

The overall referral percentages are not expected to be 100%: only crisis
categories should receive crisis resources. The deterministic `safety_check`
node runs before normal chat, bypasses the RAG loop for matched threats, and
appends resource details independently of model wording.

The marker rubric treats `mental health professional`, `crisis line`, and `988`
as professional referrals. The phrase `take the weight off` is not treated as
prescription advice.

Evidence:

- `evaluation/results/safety_eval_results.csv`
- `evaluation/results/safety_eval_summary.csv`

## 3. Memory product acceptance

The memory benchmark uses the public API and disposable accounts. Each case
passes through chat persistence, the transactional outbox, a leased worker,
LangGraph Store, and a new conversation. DeepSeek performs chat and episode
Extraction; LM Studio serves `text-embedding-nomic-embed-text-v1.5` at 768
dimensions for episodic Selection.

### Result

| Metric | Result |
|---|---:|
| Episodic recall with memory on | **6/6** |
| Episodic recall with memory off | **0/6** |
| Grounded episodic storage | **6/6** |
| Paired expected behavior | **12/12** |
| Semantic storage and recall | **4/4** |
| Privacy/safety/idempotency gates | **8/8** |
| Overall acceptance | **24/24** |
| Execution errors | **0/24** |

The four Semantic kinds are explicit preference, goal, helpful strategy, and
important person. The eight hard gates cover consent-off behavior, physical
clear plus data-epoch invalidation, user isolation, crisis exclusion,
irrelevant-memory rejection, semantic rejection, duplicate-delivery
idempotency, and account deletion with Store cleanup.

Evidence:

- `evaluation/results/memory_eval_results.csv`
- `evaluation/results/memory_eval_summary.csv`

## Limitations

- The routing and safety sets are small and were labeled by one author.
- DeepSeek is a hosted model name rather than an immutable model artifact; a
  provider update can change generative behavior even with `temperature=0`.
- Marker and expected-group scoring is auditable but lexical. Human review is
  still required for clinical appropriateness and nuanced response quality.
- The memory suite verifies supported behaviors and hard gates, not long-term
  contradiction resolution. Conflict-aware Consolidation is not implemented.
- A retrieval-ranking score is intentionally not reported until the optional IR
  benchmark is rerun against the release configuration.

## Reproduction

Copy `backend/.env.example` to `backend/.env`, add a local
`DEEPSEEK_API_KEY`, and start PostgreSQL. Routing and safety also require the
configured pgvector collection to be populated with its matching embedding.

```bash
cd evaluation
uv run --project ../backend python eval_routing.py
uv run --project ../backend python eval_safety.py
```

For memory acceptance, start the full application with
`MEMORY_ENABLED=true`, run the configured LM Studio embedding model, and then
execute:

```bash
uv run --project ../backend python eval_memory.py --max-recall 1 --skip-gates
uv run --project ../backend python eval_memory.py
```

The live runner paces disposable account creation and cleanup to stay within the
configured authentication rate limit.
