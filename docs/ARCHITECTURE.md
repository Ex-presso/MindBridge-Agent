# Architecture

MindBridge uses a Next.js client, a FastAPI API, and a LangGraph agent.
PostgreSQL stores application data, graph checkpoints, and user-scoped memory.
The overview diagram is in the [README](../README.md#architecture).

## Agent workflow

![Agent workflow: safety check, context compaction, memory selection, and the chat-to-retrieval loop](img/agent-graph.svg)

[Interactive diagram](img/skill/agent-graph.html).

The API authenticates the user and loads the conversation. The graph checks
for crisis phrases, manages the context budget, and selects eligible memory
before normal chat. The model chooses whether to call the retrieval tool for
counselor-style examples. Matched crisis requests take a separate response
path with support resources and bypass retrieval.

The knowledge base holds MentalChat16K question/answer pairs. Only the client
question is embedded, because the query the model writes describes a concern
the way a client would; the counselor's answer is returned whole. Matches
below a similarity floor are dropped, and a search refuses to run against an
index that is missing, incomplete, or built with another embedding model.

For longer conversations, compaction first replaces older tool results with
placeholders, then summarizes older turns if needed. Recent messages remain
verbatim. Recalled memory is added to the current model prompt rather than
copied into checkpointed graph state. A model reply may repeat recalled
details, and that reply becomes part of the saved conversation.

## Conversation state and memory

| Storage | Purpose |
|---|---|
| LangGraph checkpoints | Resume the graph state of one conversation |
| Semantic facts | Carry explicit preferences, goals, helpful strategies, and important people across conversations |
| Episodes | Recall relevant past conversations through similarity search over validated summaries |

Memory is off by default, both globally and per user. Inspection and deletion
remain available when memory is disabled. Clearing memory disables consent
and advances a data epoch. Queued jobs and stored items from the old epoch
are rejected for subsequent writes and recall. An already running chat
may finish using memory selected before the consent change.

Episode extraction accepts claims attributed to user messages. Each claim
must be an exact span of its evidence quote, and the worker checks that quote
against the relational source message before storage. Semantic facts use a
small set of explicit English patterns. This limits what the application
stores without relying on an unconstrained model-generated user profile.

## Background memory writes

![Memory write path: atomic reply and job persistence, leased processing, consent checks, and validated Store writes](img/memory-write.svg)

[Interactive diagram](img/skill/memory-write.html).

For an eligible turn, the assistant reply and extraction job commit in the
same database transaction. The job stores message IDs and version snapshots.
The worker reloads the source messages when processing it. This keeps the
extraction call off the conversation request path and preserves pending work
if the backend restarts.

The worker claims a job with `FOR UPDATE SKIP LOCKED` and commits the claim.
A lease allows another worker to reclaim abandoned work. Processing then
takes locks in `User → Conversation → Job` order and checks consent, data
epoch, conversation revision, and lease ownership before calling the model.
Writes use the conversation ID and revision to handle repeated delivery.
Completion and retry updates must match the current lease token.

The worker runs inside the backend process and polls PostgreSQL. Keeping the
queue in the application database allows the reply and job to share a
transaction. Processing holds locks across the extraction call, so consent
changes, deletion, and other operations on the affected rows can wait for it
to finish. This is a throughput and latency trade-off in the current design.

## Observability

| Signal | Where | Use |
|---|---|---|
| Request id | `X-Request-ID` response header and every log line | Join the log lines of one request |
| Metrics | `GET /metrics`, Prometheus format | Latency by route, time to first token, token usage, model call latency, retrieval outcomes |
| Traces | OTLP/HTTP export when `OTEL_EXPORTER_OTLP_ENDPOINT` is set | One trace per request, with a span for each model call and retrieval |

Logs, metric labels, and span attributes carry identifiers, counts, and
scores, never message text. Metric labels take values from fixed sets such
as route templates and provider names; model names appear only on spans.
`docker compose --profile observability up -d` starts Jaeger and Prometheus.

## Current limitations

- Crisis detection and memory content filters use English patterns. They can
  miss indirect phrasing and do not establish clinical safety.
- Contradiction resolution and stale-fact consolidation are not implemented.
- If embedding initialization fails, inspection and deletion remain available
  through the key-value Store, but episode similarity recall is disabled.
- Graph checkpoints and relational messages commit through separate clients.
  A failure between those commits can leave graph history ahead of the UI.
- Credential barriers protect outbound calls, but an idle decrypted agent
  object on another replica can remain cached until eviction. Immediate
  invalidation across replicas would need a shared channel or credential epoch.
