# MindBridge Memory Architecture

MindBridge separates conversation continuity from durable user memory. This
document describes the implemented baseline and the planned long-term memory
work. It is the public source of truth for memory-related code changes.

## Current state

The application currently has two working memory mechanisms:

- A PostgreSQL LangGraph checkpointer stores graph state per conversation
  (`thread_id`), including messages, tool calls, risk state, and the running
  summary.
- Token-based compaction keeps recent messages verbatim, replaces old
  re-derivable tool results with placeholders, and summarizes older turns when
  the working context exceeds its budget.

An `AsyncPostgresStore` is initialized alongside the checkpointer and passed to
the compiled graph. The privacy API can inspect and clear it, and conversation
deletion removes the matching episode key. The graph now performs read-only,
user-scoped Selection before normal chat. There is still no automatic or
user-facing long-term writer. The durable write-safety outbox and invalidation
versions are in place; structured episode Extraction is the next delivery unit.

The privacy foundation is implemented:

- The global `MEMORY_ENABLED` kill switch defaults to `false`.
- Each user's `memory_enabled` consent flag also defaults to `false`.
- `GET /api/v1/memory` lists the authenticated user's stored items, even while
  memory is disabled, so disabling never hides retained data.
- `PATCH /api/v1/memory` changes consent without depending on the Store.
- `DELETE /api/v1/memory` commits consent off before Store access, then clears
  every item under the user's prefix. It advances a data epoch and cancels
  unfinished jobs in the same first transaction. Store failure is fail-closed:
  the request reports an error, consent remains disabled, and any residual item
  belongs to an old epoch that Selection rejects after re-enabling.
- The settings UI exposes status, enable/disable, stored-item count, and clear
  controls; the authenticated API returns the transparent item payloads.

Read-only Selection is also implemented:

- Immediately before each graph run, the chat transaction reads consent,
  consent version, and data epoch in one scalar row query. This is the run's
  linearization point: a disable that completed first prevents Store access; a
  request that already observed enabled consent may finish as an in-flight
  request.
- `safety_check → summarize → select_memory → chat` is deterministic harness
  flow. Crisis turns, either disabled switch, a missing user, or a missing Store
  cause zero Selection reads.
- Selection writes only to a fresh, invocation-scoped runtime context buffer
  and returns an empty graph-state update. Tests scan every checkpoint field and
  `pending_writes` entry to ensure application state, runner metadata, and model
  error serialization never directly copy selected values or the rendered
  prompt block. A model may still repeat remembered content in its normal reply
  or tool-call arguments; those outputs are ordinary conversation state and are
  persisted by design.
- Semantic facts use exact namespace reads and strict, allow-listed schemas.
  Episodes use cosine similarity over `summary`, reject missing/invalid scores,
  and apply the configurable `MEMORY_EPISODE_MIN_SCORE` floor before rendering.
  The conservative default `0.55` is an initial Qwen3-Embedding-0.6B calibration
  and must be re-evaluated when the embedding model changes. If the embedding or
  index path is unavailable, episode recall is skipped rather than silently
  using unrelated recency results. Startup performs a content-free embedding
  capability and dimension probe; semantic facts and all privacy APIs remain
  available through the key-value Store on failure.
- Both semantic and episode Store values carry `data_epoch` (legacy/manual
  values default to epoch `0`). Selection accepts only the authenticated user's
  current epoch, so a failed clear cannot resurrect residual data after a later
  re-enable.
- Episode candidates pass a second, request-local relational guard. IDs must be
  owned by the same user, still exist, and have `memory_crisis_seen=false`.
  Missing or failed guards remove the whole episode batch while preserving
  independently validated semantic facts.
- Only approved `kind/content` and `summary/topics` fields enter a budgeted JSON
  block. Both the base system policy and the block label it as untrusted data,
  so embedded role changes, tool requests, policies, and instructions are not
  authoritative.
- LangSmith remains off by default. Enabling prompt tracing exports the rendered
  memory block with the rest of the LLM input, so a production deployment must
  apply the same consent, retention, and data-processor review to tracing.

The current fields and `memory_jobs` outbox are delivered through Alembic
revision `004`. With the default
`AUTO_CREATE_TABLES=true`, development and Docker startup safely adopt a known
legacy `create_all` schema (when no Alembic revision exists) and upgrade it to
head. Production keeps `AUTO_CREATE_TABLES=false` and runs
`uv run alembic upgrade head` as an explicit deployment step.

All durable memory belongs under one application-owned namespace:

```text
("memory", str(user_id), "semantic")
("memory", str(user_id), "episodes")
```

The namespace root prevents collisions with LangGraph or future subsystems;
the normalized user ID is the isolation boundary, and the final component is
the memory category. Inspection and deletion validate returned namespaces
before exposing or mutating them.

## Implemented write-safety foundation

The writer is not enabled yet, but its concurrency and deletion primitives are
implemented before any transcript can be sent to an extractor:

- `memory_consent_version` advances only when the enabled state really changes.
  A job captures it so disable then re-enable cannot authorize an older job.
- `memory_data_epoch` advances on clear or account deletion. It invalidates
  queued work and already-stored values independently of ordinary disable.
- `conversation.memory_revision` advances only in the same SQL transaction as
  a successfully persisted assistant reply. Failed generation, persistence,
  commit, and canceled streams do not advance it.
- `conversation.memory_crisis_seen` is sticky. The first deterministic crisis
  match commits the tombstone with the user message and enqueues a high-priority
  `delete_episode` outbox job. The relational Selection guard blocks that
  conversation immediately; physical Store deletion will be performed by the
  worker delivery unit.
- `memory_jobs` is a durable outbox with operation/status constraints, source
  and version fields, leases, retry metadata, and a unique dedupe key. Clear and
  account deletion cancel unfinished jobs transactionally.
- `DELETE /api/v1/auth/me` requires password confirmation. It first commits the
  disabled/version tombstone, then locks User → Conversations, clears the whole
  Store prefix and every known checkpoint, and only then deletes the relational
  User so conversations, messages, API keys, and jobs cascade. External failure
  leaves the disabled User row available for an idempotent retry. The Settings
  danger zone exposes this flow.

Revision `004` also makes `(user_id, provider)` unique for BYOK records. If an
older database contains duplicates, migration fails transactionally and leaves
every encrypted key untouched for an explicit operator decision.

## Architecture decision

Long-term memory will use LangGraph Store as the persistence and namespace
primitive. Memory behavior remains application-controlled:

1. **Selection** reads user-confirmed context and relevant prior episodes before
   a normal response. The read-only stage is implemented.
2. **Extraction** derives narrowly scoped, attributable memory candidates after
   completed turns.
3. **Consolidation** resolves duplicates, contradictions, and stale entries at a
   controlled cadence.

These operations are harness-triggered rather than left to optional model tool
calls. Their LLM-generated outputs are still probabilistic and must be validated
against strict schemas. A constrained `save_memory` tool may be added only after
the automatic path and privacy controls are stable.

## Privacy invariants

Mental-health conversations are sensitive. The implemented foundation enforces
consent, inspection, user-scoped deletion, and conversation cleanup before any
long-term writer exists. Extraction must not ship until the remaining
write-specific invariants below are also enforced and tested:

- Memory is explicitly enabled per user and can be disabled immediately.
- Users can inspect and delete every durable memory stored about them.
- User IDs are normalized to strings at the Store namespace boundary.
- Deleting a conversation removes its checkpoint and associated episode.
- Deleting a user removes all Store namespaces, known checkpoints, relational
  conversations/messages, API keys, and jobs owned by that user.
- Crisis-source messages and diagnostic inferences are not written to long-term
  memory in the first version.
- Stored memory is rendered as untrusted structured data, never as executable
  instructions.

Conversation generation and deletion use the same `SELECT ... FOR UPDATE` row
lock. If generation owns the lock, deletion waits and then removes the final
checkpoint and episode. If deletion owns it, a waiting generation rechecks the
row, sees that it is gone, and never invokes the graph. External Store and
checkpoint cleanup happens before the relational delete; failures propagate so
the remaining row makes the idempotent operation safe to retry.

This deliberately holds a database connection and row lock for the duration of
one generation. It serializes concurrent runs for the same conversation while
allowing different conversations to proceed independently. That cost is
acceptable for the current deployment and should be revisited if long-running
streams or per-conversation concurrency become common.

Consent changes do not cancel an already-running response. The fresh scalar
read before the graph defines the boundary: requests starting Selection after a
completed disable do not read memory, while an earlier in-flight request may
still use its prompt-local snapshot. Immediate cancellation would require a
consent epoch plus stream cancellation, not a user-row lock held for the whole
LLM call. Writer jobs use the stricter consent version and data epoch checks, so
this read allowance does not authorize a stale write.

The LangGraph checkpoint and the relational `messages` row are still committed
by separate database clients. A failure after the graph checkpoint succeeds but
before the assistant row commits can therefore make graph history lead the UI
history. The API reports the failure and never sends a successful terminal
frame, but it cannot roll the checkpoint back atomically. Before production
hardening, add an idempotent outbox/reconciliation path (or choose one store as
the sole source of truth) and fault-injection tests for this boundary.

## Planned data model

Semantic memory uses small, attributable items rather than a single unversioned
profile document. Each item records its kind, normalized content, source thread
and message IDs, whether it was explicitly stated, confidence, sensitivity,
status, and confirmation timestamps. A compact profile can be derived from
active items for prompt injection.

Episodic memory uses a dedicated rolling conversation synopsis. It must not
reuse the working-memory compaction summary: short conversations often never
compact, and a compaction summary intentionally omits details that an episode
may need for later recall.

For the current manual pilot, semantic records must be written with
`index=False` and an allow-listed value such as
`{kind, content, status="active", explicit=true, data_epoch=...}`. Episode keys
are conversation IDs and values contain at least
`{conversation_id, summary, topics, status="active", crisis=false,
data_epoch=...}`; their
`summary` field is vector-indexed. There is intentionally no public write API
until Extraction, crisis filtering, attribution, and idempotency are complete.

## Delivery order

1. **Complete:** repair checkpoint deletion and compaction boundary handling.
2. **Complete:** add consent, inspection, deletion, and schema foundations.
3. **Complete:** add user-scoped, read-only Selection with manually seeded memories.
4. **Complete:** add version/epoch gates, sticky crisis exclusion, durable outbox,
   and retry-safe account cascade.
5. **Next:** add independent episode summaries and schema-validated Extraction.
6. Add the leased outbox worker, idempotent vector upsert, and retry policy.
7. Add conflict-aware Consolidation.
8. Add a constrained explicit-memory tool if evaluation justifies it.
9. Run a frozen multi-session evaluation covering recall, contradiction updates,
   irrelevant-memory rejection, user isolation, deletion, crisis filtering, and
   persistent prompt-injection attempts.
