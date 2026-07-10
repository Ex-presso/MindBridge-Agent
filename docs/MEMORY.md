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
deletion removes the matching episode key. No graph node reads or writes
cross-conversation memory yet; read-only Selection is the next delivery unit.

The privacy foundation is implemented:

- The global `MEMORY_ENABLED` kill switch defaults to `false`.
- Each user's `memory_enabled` consent flag also defaults to `false`.
- `GET /api/v1/memory` lists the authenticated user's stored items, even while
  memory is disabled, so disabling never hides retained data.
- `PATCH /api/v1/memory` changes consent without depending on the Store.
- `DELETE /api/v1/memory` commits consent off before Store access, then clears
  every item under the user's prefix. Store failure is fail-closed: the request
  reports an error, but consent remains disabled.
- The settings UI exposes status, enable/disable, stored-item count, and clear
  controls; the authenticated API returns the transparent item payloads.

The `memory_enabled` column is delivered through Alembic. With the default
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

## Architecture decision

Long-term memory will use LangGraph Store as the persistence and namespace
primitive. Memory behavior remains application-controlled:

1. **Selection** reads user-confirmed context and relevant prior episodes before
   a normal response.
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
- Deleting a user removes all Store namespaces owned by that user.
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

## Delivery order

1. **Complete:** repair checkpoint deletion and compaction boundary handling.
2. **Complete:** add consent, inspection, deletion, and schema foundations.
3. **Next:** add user-scoped, read-only Selection with manually seeded memories.
4. Add independent episode summaries and idempotent Extraction.
5. Add conflict-aware Consolidation.
6. Add a constrained explicit-memory tool if evaluation justifies it.
7. Run a frozen multi-session evaluation covering recall, contradiction updates,
   irrelevant-memory rejection, user isolation, deletion, crisis filtering, and
   persistent prompt-injection attempts.
