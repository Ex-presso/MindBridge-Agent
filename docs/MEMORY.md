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
the compiled graph. It is infrastructure only: no graph node reads or writes
cross-conversation memory yet.

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

Mental-health conversations are sensitive. Long-term writes must not ship until
all of these invariants are enforced and tested:

- Memory is explicitly enabled per user and can be disabled immediately.
- Users can inspect and delete every durable memory stored about them.
- User IDs are normalized to strings at the Store namespace boundary.
- Deleting a conversation removes its checkpoint and associated episode.
- Deleting a user removes all Store namespaces owned by that user.
- Crisis-source messages and diagnostic inferences are not written to long-term
  memory in the first version.
- Stored memory is rendered as untrusted structured data, never as executable
  instructions.

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

1. Repair checkpoint deletion and compaction boundary handling.
2. Add consent, inspection, deletion, and schema foundations.
3. Add user-scoped, read-only Selection with manually seeded memories.
4. Add independent episode summaries and idempotent Extraction.
5. Add conflict-aware Consolidation.
6. Add a constrained explicit-memory tool if evaluation justifies it.
7. Run a frozen multi-session evaluation covering recall, contradiction updates,
   irrelevant-memory rejection, user isolation, deletion, crisis filtering, and
   persistent prompt-injection attempts.
