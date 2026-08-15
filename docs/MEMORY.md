# Memory architecture

MindBridge separates conversation continuity from durable user memory.
Conversation state is a LangGraph checkpoint per thread; durable memory is a
user-scoped store that survives across conversations and is opt-in.

This document covers the design, the invariants it holds, how it was verified,
and what is deliberately out of scope.

## Design

Memory has three operations, and the application decides when each one runs.
None of them is exposed to the model as an optional tool, so recall cannot be
skipped by a model that declines to call it:

1. **Selection** reads stored facts and relevant prior episodes before an
   eligible turn.
2. **Extraction** derives narrowly scoped, attributable candidates after a
   completed turn. An LLM may only propose a schema-bound draft; whether
   Extraction runs, what evidence is acceptable, and what reaches the store are
   all application decisions.
3. **Consolidation** would resolve contradictions and stale entries. It is not
   built; see [Limits](#limits-and-future-work).

Working memory is separate: token-based compaction keeps recent messages
verbatim, replaces re-derivable tool results with placeholders, and summarizes
older turns when the context exceeds its budget. An episode must not reuse that
compaction summary, because short conversations never compact and a compaction
summary intentionally drops detail an episode may need later.

Durable memory lives under one application-owned namespace:

```text
("memory", str(user_id), "semantic")
("memory", str(user_id), "episodes")
```

The root prevents collisions with LangGraph's own keys, the normalized user ID
is the isolation boundary, and the last component is the category. Inspection
and deletion validate returned namespaces before exposing or mutating them.

## Privacy invariants

Mental-health conversations are sensitive, so these hold before any transcript
reaches Extraction or the store:

- Memory is off by default, both globally (`MEMORY_ENABLED`) and per user.
- Users can inspect and delete everything stored about them. `GET /api/v1/memory`
  lists items even while memory is disabled, so disabling never hides retained
  data.
- Clearing commits consent off *before* touching the store, advances a data
  epoch, and cancels queued jobs in the same transaction. Store failure is
  fail-closed: the request errors, consent stays disabled, and any residual item
  belongs to an old epoch that Selection rejects.
- Deleting a conversation removes its checkpoint and its episode. Deleting an
  account removes every store namespace, checkpoint, relational row, API key,
  and job owned by that user.
- Crisis-source messages and diagnostic inferences are never written.
- Stored memory is rendered as untrusted structured data. Both the system policy
  and the rendered block label it as such, so embedded instructions, role
  changes, and tool requests carry no authority.
- The supported product language is English; the deterministic crisis and
  content filters are English-only by scope.

Selection writes only to an invocation-scoped buffer and returns an empty graph
state update, so recalled content never lands in a checkpoint by way of
Selection. A model may still repeat remembered content in its own reply, which
is ordinary conversation state and is persisted by design.

When a turn explicitly asks to remember, remind, or reflect back a saved detail,
the harness adds a prompt-local system instruction only if Selection returned
memory. It tells the model to answer from matching stored details, or say the
detail is absent, rather than incorrectly claiming that cross-session memory is
unavailable. This instruction never authorizes invention or unrelated factual
answers.

Enabling prompt tracing exports the rendered memory block along with the rest of
the LLM input, so a deployment that turns tracing on must apply the same consent
and retention review to the tracing vendor.

## Durability and concurrency

Extraction needs an LLM call, which is slow and can fail, so it cannot run on
the request path. The job is committed with the reply instead of attempted
inline.

**Transactional outbox.** A successful assistant row, its new
`conversation.memory_revision`, and the extraction job commit atomically. If the
reply is stored, the job exists. Empty, failed, canceled, disabled, and crisis
turns never enqueue. The job stores IDs and version snapshots — never
credentials or transcript bodies.

**Leased worker.** Jobs are claimed with `FOR UPDATE SKIP LOCKED`, so concurrent
workers take disjoint jobs instead of queueing behind the same row. The claim
commits immediately; `lease_until` then carries ownership for the rest of the
work. A crashed worker's lease simply lapses and another worker takes over, with
no operator involvement. Because `lease_until` must match exactly on write-back,
a stale worker that wakes up after its lease was reissued becomes a no-op rather
than overwriting the new claim.

**Lock ordering.** Every path takes `User → Conversation → Job` in that order,
which is what prevents deadlock between chat, the worker, and deletion. The
worker holds these across the provider call so that consent changes, clears,
account deletion, and credential rotation cannot interleave while a transcript
is outside PostgreSQL. That deliberately holds a connection and row lock for the
duration of one generation: acceptable at this scale, and worth revisiting if
long-running streams or per-conversation concurrency become common.

**Gates re-checked under lock.** Before the provider call the worker revalidates
global and per-user consent, consent version, data epoch, account state, the
exact conversation revision, crisis review, lease ownership, and the current
BYOK row. Credentials are decrypted only after all of them pass.

- `memory_consent_version` advances only on a real enabled/disabled change, so
  disable-then-re-enable cannot authorize an older job.
- `memory_data_epoch` advances on clear or account deletion, invalidating queued
  work and already-stored values independently of ordinary disable.
- `memory_revision` advances only in the same transaction as a persisted
  assistant reply.

**Idempotent writes.** Episode keys are conversation IDs. The same revision is an
idempotent success, a newer revision is never overwritten, and a higher data
epoch fails closed. A store write that succeeds before a database commit failure
converges on the next lease through the same-revision read. Delivery is
at-least-once; idempotency is what makes the effect exactly-once.

**Crisis exclusion.** `memory_crisis_seen` is sticky. The first deterministic
match commits a tombstone and enqueues a high-priority deletion job. The
relational guard blocks that conversation from Selection immediately, and the
deletion job retries indefinitely with capped backoff until the stored value is
physically gone — including while the global switch is off. Conversations that
predate the current `CRISIS_DETECTOR_VERSION` are ineligible until their full
history is re-reviewed, so bumping the detector automatically forces a re-scan.

**Isolation.** SQLAlchemy is pinned to `READ COMMITTED`. The barrier protocol
depends on a statement that follows a blocked row lock seeing a fresh snapshot;
moving to snapshot isolation would require an explicit credential/account epoch.

## What is stored

**Semantic facts** are small attributable items rather than one profile blob.
Fixed English patterns promote only four kinds from relational user messages —
explicit preferences, goals, helpful strategies, and important people. There is
no second model call and no diagnosis or personality inference. Each item keeps
its source thread and message IDs, data epoch, status, and version. A
deterministic key over normalized kind and content deduplicates equivalent
facts.

Promotion is additive rather than load-bearing: a candidate that cannot be
written is counted and logged, and the job proceeds to its episode, so one
unusable value cannot cost a user their episodic memory. At most 12 candidates
are promoted per turn, since each costs two store round trips inside the
lock-holding transaction. A stored value from an older data epoch is treated as
absent — its content is legitimately re-derived from the current conversation,
but metadata belonging to a record the user asked to erase is never inherited.

Facts use `index=False` because Selection reads this small allow-listed profile
exactly rather than by similarity. An unindexed namespace has no store-defined
order, so Selection reads a bounded multiple of its render budget, drops
superseded and stale-epoch values, orders the rest by last write, and only then
trims to `MEMORY_SELECT_SEMANTIC_LIMIT`. Ties break on content, so the result
never depends on the order the store happened to return. Recency is the ordering
policy, not a relevance judgment.

**Episodes** are a rolling conversation synopsis, keyed by conversation ID:

```text
{conversation_id, claims=[{claim, evidence_message_id, evidence_quote}],
 summary, topics, target_revision, status, crisis, data_epoch}
```

`summary` is a deterministic join of the validated claims rather than another
model-authored field, and it is the only vector-indexed value. Episode recall
uses cosine similarity with a configurable `MEMORY_EPISODE_MIN_SCORE` floor and
rejects missing or invalid scores; if the embedding path is unavailable, recall
is skipped rather than silently degrading to unrelated results. There is no
public direct-write API.

## Extraction guarantees

Only **user-role** messages reach the model prompt; assistant replies, the
compaction summary, and the Selection buffer can never become evidence. Each
claim carries a canonical `evidence_message_id` and an exact `evidence_quote`,
the claim must occur inside that quote, and every topic must be an exact span of
a validated claim — so a model cannot silently mutate a previously cited claim.
The worker reloads every citation from the relational `(user_id,
conversation_id)` scope and re-checks it against the full message after the
model returns.

The provider call is wrapped in strict structured output, a 2,048-token cap, a
30-second timeout, temperature `0`, and a second validation pass even when the
provider returns an apparent model instance. Tracing is explicitly disabled for
this path. Raw responses, transcript bodies, and parser exception text are never
logged or returned; failures surface as bounded, content-safe codes for the
worker's retry classifier.

DeepSeek's function-calling output can preserve the exact evidence quote while
paraphrasing the claim into third person. A narrow repair runs only after schema
parsing fails, accepts exactly one known-shape tool call, and replaces only a
non-verbatim claim with that tool call's verbatim evidence quote. The normal
relational grounding check and all privacy filters still run afterward, so
model-authored paraphrases never become stored facts.

Before any provider call the draft is rejected outright if a source trips the
deterministic crisis detector. NFKC-normalized diagnosis and
persisted-instruction filters run after parsing. Source input is capped at 40
records and 12,000 characters, and the current turn is always included.

## Verification

Claims here are limited to what was actually observed.

**Automated.** The backend suite and lint run in CI on every push.

**Concurrency, against real PostgreSQL.** Six opt-in integration tests confirm
`READ COMMITTED` sessions; that a chat's `FOR KEY SHARE` barrier genuinely
blocks credential replacement and account deletion in both orderings; that
composite ownership foreign keys reject cross-user jobs and cascade correctly;
and that concurrent workers lease disjoint jobs while a stale lease token cannot
complete a re-leased job.

The two leasing invariants were falsified before being trusted: removing
`skip_locked` makes the second worker block until the test times out, and
removing the `lease_until` equality check lets a superseded worker mark the job
succeeded. Both tests fail when the guarantee is removed, which is why they are
evidence rather than decoration.

**End to end, through the product UI.** Browser acceptance on 2026-08-16 used a
disposable account to complete registration → encrypted DeepSeek BYOK → consent
→ chat → outbox → leased worker → Store → new conversation → exact recall.
Permanent deletion returned to login, rejected the deleted credentials, and
left no matching user, Store value, or message rows.

**Release evaluation.** The current public-API suite passed 24/24 cases with
DeepSeek V4 Flash and a 768-dimensional Nomic embedding served by LM Studio:
episodic recall was 6/6 with memory on and 0/6 with memory off; all four
Semantic kinds stored and recalled; and all eight privacy, safety, isolation,
deletion, relevance, and idempotency gates passed. The runner reported zero
execution errors. See [EVALUATION.md](EVALUATION.md) for the rubric and limits.

### Running the PostgreSQL integration suite

`tests/test_memory_postgres_integration.py` is skipped by default because the
lock, constraint, and lease behavior it asserts cannot be reproduced by an
in-memory double. Run it against a throwaway database:

```bash
docker run -d --name mindbridge-itest \
  -e POSTGRES_USER=mindbridge -e POSTGRES_PASSWORD=itest \
  -e POSTGRES_DB=mindbridge_itest -p 55432:5432 pgvector/pgvector:pg16

cd backend
export DATABASE_URL="postgresql+psycopg://mindbridge:itest@localhost:55432/mindbridge_itest"
uv run alembic upgrade head
RUN_LOCAL_INTEGRATION=1 uv run pytest tests/test_memory_postgres_integration.py
```

Point it at a disposable database rather than a development one. The suite
creates and deletes its own `@example.com` accounts, and an autouse fixture
removes accounts a failed run left behind so a re-run cannot lease an orphaned
job and report a misleading result.

## Operations

Write-safety fields and the `memory_jobs` outbox arrive in Alembic revision
`004`; ownership and account-deletion hardening in `005`. With
`AUTO_CREATE_TABLES=true`, development and Docker startup adopt a known legacy
`create_all` schema and upgrade it to head; an unversioned schema carrying
write-safety markers stops with an actionable error rather than being stamped
from column names alone. Production sets `AUTO_CREATE_TABLES=false` and runs
`uv run alembic upgrade head` as an explicit release step.

Downgrade preflights refuse to erase non-default versions, epochs, crisis
tombstones, revisions, jobs, or an account-deletion tombstone, locking the
affected tables first so concurrent writes cannot cross the check-to-DDL
boundary. An operator must migrate that state explicitly before rolling back.

## Limits and future work

**Conflict-aware consolidation is not built.** Deduplication is already handled
by the deterministic key over kind and normalized content, so what remains is
superseding contradictory facts and pruning stale ones. Both need either a
second probabilistic call or brittle hand-written rules, and neither is
justified at this scale: Selection renders the most recently written facts up to
a fixed budget, which is deterministic and explainable, and the evaluation
surfaced no contradiction failures. The concrete trigger for revisiting this is
a user profile that regularly exceeds the render budget.

**No model-invoked memory write.** A constrained `save_memory` tool would let
the model request a durable write. It stays unbuilt because the automatic path
should demonstrate value first, and because every write currently passes gates
the model cannot influence.

**Checkpoint and relational history can diverge.** The LangGraph checkpoint and
the `messages` row are committed by separate database clients. A failure after
the checkpoint commits but before the assistant row does can leave graph history
ahead of the UI. The API reports the failure and never sends a successful
terminal frame, but it cannot roll the checkpoint back atomically. Closing this
needs an idempotent reconciliation path, or choosing one store as the sole
source of truth, plus fault-injection tests at that boundary.

**Cross-replica credential erasure is process-local.** The database barrier and
fresh credential lookup protect outbound calls across processes, but another
replica can retain an idle decrypted agent object until its LRU eviction. A
deployment needing prompt cross-replica erasure requires a shared invalidation
channel or a credential epoch.

**Consent changes do not cancel an in-flight response.** A request that already
observed enabled consent may finish using its prompt-local snapshot. Writer jobs
use the stricter consent-version and epoch checks, so this read allowance never
authorizes a stale write.
