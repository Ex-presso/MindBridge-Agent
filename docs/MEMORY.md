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
the compiled graph. The privacy API can inspect and clear it, conversation
deletion removes the matching episode key, and the graph performs user-scoped
Selection before normal chat. Successful opted-in turns now enqueue grounded
Episode Extraction in the assistant persistence transaction. A leased worker
reloads relational evidence, applies the privacy gates, and writes an idempotent
conversation Episode to Store. There is still no public direct-memory write API
or automatic semantic-fact writer.

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

- Immediately before each graph run, the chat transaction takes a
  `User FOR KEY SHARE` barrier and reads consent, consent version, data epoch,
  and the account-deletion tombstone from that row. It then reloads the current
  BYOK record before constructing the Agent. This statement is the run's
  memory linearization point: a disable that completed first prevents Store
  access; a request that already observed enabled consent may finish as an
  in-flight request.
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
  owned by the same user, still exist, have `memory_crisis_seen=false`, and have
  completed a review with the current deterministic crisis-detector version.
  Missing or failed guards remove the whole episode batch while preserving
  independently validated semantic facts.
- Only approved `kind/content` and `summary/topics` fields enter a budgeted JSON
  block. Both the base system policy and the block label it as untrusted data,
  so embedded role changes, tool requests, policies, and instructions are not
  authoritative.
- LangSmith remains off by default. Enabling prompt tracing exports the rendered
  memory block with the rest of the LLM input, so a production deployment must
  apply the same consent, retention, and data-processor review to tracing.

The write-safety fields and `memory_jobs` outbox start at Alembic revision `004`;
the ownership and account-deletion hardening is revision `005`. With the default
`AUTO_CREATE_TABLES=true`, development and Docker startup safely adopt a known
pre-write-safety `create_all` schema (when no Alembic revision exists) and
upgrade it to head. An unversioned schema with any write-safety marker is not
stamped from columns alone because that cannot prove checks, foreign keys, or
unique constraints. Production keeps `AUTO_CREATE_TABLES=false` and runs
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

## Implemented production writer

The Episode writer uses the following production gates and deletion primitives:

- `memory_consent_version` advances only when the enabled state really changes.
  A job captures it so disable then re-enable cannot authorize an older job.
- `memory_data_epoch` advances on clear or account deletion. It invalidates
  queued work and already-stored values independently of ordinary disable.
- `conversation.memory_revision` advances only in the same SQL transaction as
  a successfully persisted assistant reply. Failed generation, persistence,
  commit, and canceled streams do not advance it.
- A stream or non-stream provider response containing no non-whitespace
  assistant content is also a failed generation: no assistant row is written
  and `memory_revision` does not advance. The streaming endpoint emits a safe
  error frame and never emits `[DONE]` on that path.
- `conversation.memory_crisis_seen` is sticky. The first deterministic crisis
  match commits the tombstone with the user message and enqueues a high-priority
  `delete_episode` outbox job. The relational Selection guard blocks that
  conversation immediately; the high-priority worker job physically deletes
  the Store value and retries with capped backoff until it succeeds.
- Existing conversations are migrated with `memory_crisis_reviewed=false` and
  `memory_crisis_review_version=0`, so they remain ineligible. On their next
  user turn, the complete relational user history is scanned by the current
  deterministic crisis detector under the Conversation lock. The current
  `CRISIS_DETECTOR_VERSION` is `4` (version 4 narrowed detection to English
  only, matching the product's supported language); only a completed clean
  review at that version makes the conversation eligible. A detector-version
  bump automatically forces another review before Selection or Extraction.
- `memory_jobs` is a durable outbox with operation/status constraints, source
  and version fields, leases, retry metadata, and a unique dedupe key. Clear and
  account deletion cancel unfinished jobs transactionally. Composite foreign
  keys prove that its Conversation, source messages, API key, and User belong
  together; matching child indexes keep cascades bounded.
- A successful assistant row, the new `memory_revision`, and its
  `extract_episode` job commit atomically. Empty, failed, canceled, disabled, or
  crisis turns never enqueue Extraction. The job stores IDs and version/config
  snapshots, never plaintext credentials or transcript bodies.
- FastAPI lifespan runs one lightweight asyncio worker. Jobs are leased with
  `FOR UPDATE SKIP LOCKED`; `lease_until` is the ownership token, while
  `attempts` counts actual processing failures rather than queue or lock wait.
  Extraction failures stop after three attempts. Privacy `delete_episode` jobs
  remain higher priority and retry indefinitely with a capped delay, even when
  the global memory switch is off.
- The worker locks `User FOR SHARE → Conversation FOR UPDATE → MemoryJob FOR
  UPDATE`, then rechecks global/user consent, consent version, data epoch,
  account state, exact conversation revision, crisis review, lease ownership,
  and the current owned BYOK row. Provider/base-URL changes supersede the job;
  the current credential is decrypted only after all gates pass.
- Store `aget`, `aput`, and `adelete` calls have a 15-second boundary. Episode
  writes use conversation ID as the key. The same revision is an idempotent
  success, a newer revision is never overwritten, and a higher data epoch fails
  closed. A Store success followed by database commit failure converges on the
  next lease through the same-revision read.
- `DELETE /api/v1/auth/me` requires password confirmation. It first commits the
  disabled/version tombstone with `account_deletion_pending=true`, then locks
  User → Conversations, clears the whole Store prefix and every known
  checkpoint, and only then deletes the relational User so conversations,
  messages, API keys, and jobs cascade. The tombstone blocks new chat and memory
  consent work; per-user Agent cache entries and title tasks are purged before
  and after cleanup. External failure leaves the disabled pending User row
  available for an idempotent retry. The endpoint has a bounded deadline and
  returns retryable `503` rather than waiting forever on an uncooperative task.
  The Settings danger zone exposes this flow.
- API-key replacement and deletion purge the user's cached Agents and title
  tasks, then take `User FOR UPDATE`. Chat and title generation hold
  `User FOR KEY SHARE`, reload the current credential, and explicitly close
  provider streams before releasing the transaction. A successful credential
  mutation therefore cannot be followed by a new outbound call using the old
  key. Runtime eviction and the subsequent strong-lock wait are both bounded;
  timeout rolls back and returns retryable `503`.
- SQLAlchemy is pinned to PostgreSQL `READ COMMITTED`. The barrier protocol
  depends on the statement after a blocked row lock receiving a fresh snapshot;
  moving to snapshot isolation requires an explicit credential/account epoch.

Revision `004` also makes `(user_id, provider)` unique for BYOK records. If an
older database contains duplicates, migration fails transactionally and leaves
every encrypted key untouched for an explicit operator decision. Runtime upsert
infers those columns instead of relying on a particular historical constraint
name. Revision `005` adds the composite ownership constraints and deletion
barrier described above. Downgrade preflights refuse to erase non-default
versions, epochs, crisis tombstones, revisions, jobs, or an account-deletion
tombstone, and revision `005` also refuses to discard an incomplete/outdated
historical crisis review. The preflight first locks affected parent and child
tables so concurrent writes cannot cross the check-to-DDL boundary. An operator
must migrate that state explicitly before rolling back. The BYOK uniqueness
constraint is intentionally retained across a `004 → 003` downgrade because
the migration cannot prove whether an equivalent legacy constraint predated it.

## Architecture decision

Long-term memory will use LangGraph Store as the persistence and namespace
primitive. Memory behavior remains application-controlled:

1. **Selection** reads user-confirmed context and relevant prior episodes before
   a normal response. The read-only stage is implemented.
2. **Extraction** derives narrowly scoped, attributable memory candidates after
   completed turns. The structured-draft core, transactional enqueue, leased
   execution, relational revalidation, and Episode Store write are implemented.
3. **Consolidation** resolves duplicates, contradictions, and stale entries at a
   controlled cadence.

These operations are harness-triggered rather than left to optional model tool
calls. Their LLM-generated outputs are still probabilistic and must be validated
against strict schemas. The constrained `save_memory` tool remains the one
agentic hook in the C′ target, but is deferred until the automatic path and
privacy controls are stable.

### C′ alignment

The implementation still follows the original **C′ (Claude Code-aligned)**
decision. “Deterministic” describes who decides whether a memory operation runs;
it does not pretend that an internal LLM draft is deterministic:

| Subsystem | C′ trigger/authority | Current status |
|---|---|---|
| Selection | Harness runs it before every eligible chat; recall is not an optional model tool | Complete |
| Extraction | Harness runs it after every eligible completed turn; the LLM may only propose a schema-bound draft | Episode path complete; semantic path planned |
| Consolidation | Harness will run it at an explicit threshold/cadence | Planned after frozen evaluation |
| `save_memory` | The only agentic memory action in the complete C′ target | Deliberately deferred |

The original sketch placed `extract_memory` at the end of the LangGraph. The
production implementation moves that same deterministic trigger to the
assistant SQL transaction and durable outbox. This is an implementation
hardening, not an architectural drift: the model still cannot choose whether
Selection, Extraction, evidence validation, privacy filtering, leasing, or the
Store write happens. Moving the side effect out of the graph adds retries,
idempotency, deletion ordering, and crash recovery without putting memory policy
under model control.

The current milestone is therefore the **C′ episodic production slice**, not the
complete three-layer endpoint. Semantic Extraction, Consolidation, and the
single constrained `save_memory` hook remain open. Omitting that hook permanently
would be a deliberate departure from the original full C′ definition and must
be documented as such rather than silently relabeled.

The crisis policy is intentionally stricter than the early sketch: a detected
crisis does not merely skip one turn. It sets a sticky conversation tombstone,
excludes the whole Episode, and schedules physical deletion. This is a safety
hardening within the deterministic harness, not a shift of authority to the
model.

The implemented Extraction core accepts supplied source-message records plus an
optional prior `EpisodeDraft`, but places only **user-role** records in the model
prompt; supplied assistant records are ignored and cannot become evidence. The
worker reloads every current and historical citation from the relational
`(user_id, conversation_id)` scope before calling this pure core. The
core exact-matches those supplied records and never consumes assistant replies,
the LangGraph compaction summary, or the current Selection buffer. Each claim
carries a canonical `evidence_message_id` and an exact `evidence_quote`; the
claim itself must occur inside that quote, and every topic must be an exact span
of a validated claim. A model cannot silently mutate a previously cited claim.
`summary` is a deterministic join of the validated claims rather than another
model-authored field. Store values persist those immutable claims together with
`target_revision`, so stale overwrites are rejected.

OpenAI, Anthropic, Gemini, and compatible BYOK endpoints are wrapped with strict
structured output, a 2,048-token cap, a 30-second invocation timeout, and a
second Pydantic validation pass even when the provider returns an apparent model
instance. Extraction explicitly disables LangSmith callbacks and tracing. Raw
responses, transcript bodies, parsing exception text, and provider payloads are
never logged or returned. Endpoint schema capability is checked at invocation
time. Parsing/filtering failures return bounded content-safe codes; unsupported
schema, timeout, and invocation failures raise sanitized typed exceptions for
the worker's retry classifier.

Some OpenAI-compatible servers return a successful structured response with an
empty standard `content` field while placing the exact JSON object in
`reasoning_content`. MindBridge handles that response-envelope mismatch only in
the `openai_compatible` adapter: the SDK call must be a structured parse,
`content` must be empty, there must be no refusal, and the entire reasoning
field must decode with one `json.loads` call to an object. The raw reasoning is
not copied into message content or logs. Official OpenAI calls keep the native
client path, and every recovered object still passes the same `EpisodeDraft`,
grounding, crisis, diagnosis, and instruction gates. This is provider
compatibility, not additional model authority over memory.

Before any provider call, the whole draft is rejected if a supplied or prior
source trips the deterministic crisis detector. NFKC-normalized
diagnosis and persisted-instruction filters run after parsing; common text
whitespace is JSON-escaped while NUL, format, and other unsafe control
characters are rejected. Filtered or invalid drafts are not persisted. Prior
quotes are checked against full relational messages, projected into a bounded
prompt, then every returned citation is checked against the full message again.
The worker always includes the current turn and caps source input at 40 records
and 12,000 characters.

LM Studio probes on 2026-07-12 confirmed the host endpoint at
`http://127.0.0.1:1234`, live LangChain recovery for
`qwen3.5-27b-claude-4.6-opus-distilled-mlx@4bit`, and a 1,024-dimensional
response from `text-embedding-mxbai-embed-large-v1`. OrbStack services use
`http://host.docker.internal:1234/v1`. The local embedding adapter disables
client-side token-ID batching so LM Studio receives its supported string input.
The Qwen preset ignores both tested no-thinking flags, so Extraction keeps the
scoped envelope adapter and the existing 30-second timeout instead of relying
on model-specific prompt switches.

## Privacy invariants

Mental-health conversations are sensitive. The implemented writer enforces the
following invariants before a transcript reaches Extraction or Store:

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
- The supported product language is English. The crisis detector and the
  diagnosis/instruction filter vocabularies are English-only by scope; text in
  other languages is not covered by these deterministic gates.

Every chat transaction takes a `User FOR KEY SHARE` deletion barrier before any
Conversation lock or child-FK insert. Account deletion takes `User FOR UPDATE`,
commits the pending tombstone, and then reacquires User → Conversations in the
same global order. A chat that entered first completes or releases its barrier;
a chat that arrives later sees pending/deleted state and never invokes the
graph. Generation and deletion also share the Conversation `FOR UPDATE` lock,
so deletion waits for an in-flight graph and removes its final checkpoint.
External Store and checkpoint cleanup happens before the relational delete;
failures propagate so the pending row makes the idempotent operation safe to
retry.

This deliberately holds a database connection and row lock for the duration of
one generation. It serializes concurrent runs for the same conversation while
allowing different conversations to proceed independently. That cost is
acceptable for the current deployment and should be revisited if long-running
streams or per-conversation concurrency become common. Stream cancellation
explicitly closes the router, service, Agent, and provider async generators
inside-out before the transaction releases its row locks.

The database barrier and fresh credential lookup protect outbound calls across
backend processes. Per-user Agent eviction and title-task cancellation are
process-local: another replica can retain an idle decrypted Agent object until
its LRU eviction even though it cannot use that object for a post-mutation call.
A deployment that requires prompt cross-replica in-memory erasure needs a shared
invalidation channel or credential epoch.

Consent changes do not cancel an already-running response. The fresh
`READ COMMITTED` User-row read before the graph defines the boundary: requests
starting Selection after a completed disable do not read memory, while an
earlier in-flight request may still use its prompt-local snapshot. The
generation's `FOR KEY SHARE` barrier blocks account deletion and API-key
mutation but remains compatible with a non-key consent update; immediate
response cancellation would require a consent epoch plus stream cancellation.
Writer jobs use the stricter consent version and data epoch checks, so this read
allowance does not authorize a stale write.

The LangGraph checkpoint and the relational `messages` row are still committed
by separate database clients. A failure after the graph checkpoint succeeds but
before the assistant row commits can therefore make graph history lead the UI
history. The API reports the failure and never sends a successful terminal
frame, but it cannot roll the checkpoint back atomically. Before production
hardening, add an idempotent outbox/reconciliation path (or choose one store as
the sole source of truth) and fault-injection tests for this boundary.

## Stored data model

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
`{conversation_id, claims=[{claim, evidence_message_id, evidence_quote}],
summary, topics, target_revision, status="active", crisis=false,
data_epoch=...}`; `summary` must equal the deterministic join of `claims`, and its
`summary` field is vector-indexed. The automatic Episode writer produces this
shape; there is intentionally no public direct write API.
Legacy manual episode values containing only `summary/topics` remain visible to
the privacy inspection and deletion APIs, but Selection now rejects them
fail-closed. They must be re-derived with grounded `claims` and a
`target_revision` before they can be recalled again.

## Delivery order

1. **Complete:** repair checkpoint deletion and compaction boundary handling.
2. **Complete:** add consent, inspection, deletion, and schema foundations.
3. **Complete:** add user-scoped, read-only Selection with manually seeded memories.
4. **Complete:** add version/epoch gates, sticky crisis exclusion, durable outbox,
   and retry-safe account cascade.
5. **Complete:** add the pure structured episode draft, user-evidence grounding,
   deterministic crisis/diagnosis/instruction filters, and sanitized failures.
6. **Complete:** enqueue completed turns and add the leased worker, bounded
   retries, relational source reload, and idempotent Episode upsert/deletion.
7. **Complete:** accept the production paths in OrbStack through the public API
   and browser: write/recall, consent-off, clear, crisis exclusion, account
   deletion, and Store cleanup.
8. **Complete:** freeze and run a 17-case multi-session memory on/off evaluation
   covering grounded recall, irrelevant-memory rejection, user isolation,
   consent, clear, and crisis filtering. The committed local-model snapshot is
   4/6 memory-on recall, 0/6 memory-off recall, and 5/5 gates.
9. **Next:** use those results to implement only the minimum explicit semantic facts and
   conflict-aware Consolidation needed for a complete three-layer claim.
10. Implement the constrained C′ `save_memory` hook last; evaluation determines
    its narrow scope and priority, not whether an unimplemented system may be
    described as the complete C′ target.

## Next delivery plan

1. **Complete — production smoke:** `chat → outbox → worker → Store → later
   recall`, consent-off, clear, crisis exclusion, and account deletion were
   accepted through the running OrbStack stack.
2. **Complete — frozen memory evaluation:** the repository now contains the
   17-case dataset, live API runner, exact local-model configuration, row-level
   results, and summary metrics. The two retained misses distinguish response
   omission from bounded structured-Extraction failure.
3. **Next — minimal semantic layer:** store only explicit preferences, goals, helpful
   strategies, and important people with relational evidence. Never infer a
   diagnosis. Deduplicate equivalent facts and supersede conflicts.
4. **Deterministic Consolidation:** trigger by an application threshold or
   cadence, not a model tool decision. Keep the scope limited to duplicates,
   contradictions, and stale active records exposed by the evaluation.
5. **Portfolio handoff:** add one Agent graph, one memory-write sequence diagram,
   an evaluation table, reproducible commands, and resume/interview bullets.

## Running the PostgreSQL integration suite

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

Point it at a disposable database rather than a development one: the suite
creates and deletes its own `@example.com` accounts, and an autouse fixture
removes accounts a previously failed run left behind so a re-run cannot lease
an orphaned job and report a misleading result.

## Human review handoff (updated 2026-08-14)

The current review boundary adds production enqueue, the leased worker, and the
Episode Store writer to the previously reviewed safety and Extraction core.
Verify assistant/revision/job transactionality, User → Conversation → Job lock
order, lease-token ownership, exact-revision/data-epoch gates, infinite privacy
deletion retries, relational evidence matching, and Store idempotency.

The current automated evidence is `298 passed, 6 skipped` for the default
backend suite with `ruff check backend` clean under the default rules; CI now
gates the backend on that baseline. The six opt-in PostgreSQL integration tests
also pass against a migrated throwaway database: they confirm `READ COMMITTED`
sessions, that a chat's `FOR KEY SHARE` barrier genuinely blocks credential
replacement and account deletion in both orderings, that composite ownership
foreign keys reject cross-user jobs and cascade correctly, and that concurrent
workers lease disjoint jobs while a stale lease token cannot complete a
re-leased job. The two leasing invariants were falsified before being trusted:
removing `skip_locked` makes the second worker block until the test times out,
and removing the `lease_until` equality check lets a superseded worker mark the
job succeeded. The suite count reflects two cleanups: a
`tests/conftest.py` baseline now pins `MEMORY_ENABLED` so a developer `.env`
cannot change test behavior (a `.env` flip had silently broken nine chat-lock
tests), and the English-only product-scope decision removed the Chinese
detector/filter branches, their test cases, and bumped
`CRISIS_DETECTOR_VERSION` to `4`. The LM Studio compatibility increment adds
two focused regression tests, and a live LangChain schema probe recovers
`{"status":"ok"}` without copying reasoning text.

The API/browser runtime boundary was accepted on 2026-08-14 with OrbStack,
PostgreSQL/pgvector, and LM Studio. A disposable user completed
chat → transactional outbox → leased worker → Episode Store, then a second
conversation recalled both the interview topic and preferred response style.
The UI observed Store counts 0 → 1 → 2 → 0 across writes and clear-and-disable;
account deletion returned to login and the user row was physically absent.
The frozen live API evaluation now extends that trace with six memory-on/off
pairs and five privacy/Selection gates. With
`nvidia/nemotron-3-nano-4b` and `text-embedding-mxbai-embed-large-v1`, it
recorded 4/6 memory-on recalls, 0/6 memory-off recalls, 5/6 grounded writes,
and 5/5 gates. The committed row-level evidence retains one response omission
and one bounded structured-Extraction failure; see `docs/EVALUATION.md`.

Production writes occur only when both global and user consent are enabled and
all fresh relational gates pass. Remaining work is the minimal semantic layer,
conflict-aware Consolidation, and optional explicit-memory tooling.

`README.md` and this file are the tracked, authoritative documentation in the
handoff commits. `docs/develop.md`, `docs/system_arch.md`, and
`docs/memory_design.md` are intentionally excluded by the repository's existing
`.gitignore`; the first two are synchronized local workspace references, while
`memory_design.md` preserves the earlier design exploration rather than current
implementation status.
