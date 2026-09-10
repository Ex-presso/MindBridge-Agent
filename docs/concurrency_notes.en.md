# Background job concurrency notes: outbox, leases, `FOR UPDATE SKIP LOCKED`

> This covers the concurrency design of MindBridge's memory write path. Every code
> reference points at a real location in this repository, and every behavioral claim
> was verified against real PostgreSQL (see "Falsification record" at the end).

---

## 1. The problem this all exists to solve

After a user turn completes and the assistant reply succeeds, we want to distill that
turn into a durable memory. The naive approach does it inline:

```python
# Naive version — NOT what this project does
await save_assistant_message(db, reply)      # persist the reply
draft = await llm.extract_episode(messages)  # LLM call, up to 30 seconds
await store.put(namespace, key, draft)       # write the memory
```

Four problems appear immediately:

1. **The user waits.** They sit through 30 extra seconds before the turn finishes.
2. **Work gets lost.** An LLM timeout or network blip destroys the memory, and nothing
   records that it was ever owed.
3. **Work gets duplicated.** Add a retry and you may distill twice and write twice.
4. **A crash is unrecoverable.** Die on line two and the reply is stored, the memory is
   gone, and there is no record that anything is outstanding.

The fix is to move slow work into the background. But "move it to the background" raises
its own questions: **where does the task live? who runs it? what if two processes run it
at once? what if one dies halfway through?** The three concepts below answer these in
order.

---

## 2. The outbox: write the intent into the database first

### The core idea

Don't perform the side effect directly. Instead, **write "I need to do this" as a row, in
the same database transaction as the business data**.

"Same transaction" is the whole point. A transaction is all-or-nothing, so:

- The assistant reply committed → the distillation task is definitely there (nothing lost)
- The transaction rolled back → neither exists (never a task without a reply)

This downgrades "consistency between two systems" (database + LLM/memory store) into
"atomicity inside one database", and databases give you the latter for free. The pattern
is called the **transactional outbox**.

### The code

`backend/app/api/v1/chat.py:82`, `_enqueue_episode_after_success`:

```python
async def _enqueue_episode_after_success(db, *, user_id, conversation, ...):
    """Capture a successful turn in the same transaction as its revision."""
    if not settings.MEMORY_ENABLED:
        return
    access = await user_repo.get_memory_access_snapshot(db, user_id)
    if (access is None or not access.enabled or ...):
        return                       # without consent, nothing is ever enqueued
    await memory_job_repo.enqueue_extract_episode(
        db,                          # the caller's session, deliberately not a new one
        user_id=user_id,
        conversation_id=conversation.id,
        target_revision=target_revision,
        consent_version=access.consent_version,   # snapshot the consent version
        data_epoch=access.data_epoch,             # snapshot the data epoch
        source_user_message_id=user_message.id,
        ...
    )
```

**Point by point:**

- `db` is **the caller's session**, not a fresh one, so the job row and the assistant
  message row commit together. This is the linchpin: open a new connection here and the
  atomicity guarantee is gone.
- What gets enqueued is **references and snapshots, not content**. There is no
  `content=...`, only `source_user_message_id`. The queue table never holds sensitive
  transcript text; the worker reloads it from the database when it actually needs it.
- `consent_version` and `data_epoch` capture **the authorization as of enqueue time**. If
  the user disables memory while the job is queued, those numbers advance, and the worker
  compares them and cancels the job. This is what stops a queued job from quietly writing
  a memory after consent was withdrawn.

The queue table itself is defined in `backend/app/db/models/memory_job.py`: status,
priority, availability time, lease, attempt count, dedupe key, plus composite foreign keys
proving the job's conversation, messages, API key, and user all belong to the same person.

---

## 3. Leases: who owns the job, and what if they die

The task is in the table; now a worker picks it up. First question: **how do you mark
"someone is already working on this"?**

### The naive approach and why it breaks

The obvious move is a boolean-ish flag:

```sql
UPDATE memory_jobs SET status = 'processing' WHERE id = ...;
```

The problem: **what if the worker process is killed after claiming?** That row stays
`processing` forever and nobody will ever touch it again. The job is **permanently
stuck**, and recovery means a human editing the database.

### The lease approach

A lease is **a claim with an expiry**. The semantics are not "this is mine" but "**this is
mine until time T**".

```python
job.status = "processing"
job.lease_until = now + timedelta(seconds=lease_seconds)  # e.g. expires in 90 seconds
```

So:

- Worker finishes normally → it clears the lease and marks the job done
- **Worker crashes or hangs** → nobody renews, the lease expires after 90 seconds →
  another worker sees "a processing job whose lease has expired" and legitimately takes over

**No human is involved.** That is the entire value of a lease over a boolean lock: it turns
failure recovery from a manual operation into something the passage of time handles.

Configured in `backend/config/settings.py`: `MEMORY_WORKER_LEASE_SECONDS = 90` (the LLM
call is capped at 30 seconds, so this leaves 3x headroom).

---

## 4. `FOR UPDATE SKIP LOCKED`: two workers racing for the same job

Build it up in three steps.

### Step 1: no locking at all

Two workers run this simultaneously:

```sql
SELECT * FROM memory_jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1;
```

Both read **the same row**, both call the LLM, both write a memory. The job is **done
twice**: one wasted LLM call, plus potentially duplicated data.

### Step 2: `FOR UPDATE` prevents duplicates but serializes everything

`SELECT ... FOR UPDATE` means "I intend to modify these rows, so **lock them, nobody else
may touch them**". The lock lives exactly as long as the **current transaction** and is
released on commit or rollback.

```sql
SELECT * FROM memory_jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1 FOR UPDATE;
```

No more duplicates — but now something **counterintuitive** happens:

| Time | Worker A | Worker B |
|---|---|---|
| T0 | Query runs, locks job #1 ✅ | |
| T1 | Starts processing job #1 | Runs the identical query |
| T2 | | Its `ORDER BY` also **targets job #1** → sees the lock → **blocks** ⏳ |
| T3 | | Still waiting… while job #2 sits idle next to it |
| T4 | Commits, releases the lock | Unblocks, but #1 is already processing — it waited for nothing |

**B does not automatically move on to job #2.** Because `ORDER BY ... LIMIT 1` points it at
the first row too, all it can do is queue behind that one row.

Result: **run 10 workers and your throughput is still 1**, with 9 of them queueing. This is
locking turning parallelism into serialization.

An analogy: at a buffet, `FOR UPDATE` means everyone must queue behind the first dish, even
though nine other dishes are untouched.

### Step 3: `SKIP LOCKED` walks past locked rows

```sql
SELECT ... LIMIT 1 FOR UPDATE SKIP LOCKED;
```

The semantics: **if a row is already locked by someone else, don't wait — skip it and find
the next one you can take.**

| Time | Worker A | Worker B |
|---|---|---|
| T0 | Locks job #1 ✅ | |
| T1 | Processing #1 | Query: #1 is locked → **skip** → locks job #2 ✅ |
| T2 | Processing #1 | Processing #2 — **genuinely parallel** |

Back to the analogy: `SKIP LOCKED` is walking past the dishes someone is already serving
themselves from and taking the next one.

**In one sentence:** `SKIP LOCKED` lets N workers claim N **distinct** jobs, with neither
duplication nor queueing. This is the standard way to build a job queue on a relational
database — a reliable queue without introducing Redis or RabbitMQ.

### The actual code

`backend/app/db/repositories/memory_job_repo.py:98`:

```python
async def claim_next(db, *, lease_seconds, max_attempts):
    """Lease the next due job without blocking another worker."""
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(MemoryJob)
        .where(
            or_(
                and_(                                        # case 1: a fresh job
                    MemoryJob.status.in_(("pending", "retry")),
                    MemoryJob.available_at <= now,           # backoff has elapsed
                    or_(
                        MemoryJob.operation == "delete_episode",  # deletions retry forever
                        MemoryJob.attempts < max_attempts,        # everything else is capped
                    ),
                ),
                and_(                                        # case 2: reclaiming a dead lease
                    MemoryJob.status == "processing",
                    MemoryJob.lease_until <= now,            # the lease has expired
                ),
            )
        )
        .order_by(
            MemoryJob.priority.asc(),      # lower number first (delete=0, extract=100)
            MemoryJob.available_at.asc(),
            MemoryJob.created_at.asc(),
        )
        .limit(1)
        .with_for_update(skip_locked=True)   # <-- the line that matters
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None
    ...
    job.status = "processing"
    job.lease_until = now + timedelta(seconds=lease_seconds)   # issue the lease
    await db.flush()
    return job
```

The **actual SQL** that `.with_for_update(skip_locked=True)` compiles to:

```sql
SELECT memory_jobs.*
FROM memory_jobs
WHERE memory_jobs.status IN (...) AND memory_jobs.available_at <= %(available_at_1)s
      AND (memory_jobs.operation = %(operation_1)s OR memory_jobs.attempts < %(attempts_1)s)
   OR memory_jobs.status = %(status_2)s AND memory_jobs.lease_until <= %(lease_until_1)s
ORDER BY memory_jobs.priority ASC, memory_jobs.available_at ASC, memory_jobs.created_at ASC
LIMIT %(param_1)s FOR UPDATE SKIP LOCKED
```

Note `FOR UPDATE SKIP LOCKED` on the last line, and note **case 2** in the `WHERE` clause
(`status = 'processing' AND lease_until <= now`) — that is automatic takeover after a lease
expires.

---

## 5. The key distinction: what the lock covers vs. what the lease covers

The thing most easily confused: **if we already have a row lock, why do we need a lease?**

They are not redundant. They protect **two different spans of time**:

```
timeline ────────────────────────────────────────────────────────►

  |--tx 1: claim--|          |------ tx 2: process (incl. 30s LLM) ------|
     ^ row lock here       ^ commit, row lock released
       (milliseconds)         no claim lock on the row here -> the lease marks ownership
```

- **The row lock (`FOR UPDATE SKIP LOCKED`)** protects only the **claim**, for
  milliseconds. It answers "when two workers race at the same instant, who wins?"
- **The lease (`lease_until`)** protects the whole **processing** span, possibly tens of
  seconds. It answers "who owns this job during that span, and what if the owner vanishes?"

The worker loop, `backend/app/services/memory_worker.py:601`:

```python
while not stop_event.is_set():
    async with session_factory() as db:
        async with db.begin():                      # transaction 1: claim only
            claimed = await memory_job_repo.claim_next(db, ...)
    # transaction 1 commits here: claim lock released, connection back to the pool

    if claimed is not None:
        await process_claimed_job(session_factory, store, claimed, ...)  # transaction 2
        continue
```

### Why the claim must commit first

The decisive reason: **failure has to be recordable**. Look at the exception handling in
`process_claimed_job` (`memory_worker.py:514`):

```python
    except (_RetryableJobError, EpisodeModelInvocationError, EpisodeModelTimeoutError) as exc:
        code = exc.code if isinstance(exc, _RetryableJobError) else "model_failed"
        async with session_factory() as db:          # a brand new session
            async with db.begin():
                await memory_job_repo.retry_claim(
                    db, job_id=claimed.id, lease_until=lease_until,
                    max_attempts=max_attempts, error_code=code,
                )
```

If claiming and processing shared **one transaction**, then a processing failure would roll
back — and **the fact that we tried and failed would roll back with it**. `attempts` would
stay 0 forever and the job would retry endlessly. So "I claimed it, status is processing"
must be committed as a durable fact before there is anywhere to record the failure.

The exponential backoff in `retry_claim` (`memory_job_repo.py:230`):

```python
    if count_failure:
        job.attempts += 1
    if job.operation != "delete_episode" and not force_retry and job.attempts >= max_attempts:
        job.status = "blocked"                  # extraction gives up after 3 failures
    else:
        job.status = "retry"
        delay_seconds = min(300, 5 * (2 ** min(max(job.attempts - 1, 0), 6)))
        job.available_at = now + timedelta(seconds=delay_seconds)   # 5s -> 10s -> 20s ... cap 300s
    job.lease_until = None                      # hand the lease back voluntarily
```

Note that `delete_episode` (physically deleting crisis content) is excluded from
"give up after 3 failures" — **privacy deletion must retry indefinitely until it
succeeds**. That is a deliberate safety decision, not an oversight.

### One honest trade-off

The processing stage (`_process_locked_job`, `memory_worker.py:259`) **deliberately** holds
three locks — `User FOR SHARE → Conversation FOR UPDATE → MemoryJob FOR UPDATE` — across
the entire LLM call. The goal: while transcript content is outside PostgreSQL at the
provider, no consent change, memory clear, account deletion, or credential rotation can
interleave.

The cost is one connection and row lock held for the duration of a generation, which
serializes concurrent requests for the same conversation. That is acceptable for the
current deployment, and `docs/MEMORY.md` records the trade-off and the conditions for
revisiting it. This makes good interview material: **knowing what you traded for what**.

---

## 6. `lease_until` as an ownership token

One more subtle scenario: the **zombie worker**.

```
T0   Worker A claims the job, lease = 12:00:00
T1   A stalls (network hang, GC pause, host load)
T2   12:00:00 passes; the lease expires
T3   Worker B takes over the same job, lease = 12:01:30
T4   A suddenly wakes up, believes it still owns the job, tries to write "succeeded"  <-- danger
```

If A's write lands at T4 it **overwrites work B is still doing**: the job gets marked
complete while B is mid-flight, so B's result may end up orphaned, or A reports a
conclusion that expired long ago.

The fix: **make the lease timestamp itself the proof of ownership.**
`memory_job_repo.py:150`:

```python
async def get_processing_for_update(db, job_id, *, lease_until):
    """Lock a lease only when its timestamp token still belongs to this worker."""
    result = await db.execute(
        select(MemoryJob)
        .where(
            MemoryJob.id == job_id,
            MemoryJob.status == "processing",
            MemoryJob.lease_until == lease_until,   # exact equality, not >= and not IS NOT NULL
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()
```

Every completion and retry write-back goes through this gate (`finish_claim` and
`retry_claim` both call it). A holds the token `12:00:00`; the database now says
`12:01:30`; **not equal → no row matches → returns None → the write-back becomes a
no-op**. A quietly does nothing and B is unaffected.

So `lease_until` is not merely an expiry — it is also a **version number**. Every reissued
lease changes the token, which invalidates the previous owner automatically. In
distributed-systems terms this is a **fencing token**.

---

## 7. Code map

| Concept | Location |
|---|---|
| Enqueue (outbox write, same transaction as the assistant message) | `backend/app/api/v1/chat.py:82` |
| Queue table (statuses, constraints, indexes, composite FKs) | `backend/app/db/models/memory_job.py` |
| **`FOR UPDATE SKIP LOCKED` claim + lease issue** | `backend/app/db/repositories/memory_job_repo.py:98` |
| **Lease token validation** | `backend/app/db/repositories/memory_job_repo.py:150` |
| Completion write-back / retry backoff | `memory_job_repo.py:202` / `:230` |
| Worker loop (two-phase transactions) | `backend/app/services/memory_worker.py:601` |
| Failure classification, recorded in fresh sessions | `backend/app/services/memory_worker.py:514` |
| Processing stage (re-lock → revalidate gates → LLM → write) | `backend/app/services/memory_worker.py:259` |
| Real-database concurrency test | `backend/tests/test_memory_postgres_integration.py:541` |

Commands for running the real-database suite are in `docs/MEMORY.md`, under "Running the
PostgreSQL integration suite".

---

## 8. Falsification record: why the claims above are trustworthy

None of this is "should hold by design" — it was executed against real PostgreSQL 16. And
the tests themselves were **mutation-checked**: the code was deliberately broken to confirm
the tests actually fail.

| Break applied to the code | How the test reacted |
|---|---|
| Removed `skip_locked=True` | The second worker blocked → `wait_for` timed out → failed ✅ |
| Removed `lease_until == lease_until` | The zombie worker successfully overwrote the new lease → assertion failed ✅ |

Both failed as predicted, which proves the test measures these two properties rather than
passing by coincidence. Restored, 6/6 pass.

An incidental lesson: the first mutation run timed out mid-test and **left orphan rows in
the database**, which made the second verification fail at the wrong assertion and produce
an invalid conclusion. Only after cleaning the database and redoing it did the result
become meaningful. An autouse fixture now removes leftover `@example.com` accounts (scoped
deletion, no `TRUNCATE`, so pointing it at the wrong database cannot wipe real data).
**"The test passes" and "the test means something" are two different claims.**

---

## 9. The 30-second interview version

> Memory distillation needs an LLM call, which is slow and can fail, so it can't live on
> the request path. I used a transactional outbox: the job row is written in the same
> database transaction as the assistant message, so work is never lost. A background worker
> claims jobs with `FOR UPDATE SKIP LOCKED` — `SKIP LOCKED` is the important part, because
> with plain `FOR UPDATE` multiple workers queue behind the same row and parallelism
> collapses to one. On claim it issues a lease with an expiry, so if a worker crashes the
> lease lapses and another worker takes over with no human intervention. The lease timestamp
> doubles as a fencing token: when a zombie worker wakes up its timestamp no longer matches,
> so its write-back is a no-op. Since delivery is at-least-once, the Store write is
> idempotent on revision, so a repeat never produces duplicate data.

That last sentence is the differentiator: **at-least-once delivery plus idempotent writes
gives you effectively-once**. That argument is far stronger than claiming you "implemented
exactly-once", which strictly speaking is not possible.

---

## 10. Follow-up questions worth preparing

Common next-layer interview questions, not yet covered in these notes:

1. **Why does the processing stage take three locks in a fixed `User → Conversation → Job`
   order?** A globally consistent lock order is the standard way to prevent deadlock; two
   paths that acquire the same locks in different orders can wait on each other forever.
2. **Database-as-queue vs. Redis / RabbitMQ / SQS.** The database queue's decisive advantage
   is exactly section 2: it can share a transaction with the business data, so nothing is
   ever lost. The costs are a throughput ceiling and polling overhead.
3. **Fairness under `SKIP LOCKED`.** It does not guarantee FIFO; under heavy contention some
   rows can be skipped repeatedly. Not a concern at this project's scale.
