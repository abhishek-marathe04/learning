# System Design Study Guide
> Abhishek's personal interview prep notes — updated after each study session.
> Last updated: 2026-05-06 (Session 4 added)

---

## How to use this guide
- Each session adds a new dated section under the relevant topic
- Use the Table of Contents to jump to any topic
- Topics build on each other — read in order for first pass, jump around for revision

---

## Table of Contents
1. [Database Selection](#1-database-selection)
2. [Distributed Systems Failure Modes](#2-distributed-systems-failure-modes)
3. [Latency Diagnosis & Performance](#3-latency-diagnosis--performance)
4. [Database Principles — Battle-Hardened Rules](#4-database-principles--battle-hardened-rules)

---

## 1. Database Selection
*Session: 2026-05-06*

### The 5-axis framework

When picking a database in a system design interview, reason through these five dimensions in order:

#### 1. Data model
What shape is your data?
- **Relational (Postgres, MySQL)** — structured, tabular, clear relationships between entities
- **Document (MongoDB)** — flexible, nested/hierarchical data, schema-free
- **Graph (Neo4j)** — highly connected entities where relationships are first-class (social graphs, fraud detection)
- **Time-series (InfluxDB, TimescaleDB)** — append-only, timestamped metrics or events
- **Key-value (Redis, DynamoDB)** — simple lookups by a single key, no joins needed
- **Wide-column (Cassandra, HBase)** — large-scale, sparse data with known access patterns

#### 2. Read/write pattern
- **Read-heavy** → add read replicas, caching layer (Redis), or CDN for static data
- **Write-heavy** → needs high write throughput; Cassandra, DynamoDB are designed for this
- **Mixed** → relational DBs handle this well at moderate scale; partition reads/writes at high scale
- **Append-only** → time-series or log databases; avoid update-heavy designs

#### 3. Scale
- **Vertical scaling** — bigger machine; works for SQL up to a point (~TBs, thousands of QPS)
- **Horizontal scaling** — add more nodes; NoSQL (Cassandra, DynamoDB) designed for this natively
- **Sharding SQL** — possible but operationally complex; avoid unless necessary
- Key signals to note: DAU, QPS, data size now vs 5 years from now

#### 4. Consistency needs (CAP theorem)
- **Strong consistency / ACID** — banking, inventory, bookings, anything where stale reads cause business loss → use relational DB
- **Eventual consistency / BASE** — social feeds, analytics dashboards, likes/views counters → NoSQL is fine
- Remember: CAP theorem says you can only have 2 of: Consistency, Availability, Partition tolerance. Networks always partition → real choice is C vs A.

#### 5. Query pattern
- **Complex joins + aggregations** → SQL is far more expressive
- **Single-key lookups, no joins** → key-value store (Redis, DynamoDB) — simpler and faster
- **Full-text search** → Elasticsearch or OpenSearch
- **Range queries on time** → time-series DB
- **Graph traversal** → Neo4j or graph extensions

---

### Quick reference: Database → use case mapping

| Database | Best for |
|---|---|
| PostgreSQL | Transactions, complex queries, general-purpose relational |
| MySQL | Web apps, read-heavy relational workloads |
| MongoDB | Flexible schema, nested documents, rapid iteration |
| Cassandra | High write throughput, multi-region, wide-column |
| DynamoDB | Serverless, key-value at massive scale, AWS ecosystem |
| Redis | Caching, session store, pub/sub, leaderboards |
| Elasticsearch | Full-text search, log analytics |
| Neo4j | Social graphs, recommendation engines, fraud detection |
| InfluxDB / TimescaleDB | Metrics, IoT, monitoring dashboards |

---

### Interview tips for database questions

1. **Always clarify before choosing** — ask about scale, consistency requirements, and access patterns before naming a DB
2. **Justify the choice** — don't just say "I'd use Postgres"; say "because we need ACID transactions and our data is relational"
3. **Mention trade-offs** — every choice has one; show you understand what you're giving up
4. **Polyglot persistence is valid** — real systems often use multiple DBs (e.g. Postgres for core data + Redis for caching + Elasticsearch for search)
5. **Don't over-engineer** — start with the simplest option that meets requirements; scale only when needed

---

*— End of Session 1 —*

---

## 2. Distributed Systems Failure Modes
*Session: 2026-05-06*

### 2a. Race Conditions & Connection Bottlenecks

**The OTP problem:** Send OTP → nothing. Resend OTP → both arrive together.

This is NOT a UI bug. It's a distributed systems bottleneck.

#### Root causes ranked by likelihood

| Cause | What happens |
|---|---|
| HTTP connection pool exhaustion | First request grabs a connection, stalls on downstream call; second request queues; both flush together when bottleneck clears |
| SMS gateway batching | Gateway buffers outbound SMS for cost/rate reasons; resend triggers a flush of the buffer |
| DB row-level lock | UPDATE otp WHERE user_id=X holds a write lock; second request waits; both complete milliseconds apart when lock releases |
| Async queue starvation | OTP task queued in Celery/BullMQ; no workers free; second task enqueues; worker processes both back-to-back |
| Client-side premature timeout | Client fires retry before server actually failed; server was never told to stop — both requests eventually return |

#### How to diagnose
Compare timestamps: when was the OTP *generated* on the server vs when was it *dispatched* to the SMS gateway. That gap tells you exactly which layer is stuck.

---

### 2b. Idempotency Key Pattern

**Core idea:** Every request carries a unique client-generated ID. If the server sees the same ID twice, it returns the cached result instead of executing the operation again.

**Why it matters:** Retries are inevitable in distributed systems. Without idempotency, a retry = a duplicate side effect (two charges, two OTPs, two emails).

#### How it works

```
Client generates UUID → attaches as X-Idempotency-Key header → sends request
Server checks cache:
- Key seen before? → return cached result, skip execution
- New key? → execute, store result in cache with TTL, return result
```

#### Minimal implementation

```python
@app.post("/send-otp")
def send_otp(request):
    key = request.headers.get("X-Idempotency-Key")

    cached = redis.get(f"idem:{key}")
    if cached:
        return cached  # replay — no side effects

    result = generate_and_send_otp(request.phone)
    redis.setex(f"idem:{key}", 86400, result)  # TTL: 24h
    return result
```

#### Key properties
- **Client generates the key** (UUID) before the first attempt, reuses it on every retry
- **Server stores result with TTL** — not forever, just long enough to cover retry windows
- **Safe to retry freely** — client doesn't need to know if the first attempt succeeded
- Used mandatorily by Stripe, Razorpay, and most payment APIs

---

### 2c. Webhook Reliability — The Stripe Interview Question

**Scenario:** Payment succeeds. Customer charged. DB shows order as `pending`. Webhook received. Handler returned `200`. Order never moved to `paid`.

**The trap:** Everything looks fine. The `200` is the lie.

#### The core principle

> `200` means "I processed this successfully" — not "I received it", not "I tried".

Returning `200` regardless of whether the DB write succeeded is the bug. Stripe sees `200` and will never retry. The order is stuck forever.

#### All the places the bug hides

```python
# Bug 1: Silent exception swallow
try:
    db.update_order(event.order_id, status="paid")
except Exception:
    pass  # swallowed — 200 still returned

# Bug 2: Wrong event type string — silent skip
if event.type == "payment_intent.succeeded":  # Stripe sent "charge.succeeded"
    update_order()
# no match → skipped → 200

# Bug 3: Early return on missing order
order = db.find_order(event.metadata["order_id"])
if not order:
    return 200  # order_id key was wrong/missing in metadata

# Bug 4: Async job enqueued but worker was down
task_queue.enqueue(update_order, order_id)  # worker never ran
return 200

# Bug 5: Idempotency check short-circuits before actual processing
if already_processed(event.id):
    return 200  # but it was never actually processed — just marked "seen"
```

#### The correct pattern

```python
@app.post("/webhook")
def handle_webhook(payload):
    event = parse_and_verify(payload)

    if event.type == "payment_intent.succeeded":
        try:
            with db.transaction():
                order = db.get_order_for_update(event.metadata["order_id"])
                if order.status == "paid":
                    return 200  # idempotent — already done, safe to ack

                order.status = "paid"
                db.save(order)
                # transaction commits here — exception raised if it fails

        except Exception as e:
            log.error(e)
            return 500  # tell Stripe to RETRY — work was NOT done

    return 200  # only reached after confirmed successful commit
```

#### The Kafka analogy
Same as Kafka's manual offset commit — you don't commit the offset until you've durably processed the message. ACK = commit. Never ACK before the side effect is written.

#### Fix checklist for webhook handlers
- [ ] Never swallow exceptions in the handler body
- [ ] Return `500` (not `200`) when DB write fails — let the processor retry
- [ ] Verify the exact event type string against provider docs
- [ ] Check that all `metadata` keys exist before accessing them
- [ ] Use `SELECT FOR UPDATE` to lock the row during status transition
- [ ] Add idempotency check *after* confirming the event is real, not before processing
- [ ] Log the gap between event received and DB write committed

---

*— End of Session 2 —*

---

## 3. Latency Diagnosis & Performance
*Session: 2026-05-06*

### 3a. Reading p50 vs p99 — The Tail Latency Signal

**The scenario:** 50K RPS. p50 latency = 12ms. p99 = 4 seconds. CPU at 30%. Memory fine. No GC pauses. No slow queries.

**The key insight:** CPU at 30% means the machine is not busy computing. Something is *waiting*. Fast p50 + brutal p99 is the classic signature of **queuing**, not slowness.

#### The five hypotheses (in order of likelihood for this symptom profile)

| Hypothesis | Root cause | Signature |
|---|---|---|
| Connection pool exhaustion | All DB/cache connections in use; new requests queue for a slot | Fast once started; wait time = the 4s |
| Thread pool saturation | Worker threads all occupied; requests queue at front door | CPU low (threads idle-waiting, not computing) |
| External dependency tail latency | A downstream service has bad p99; your p99 = their p99 | Per-hop trace shows fat span on dependency |
| Retry storms | Failed requests retrying amplify load on struggling system | p99 worsens over time; retry rate spikes |
| Lock contention | Hot DB row or mutex; threads queue to acquire lock | No "slow queries" but high lock wait time |

#### The diagnostic question that cuts search space in half
*"Do you have per-hop latency breakdown — is the 4s happening before my service logic, or inside it?"*

Distributed tracing (Jaeger, Zipkin, Tempo) shows exactly which span is fat at p99.

---

### 3b. Connection Pool Sizing — Little's Law

**Formula:** `pool_size = (RPS × avg_query_time_s) × safety_factor`

**Little's Law:** N = λ × W
- N = concurrent connections needed
- λ = requests per second
- W = average time per DB operation (seconds)

**Example:** 50K RPS, 12ms DB time → minimum pool = 50,000 × 0.012 = **600 connections**

#### The counter-intuitive truth
Bigger pool ≠ better. Each Postgres connection holds ~5–10 MB RAM. At 500 connections = ~4 GB overhead before any data. HikariCP formula: `pool_size = (core_count × 2) + spindle_count` — often just 10–20 connections on OLTP workloads, paired with a request queue for bursts.

#### What to monitor
- `pool.wait_time` — time waiting for a connection slot (non-zero at p50 = pool too small)
- `pool.active_connections` — if pegged at max constantly, increase pool
- `pool.pending_queue_depth` — growing queue = heading for OOM or timeouts

---

### 3c. Thread Pool vs Connection Pool — Key Distinction

| | Thread pool | Connection pool |
|---|---|---|
| What it is | Workers that accept and process incoming requests | Slots to talk to external systems (DB, Redis, APIs) |
| Scope | Internal to your service | Between your service and downstream |
| Saturates when | More concurrent requests than threads | More concurrent DB calls than pool slots |
| Symptom | Requests queue before any work starts | Threads sit idle holding a slot, waiting for DB |

**They compound:** An exhausted connection pool causes threads to block waiting. Those blocked threads fill the thread pool. One undersized connection pool can cascade into thread pool saturation.

**Python/FastAPI note:** Python's GIL means threads don't give true CPU parallelism. FastAPI uses async/await — a single-thread event loop handles thousands of concurrent I/O waits. Thread pool saturation manifests as the Uvicorn/Gunicorn worker pool filling up, or the threadpool executor for sync route handlers blocking.

---

### 3d. Exponential Backoff with Jitter — Preventing Retry Storms

**Why exponential alone fails:** All clients do identical math → all retry at identical times → thundering herd moves forward in time, not eliminated.

**Why jitter alone fails:** Random spread stays constant across attempts → no backing off → server never gets a recovery window.

**Full jitter (AWS-recommended):** `sleep(random(0, min(cap, base × 2ⁿ)))` — window grows exponentially, each client picks a random point within it → server load becomes flat across the window.

#### Production implementation (Python/asyncio)

```python
import asyncio, random

async def retry_with_backoff(
    fn,
    max_attempts: int = 5,
    base_ms: float = 500,
    cap_ms: float = 10_000,
    retryable: tuple = (Exception,),
):
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except retryable as e:
            last_exc = e
            if attempt == max_attempts - 1:
                break
            expo_cap = min(cap_ms, base_ms * (2 ** attempt))
            delay_ms = random.uniform(0, expo_cap)  # full jitter
            await asyncio.sleep(delay_ms / 1000)
    raise last_exc
```

#### What to retry vs not
- **Retry:** 429 (rate limit), 503 (unavailable), 504 (timeout), transient network errors
- **Never retry:** 4xx client errors (bad request, auth failure, not found) — retrying won't fix them
- **Only retry idempotent operations** — or use an idempotency key if there are side effects

#### The three parameters that matter
- `base_ms` — starting window (500ms is standard)
- `cap_ms` — maximum delay before giving up on backing off further (10–30s depending on SLA)
- `max_attempts` — 5 is usually right; more just delays the inevitable failure signal

---

### 3e. OOM (Out of Memory) in Kubernetes

**What happens:** Process allocates more memory than the pod limit → OS sends SIGKILL → pod restarts → if it keeps happening: CrashLoopBackOff.

**Config-triggered OOM — what to suspect:**
A 2KB config change OOMing pods 30 seconds after deploy means the config changed *runtime allocation behaviour*, not just a setting. The delay = time for traffic to fill the new slots.

| Config change | Why it causes OOM |
|---|---|
| Raised worker/thread count | More concurrent work held in memory |
| Raised connection pool size | Each connection holds buffers |
| Enabled a feature flag | New code path allocates large objects |
| Raised request size limit | Larger payloads buffered |
| Unbounded cache size | Cache grows without eviction |
| Raised timeout values | Slow requests accumulate in memory longer |

#### Debug toolkit

```bash
kubectl top pods                              # live memory/CPU per pod
kubectl describe pod <pod-name>              # OOMKilled events, configured limits
kubectl get events --sort-by='.lastTimestamp' # recent OOM events
```

```python
# Find what's allocating memory inside the process
import tracemalloc
tracemalloc.start()
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics('lineno')[:10]:
    print(stat)
```

Grafana: plot `container_memory_working_set_bytes` against deploy timestamp. Steady climb = pool/cache filling. Sudden spike = specific traffic pattern hitting new code path.

---

*— End of Session 3 —*

---

## 4. Database Principles — Battle-Hardened Rules
*Session: 2026-05-06*

Nine principles that separate engineers who've been burned from those who haven't. Grouped by the layer they bite you at.

---

### Design phase — before you write a line

#### Principle 1: Know your access pattern before you pick the DB

This is the most violated rule in early-career system design. People pick Postgres because they know it, or MongoDB because it sounds modern — before asking the only question that matters: *how will this data be read and written?*

Access patterns determine everything:
- Always fetch a user by `user_id` → key-value store wins, relational is overkill
- Complex aggregations across many rows → columnar (Redshift, BigQuery) beats row-oriented Postgres
- Writes are 10x reads → Cassandra's LSM-tree is built for this; Postgres B-tree indexes will groan
- Query by multiple arbitrary fields → document store with flexible indexing beats a rigid schema

In an interview, before naming any database say: *"Let me first understand the read/write ratio and what queries we'll actually run."* That one sentence signals seniority.

#### Principle 2: Always state the consistency model out loud

Consistency is a choice, not a default. The mistake is leaving it implicit.

- **Strong consistency** — every read sees the most recent write
- **Eventual consistency** — reads *will* converge to the latest write, but maybe not immediately

Where this bites: a user places an order, refreshes order history, doesn't see it. That's replication lag + eventual consistency — expected behaviour if you designed for it, a bug if you didn't.

In interviews, say it explicitly: *"For the payment service I'll use strong consistency — we cannot show a stale balance. For the activity feed, eventual consistency is fine — a 2-second lag on likes is acceptable."* Interviewers reward engineers who make this tradeoff conscious.

#### Principle 3: Sharding is not free — pick the shard key like your job depends on it

Sharding splits data across multiple nodes. The shard key decides which node each row lives on. A bad key creates hotspots — one node drowns while others idle.

Classic mistakes:
- Sharding by `created_at` → all new writes hit the same shard (temporal hotspot)
- Sharding by `user_country` → US shard gets 60% of traffic (geographic skew)
- Sharding by `user_id` (random hash) → writes spread evenly, but cross-shard queries become expensive scatter-gather

A good shard key: high cardinality, evenly distributed, aligns with your most common query pattern.

The interview test: *"I'm sharding by `user_id` because 90% of queries are scoped to a single user."* If you can't say that sentence, you don't have a shard key yet.

---

### Runtime costs — every operation has a price

#### Principle 4: Indexes are a write tax

Every index you create is a separate data structure the DB maintains. When you `INSERT` a row, the DB writes to the table *and* updates every index. Same for `UPDATE` and `DELETE`.

Indexes = faster reads, slower writes, more storage. Every time.

Where this matters: designing a write-heavy system (logs, events, IoT telemetry), aggressive indexing kills write throughput. The right answer: write to an append-only store with minimal indexing, then asynchronously build read-optimised indexes in a separate store.

Rule of thumb: index columns you filter on in `WHERE` and `JOIN`. Don't index everything because you might query it someday.

#### Principle 5: Replication lag is not a corner case — it's the default

Almost every production database uses primary-replica. Writes go to primary; reads can go to replicas. Replicas are always slightly behind. In high-throughput systems, "slightly" can mean seconds.

This creates bugs invisible in development (single node) and nasty in production:
- User updates profile → immediately fetches it → gets old version (read-your-own-write problem)
- User completes checkout → inventory service reads replica → sees stock already sold

Solutions: read-your-writes consistency (route user's reads to primary for a short window after they write), sticky sessions, or designing UX around the lag (optimistic updates).

In an interview: when you say "I'll add read replicas," immediately follow with "and I'll handle replication lag by..."

---

### Operational reality — what breaks at 3am

#### Principle 6: Backups ≠ disaster recovery

A backup is a snapshot of your data at a point in time. Disaster recovery is your *ability to restore service* from that snapshot within an acceptable window.

The gap is huge:
- A backup on the same data centre as your primary is destroyed in the same flood
- A backup you've never restored may be corrupt or missing critical tables
- A backup from 24 hours ago means 24 hours of lost data — acceptable?

The real questions are **RTO** (Recovery Time Objective — how fast must you recover?) and **RPO** (Recovery Point Objective — how much data can you afford to lose?). These define your DR strategy. A 1-hour RTO requires a hot standby. A 24-hour RPO means daily backups might be enough.

In interviews: when asked about reliability, bring up RTO/RPO. Most candidates forget this layer entirely.

---

### Application layer bugs — the ones you will keep hitting

#### Principle 7: The N+1 query problem is the most common bug in your career

Classic scenario: fetch 100 posts (1 query), then for each post fetch the author (100 queries). Total: 101 queries. N+1.

ORMs make this trivially easy to do by accident. The code looks clean; the query log is a disaster.

The fix — eager loading, fetch all related data in one query:
```sql
SELECT posts.*, users.name
FROM posts
JOIN users ON posts.user_id = users.id
LIMIT 100;
```

Or in an ORM: `.select_related('author')` / `options(joinedload(...))` / `.include(:author)`

Spotting it in interviews: if a candidate says "for each item, fetch its metadata" without saying "in a single batched query," flag it.

#### Principle 8: "Just add a cache" is the most expensive cheap sentence in tech

Caching solves a read performance problem and immediately creates three new problems:

- **Cache invalidation** — when does it go stale? Who invalidates it? What if two services write to the same cached data?
- **Cold start** — cache is empty after deploy or restart; every request hits DB; this is often when systems fall over
- **Consistency** — cache and DB can now disagree; for how long is that acceptable?

Common invalidation strategies:
- **TTL** — expire after N seconds. Simple, but stale.
- **Write-through** — update cache on every write. Consistent, but adds write latency.
- **Cache-aside** — app checks cache, misses go to DB, app populates cache. Most common, but creates a consistency window.

Never say "cache it" without immediately saying which invalidation strategy and how you handle cold starts.

#### Principle 9: Soft deletes look smart for 2 years and stupid forever after

Soft deletes add `is_deleted = true` (or `deleted_at`) instead of removing rows. The idea: preserve data for audit trails, allow undo, avoid cascade issues.

The reality after two years:
- Every query needs `WHERE is_deleted = false` — forget it once and you're showing deleted users their deleted content
- Every index needs to include `is_deleted` or it becomes useless
- Your table has 10x the rows it needs; deleted rows are noise you scan forever
- GDPR right-to-erasure requests become a nightmare — "deleted" rows aren't actually deleted

Better alternatives:
- **Event sourcing** — append-only log of state changes; reconstruct history without polluting the live table
- **Archive table** — move deleted records to a separate cold storage table on delete
- **Hard delete + audit log** — actually delete, but write a separate audit event before doing so

In interviews: if you propose soft deletes, an experienced interviewer will ask "how do you handle compliance deletion requests?" Have an answer.

---

### Quick-recall cheat sheet

| # | Principle | The one-liner |
|---|---|---|
| 1 | Access pattern first | Name the query before you name the DB |
| 2 | State consistency out loud | Strong or eventual — make it a conscious decision |
| 3 | Shard key is critical | One sentence justification, or you don't have one |
| 4 | Indexes are a write tax | Every index slows every write |
| 5 | Replication lag is default | Design for stale reads, not against them |
| 6 | Backups ≠ DR | Know your RTO and RPO |
| 7 | N+1 is everywhere | Batch your fetches; ORMs hide this |
| 8 | Cache is not free | Invalidation + cold start + consistency = three new problems |
| 9 | Soft deletes rot | `is_deleted` pollutes every query forever |

---

*— End of Session 4 —*
