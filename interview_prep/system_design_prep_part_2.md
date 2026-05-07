# System Design Prep — Part 2

---

## Q: Given a 100GB log file of URLs, find all duplicate URLs using only 1GB of RAM.

### The Core Constraint

You can't load 100GB into 1GB of RAM. You need a strategy that makes multiple passes over the data or partitions it intelligently.

---

### Approach: External Hash Partitioning (Exact, O(N))

The cleanest solution for **exact** duplicate detection. Two phases:

1. **Partition** — Stream the 100GB file and write each URL to one of ~150 bucket files on disk, determined by `hash(url) % 150`.
2. **Detect** — Load each bucket into RAM one at a time, use an in-memory hash set to find exact duplicates.

---

### Diagram

```
PHASE 1 — PARTITION
─────────────────────────────────────────────────────────────────

┌──────────────────┐   stream   ┌─────────────────────┐
│  100 GB log file │ ─────────► │  hash(url) % 150    │
│  URLs, one/line  │            │  deterministic       │
└──────────────────┘            │  bucket ID           │
                                └──────────┬──────────┘
                                           │
                          ┌────────────────┼────────────────┐
                          ▼                ▼                ▼
                    ┌──────────┐    ┌──────────┐    ┌──────────┐
                    │bucket_000│    │bucket_001│    │bucket_149│
                    │  ~667 MB │    │  ~667 MB │    │  ~667 MB │
                    │  on disk │    │  on disk │    │  on disk │
                    └──────────┘    └──────────┘    └──────────┘
                          · · · (150 buckets total) · · ·


PHASE 2 — DETECT DUPLICATES (repeat for all 150 buckets)
─────────────────────────────────────────────────────────────────

┌────────────┐  load  ┌──────────────────────┐         ┌─────────────────┐
│  bucket_k  │ ─────► │  in-memory hash set  │ ──────► │ duplicates found │
│  from disk │        │  fits in < 1 GB RAM  │         │ exact matches   │
└────────────┘        └──────────────────────┘         └─────────────────┘
      ▲                                                         │
      └──────────────── repeat for next bucket ◄────────────────┘


KEY INVARIANT
─────────────────────────────────────────────────────────────────
if url_A == url_B  →  hash(url_A) % 150 == hash(url_B) % 150

Duplicates are ALWAYS co-located in the same bucket.
They can never be split across two buckets.
```

---

### Sizing the Math

| Variable              | Calculation                          | Value       |
|-----------------------|--------------------------------------|-------------|
| Input size            | given                                | 100 GB      |
| Number of buckets     | 100 GB / 0.6 GB (safe margin)        | ~150–200    |
| Avg bucket size       | 100 GB / 150                         | ~667 MB     |
| Peak RAM per bucket   | hash set overhead ~2–3× raw size     | must ≤ 1 GB |
| Total disk needed     | 100 GB input + 100 GB buckets        | ~200 GB     |

> **Skew risk**: If one domain dominates the log, a bucket can grow too large.  
> **Mitigation**: Use a uniform hash (MurmurHash3, xxHash) or increase bucket count.

---

### Why Not External Sort?

External sort (sort the 100GB on disk → scan for adjacent duplicates) also works, but:

- **Time complexity**: O(N log N) vs O(N) for hashing
- **Disk I/O**: Multiple merge passes vs two linear passes

Hashing wins unless you also need the URLs in sorted order.

---

### Alternative: Bloom Filter (Approximate, One-Pass Friendly)

1. **Pass 1** — Stream all URLs into a Bloom filter (~1 GB of bits). Flag "probably seen" hits.
2. **Pass 2** — Collect flagged URLs, hash/sort to confirm true duplicates.

**Trade-off**: Bloom filters have false positives → requires a verification pass. Exact method is preferred when correctness is required.

---

### Follow-Up Interview Questions

| Question | Answer |
|---|---|
| What if a single bucket is still too large? | Recursively re-partition that bucket with a second hash function |
| Can you do it in one pass? | Only approximately — Bloom filter or HyperLogLog (counts distinct, doesn't find duplicates) |
| Time complexity? | O(N) amortized — two linear passes + O(N/k) per bucket for hashing |
| How does this map to distributed systems? | Directly to MapReduce: Map = `emit(hash(url), url)`, Reduce = find duplicates per group |

---

## Q: If cache is faster than DB, why not store everything in cache?

### The Short Answer

Cache is fast because it's expensive, volatile, and small. You can't store everything there — and even if you could, you shouldn't.

---

### The Real Reasons, Layered

**1. Cost**
RAM costs ~10–50× more per GB than SSD, and ~100× more than spinning disk. A 10TB database would cost a fortune to replicate entirely in Redis or Memcached. Storage tiers exist because not all data is accessed equally.

**2. Volatility — caches don't persist**
RAM is wiped on restart. If your cache *is* your database, a pod crash or power failure means total data loss. Databases are built around durability guarantees — WAL (write-ahead logs), fsync, replication. Caches have none of that by default.

**3. Cache is not the source of truth**
The DB is canonical. The cache holds a *copy* — and copies go stale. You need a system that can reconstruct the cache from somewhere when it's cold, evicted, or corrupted. If the cache *is* the source of truth, you've just built a worse database.

**4. Eviction — caches forget things**
Caches use eviction policies (LRU, LFU, TTL) to manage limited memory. If you store everything in cache, you either need infinite RAM or start evicting things — and now your "database" has random data loss. That's not a database anymore.

**5. Not all data is hot**
The 80/20 rule applies strongly to data access patterns. ~20% of your data handles ~80% of your reads. Caching that hot 20% gives most of the performance benefit at a fraction of the cost. The cold 80% — historical records, archived orders, old user data — rarely gets read and doesn't need to live in RAM.

**6. Cache doesn't support rich queries**
Databases give you indexes, joins, full-text search, aggregations, transactions, and ACID guarantees. Caches are essentially key-value lookups. You can't run `SELECT SUM(revenue) WHERE region='APAC' AND date > '2024-01-01'` against Redis without loading everything into memory and doing it yourself.

**7. Consistency and concurrency**
Databases handle concurrent writes with locking, MVCC, and isolation levels. A distributed cache has no native notion of transactions — you'd have to build all of that yourself, and you'd just be reinventing a database.

---

### Storage Tier Diagram

```
Speed ↑ / Cost ↑ / Capacity ↓
────────────────────────────────────────────────────
  CPU registers       < 1 ns      (bytes)
  L1/L2/L3 cache      1–10 ns     (KB–MB)
  RAM / Redis         ~100 ns     (GB)        ← cache lives here
  NVMe SSD            ~100 µs     (TB)
  HDD                 ~10 ms      (TB)        ← DB often here
  Network storage     variable    (PB+)
────────────────────────────────────────────────────
Cost ↓ / Capacity ↑ / Speed ↓

Caching = promoting hot data UP the hierarchy temporarily.
          Not replacing the lower tiers.
```

---

### When People *Do* Use Cache as Primary Storage

Redis supports persistence (RDB snapshots + AOF logs) and is used as a primary store for specific use cases:

| Use case | Why it works |
|---|---|
| Session tokens | Loss is tolerable; user just re-logs in |
| Rate limiters | Counters are reconstructable |
| Leaderboards | Sorted sets; data can be rebuilt |
| Pub/sub queues | Ephemeral by nature |

Even then, serious production setups pair Redis with a durable backing store. And durability requires explicit config (`AOF fsync=always`), which kills most of the performance advantage.

---

### One-Line Answer for the Interviewer

> *"Cache is a performance layer, not a storage layer. It's fast because it trades durability, capacity, and query richness for speed. The database is the source of truth — the cache holds a warm copy of what's accessed most."*

---

### Follow-Up Interview Questions

| Question | Answer |
|---|---|
| What happens on a cache miss? | Fall through to DB, populate cache, return result (cache-aside pattern) |
| What is cache invalidation? | The hard problem — keeping cache consistent when DB changes. Strategies: TTL, write-through, write-behind |
| What is a cache stampede? | Many requests hit DB simultaneously on cache miss. Fix: mutex/lock, probabilistic early expiry |
| When would you NOT use a cache? | Write-heavy workloads, data that must always be fresh, highly unique access patterns (no hot keys) |
| Redis vs Memcached? | Redis: richer data types, persistence, pub/sub. Memcached: simpler, multi-threaded, pure caching |

---

## Q: What's the difference between REST and GraphQL? Why would you pick one over the other?

### The Core Difference

Both are ways to expose data over HTTP, but they differ in **who decides the shape of the response**.

- **REST** — server defines the shape. Each resource has its own endpoint.
- **GraphQL** — client defines the shape. One endpoint, client asks for exactly what it needs.

---

### REST: Server Defines the Shape

```
GET /users/123          → returns full user object
GET /users/123/posts    → returns all posts for that user
GET /posts/456/comments → returns comments for a post
```

To render a profile page (user info + last 3 posts + comment counts) = 3 separate requests. You often get too much (over-fetching) or too little (under-fetching).

### GraphQL: Client Defines the Shape

```graphql
query {
  user(id: "123") {
    name
    email
    posts(last: 3) {
      title
      commentCount
    }
  }
}
```

Single round trip. Exactly those fields — nothing more, nothing less.

---

### Core Differences

| | REST | GraphQL |
|---|---|---|
| Endpoints | One per resource | Single `/graphql` endpoint |
| Response shape | Server-defined | Client-defined |
| Over-fetching | Common | Eliminated |
| Under-fetching / N+1 | Common (multiple calls) | Solved in one query |
| Versioning | `/v1/`, `/v2/` routes | Schema evolution via deprecation |
| Caching | HTTP caching works naturally | Harder — everything is POST |
| Error handling | HTTP status codes | Always 200, errors in response body |
| Learning curve | Low | Higher — schema, resolvers, queries |

---

### When to Pick REST

- **Public APIs** — universally understood, any HTTP client works
- **Simple CRUD** — endpoints map cleanly to resources, less overhead
- **Caching matters** — HTTP GET caching (CDN, browser) is trivial
- **Small teams / fast MVPs** — no schema to define, no resolvers to wire up
- **Microservices talking to each other** — REST is fine for service-to-service calls

### When to Pick GraphQL

- **Multiple client types** — mobile, web, third parties all need different shapes of the same data
- **Highly relational data** — social graphs, dashboards, anything requiring 4–5 chained REST calls
- **Rapid frontend iteration** — frontend fetches new fields without waiting for backend endpoint changes
- **Avoid API versioning** — deprecate fields in schema instead of maintaining `/v1` and `/v2` forever

---

### The N+1 Problem — Where GraphQL Shines Most

Classic REST pain: fetch 50 posts, then fetch the author for each = 51 requests. In GraphQL, one query resolves everything — and with DataLoader batching, 50 author lookups collapse into a single DB query.

---

### One-Line Answer for the Interviewer

> *"REST is resource-oriented — the server defines what you get. GraphQL is query-oriented — the client defines what it needs. Pick REST for simplicity and public APIs, GraphQL when multiple clients need different shapes of the same data."*

---

## Q: Two transactions read inventory = 1 simultaneously. Both allow a purchase. You've now sold something you don't have. How do you prevent this?

### What Actually Happened

```
T1 reads stock = 1  ─┐
T2 reads stock = 1  ─┤  both see 1, both think "ok to sell"
T1 writes stock = 0 ─┤
T2 writes stock = 0 ─┘  stock is 0 but TWO purchases went through
```

This is the **lost update / race condition** problem. The gap between read and write is where the bug lives.

---

### Solution 1: Pessimistic Locking — `SELECT FOR UPDATE`

Lock the row at read time. T2 blocks until T1 commits.

```sql
BEGIN;
SELECT stock FROM inventory WHERE product_id = 42 FOR UPDATE;
-- T2 is blocked here until T1 finishes
UPDATE inventory SET stock = stock - 1 WHERE product_id = 42;
COMMIT;
-- T2 unblocks, reads stock = 0, rejects purchase
```

**Use when**: High contention (flash sales, limited stock). Orderly but slower.

---

### Solution 2: Optimistic Locking — Version Column

Don't lock at read time. Detect conflict at write time using a version counter.

```sql
-- Read
SELECT stock, version FROM inventory WHERE product_id = 42;
-- returns stock=1, version=7

-- Write — only succeeds if version hasn't changed
UPDATE inventory
SET stock = stock - 1, version = version + 1
WHERE product_id = 42 AND version = 7;

-- 1 row updated → success
-- 0 rows updated → someone else got there first, retry or reject
```

**Use when**: Low contention. High throughput, no blocking — but needs retry logic.

---

### Solution 3: Atomic Operation — Skip the Read Entirely

```sql
UPDATE inventory
SET stock = stock - 1
WHERE product_id = 42 AND stock > 0;

-- 1 row updated → success
-- 0 rows updated → out of stock
```

Single atomic operation. No gap between read and write. Simplest fix for pure inventory decrement.

---

### Solution 4: Serializable Isolation

```sql
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

DB treats concurrent transactions as if they ran one at a time. Automatically aborts conflicting transactions. Most expensive — use targeted, not globally.

---

### Which to Use When

| Scenario | Best approach |
|---|---|
| Low contention, high read throughput | Optimistic locking (version column) |
| High contention, must not oversell | Pessimistic locking (`SELECT FOR UPDATE`) |
| Simple decrement with floor | Atomic `UPDATE WHERE stock > 0` |
| Complex multi-row invariants | Serializable isolation |
| Distributed / microservices | Redis `SETNX` / Redlock, or Saga pattern |

---

### Bonus: What Does "Contention" Mean?

**Contention = how many transactions are simultaneously fighting over the same row.**

- **Low contention** — a niche product, 2–3 purchases a day. Collision chance near zero. Optimistic locking is fine.
- **High contention** — last PS5 on a flash sale. Thousands hit the same row at the same millisecond. Optimistic locking causes retry chaos. Pessimistic locking queues them up cleanly.

> Analogy: one toilet at a concert. 10 people = no queue. 10,000 people = the system breaks down. The toilet is your DB row. The queue is lock wait time.

---

### One-Line Answer for the Interviewer

> *"This is a lost update race condition. Fix it with pessimistic locking (`SELECT FOR UPDATE`) when contention is high, optimistic locking (version column) when it's low, or an atomic `UPDATE WHERE stock > 0` when the logic is simple enough to skip the read entirely."*

---

## Message Queues — Scenario-Based Deep Dive

---

### Scenario 1: The Overloaded Checkout Service

**Your e-commerce site runs fine normally. On a flash sale, 50,000 users hit "Buy" simultaneously. Your order service crashes.**

**Q: How do message queues fix this?**

Without a queue, every user request hits your order service directly. The service can handle maybe 500 requests/second. At 50,000 simultaneous — it falls over.

With a queue:

```
Users → [Place Order API] → Queue → [Order Service workers]
                                         (consumes at own pace)
```

The API instantly accepts every request and drops a message into the queue. Response to user: *"Your order is being processed."* The order service pulls messages off the queue at whatever rate it can handle — say 500/sec. The queue absorbs the spike. Nothing crashes. It just takes a bit longer to process during the surge.

**This is the core value of a queue: decoupling the rate of production from the rate of consumption.**

Producers don't care how fast consumers are. Consumers don't care how fast producers are. They're completely independent.

---

### Scenario 2: The Failed Payment Processor

**Your order is in the queue. The payment service pulls it, tries to charge the card — and the payment gateway times out. The message is gone. The order is lost.**

**Q: How do queues handle failure without losing messages?**

This is where **acknowledgements (acks)** come in.

A consumer doesn't delete a message just by reading it. It reads it, processes it, then explicitly sends an **ack** back to the queue saying "done, remove this." If the consumer crashes mid-processing — no ack sent — the queue re-delivers to another consumer.

```
Queue sends message → Consumer receives it
                    → Consumer processes (payment call)
                    → Success: sends ACK → queue deletes message
                    → Crash/timeout: no ACK → queue re-delivers after timeout
```

This is called **at-least-once delivery**. The message will arrive at least once — possibly more if the consumer crashes after processing but before acking.

**The follow-up problem**: what if the payment went through but the ack was lost? Queue re-delivers, you charge the card twice.

**The fix**: make your consumer **idempotent** — processing the same message twice produces the same result as once. Store a `processed_order_ids` set. Before charging, check if this order ID was already handled.

---

### Scenario 3: The Notification Fan-Out

**When an order is placed, you need to: send a confirmation email, update inventory, notify the warehouse, and log analytics. All of this happens synchronously in your order service. Adding a new step means changing order service code every time.**

**Q: How do you architect this cleanly with queues?**

This is the **pub/sub (publish-subscribe)** pattern.

```
Order Service
     │
     │ publishes event: "order.placed"
     ▼
  [Topic / Exchange]
     │
     ├──► Email Queue        → Email Service
     ├──► Inventory Queue    → Inventory Service
     ├──► Warehouse Queue    → Warehouse Service
     └──► Analytics Queue    → Analytics Service
```

The order service publishes one event. The broker fans it out to every subscriber. Each service has its own queue — they consume independently, at their own pace, and failures in one don't affect the others.

**The key win**: the order service doesn't know or care who's listening. Tomorrow you add a loyalty points service — just subscribe it to `order.placed`. Zero changes to the order service. This is loose coupling at the architecture level.

RabbitMQ calls this an **exchange**. Kafka calls the central log a **topic**. Different implementations, same concept.

---

### Scenario 4: The Poison Message

**One specific order message keeps crashing your consumer. It gets re-delivered, crashes again, re-delivered again — infinite loop. Your queue is stuck and real orders aren't being processed.**

**Q: How do you handle a message that can never be successfully processed?**

This is a **poison message** — a malformed or unprocessable message that kills every consumer that touches it.

The fix is a **Dead Letter Queue (DLQ)**.

```
Main Queue → Consumer crashes → retry (attempt 2)
                             → Consumer crashes → retry (attempt 3)
                             → Consumer crashes → move to DLQ
```

After N failed attempts (typically 3–5, you configure the threshold), the message is moved out of the main queue into a Dead Letter Queue. Normal processing continues unblocked.

The DLQ is handled separately:
- Engineer inspects the messages to find the bug
- Fix the bug and replay messages back into the main queue
- Or discard them if genuinely corrupt

Without a DLQ, one bad message halts an entire queue indefinitely. With one, poison messages are isolated and the system degrades gracefully.

---

### Scenario 5: The Ordering Problem

**You're processing bank transactions from a queue: deposit ₹1000, withdraw ₹1500, deposit ₹200. If processed out of order — withdraw before deposit — the account goes negative and the transaction is incorrectly rejected.**

**Q: How do you guarantee message ordering in a queue?**

Standard queues with multiple consumers don't guarantee order. Messages can arrive out of sequence, especially with retries and parallel consumers.

**Option 1: Single consumer**
One consumer, one queue, processes serially. Order guaranteed. Throughput limited. Fine for low-volume use cases.

**Option 2: Partition by key (Kafka's approach)**

```
Transactions for Account A → Partition 1 → Consumer 1
Transactions for Account B → Partition 2 → Consumer 2
Transactions for Account C → Partition 3 → Consumer 3
```

Kafka partitions a topic by a key (e.g. `account_id`). All messages for the same key always go to the same partition, consumed by the same consumer, in order. You get parallelism across accounts while preserving order within each account.

**Option 3: Sequence numbers + reorder buffer**
Each message carries a sequence number. Consumer buffers out-of-order messages and waits for the missing one before processing. Complex to implement — usually better to fix at the infrastructure level.

---

### When to Reach for a Queue

| Signal | Queue solves it |
|---|---|
| Producer is faster than consumer | Buffer the spike — queue absorbs it |
| A task can be done asynchronously | Don't block the user — queue it |
| Multiple services need the same event | Pub/sub fan-out |
| A step can fail and must be retried | Ack + re-delivery |
| You want services to not know about each other | Queue as the contract between them |

---

### Concept Cheat Sheet

| Concept | What it means |
|---|---|
| Queue | Buffer between producer and consumer |
| Ack | Consumer's receipt — message deleted only after this |
| At-least-once delivery | Message guaranteed to arrive, may arrive twice — make consumers idempotent |
| Idempotency | Processing the same message twice = same result as once |
| Pub/Sub | One event, many subscribers — loose coupling |
| Dead Letter Queue | Isolation ward for poison messages |
| Partition | Ordering guarantee within a key, parallelism across keys |

---

*System Design Prep series — Abhishek Marathe*
