# Database Systems — Scenario-Based Interview Prep

A deep-dive guide covering Indexing, Sharding, Replication, CAP Theorem, and NoSQL trade-offs. Each section starts with a quick mental model, then walks through realistic scenarios with answers structured the way you'd actually deliver them in an interview.

---

## 1. Indexing

### Mental model

A B+ tree (what most relational DBs actually use, not pure B-trees) is a balanced tree where:
- Internal nodes hold *routing keys* — they tell you which child to descend into.
- Leaf nodes hold the actual key→row-pointer entries and are linked together as a doubly-linked list.
- Tree height is typically 3–4 levels even for billions of rows because branching factor is high (hundreds of keys per node).

So a lookup is `O(log n)` disk reads — usually 3–4 page reads to find any row.

The trade-off: every index is a *separate* B+ tree the database has to keep in sync. Every `INSERT`, `UPDATE` (to indexed columns), and `DELETE` modifies every relevant index. Indexes speed up reads at the cost of writes and storage.

---

### Scenario 1.1 — The mysterious slow query

> Your e-commerce app has an `orders` table with 50M rows. The query `SELECT * FROM orders WHERE user_id = ? AND status = 'pending' ORDER BY created_at DESC LIMIT 20` is taking 8 seconds. You already have an index on `user_id`. What's happening, and how do you fix it?

**Answer:**

The single-column index on `user_id` is doing only the first part of the work. Here's the execution flow:

1. Index lookup on `user_id` returns, say, 5,000 row pointers (this user is a heavy buyer).
2. The DB does 5,000 random I/Os back to the heap to fetch each row.
3. It filters those 5,000 rows down to ones with `status = 'pending'`.
4. It sorts the survivors by `created_at`.
5. It returns the top 20.

The 8 seconds is mostly steps 2 and 4 — random I/O for the heap fetches plus an in-memory sort.

**The fix is a composite index designed for this access pattern:**

```sql
CREATE INDEX idx_orders_user_status_created
ON orders (user_id, status, created_at DESC);
```

Why this order matters — the rule is **equality columns first, then range/sort columns**:
- `user_id` is equality → leftmost.
- `status` is equality → next.
- `created_at` is the sort key → last, with `DESC` matching the query's sort direction so the DB can read the index in physical order without a sort step.

With this index, the DB descends to `(user_id=X, status='pending')` and reads 20 entries off the leaf chain in sorted order. Done. ~3-4 page reads instead of 5,000.

**Bonus optimization — covering index:**

If the query selects only a few columns, include them in the index:

```sql
CREATE INDEX idx_orders_covering
ON orders (user_id, status, created_at DESC)
INCLUDE (order_total, item_count);  -- Postgres syntax
```

Now the index alone answers the query — no heap visit at all. This is called an *index-only scan*.

**The trade-off to mention:** every new index adds write amplification. On a write-heavy table, blindly creating composite indexes for every query pattern slows down inserts and bloats storage. You profile, you measure, you index the queries that matter.

---

### Scenario 1.2 — Why did adding an index make things worse?

> A junior engineer added indexes on every column of a `transactions` table because "indexes make things faster." Write throughput dropped from 8K TPS to 1.5K TPS. Explain.

**Answer:**

Each index is a separate B+ tree. When you `INSERT` a row:

- The heap gets one write.
- *Every* index gets a write — locate the right leaf, possibly split it, possibly cascade splits up the tree.
- Each of those writes goes through the WAL (write-ahead log) for durability.

If the table has 12 indexes, each insert is doing roughly 13× the work of an unindexed insert. Plus:

- **Index page splits** — when a leaf is full and a new key needs to fit, the page splits into two, and the parent gets a new pointer. Splits cause random I/O and temporary lock contention.
- **Cache pollution** — every index competes for buffer pool space. Indexes nobody queries still get loaded when their pages are touched during writes.
- **Vacuum / autovacuum pressure** (Postgres) or page consolidation (others) — every index needs maintenance.

**The fix:** drop indexes that aren't actually used.

```sql
-- Postgres: find unused indexes
SELECT schemaname, relname, indexrelname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

Rule of thumb: an index needs to pay for itself. If it serves <1% of the query workload but participates in 100% of writes, it's net-negative.

---

### Scenario 1.3 — Composite index column order

> You have queries like:
> - `WHERE country = 'US' AND city = 'NYC'`
> - `WHERE country = 'US'`
> - `WHERE city = 'NYC'`
>
> Will `INDEX (country, city)` serve all three? What about `INDEX (city, country)`?

**Answer:**

The **leftmost prefix rule** governs composite indexes. A B+ tree is sorted by `(country, city)` lexicographically, so the index can only be used efficiently when the query constrains the leading column(s).

With `INDEX (country, city)`:
- ✅ `country = 'US' AND city = 'NYC'` — uses both columns.
- ✅ `country = 'US'` — uses leading column, scans the `country='US'` slice.
- ❌ `city = 'NYC'` alone — **cannot use this index efficiently**. The DB would have to scan every leaf to find NYC entries scattered across countries. Most planners will fall back to a full table scan or pick a different index.

With `INDEX (city, country)`:
- The third query works, the second doesn't.

**Practical answer:** if all three patterns matter, you need *two* indexes:

```sql
CREATE INDEX idx_country_city ON locations (country, city);
CREATE INDEX idx_city ON locations (city);  -- or (city, country) if you also filter
```

The `(country, city)` index already serves `country` alone, so you don't need a separate index for that.

**One nuance** — some DBs (Postgres in particular) can use an index "skip scan" or bitmap scan even when leading columns aren't constrained, but it's much less efficient than a proper index. Don't rely on it.

---

## 2. Sharding

### Mental model

Sharding means partitioning a single logical dataset across multiple physical database nodes, where each node holds a *disjoint subset* of the rows. It's how you scale beyond what a single machine can handle for storage, write throughput, or working-set RAM.

The hard part isn't slicing the data — it's that suddenly:
- Cross-shard queries are expensive (scatter-gather across N nodes).
- Cross-shard transactions need 2PC or sagas (distributed coordination).
- Re-sharding when you outgrow your scheme is painful.
- Joins between sharded tables may become impossible without colocation.

Pick your shard key carefully. It's the most consequential decision in the system.

---

### Scenario 2.1 — Choosing a shard key

> You're sharding a multi-tenant SaaS database. Tables include `tenants`, `users` (each user belongs to a tenant), `documents` (each document belongs to a user), and `audit_logs`. Some tenants are 100× larger than others. How do you shard?

**Answer:**

Three candidates worth weighing:

**Option A — shard by `tenant_id`:**

Pros:
- Most queries are tenant-scoped (`WHERE tenant_id = ?`). They route to a single shard. No scatter-gather.
- Tenant data lives together, so tenant-scoped joins (`users JOIN documents`) stay on one node.
- Strong tenant isolation — easy to back up, restore, or move a single tenant.

Cons:
- **Hotspots** — that 100× large tenant will overload its shard while smaller tenants leave their shards idle. Skew is the killer.
- Cross-tenant analytics (rare in SaaS, but real for billing or admin dashboards) need scatter-gather.

**Option B — shard by `user_id` (hash):**

Pros:
- Even distribution — hashing flattens skew.
- High write throughput.

Cons:
- Tenant queries (`WHERE tenant_id = X`) hit every shard. Every read becomes a fan-out.
- Joins across `users` and `documents` cross shards.
- This is usually a bad fit for multi-tenant SaaS.

**Option C — shard by `tenant_id`, but isolate the whales:**

This is what most production multi-tenant systems land on. Shard by `tenant_id` for the 99% of normal-sized tenants, but give the top 1% (the whales) their own dedicated shards. A lookup table maps `tenant_id → shard_id`. When a tenant outgrows their shared shard, you migrate them to a dedicated one.

**My recommendation:** Option C. Shard by `tenant_id` with an explicit directory-based mapping rather than pure hash. The directory lets you handle hotspots, do tenant migrations, and run "noisy neighbor" tenants on isolated hardware.

**For `audit_logs`:** these are usually append-heavy, time-series-shaped, and rarely queried cross-tenant in real time. Shard those by `tenant_id` too (so audit queries stay local), but partition each shard's audit table by month for retention management.

**Things I'd ask before committing:**
1. What's the tenant size distribution? (Median, p99, max.)
2. Is there a cross-tenant query path that has to be fast?
3. Is tenant deletion / data export a frequent operation?

---

### Scenario 2.2 — Hash vs range vs directory sharding

> Compare the three sharding strategies. When does each shine, and when does each fail?

**Answer:**

**Hash sharding** — `shard = hash(key) % N`

- Best for: even distribution when the key has high cardinality and the access pattern is point lookups by that key. User-keyed social graphs, key-value workloads.
- Fails at: range queries (`WHERE created_at BETWEEN ...`) become scatter-gather across all shards. Resharding is brutal — `% N` changing means almost every key moves. Mitigation: **consistent hashing**, which only moves `1/N` of keys when you add a node.

**Range sharding** — assign contiguous key ranges to shards

- Best for: range queries (time-series, alphabetical lookups). Easy to add a new shard for new data (e.g., a new month).
- Fails at: hotspots. If you range-shard by `user_id` and user IDs are monotonically increasing, all new writes hit the last shard. Same with timestamp-based sharding for current data — the "today" shard burns while yesterday's shard sleeps.

**Directory sharding** — explicit lookup table mapping key → shard

- Best for: irregular distributions, multi-tenant whales, gradual migrations. Maximum flexibility.
- Fails at: the directory itself becomes a critical dependency. It needs to be highly available, cached aggressively, and consistent. If it goes down, every query is blind.

**The honest answer in an interview:** real production systems mix these. Vitess (YouTube/Slack) uses a directory of "keyspaces" with hash-distributed vindexes inside. DynamoDB hashes by partition key but range-orders within a partition by sort key. Cassandra does both: partition key (hash) plus clustering key (range).

---

### Scenario 2.3 — The cross-shard transaction

> A user transfers money from their account (on shard A) to another user's account (on shard B). How do you make this atomic without sacrificing throughput?

**Answer:**

There's no free lunch here. Three real approaches:

**Approach 1 — Two-Phase Commit (2PC):**

Coordinator asks both shards to *prepare* (lock rows, write to log, vote yes/no). If both vote yes, coordinator tells both to *commit*. If anyone votes no or times out, both roll back.

- Correctness: strong. Transactions are atomic across shards.
- Cost: high. Every transaction holds locks for two network round-trips. Coordinator failure during commit phase leaves shards in an indeterminate state ("blocking" — known weakness of 2PC).
- When to use: low-volume, high-value operations where correctness trumps everything. Bank wire transfers, tax filings.

**Approach 2 — Saga pattern:**

Break the transaction into a sequence of local transactions, each with a compensating action.

```
1. Debit sender (local txn on shard A). On failure, abort.
2. Credit receiver (local txn on shard B). On failure, run compensating "refund sender" on shard A.
3. Mark transfer complete.
```

- Correctness: eventually consistent. There's a window where sender is debited but receiver isn't credited yet.
- Cost: low. Each step is a local transaction.
- Watch out: compensating actions must be idempotent and must always succeed (or you need human intervention). State machines and an outbox pattern help.
- When to use: high-volume operational systems where eventual consistency is acceptable for seconds. Most fintech transfers, e-commerce checkouts.

**Approach 3 — colocation (avoid the problem):**

Shard so that frequently-related data lives on the same node. If both accounts in a transfer can be routed to the same shard (e.g., shard by `account_pair` or use a wallet abstraction that batches before settlement), it's a single local transaction.

Many real systems do this: shard the *operational* tables together, and use async replication to a different "analytical" sharding scheme for reporting.

**My production answer:** for money movement at scale, saga + idempotency keys + an outbox pattern. 2PC is technically correct but operationally painful. The "money in flight" window is usually fine if the UX shows pending state and reconciliation runs continuously.

---

## 3. Replication

### Mental model

Replication is keeping multiple copies of the same data on different nodes. Sharding is "different data on different nodes"; replication is "same data on different nodes." Most real systems combine both — each shard is replicated.

You replicate for three reasons (in order of importance for most systems):
1. **Durability** — survive node failure.
2. **Read scale** — serve reads from replicas.
3. **Geographic locality** — replicas closer to users.

The fundamental tension: **synchronous replication is slow but consistent; asynchronous replication is fast but allows lag and lost writes on failover.**

---

### Scenario 3.1 — The "I just posted but I don't see it" bug

> Your social app uses Postgres with one primary and three async read replicas. Reads go to a random replica via a load balancer; writes go to primary. Users complain that after they post, sometimes they refresh and their post is missing for a few seconds. Diagnose and fix.

**Answer:**

This is **read-your-own-writes** consistency violation, caused by **replication lag**.

The flow:
1. User posts → write goes to primary → primary acks success.
2. User's UI refreshes → read goes to a random replica.
3. That replica hasn't received the write yet (lag of, say, 200ms).
4. User sees no post.
5. They refresh again, hit a different replica or the same one a moment later, post appears.

**Fixes, in order of operational complexity:**

**Fix A — read your own writes from primary:**

For a window after a user writes (say, 10 seconds), route their reads to the primary. Implement with a session flag or a sticky cookie:

```python
if request.user.last_write_at and (now - request.user.last_write_at) < 10s:
    db = primary
else:
    db = replica
```

Cheap and works for most cases. Cost: slightly more primary load.

**Fix B — wait-for-LSN (log sequence number) routing:**

When the primary commits a write, it returns an LSN. The client stores this and sends it with subsequent reads. The replica check its own LSN; if it's behind, either wait or fall back to primary.

Postgres supports this via `pg_wait_for_replay_lsn`. More precise than time-based routing, but more code.

**Fix C — synchronous replication (at least one replica):**

Configure Postgres `synchronous_standby_names` so the primary waits for at least one replica to confirm before acking the write. Now any read that hits *that* replica is guaranteed fresh. But: writes are slower (paying for a network round-trip), and if the sync replica goes down, writes block.

**Production answer:** Fix A for 95% of cases. Fix B for systems where staleness isn't tolerable (financial dashboards). Fix C only when you genuinely need synchronous durability across nodes — rarely worth the latency for social-app workloads.

**The deeper point:** "eventual consistency" is the default for async replication, and apps need to be designed for it. Showing optimistic UI updates ("your post is published" before the replica round-trip completes) often beats trying to engineer strict consistency.

---

### Scenario 3.2 — Replication topologies

> Walk me through the difference between primary-replica, multi-primary, and quorum-based replication. When would you pick each?

**Answer:**

**Primary-replica (also called primary-secondary or leader-follower):**

One node accepts writes; others replicate from it and serve reads.

- Pros: simple consistency model, easy to reason about, no write conflicts.
- Cons: write throughput capped by primary. Failover requires elections and may lose recent async-replicated writes.
- Use when: read-heavy workload, single-region or single-writer is acceptable. Default choice for Postgres, MySQL, MongoDB.

**Multi-primary (multi-leader):**

Multiple nodes accept writes; they replicate to each other.

- Pros: write scale, geographic locality (each region writes locally), survives any single node failure with no failover.
- Cons: **conflict resolution**. If two primaries both update the same row, what wins? Strategies: last-write-wins (LWW, often wrong), CRDTs (correct but limited data types), application-level merging (correct but expensive).
- Use when: geo-distributed systems where each region has mostly local writes. Active-active deployments. Examples: Cassandra (effectively multi-leader per partition), CockroachDB.

**Quorum-based (Paxos/Raft):**

A write is acked when a majority (quorum) of nodes confirm. Reads also go through a quorum to ensure freshness.

- Pros: strong consistency without a single point of failure. Survives minority node failures transparently.
- Cons: latency = slowest node in the quorum. Requires odd node counts (3, 5, 7) to avoid split-brain.
- Use when: you need strong consistency *and* HA. Examples: etcd, Consul, ZooKeeper, CockroachDB at the range level, Spanner.

**Quick decision:**

| Need | Pick |
|---|---|
| Read scale, simple ops, OK with async lag | Primary-replica |
| Geo-distributed writes, can tolerate conflicts | Multi-primary |
| Strong consistency + HA, latency budget allows | Quorum |

---

### Scenario 3.3 — The split-brain incident

> Your primary database becomes network-partitioned from its replicas. The replicas elect a new primary. The original primary doesn't know it's been demoted and keeps accepting writes. The partition heals 30 seconds later. What's the state of the system, and how do you prevent this?

**Answer:**

This is the classic **split-brain** scenario. State after partition heals:

- Old primary has 30 seconds of writes that no replica has.
- New primary has 30 seconds of writes that the old primary doesn't have.
- Two divergent histories. Some clients talked to old primary, some to new.

You cannot reconcile this losslessly without application-level merge logic. Some writes will be lost. This is why split-brain is feared.

**Prevention:**

**1. Fencing tokens / generation numbers:**

Every primary election produces a monotonically increasing generation number. The old primary, when it tries to write to disk or to the WAL, finds that the new generation has already advanced and refuses the write. Now even if it accepts client requests, they fail at the storage layer.

**2. Quorum writes (don't accept writes without a majority):**

Old primary can no longer reach a majority of the cluster (it's the partitioned minority). It must refuse writes. New primary is on the majority side and can accept. When partition heals, old primary syncs from new primary. No split-brain by construction.

This is the Raft / Paxos approach. **It's why these protocols exist.** You sacrifice availability on the minority side (CAP — see next section) to prevent split-brain.

**3. STONITH ("Shoot The Other Node In The Head"):**

When a new primary is elected, it forcibly powers off or fences the old one (via IPMI, hypervisor API, etc.) before accepting writes. Brutal but effective. Common in HA Postgres setups.

**The real lesson:** any HA setup using async replication and naive failover is at risk of split-brain. Either use quorum protocols (Raft/Paxos) or accept that you'll need careful fencing + manual reconciliation. There's no middle ground.

---

## 4. CAP Theorem

### Mental model

CAP says: in the presence of a **network Partition**, a distributed system must choose between **Consistency** (all reads see the latest write) and **Availability** (every request gets a non-error response).

The common misreading is "pick 2 of 3." The accurate reading: **partitions are inevitable, so you're really choosing between C and A when one occurs.** During normal operation (no partition), you can have both.

A more useful frame is **PACELC**: during a Partition, choose A or C; Else (during normal operation), choose Latency or Consistency.

This makes the design space concrete:
- **CP + EC**: Spanner, etcd. Strict consistency always; rejects writes during partitions.
- **CP + EL**: rare, hard to do.
- **AP + EL**: Cassandra, DynamoDB. Stays available during partitions; favors latency over consistency normally.
- **AP + EC**: rare; you'd want consistency normally but availability during partitions.

---

### Scenario 4.1 — The CP vs AP product decision

> You're designing two systems:
> 1. A banking ledger that tracks account balances.
> 2. A shopping cart for an e-commerce site.
>
> For each, would you pick CP or AP? Defend your choice.

**Answer:**

**Banking ledger → CP.**

Showing a wrong balance is worse than showing no balance. If the system is partitioned and we can't be sure what the true balance is, we should refuse the operation rather than allow a double-spend. Specifically:

- If the user tries to withdraw and we can't reach quorum, return an error: "Service temporarily unavailable, try again." Annoying, but no money is created or destroyed incorrectly.
- An AP design here would mean two partitioned nodes both authorizing withdrawals against the same account, then needing to "merge" balances later — which means either reversing transactions or going negative. Both are unacceptable.

Use a CP system: Postgres with synchronous replication, Spanner, CockroachDB. Accept some downtime during partitions.

**Shopping cart → AP.**

A user adding items to their cart and getting an error is worse than a brief inconsistency. If they're shopping and the system is partitioned, let them keep adding items — we'll reconcile when partition heals. Specifically:

- Two replicas accept "add item X" during partition. When they sync, both items appear in cart. Worst case: user sees an extra item, removes it. Recoverable.
- Even simpler: cart is a set, and set union is a CRDT — automatic conflict-free merge.

A famous case study: Amazon DynamoDB's predecessor, Dynamo, was built explicitly for this — the shopping cart had to stay available. They tolerated the rare "deleted item resurrects" bug as preferable to "user can't add to cart."

**The general principle:** ask what the cost of unavailability is vs. the cost of inconsistency, *for this specific operation*. The answer depends on the operation, not the company. The same e-commerce site uses CP for order placement (don't double-charge) and AP for cart, recommendations, and reviews.

---

### Scenario 4.2 — "Eventually consistent" — what does that actually mean?

> Marketing materials for NoSQL databases love "eventual consistency." What does it actually guarantee, and what doesn't it guarantee?

**Answer:**

Eventual consistency guarantees: **if no new updates are made to a data item, eventually all reads will return the same value.**

That's it. That's the entire guarantee.

What it does *not* guarantee:

1. **A bound on how long "eventually" takes.** Could be 10ms, could be hours. Most production systems are sub-second, but the guarantee doesn't say so.
2. **Monotonic reads.** You might read value V2, then read V1 (older) on the next request, then V2 again. Replicas can serve different versions.
3. **Read-your-own-writes.** You write, then immediately read, and might see the old value.
4. **Causal consistency.** You comment "+1 to that!" on a post, and someone might see your comment before they see the post.

This is why "eventual consistency" alone is rarely enough for a UX. Real systems layer on stronger guarantees:

- **Read-your-own-writes**: route a user's reads to the same replica or to primary for a window.
- **Monotonic reads**: pin a session to one replica.
- **Bounded staleness**: guarantee replication lag < N seconds (Cosmos DB offers this as a tier).
- **Causal consistency**: track logical clocks (vector clocks, version vectors) so reads respect happens-before relationships. Implemented in MongoDB causal sessions, Riak, and others.

When evaluating an "eventually consistent" datastore, the right questions are:
- What's the *typical* convergence time? (p50, p99)
- What's the worst case under partition?
- What stronger session-level guarantees does it offer?
- How does it handle conflicts (LWW, vector clocks, CRDTs)?

---

### Scenario 4.3 — The CAP myth

> Some people say "CAP theorem is misunderstood / outdated." What's the steelman version of that critique?

**Answer:**

The critique has a few solid points:

**1. "Pick 2 of 3" is misleading.**

You don't pick CA. CA means "no partition tolerance," which translates to "single node" — not a distributed system at all. Once you're distributed, partitions will happen, so the real choice is C vs A *during* a partition.

**2. C and A aren't binary.**

The original formulation treats them as on/off. Reality is a spectrum:
- Consistency has many flavors: linearizability (strongest), sequential, causal, read-your-writes, eventual.
- Availability is rarely 0% or 100% — it's "what fraction of requests succeed within latency budget."

PACELC (mentioned above) captures more of the trade-off.

**3. The "during a partition" framing ignores normal-operation trade-offs.**

99.9% of the time, your system is *not* partitioned. The interesting trade-offs are usually about latency and throughput in normal operation, not in the rare partition. Spanner uses TrueTime hardware to achieve strong consistency in normal operation at the cost of waiting for clock uncertainty windows. That's a real trade-off CAP doesn't speak to.

**4. Modern systems blur the lines.**

CockroachDB, Spanner, and FaunaDB offer "strong consistency at scale" — claiming both C and A under most circumstances by using quorum protocols cleverly. They still bow to CAP during a real majority-side partition (writes block), but the practical envelope is much better than 2010-era reasoning suggested.

**Honest synthesis:** CAP is a useful starting framework but not a design guide. Use it to recognize that *some* trade-off is necessary in distributed systems, then reach for richer models (PACELC, consistency hierarchies, latency-vs-staleness curves) for actual design decisions.

---

## 5. NoSQL Trade-offs

### Mental model

"NoSQL" is a marketing term covering very different systems. The useful taxonomy:

- **Key-value** (Redis, DynamoDB) — `key → value`. O(1) lookups, no relations.
- **Wide-column** (Cassandra, HBase, ScyllaDB) — `partition_key → (clustering_key → value)`. Optimized for time-series and write-heavy workloads.
- **Document** (MongoDB, Couchbase) — JSON documents with flexible schemas, queryable on nested fields.
- **Graph** (Neo4j, Neptune) — nodes and edges, optimized for traversal queries.

The thing they share: they relax some property that relational DBs hold (joins, ACID, schema, single-node) in exchange for some property they value more (horizontal scale, flexibility, write throughput, traversal speed).

The mistake is picking NoSQL because it sounds modern. Postgres has, for many use cases, eaten NoSQL's lunch with `JSONB`, partial indexes, partitioning, and logical replication.

---

### Scenario 5.1 — The "pick a database" interview question

> You're designing storage for these four systems. Pick a database for each and justify:
>
> 1. IoT sensor telemetry — 1M devices, each emitting a reading every 10 seconds. Mostly written, occasionally queried by device + time range.
> 2. User profile store for a social app — 100M users, each profile is a JSON blob with varying fields. Read-heavy, queried by user ID.
> 3. Friend-of-friend recommendations on a social network — needs to traverse 2-3 hops in the social graph.
> 4. Product catalog for an e-commerce site — 1M products, complex filters (price + category + brand + size + in-stock), needs transactions for inventory.

**Answer:**

**1. IoT telemetry → Cassandra or a purpose-built TSDB (TimescaleDB, InfluxDB).**

Why Cassandra:
- Write-optimized via LSM-trees (memtable + SSTables) — handles 1M writes/sec on modest hardware.
- Natural data model: partition by `device_id`, cluster by `timestamp DESC`. Queries like "last 24h of readings for device X" hit one partition with a sequential scan.
- Linear scalability — add nodes, capacity grows.
- No need for joins or transactions; this is append-only telemetry.

Why TimescaleDB might win:
- It's Postgres underneath, so SQL, joins, and transactions if you ever need them.
- Hypertables auto-partition by time. Compression for old data is excellent (10x+).
- For 1M devices × 6 readings/min = 6M writes/min = 100K/s. Comfortably within a beefy single TimescaleDB node, with horizontal scaling available.

**My pick:** TimescaleDB unless I expect 10M+ devices, in which case Cassandra. The Postgres ecosystem benefit is huge.

**2. User profile store → MongoDB or Postgres with JSONB.**

Why MongoDB:
- Document model fits "varying fields" naturally — no schema migrations as fields evolve.
- Sharding by `user_id` is built in.
- Per-document atomicity is enough for "update my profile" operations.

Why Postgres + JSONB might win:
- JSONB supports the same flexible-schema use case with full SQL on top.
- GIN indexes on JSONB fields → fast queries on nested attributes.
- Multi-statement ACID transactions if needed.
- Operationally simpler than running a sharded MongoDB cluster.

**My pick:** Postgres with JSONB unless write volume genuinely exceeds single-node capacity. MongoDB shines specifically when document shapes are wildly heterogeneous and at scales where Postgres horizontal scaling becomes painful.

**3. Friend-of-friend → Neo4j (or another graph DB).**

This is the textbook graph use case. In SQL, "find friends of friends" is a self-join on the friendship table, then another self-join, getting expensive fast at 3+ hops. In a graph DB:

```cypher
MATCH (me:User {id: $userId})-[:FRIEND]->(friend)-[:FRIEND]->(fof)
WHERE fof <> me AND NOT (me)-[:FRIEND]->(fof)
RETURN fof, count(friend) AS mutualCount
ORDER BY mutualCount DESC LIMIT 50
```

Native graph storage means traversals are pointer-chasing, not joins. 100x+ faster than SQL at multi-hop queries.

Caveat: at very large scale (Facebook-size), even graph DBs struggle, and people build custom solutions (TAO at Facebook). For most apps, Neo4j or Neptune is the right call.

**4. Product catalog → Postgres.**

Boring answer, correct answer.

- Complex multi-attribute filters → SQL with proper indexes (B-tree or GIN for trigram/array).
- Inventory transactions → ACID. You don't want to oversell a product because two checkouts both read "1 in stock."
- 1M products fits comfortably in a single Postgres instance with room to spare.
- Full-text search built in (`tsvector`); add Elasticsearch only if you outgrow it.

The mistake people make: reaching for Elasticsearch as the *primary* product catalog because "search is hard." ES is wonderful as a *search index* synced from a primary store, but using it as source-of-truth for inventory is a recipe for consistency bugs.

---

### Scenario 5.2 — Cassandra vs DynamoDB vs MongoDB

> All three are commonly thrown around as "NoSQL options." Compare them on architecture, consistency model, and where each excels.

**Answer:**

**Cassandra (wide-column, AP):**

- Architecture: peer-to-peer, no leader. Every node can serve any read/write. Data partitioned via consistent hashing of partition key, replicated to N nodes.
- Consistency: tunable — `ONE`, `QUORUM`, `ALL` per operation. Default is eventual.
- Storage: LSM-tree (writes go to memtable + WAL, flushed to SSTables, compacted in background). Insanely fast writes.
- Best at: write-heavy time-series, event logging, IoT, fraud detection where data is partition-key-keyed and you query by that key. Self-managed, runs anywhere.
- Pain points: no ad-hoc queries (you must design tables for queries), painful operations (compaction tuning, repair), no joins, eventual consistency surprises.

**DynamoDB (key-value + wide-column hybrid, AP→CP tunable, fully managed):**

- Architecture: managed by AWS, hash-partitioned, replicated 3x within a region.
- Consistency: eventually consistent reads by default; strongly consistent reads available (cost: 2x RCU, slightly higher latency).
- Storage: similar LSM ideas internally, but it's a black box.
- Best at: predictable-traffic AWS workloads where you want zero ops. Single-digit-ms latency at any scale. Great for session stores, user profiles, gaming leaderboards.
- Pain points: query patterns must be designed around partition keys + sort keys + secondary indexes. Bad query patterns lead to "hot partition" throttling. Cost can balloon at high read/write rates if you don't tune capacity. Vendor lock-in.

**MongoDB (document, primary-replica, CP-leaning with tunable reads):**

- Architecture: replica sets (one primary + secondaries), sharded clusters for horizontal scale.
- Consistency: writes go to primary; reads can be configured for primary (strong) or secondary (eventual). Causal consistency available within a session.
- Storage: WiredTiger (B-tree based, with compression). Document-shaped.
- Best at: applications with flexible/evolving schemas, content management, catalogs with nested attributes, analytics dashboards.
- Pain points: sharding is more complex to operate than Cassandra's. Historically had reputation for data durability issues (largely fixed). Aggregation pipeline is powerful but not as expressive as SQL.

**Quick comparison:**

| | Cassandra | DynamoDB | MongoDB |
|---|---|---|---|
| Data model | Wide-column | KV + sort key | Document |
| Topology | Peer-to-peer | Managed | Primary-replica |
| Default consistency | Eventual | Eventual | Strong (primary reads) |
| Write throughput ceiling | Highest | Very high | High |
| Query flexibility | Low | Low | High |
| Operational burden | High | Zero | Medium |
| Sweet spot | Time-series, logs | AWS apps | Flexible schemas |

---

### Scenario 5.3 — When NOT to use NoSQL

> A team wants to migrate their Postgres database to MongoDB because "Postgres won't scale." The data is 200GB, growing 20% per year. The team is 8 engineers. What do you tell them?

**Answer:**

I'd push back hard, with specifics.

**1. Postgres absolutely scales to this size and well beyond.**

200GB is small. Single Postgres instances comfortably run on 1-10TB working sets with adequate hardware. Discord runs trillions of messages on Cassandra now, but they were on a single Postgres instance for years through massive growth. The ceiling for vertical Postgres is high — modern instances have 24TB+ of NVMe and 1TB+ of RAM available.

**2. The migration cost is enormous.**

- Schema redesign: relational → document is not a 1:1 map. They'll spend months figuring out denormalization patterns.
- Application rewrite: every query, every transaction, every join becomes new code.
- New operational expertise: replica sets, sharding, ops tooling, monitoring — none of which the team has.
- Data migration: 200GB of live data with downtime budget approaching zero.
- Realistic estimate: 6-12 months of senior eng time. That's 4-8 person-years.

**3. What's the *actual* problem?**

"Won't scale" is vague. Push for specifics:
- Are queries slow? That's an indexing or query plan problem.
- Are writes saturating? Partition the table, add replicas for reads, consider a write-side cache.
- Is the schema too rigid? Use JSONB for the flexible parts.
- Is the team overwhelmed by ops? Move to managed Postgres (RDS, Crunchy, Aiven).

In 95% of "we need NoSQL" cases, the actual fix is a Postgres index, a query rewrite, partitioning, or a read replica.

**4. When would I actually agree?**

- Sustained write rate that no single Postgres instance can absorb (>50K writes/sec sustained).
- Data model that's genuinely document- or graph-shaped (heavy nested structures, deep traversals).
- Multi-region active-active with conflict tolerance.
- Operationally certain that the migration cost is less than the next 5 years of trying to scale Postgres.

For a team of 8 engineers running 200GB of relational data, none of these apply. **Stay on Postgres. Profile the slow queries. Add the right indexes. Move on.**

This is one of those moments in an interview where the *real* signal is engineering judgment, not knowing buzzwords. The strongest answer is "I'd want to understand the actual constraint before recommending a database change."

---

## How to use this for actual interview prep

A few things that worked for me with this kind of material:

1. **Read each scenario, close the doc, talk through the answer out loud.** Interviews are spoken, not written. Practice the verbal flow.
2. **Know one production-grade story per topic.** "I once tuned a composite index that took a query from 8s to 30ms." Real anecdotes beat textbook answers.
3. **Always close with trade-offs.** "I'd pick X, but I'd reconsider if [specific condition]." That's the senior-engineer signal.
4. **Be willing to disagree with the question.** If someone asks "should you use MongoDB here?" and the answer is no, say so and explain why. Not knowing is fine; not having opinions is a red flag at senior level.


