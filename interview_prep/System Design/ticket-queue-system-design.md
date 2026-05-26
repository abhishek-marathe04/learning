# High-Concurrency Ticket Queue System Design
### How BookMyShow / District Handle 10 Lakh Users at 8 PM

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Why Naive Solutions Fail](#why-naive-solutions-fail)
3. [Core Architecture](#core-architecture)
4. [Queue Assignment — How Your Number is Decided](#queue-assignment)
5. [The Waiting Room Pattern](#the-waiting-room-pattern)
6. [How Queue Position is Tracked and Updated](#how-queue-position-is-tracked-and-updated)
7. [Seat Reservation with TTL](#seat-reservation-with-ttl)
8. [Protecting the Database](#protecting-the-database)
9. [Handling the Thundering Herd](#handling-the-thundering-herd)
10. [The Abandonment Cascade](#the-abandonment-cascade)
11. [UX vs Reality — What the Numbers Mean](#ux-vs-reality)
12. [What Actually Goes Wrong](#what-actually-goes-wrong)
13. [Full End-to-End Flow](#full-end-to-end-flow)
14. [Key Takeaways](#key-takeaways)

---

## The Problem

IPL ticket sales open at **8:00 PM sharp**.

At that exact moment:
- **1–10 lakh users** click "Book Now" simultaneously
- Only **30,000 tickets** are available
- More users are joining **every passing second**
- The system must be **fair, fast, and not crash**

This is called the **Thundering Herd Problem** — a massive burst of concurrent traffic hitting a system at a single point in time.

### Scale of the challenge

```
Users hitting the system: 5,00,000 to 10,00,000
Tickets available:               30,000
Ratio:                        ~17x oversubscribed

Peak requests per second at 8:00 PM: 50,000–2,00,000 RPS
Normal system capacity:                       ~5,000 RPS
```

A naive implementation simply **dies** at this load.

---

## Why Naive Solutions Fail

### Option 1: First-come-first-served via Database

```sql
-- 1 lakh threads trying to do this simultaneously
BEGIN TRANSACTION;
SELECT count FROM tickets WHERE event_id = 'ipl_rr_2026';
IF count > 0:
    UPDATE tickets SET count = count - 1;
    INSERT INTO bookings ...
COMMIT;
```

**Why it fails:**
- Database row-level locks create a queue at the DB layer
- At 1 lakh concurrent transactions, connection pool exhausts in milliseconds
- DB CPU spikes to 100%, query times go from 5ms to 30 seconds
- Most users get **connection timeout errors**
- System crashes within 30–60 seconds of sale opening

### Option 2: In-memory counter on a single server

- Works until that server restarts — all queue state lost
- Doesn't scale horizontally
- Single point of failure

### Option 3: Just add more servers

- More servers doesn't solve the **coordination problem**
- Two servers can both assign the same seat to different users
- Race conditions everywhere

**The real solution needs:**
1. Atomic operations (no race conditions)
2. Microsecond latency (not milliseconds)
3. Horizontal scalability
4. State that survives server restarts
5. A way to protect the database from direct traffic

---

## Core Architecture

```
                    ┌─────────────────────────────────┐
                    │         5 Lakh Users             │
                    └──────────────┬──────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │   CDN (Cloudflare / Akamai)      │
                    │   Static assets cached at edge   │
                    │   Sale page, seat map images     │
                    └──────────────┬──────────────────┘
                                   │ Only API calls pass through
                                   ▼
                    ┌─────────────────────────────────┐
                    │      Load Balancer               │
                    │   (AWS ALB / NGINX)              │
                    │   Distributes to 100s of servers │
                    └──────┬──────────────────┬───────┘
                           │                  │
               ┌───────────▼──┐          ┌────▼──────────────┐
               │  Rate Limiter │          │  Queue Assignment  │
               │  DDoS filter  │          │  Service           │
               │  Bot killer   │          │  (Redis INCR)      │
               └───────────────┘          └────────┬──────────┘
                                                   │
                           ┌───────────────────────┼──────────────────────┐
                           │                       │                      │
               ┌───────────▼──────┐   ┌────────────▼──────┐  ┌──────────▼──────┐
               │  Waiting Room     │   │   Booking Engine   │  │   Payment GW    │
               │  (WebSocket)      │   │   Seat Selection   │  │  Razorpay etc.  │
               │  positions > 30k  │   │   positions ≤ 30k  │  │                 │
               └───────────────────┘   └────────────────────┘  └─────────────────┘
                                                   │
                                        ┌──────────▼──────────┐
                                        │   Redis Cluster      │
                                        │   - Queue counter    │
                                        │   - Session mapping  │
                                        │   - Seat locks (TTL) │
                                        │   - Threshold state  │
                                        └──────────┬──────────┘
                                                   │ Only confirmed bookings
                                        ┌──────────▼──────────┐
                                        │   PostgreSQL / MySQL  │
                                        │   Confirmed bookings  │
                                        │   User accounts       │
                                        └─────────────────────┘
```

---

## Queue Assignment

### The Core Trick: Redis `INCR`

When you click "Book Now", the server executes a single Redis command:

```
INCR global:ipl_rr_2026:queue
```

**Why this works:**
- Redis is **single-threaded** for command execution
- `INCR` is **atomic** — no two clients ever get the same number
- Executes in **~1 microsecond**
- Can handle **1 million operations per second** easily

Even if 5 lakh people click simultaneously, Redis serializes them internally and hands out sequential numbers: 1, 2, 3, 4... 5,00,000.

### Session Mapping

Immediately after getting your queue number, the server maps it to your session:

```
Redis HSET queue:sessions  <your_session_token>  45231
```

Your **browser cookie token** → your **queue number**. One atomic write. Done.

```
Browser Cookie: "sess_xyz_abc_789..."
         │
         ▼
Redis Hash: queue:sessions
         │
         └─► sess_xyz_abc_789 = 45231
```

### What determines your queue number?

Your position is **not just about when you clicked.** It's the total journey time:

```
Queue Position = Click time
              + WiFi / 4G first hop latency    (~5–20ms)
              + ISP routing latency            (~10–40ms)
              + Geographic distance to server  (~10–50ms)
              + Load balancer processing       (~1–2ms)
              + Queue service processing       (~1–2ms)
```

Two people clicking at the **exact same millisecond** can get very different numbers because:

- **Mumbai user vs Pune user** — BMS servers likely hosted in Mumbai, Mumbai user wins by 15–20ms
- **Airtel vs BSNL** — routing efficiency differs significantly  
- **4G vs WiFi** — sometimes 4G wins on first hop
- **Which server you land on** — pure load balancer luck

**Beyond ~position 50,000 in a 5 lakh queue, it's essentially a lottery.** Network physics, not reflexes, decides who wins.

---

## The Waiting Room Pattern

Users are split into two buckets immediately:

```
Queue position ≤ 30,000  →  Allowed into booking flow
Queue position > 30,000  →  Waiting Room
```

### The Waiting Room is a separate, lightweight system

This is critical. The waiting room does **not** touch the booking engine or database. It's just:

- A static-ish page
- A persistent WebSocket connection to a notification server
- Your queue number stored in Redis

This means **4,70,000 waiting users add almost zero load** to the booking infrastructure.

### What "You are #12,000 in queue" actually means

District and BMS show a **relative position**, not your absolute queue number:

```
Your absolute queue number: 42,231
Current threshold:          30,231
Display shows:              42,231 - 30,231 = 12,000 ahead of you
```

As the threshold advances, your displayed number drops — even though your actual queue number never changes.

```
Your number:    42,231  (permanent, never changes)
Threshold:      30,231  → 32,231  → 40,000  → 42,100  → 42,231
Display shows:  12,000  → 10,000  →  2,231  →   131   →  "You're in!"
```

---

## How Queue Position is Tracked and Updated

### The server only tracks ONE number

The entire queue state for 5 lakh users is managed by tracking **a single threshold value** in Redis:

```
Redis GET queue:ipl_rr_2026:threshold
→ 30,047
```

This number represents: *"Everyone with queue position ≤ 30,047 has been let in."*

### Updates via WebSocket, not polling

Your browser does NOT ask "what's my position?" every second. That would be 5 lakh requests/second — instant collapse.

Instead:

```
Browser ◄──── WebSocket ──── Notification Server
              persistent
              connection

Push events sent only when something changes:
- Heartbeat: "Still waiting" every 30 seconds  
- Update:    "10,000 ahead of you" every ~30 seconds
- Alert:     "You're in! 8 minutes to complete booking"
```

### How threshold advances

A background job runs every few seconds:

```python
def advance_threshold():
    # Count abandoned sessions (expired TTLs, closed tabs)
    recovered_slots = count_released_seats()
    
    # Count completed bookings
    completed = get_recent_completions()
    
    # Advance threshold by total freed capacity
    new_threshold = current_threshold + recovered_slots + completed
    redis.set("queue:threshold", new_threshold)
    
    # Notify users who just crossed the threshold
    notify_newly_eligible_users(old_threshold, new_threshold)
```

### The "single digit decrement" — UX theater vs real updates

**BookMyShow style (cosmetic):**
```
45,231 → 45,230 → 45,229 → 45,228
```
Decrements by 1 every few seconds regardless of actual throughput. Pure psychology — tells your brain "system is alive, don't close the tab." The actual threshold may have barely moved.

**District style (real):**
```
12,000 → 10,500 → 8,200 → 2,000 → 131 → "You're in!"
```
Shows the actual relative position. Big drops reflect real throughput — hundreds of people booking or abandoning simultaneously.

**Why the cosmetic version exists:**
- Without any visible change, users think the app is frozen
- They refresh → lose their queue position → call support
- A fake decrement costs nothing computationally and prevents this

---

## Seat Reservation with TTL

When you enter the booking flow, your chosen seat is **soft-reserved** in Redis:

```
Redis SET seat:M12:row5:seat23  user_session_xyz  EX 480
                                                      └── 8 minutes TTL
```

This means:
- The seat is **held for you for 8 minutes**
- Nobody else can select it during this time
- If you don't pay within 8 minutes → **key auto-expires → seat released automatically**
- Redis handles expiry natively — no cron job needed, no cleanup code

### Why 8 minutes?

```
Seat selection:     ~2 minutes
Payment details:    ~2 minutes  
OTP / 2FA:         ~1 minute
Payment processing: ~30 seconds
Buffer:             ~2.5 minutes
─────────────────────────────
Total:              8 minutes
```

### The countdown timer is real

That "Complete booking in 07:43" timer on BookMyShow is directly tied to this Redis TTL. It's not cosmetic. When it hits zero, your seats are genuinely gone.

---

## Protecting the Database

The database never sees the traffic spike. This is the most important architectural decision.

```
┌─────────────────────────────────────────────────────┐
│                   REDIS LAYER                        │
│                                                      │
│  Queue counter:    INCR → 1 operation                │
│  Session mapping:  HSET → 1 operation                │
│  Seat hold:        SET EX → 1 operation per seat     │
│  Threshold:        GET/SET → background job only     │
│                                                      │
│  All 5 lakh users handled here                       │
└──────────────────────────┬──────────────────────────┘
                           │
                           │ Only on successful payment
                           │ ~30,000 writes total
                           ▼
┌─────────────────────────────────────────────────────┐
│                  DATABASE (PostgreSQL)               │
│                                                      │
│  INSERT INTO bookings (user_id, seat_id, ...)        │
│  UPDATE ticket_inventory SET sold = sold + 1         │
│                                                      │
│  Sees ~30,000 writes over 30–60 minutes              │
│  Normal load, no spike                               │
└─────────────────────────────────────────────────────┘
```

### Comparison

| Layer | Users hitting it | Writes |
|---|---|---|
| CDN | 10,00,000 | 0 (read cache) |
| Load Balancer | 10,00,000 | — |
| Redis | 10,00,000 | ~10,00,000 (microsecond each) |
| Booking Engine | 30,000 | — |
| Database | 30,000 | 30,000 over 30-60 mins |

---

## Handling the Thundering Herd

Three layers prevent the system from being overwhelmed:

### Layer 1: CDN — Absorb static traffic at edge

```
Seat map images, CSS, JS, sale page HTML
→ Cached at 200+ edge locations globally
→ Never reaches origin servers
→ Serves millions of requests with zero backend load
```

### Layer 2: Rate Limiting — Kill bots and burst traffic

Token bucket algorithm per user/IP:

```python
# Each user gets 1 token per second
# "Book Now" click costs 1 token
# If token bucket empty → 429 Too Many Requests

if not rate_limiter.consume(user_ip, cost=1):
    return Response(status=429, body="Too many requests")
```

- Prevents a single user/bot from flooding the queue endpoint
- Bots running scripts get blocked here
- Real users get through

### Layer 3: Backpressure — Protect the booking engine

If the booking engine is overwhelmed, stop letting people in:

```python
def should_admit_user(queue_position):
    current_load = booking_engine.get_active_sessions()
    
    if current_load > MAX_CONCURRENT_SESSIONS:
        # Don't advance threshold — just hold everyone in waiting room
        return False
    
    return queue_position <= current_threshold
```

The booking engine processes at its **own comfortable pace**. The waiting room absorbs overflow indefinitely.

---

## The Abandonment Cascade

This explains the experience of watching your position drop from 12,000 to 2,000 suddenly.

### Why it happens

The first wave of users (positions 1–30,000) all enter booking at 8:00 PM. They have **8 minutes** each. Many of them:

```
→ Can't find their preferred stand / category
→ Get price shock (Category A at ₹8,000)
→ Payment fails (wrong CVV, OTP timeout)
→ Close the tab to try a different device
→ Internet drops during payment
→ Simply give up
```

### The timing creates a cascade

```
8:00 PM  — First 30,000 users enter booking
8:08 PM  — First wave of 8-min TTLs start expiring
           Hundreds of seats release simultaneously
           Threshold jumps by 500–1000 at once
           
8:10 PM  — More payment failures accumulate
           Another burst of releases
           
8:12–15 PM — The "abandonment cascade" peaks
           Thousands of positions release in minutes
           Queue position drops rapidly: 12,000 → 2,000
           
8:15–20 PM — Genuine scarcity kicks in
           Only real buyers left
           Queue slows or stops
```

This is exactly the experience of: slow progress → sudden big drop → you're inside within seconds.

---

## UX vs Reality

| What you see | What's actually happening |
|---|---|
| "You are #45,231" | Your absolute position in the atomic Redis counter |
| "12,000 people ahead of you" | `your_position - current_threshold` — a simple subtraction |
| Number dropping by 1 every few seconds | Cosmetic decrement (BMS) OR real throughput (District) |
| Big drop from 12k → 2k suddenly | Abandonment cascade — TTL expiries firing in a burst |
| "Complete booking in 07:43" | Real Redis TTL countdown — it genuinely expires |
| "Your seats are no longer available" | Redis EX key expired, seat released back to pool |

---

## What Actually Goes Wrong

Even with this architecture, real incidents happen:

### Redis cluster failover
If the Redis primary fails mid-sale and the replica takes over, queue numbers can reset or session mappings can be lost. Users lose their position. Chaos ensues.

**Mitigation:** Redis Sentinel or Redis Cluster with AOF persistence. But failover still takes 10–30 seconds during which the system is in a degraded state.

### Payment gateway becomes the bottleneck
This is the most common real-world failure. The entire BMS infrastructure handles load fine, but Razorpay / CC Avenue / PayU gets overwhelmed by 30,000 near-simultaneous payment initiations.

**The 2023 World Cup Final sale** — BMS itself handled load fine. The payment gateway buckled. Users completed seat selection, hit "Pay", and got timeouts. Seats got released. Frustration peaked.

**Mitigation:** Payment gateway rate limiting from BMS side (release users to payment in batches), multiple gateway fallbacks.

### Seat map rendering kills browsers
The SVG seat map for a 50,000 capacity stadium is a massive DOM tree. Loading and rendering it for 30,000 concurrent users spikes browser memory and CPU.

**Mitigation:** Virtualised seat map rendering (only render visible seats), progressive loading, simplified mobile view.

### WebSocket connection limits
At 5 lakh concurrent WebSocket connections, you need serious infrastructure. Each connection holds a file descriptor on the server.

**Mitigation:** Horizontal scaling of WebSocket servers with sticky sessions, or switching to SSE (Server-Sent Events) which is lighter.

---

## Full End-to-End Flow

```
7:58 PM  — User opens BMS/District app
           Static page served from CDN
           User logs in, pre-fills payment

8:00:00 PM — User clicks "Book Now"
           │
           ▼
    Load Balancer routes to Queue Service
           │
           ▼
    Rate Limiter checks: bot? throttled? → No → proceed
           │
           ▼
    Redis: INCR global:ipl_rr_2026:queue → returns 45,231
    Redis: HSET queue:sessions sess_xyz 45231
           │
           ▼
    Is 45,231 ≤ 30,000?
           │
     NO    │    YES
     │     │     └──→ Enter booking flow immediately
     ▼     │
  Waiting Room
  WebSocket connection opened
  Display: "12,000 people ahead of you"
           │
           │  Every 30 seconds:
           │  Server: threshold = 30,231
           │  Display: 45,231 - 30,231 = 12,000
           │
8:08 PM  — First TTL wave expires
           Threshold jumps to 43,000
           WebSocket push: "2,231 ahead of you"
           │
8:10 PM  — More abandonments
           Threshold jumps to 45,100
           WebSocket push: "131 ahead of you"
           │
8:11 PM  — Threshold hits 45,231
           WebSocket push: "You're in! 8 minutes to complete"
           │
           ▼
    Booking Engine: Load seat map
    User selects seats M12, M13
           │
           ▼
    Redis: SET seat:M12 sess_xyz EX 480
    Redis: SET seat:M13 sess_xyz EX 480
    Countdown timer starts: 07:59... 07:58...
           │
           ▼
    User enters payment details
    Hits "Pay ₹4,200"
           │
           ▼
    Payment Gateway: processes card
           │
     FAIL  │  SUCCESS
      │    │     │
      │    │     ▼
      │    │  Redis: DEL seat:M12, DEL seat:M13  (explicit delete)
      │    │  PostgreSQL: INSERT INTO bookings ...
      │    │  Email / SMS confirmation sent
      │    │  Ticket PDF generated
      │    │
      ▼    │
   Seats auto-release
   (or user retries payment)
   Threshold advances
   Next user in queue notified
```

---

## Key Takeaways

### System Design Principles Demonstrated

| Principle | How it's applied |
|---|---|
| **Separate hot path from cold path** | Redis handles all queue ops; DB only sees confirmed bookings |
| **Atomic operations over locks** | Redis INCR instead of DB transactions for queue numbering |
| **Push over poll** | WebSocket notifications instead of client polling |
| **TTL-based cleanup** | Seat expiry is automatic, no cron jobs or cleanup code |
| **Backpressure** | Waiting room absorbs overflow without stressing booking engine |
| **Stateless servers** | All state in Redis; any server can handle any request |
| **CDN for static assets** | Millions of page loads never hit origin |

### The Elegant Core Insight

The entire system — handling 10 lakh users fairly — reduces to:

```
One Redis counter (queue assignment)     → O(1), microseconds
One Redis hash (session mapping)         → O(1), microseconds  
One Redis integer (threshold)            → O(1), microseconds
One Redis key per seat (TTL hold)        → O(1), microseconds
```

**The database sees none of this.** It only sees the 30,000 happy endings.

### Why your queue position feels unfair (and is)

For a 5 lakh user sale with 30,000 tickets, whether you're #28,000 or #35,000 is decided by:
- Your ISP's routing efficiency (you don't control)
- Your physical distance from the server (you don't control)
- Which load balancer instance you land on (random)
- Network congestion at 8:00:00 PM (you don't control)

It's a **lottery disguised as a queue.** The ordered numbering creates a perception of fairness. The engineering underneath is genuinely world-class. Whether the outcome is fair is a different question entirely.

---

*Discussion and analysis of BookMyShow / District ticket sale queue system*  
*Based on: system design principles, distributed systems patterns, Redis internals*
