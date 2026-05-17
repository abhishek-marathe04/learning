# System Design Study Guide: Load Balancing & API Design Patterns

> **Target audience:** Senior Software Engineers preparing for system design interviews
> **Prerequisites:** Familiarity with HTTP, REST, basic distributed systems concepts
> **Goal:** Deep conceptual understanding, not memorization

---

## Table of Contents

1. [Load Balancing](#part-1-load-balancing)
   - [What is a Load Balancer, Really?](#what-is-a-load-balancer-really)
   - [Layer 4 vs Layer 7 Load Balancing](#layer-4-vs-layer-7-load-balancing)
   - [Load Balancing Algorithms](#load-balancing-algorithms)
   - [Consistent Hashing — Deep Dive](#consistent-hashing--deep-dive)
   - [Health Checks & Failure Detection](#health-checks--failure-detection)
   - [Sticky Sessions](#sticky-sessions)
   - [Global Load Balancing (GeoDNS, Anycast)](#global-load-balancing)
2. [API Design Patterns](#part-2-api-design-patterns)
   - [gRPC Deep Dive](#grpc-deep-dive)
   - [WebSockets Deep Dive](#websockets-deep-dive)
   - [Webhooks Deep Dive](#webhooks-deep-dive)
   - [Server-Sent Events (SSE)](#server-sent-events-sse)
   - [Rate Limiting Algorithms](#rate-limiting-algorithms)
3. [Scenario-Based Interview Questions](#part-3-scenario-based-interview-questions)
4. [Cheat Sheet & Decision Trees](#part-4-cheat-sheet)

---

# Part 1: Load Balancing

## What is a Load Balancer, Really?

A load balancer is a **traffic cop** that sits between clients and a pool of servers. But that's the surface description. Let me give you the deeper mental model.

Think of it as solving **three fundamental problems simultaneously**:

1. **Scalability** — A single server can handle ~10K-100K connections. Beyond that, you need horizontal scaling, which is impossible without something distributing the work.
2. **Availability** — If one server dies, traffic should automatically route around it. The LB is the failover mechanism.
3. **Abstraction** — Clients see ONE endpoint (`api.google.com`). The fact that there are 50 servers behind it is hidden. This decoupling is enormous — you can add/remove servers without clients knowing.

### The Architectural Position

```
                    ┌─────────────┐
   Clients ───────► │    DNS      │
                    └──────┬──────┘
                           │ (returns LB IP)
                           ▼
                    ┌─────────────┐
                    │ Load        │  ◄─── This is your control plane
                    │ Balancer    │       (routing decisions happen here)
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐   ┌────────┐   ┌────────┐
         │Server 1│   │Server 2│   │Server 3│
         └────────┘   └────────┘   └────────┘
```

### Hardware vs Software vs Cloud LBs

| Type | Examples | When to Use |
|------|----------|-------------|
| **Hardware** | F5 BIG-IP, Citrix NetScaler | High-throughput data centers, low latency requirements |
| **Software** | NGINX, HAProxy, Envoy | Most modern setups, runs on commodity hardware |
| **Cloud-managed** | AWS ALB/NLB, GCP Cloud Load Balancing | Serverless infrastructure, auto-scaling integration |
| **Service Mesh** | Istio, Linkerd | Microservices with internal traffic management |

**Why this matters in interviews:** Knowing when each fits shows architectural maturity. Hardware LBs are dying because software LBs on Linux can now handle millions of connections (epoll, io_uring).

---

## Layer 4 vs Layer 7 Load Balancing

This is **the most important distinction** to understand. It maps directly to the OSI model.

### Layer 4 (Transport Layer)

Operates on **TCP/UDP packets**. The LB doesn't look inside the packet — it just sees:
- Source IP
- Destination IP
- Source port
- Destination port
- Protocol (TCP/UDP)

```
Client TCP packet ──► L4 LB ──► (NAT or DSR) ──► Backend Server
                       │
                       └──── Decision based on IP/port only
```

**Characteristics:**
- **Fast** — minimal processing, often hardware-accelerated
- **Protocol-agnostic** — works for any TCP/UDP traffic (databases, gaming, video)
- **Stateful** — tracks connections in a flow table
- **Cannot make routing decisions based on content** (no URL inspection, no headers)

**Real-world examples:** AWS NLB, HAProxy in TCP mode, IPVS

### Layer 7 (Application Layer)

Operates on **HTTP/HTTPS requests**. The LB **terminates the connection**, parses the HTTP request, and can route based on:
- URL path (`/api/v1/users` → user-service)
- HTTP headers (`Host`, `Authorization`, custom headers)
- Cookies (for sticky sessions)
- Query parameters
- HTTP method (GET vs POST)

```
Client HTTP request ──► L7 LB ──► [parses request] ──► Routes to specific backend
                          │
                          ├── /api/users   ──► User Service
                          ├── /api/orders  ──► Order Service
                          └── /static/*    ──► CDN/Static Server
```

**Characteristics:**
- **Smart routing** — content-aware
- **Slower than L4** (parsing overhead, but usually negligible)
- **TLS termination** happens here (so you decrypt once)
- **Enables advanced features:** request rewriting, A/B testing, canary deployments, WAF integration

**Real-world examples:** AWS ALB, NGINX, Envoy, Traefik

### Key Interview Insight

> **"When would you use L4 over L7?"**

Use **L4** when:
- Non-HTTP traffic (game servers, custom TCP protocols, MQTT, gRPC if you don't need path routing)
- Extreme low-latency needs (microseconds matter)
- You need end-to-end TLS without termination at LB
- Simple round-robin distribution

Use **L7** when:
- HTTP/HTTPS APIs (the common case)
- You need path-based or host-based routing
- Microservices with API gateways
- Need observability (per-endpoint metrics)
- Want to terminate TLS centrally

---

## Load Balancing Algorithms

The algorithm determines **which server gets the next request**. Choice depends on backend characteristics.

### 1. Round Robin

**How it works:** Cycle through servers in order. Request 1 → Server A, Request 2 → Server B, Request 3 → Server C, Request 4 → Server A...

```python
class RoundRobin:
    def __init__(self, servers):
        self.servers = servers
        self.index = 0

    def get_server(self):
        server = self.servers[self.index]
        self.index = (self.index + 1) % len(self.servers)
        return server
```

**Pros:**
- Dead simple
- No state needed beyond a counter
- Fair distribution if servers are identical

**Cons:**
- **Ignores server load** — sends requests to a struggling server just like a healthy one
- Bad for variable-cost requests (a 5ms request and a 5-second request count the same)

**When to use:** Stateless services with uniform request costs and identical server capacity.

### 2. Weighted Round Robin

**How it works:** Same as round robin, but servers have weights based on capacity. Server A (weight 3), Server B (weight 1) → A gets 3x the traffic.

**When to use:** Heterogeneous server fleet (mixing m5.large with m5.xlarge instances).

### 3. Least Connections

**How it works:** Send the next request to the server currently handling the **fewest active connections**.

```python
def get_server(self):
    return min(self.servers, key=lambda s: s.active_connections)
```

**Pros:**
- Adapts to actual load
- Great for **long-lived connections** (WebSockets, database connections)
- Self-correcting — slow servers naturally accumulate fewer connections

**Cons:**
- Requires tracking connection counts (state)
- Can cause "thundering herd" when a new server joins (all new connections go there)

**When to use:** Variable-duration requests, WebSocket servers, long-polling endpoints.

### 4. Least Response Time

**How it works:** Combines least connections with the lowest average response time. The LB picks the server that's **fastest AND least busy**.

**When to use:** Latency-sensitive APIs where you want to actively route around slow servers.

### 5. IP Hash

**How it works:** `hash(client_ip) % num_servers` → maps each client to a fixed server.

**Pros:**
- Same client always hits the same server (cheap sticky sessions)
- No state needed at LB

**Cons:**
- Uneven distribution if you have many clients behind one NAT (corporate networks)
- **Major problem when servers are added/removed** — most clients get remapped (this is what consistent hashing solves)

### 6. Random / Power of Two Choices (P2C)

**How it works:** Pick **two servers at random**, send the request to whichever has fewer connections.

**Why it's brilliant:** With just two random samples, you avoid the worst-case server most of the time. It's mathematically much better than pure random and only slightly worse than full least-connections — but with WAY less coordination.

**Used by:** NGINX (`random two least_conn`), HAProxy, many service meshes.

> **Interview gold:** Mentioning P2C signals you've read distributed systems literature. This is a real-world favorite at companies like Twitter and Netflix.

---

## Consistent Hashing — Deep Dive

This is **the most important load balancing concept** for senior interviews. It appears in caches (Memcached, Redis), DBs (Cassandra, DynamoDB), CDNs, and message queues.

### The Problem It Solves

**Naive hashing:** `server = hash(key) % N` where N = number of servers.

Suppose you have 4 cache servers and 1 million keys. Each server has ~250K keys. Now add a 5th server:
- New formula: `hash(key) % 5`
- **Almost every key now maps to a different server** (~80% of keys move)
- This causes a **cache stampede**: massive miss rate, backend gets crushed, possibly cascading failure

**This is the cardinal sin in distributed caches.** You added capacity to handle MORE load and instead caused an outage.

### The Solution: The Hash Ring

Imagine a circle representing the hash space (e.g., 0 to 2^32 - 1).

```
                        0
                        │
            S1 ●────────┼────────● S2
                  ╱     │     ╲
                ╱       │       ╲
              ╱  K1     │   K2    ╲
            ╱           │           ╲
       2^32/4 ●─────────┼──────────● 3·2^32/4
                ╲       │       ╱
                  ╲     │     ╱
                    ╲   │   ╱
                      ● S3
                       2^32/2

   Each server gets a position on the ring (hash of server ID).
   Each key gets a position on the ring (hash of key).
   A key belongs to the FIRST SERVER you encounter going CLOCKWISE.
```

### How It Works — Step by Step

1. **Hash each server** to a position on the ring: `hash("server-A")`, `hash("server-B")`, etc.
2. **Hash each key** to a position: `hash("user:123")`.
3. **For each key, walk clockwise** until you find the first server. That's its owner.

**Adding a server:**
- New server gets a position on the ring.
- It "steals" only the keys between its position and the previous server (counterclockwise).
- **Only ~1/N of keys move**, not most of them.

**Removing a server:**
- Its keys go to the next server clockwise.
- Again, only ~1/N keys move.

### The Virtual Nodes Trick (Crucial!)

Naive consistent hashing has a problem: **uneven distribution**. With 3 servers, one might get 50% of the ring just by random hash placement.

**Solution:** Each physical server gets **multiple positions** on the ring (e.g., 100-200 virtual nodes).

```
Physical Server A → V-nodes at: hash("A-1"), hash("A-2"), ..., hash("A-150")
Physical Server B → V-nodes at: hash("B-1"), hash("B-2"), ..., hash("B-150")
```

**Why this works:** With 150 virtual nodes per server, the law of large numbers smooths out the distribution. Each server ends up with roughly equal share.

### Implementation Sketch

```python
import hashlib
from sortedcontainers import SortedDict

class ConsistentHashRing:
    def __init__(self, virtual_nodes=150):
        self.ring = SortedDict()
        self.virtual_nodes = virtual_nodes

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_server(self, server):
        for i in range(self.virtual_nodes):
            vnode_key = f"{server}-{i}"
            self.ring[self._hash(vnode_key)] = server

    def remove_server(self, server):
        for i in range(self.virtual_nodes):
            vnode_key = f"{server}-{i}"
            del self.ring[self._hash(vnode_key)]

    def get_server(self, key):
        if not self.ring:
            return None
        h = self._hash(key)
        # Find first server position >= hash, wrap around if needed
        idx = self.ring.bisect_right(h)
        if idx == len(self.ring):
            idx = 0
        return self.ring.values()[idx]
```

### Where It's Used in Production

| System | How it uses Consistent Hashing |
|--------|-------------------------------|
| **Memcached** (client-side) | Distributes keys across cache nodes. Adding a node only invalidates ~1/N of cache. |
| **Redis Cluster** | Uses 16384 hash slots (a variant) — more deterministic than virtual nodes |
| **Cassandra / DynamoDB** | Token ring for partition placement. Tunable replication on the ring. |
| **CDNs** (Akamai, Cloudflare) | Maps URLs to edge cache servers. |
| **Discord** | Sharding chat servers across nodes |
| **Riak, Couchbase** | Data partitioning |

### Trade-offs / Gotchas

- **Hash function quality matters** — bad hash = uneven ring even with vnodes.
- **Vnode count is a tuning knob** — too few → unbalanced; too many → memory overhead and slow lookups.
- **Hot keys still concentrate** — consistent hashing balances *key distribution*, not *access patterns*. A viral celebrity's profile might destroy whichever node owns it. Solutions: replication, request-level load balancing.
- **Bounded loads variant:** "Consistent Hashing with Bounded Loads" (Google paper, 2017) caps the load on any node, falling back to next node if exceeded. Used in Vimeo, Google's load balancers.

---

## Health Checks & Failure Detection

A load balancer is only useful if it **knows which servers are alive**. Mechanisms:

### Active Health Checks

LB periodically pings each backend (e.g., every 5 seconds):
- **TCP check:** Can I open a connection on port 8080?
- **HTTP check:** Does GET `/health` return 200?
- **Custom check:** Run a script that validates DB connectivity, cache state, etc.

**Best practice:** Distinguish between "healthy" (`/health`) and "ready to serve traffic" (`/ready`). Kubernetes makes this distinction explicit (liveness vs readiness probes).

### Passive Health Checks

Watch real traffic — if a server returns errors or times out, mark it unhealthy.
- Faster reaction time (no waiting for next active check)
- Risk: false positives from a single bad request

**Most production LBs combine both.**

### Outlier Detection (Envoy/Istio)

Statistical detection: if a server's error rate is significantly higher than the mean across the fleet, eject it temporarily. Comes back after a cooldown.

---

## Sticky Sessions

Sometimes you NEED a client to hit the same backend repeatedly:
- **In-memory session storage** (instead of Redis)
- **WebSocket connections** (the connection lives on one server)
- **Stateful workloads** (file uploads with chunked sessions)

### Implementation Methods

1. **Cookie-based:** LB sets a cookie like `SERVERID=server-3`. Reads it on subsequent requests.
2. **IP-based:** Hash client IP. Simple but breaks behind NAT.
3. **SSL session ID:** Tied to TLS session.

### The Trade-off

Sticky sessions are an **anti-pattern at scale**. They:
- Break load balancing fairness (a popular user may overload one server)
- Make scaling harder (can't drain a server cleanly)
- Are fragile to server failures (session lost on crash)

> **Modern best practice:** Make services stateless. Store session in Redis/Memcached. Use sticky sessions only when forced to (WebSockets).

---

## Global Load Balancing

For services with a global user base, you load balance **across data centers**, not just within one.

### GeoDNS

DNS resolver returns **different IPs based on the requester's geography**:
- User in Mumbai → returns India data center IP
- User in NYC → returns US East IP

**Limitation:** DNS caching means failover is slow (TTL-bound). Granularity is rough.

### Anycast

**Same IP advertised from multiple locations** via BGP. The internet's routing protocols deliver packets to the **nearest** advertised location.

**Used by:** Cloudflare, Google DNS (8.8.8.8), AWS Global Accelerator.

**Advantages:**
- Near-instant failover (BGP withdraws bad routes)
- Automatic latency optimization
- Same endpoint everywhere (simpler client code)

### CDN as a Load Balancer

CDNs (Cloudflare, Akamai, Fastly) act as a **distributed L7 LB** for your static and increasingly dynamic content. They terminate connections at the edge, serve cached content, and forward dynamic requests to origin.

---

# Part 2: API Design Patterns

You know REST. Let's go deep on the alternatives and when each shines.

## REST vs The Alternatives — Quick Mental Map

| Style | Transport | Use Case | Strengths |
|-------|-----------|----------|-----------|
| **REST** | HTTP/1.1, JSON | Public APIs, CRUD | Simple, cacheable, ubiquitous |
| **GraphQL** | HTTP, JSON | Mobile apps, federated data | Client picks fields, single endpoint |
| **gRPC** | HTTP/2, Protobuf | Internal microservices | Fast, strongly-typed, streaming |
| **WebSockets** | WS over HTTP upgrade | Real-time bi-directional | Persistent, low-latency push |
| **Webhooks** | HTTP callbacks | Event notification | No polling, server-initiated |
| **SSE** | HTTP, text/event-stream | Server → client streams | Simple, auto-reconnect |
| **Message Queue** | AMQP/Kafka | Async backend processing | Decoupling, durability |

---

## gRPC Deep Dive

### What gRPC Actually Is

gRPC is **Remote Procedure Call** done right. Instead of designing endpoints and verbs (REST), you define **services with methods**, like calling a function across the network.

**Key components:**
1. **Protocol Buffers (protobuf)** — IDL (Interface Definition Language) for defining schemas
2. **HTTP/2** — the underlying transport (multiplexing, server push, header compression)
3. **Code generation** — write a .proto, generate clients in 11+ languages
4. **Streaming primitives** — built-in support for streaming requests/responses

### A Simple .proto Example

```protobuf
syntax = "proto3";

package fantasy.cricket;

service PlayerService {
  // Unary RPC (request/response)
  rpc GetPlayer(GetPlayerRequest) returns (Player);

  // Server streaming
  rpc StreamPlayerStats(PlayerQuery) returns (stream StatUpdate);

  // Client streaming
  rpc UploadStats(stream StatUpdate) returns (UploadSummary);

  // Bidirectional streaming
  rpc LiveMatch(stream ClientEvent) returns (stream MatchUpdate);
}

message Player {
  string id = 1;
  string name = 2;
  int32 jersey_number = 3;
  repeated string roles = 4;
}

message GetPlayerRequest {
  string player_id = 1;
}
```

### Why HTTP/2 Matters

| HTTP/1.1 | HTTP/2 |
|----------|--------|
| One request per connection (head-of-line blocking) | Multiplexed streams over one connection |
| Plain-text headers (verbose) | HPACK header compression |
| Pull-only (client requests, server responds) | Server can push |
| Text protocol | Binary protocol |

**Result:** gRPC can deliver 5-10x the throughput of equivalent REST APIs on the same hardware.

### Protobuf vs JSON — Why It's Faster

```
JSON: {"id":"player_123","name":"Virat Kohli","jersey":18}
Size: ~50 bytes, requires parsing strings

Protobuf binary: \x0a\x0aplayer_123\x12\x0bVirat Kohli\x18\x12
Size: ~25 bytes, direct binary read into structs
```

**Protobuf wins:**
- **2-5x smaller payloads** (no field names in wire format, just tag numbers)
- **10-100x faster parsing** (no string-to-type conversion)
- **Schema evolution** (add fields without breaking old clients via tag numbers)
- **Type safety** at compile time

### When to Use gRPC

✅ **Internal microservices** (service-to-service)
✅ **Polyglot environments** (one .proto serves Go, Python, Java, Node)
✅ **Streaming required** (telemetry, real-time data feeds)
✅ **Performance-critical paths** (high QPS, low latency)
✅ **Strong typing wanted** (catches bugs at compile time)

❌ **Don't use for:**
- Public APIs consumed by browsers (gRPC needs gRPC-Web proxy; complexity)
- Simple CRUD where REST is fine
- When debugging via curl is critical (binary format is opaque)
- Human-readable API contracts (JSON is easier to eyeball)

### Real-world Architecture Pattern

```
                 ┌──────────────────┐
   Browser ────► │  REST API Gateway│  (HTTP/1.1, JSON — for public)
                 └────────┬─────────┘
                          │
                          │ gRPC (HTTP/2, Protobuf)
                          ▼
              ┌───────────────────────┐
              │  Internal Microservices│
              │  ┌─────┐ ┌─────┐ ┌────┤
              │  │ Auth│ │Order│ │User│
              │  └─────┘ └─────┘ └────┤
              └───────────────────────┘
```

This is the **standard modern pattern**: REST/GraphQL at the edge, gRPC internally.

---

## WebSockets Deep Dive

### What WebSockets Solve

HTTP is **request/response** — the client must initiate every interaction. For real-time features (chat, live scores, multiplayer), this means **polling** or **long-polling**, both wasteful.

WebSockets give you a **persistent, full-duplex** connection. Either side can send data anytime.

### The Handshake

WebSockets start as HTTP, then **upgrade**:

```http
# Client request
GET /chat HTTP/1.1
Host: example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13

# Server response
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

After this handshake, the TCP connection stays open and switches to the WebSocket frame protocol (binary, lightweight).

### Frames, Not Requests

WebSocket data is sent as **frames** — small binary messages with minimal overhead (~2-14 bytes header vs HTTP's hundreds of bytes).

### Architectural Implications

WebSockets fundamentally change your backend architecture:

1. **Stateful connections** — each server holds N open sockets. You can't spread one user's traffic across servers like you can with stateless HTTP.
2. **Sticky load balancing** required — once connected, a client's traffic must keep going to the same server.
3. **Horizontal scaling needs a pub/sub bus** — if User A on Server 1 sends a message to User B on Server 5, the message needs to traverse Redis Pub/Sub or NATS or Kafka.

### Scaling WebSocket Architecture

```
                           ┌────────────┐
   Clients ───WSS────────► │ L4 LB      │ (must support WebSockets / HTTP upgrade)
                           └─────┬──────┘
                                 │
                  ┌──────────────┼──────────────┐
                  ▼              ▼              ▼
             ┌────────┐     ┌────────┐     ┌────────┐
             │WS Srv 1│     │WS Srv 2│     │WS Srv 3│
             └────┬───┘     └────┬───┘     └────┬───┘
                  │              │              │
                  └──────────────┼──────────────┘
                                 ▼
                          ┌──────────────┐
                          │  Redis Pub/Sub│ ◄── Cross-server message routing
                          │  or NATS     │
                          └──────────────┘
```

### When to Use WebSockets

✅ Chat applications (Slack, Discord)
✅ Live scoreboards / sports updates (think CricketIQ live match feed!)
✅ Collaborative editing (Google Docs, Figma)
✅ Real-time gaming
✅ Live trading dashboards
✅ Notifications (where SSE doesn't fit because client also pushes)

❌ Don't use for:
- One-way server → client streams (use SSE — simpler)
- Infrequent updates (polling every minute is fine)
- Stateless RPC (use gRPC streaming instead)

### Common Gotchas

- **Connection limits:** Default Linux allows ~65K connections per IP. Tune `ulimit`, `net.core.somaxconn`, ephemeral port range.
- **Idle connections:** Some firewalls/proxies kill idle connections after 60s. Implement **ping/pong heartbeats** (every 30s).
- **Reconnection logic:** Networks drop. Clients MUST handle reconnection with exponential backoff.
- **Message ordering:** WebSockets preserve order on a single connection but not across reconnections.
- **Backpressure:** If client is slow, server's send buffer fills. Need to either drop messages or apply backpressure.

---

## Webhooks Deep Dive

### The Reverse API Concept

Most APIs are **pull** — you ask, the server answers. Webhooks are **push** — you register a URL, and the server calls *you* when something happens.

```
Traditional API:                     Webhook:
                                     
  You ──GET /orders──► Server         You register URL once
  You ──GET /orders──► Server         ─────────────────►
  You ──GET /orders──► Server         
  (polling, wasteful)                 Server ──POST /your-callback──► You
                                      (only when an event happens)
```

### Real-world Examples

| Service | Webhook Use |
|---------|-------------|
| **Stripe** | Payment success, refund, dispute |
| **GitHub** | Push, PR opened, issue created |
| **Slack** | Slash commands, message events |
| **Twilio** | SMS received, call status changed |
| **Shopify** | Order placed, inventory low |

### A Typical Webhook Flow

```
1. Subscriber registers callback URL
   POST /webhooks
   {
     "url": "https://myapp.com/stripe-events",
     "events": ["payment.succeeded", "payment.failed"],
     "secret": "whsec_xyz..."
   }

2. Event happens at provider
   (a payment succeeds)

3. Provider POSTs to subscriber
   POST https://myapp.com/stripe-events
   X-Signature: sha256=abcdef...
   {
     "event": "payment.succeeded",
     "data": { "amount": 5000, "customer": "cus_123" }
   }

4. Subscriber processes and returns 200 OK
   (or returns 4xx/5xx → provider retries)
```

### Critical Design Considerations

#### 1. Authentication & Security

- **HMAC signatures:** Provider signs payload with shared secret. Subscriber verifies signature.
   ```python
   expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
   if not hmac.compare_digest(expected, received_signature):
       return 401  # reject
   ```
- **HTTPS only** — never accept webhooks over HTTP
- **IP allow-listing** (some providers publish their IP ranges)

#### 2. Reliability — Retries & Idempotency

Networks fail. Subscribers crash. Webhooks **must** be retried.

- Provider retries with **exponential backoff**: 1m, 5m, 30m, 2h, 1d
- Subscriber **must be idempotent**: include event_id; if already processed, ack but don't act again

#### 3. At-least-once vs Exactly-once

Webhooks are **at-least-once** by nature. You WILL receive duplicates. Design for it.

```python
def handle_webhook(event):
    if already_processed(event.id):
        return 200  # ack but skip
    process(event)
    mark_processed(event.id)
```

#### 4. Subscriber Endpoint Reliability

If your webhook handler is slow (>10s), the provider may time out. **Always** queue immediately:

```python
def webhook_endpoint(request):
    verify_signature(request)
    queue.publish("webhook_events", request.body)
    return 200, "OK"  # respond fast

# Worker processes queue separately
def worker():
    while True:
        event = queue.consume()
        process_event(event)  # may take time
```

### Webhooks vs Polling vs Streaming — Decision Matrix

| Need | Best Choice |
|------|-------------|
| Infrequent events, latency tolerant | Webhooks |
| Continuous high-frequency stream | WebSockets / SSE |
| Polling-acceptable, simple integration | REST polling |
| Internal services with reliable network | Message queue (Kafka, RabbitMQ) |

---

## Server-Sent Events (SSE)

Often forgotten in interview prep but extremely useful.

### What SSE Is

Long-lived HTTP connection where the **server pushes events** to the client. Unlike WebSockets, it's **unidirectional** (server → client only).

### How It Works

Client opens HTTP connection. Server keeps it open and writes:
```
data: {"score": 145, "wickets": 3}

data: {"score": 146, "wickets": 3}

event: wicket
data: {"player": "Kohli", "out_by": "caught"}
```

### Why SSE Is Underrated

| Feature | SSE | WebSockets |
|---------|-----|------------|
| Server → client | ✅ | ✅ |
| Client → server | ❌ (use REST for that) | ✅ |
| Auto-reconnect | ✅ Built-in | ❌ Manual |
| Plays nice with HTTP infra | ✅ Just HTTP | Requires upgrade |
| Compression / caching | ✅ | ❌ |
| Binary support | ❌ Text only | ✅ |
| Browser support | ✅ (no IE) | ✅ |

**Use SSE when:**
- One-way server → client streams
- Stock tickers, live scores, log streaming, AI token streaming (ChatGPT-style)
- You want simplicity

**Don't use SSE when:**
- You need bidirectional communication
- Binary data
- Many connections (HTTP/1.1 limits 6 SSE connections per origin per browser; HTTP/2 fixes this)

> **Pro tip:** OpenAI's streaming API uses SSE. Anthropic's streaming API uses SSE. It's the de facto standard for LLM streaming.

---

## Rate Limiting Algorithms

Rate limiting is **the gateway between scalable systems and chaos**. It protects against:
- Abuse / DDoS
- Buggy clients in retry loops
- Cost overruns (think OpenAI bill at 3am)
- Fair multi-tenant resource usage

### Where Rate Limiting Lives

```
┌──────────┐   ┌──────────┐   ┌────────┐   ┌────────────┐
│  Client  │──►│   CDN    │──►│   LB   │──►│   App     │
└──────────┘   └──────────┘   └────────┘   └────────────┘
                    ▲              ▲             ▲
                    │              │             │
                  Edge RL    API Gateway RL    App RL
                  (DDoS)     (per-user/key)   (per-feature)
```

You typically rate limit at multiple layers.

### Algorithm 1: Fixed Window Counter

**How it works:** Count requests in fixed time windows (e.g., per minute).

```
Window 12:00:00-12:00:59 → counter = 0
  Request at 12:00:30 → counter = 1
  Request at 12:00:45 → counter = 2
  ...if counter > limit, reject

Window 12:01:00-12:01:59 → counter resets to 0
```

**Pros:** Trivial to implement. One counter per user.

**Cons:** **Boundary problem** — a user can fire 100 requests at 11:59:59 and another 100 at 12:00:01, resulting in 200 requests in 2 seconds while staying within "100 per minute" limits.

### Algorithm 2: Sliding Window Log

**How it works:** Store timestamps of each request. To check, count timestamps within the last N seconds.

```python
def is_allowed(user_id, now, limit=100, window=60):
    log = redis.zrangebyscore(f"rl:{user_id}", now - window, now)
    if len(log) < limit:
        redis.zadd(f"rl:{user_id}", {now: now})
        redis.expire(f"rl:{user_id}", window)
        return True
    return False
```

**Pros:** Perfectly accurate.

**Cons:** Memory cost per request. Bad for high-traffic users.

### Algorithm 3: Sliding Window Counter

**How it works:** Hybrid — store counters per window, but interpolate based on overlap.

```
Current window (12:01:00-12:01:59) at time 12:01:30:
  - Previous window count (12:00:00-12:00:59): 80
  - Current window count: 30
  - Overlap weight: 30/60 = 0.5 of previous
  - Effective rate: 80 * 0.5 + 30 = 70 requests
```

**Pros:** Approximates sliding log accurately, very memory-efficient.

**Used by:** Cloudflare, many modern API gateways.

### Algorithm 4: Token Bucket

This is **the most important rate limiting algorithm** to understand. Used by AWS, Stripe, and most production systems.

**Mental model:** Imagine a bucket that holds tokens.
- Bucket has a **capacity** (e.g., 100 tokens — your burst limit)
- Tokens **refill** at a constant rate (e.g., 10 tokens/second — your sustained rate)
- Each request **takes 1 token**
- If bucket is empty → reject

```
        Refill rate: 10 tokens/sec
              │
              ▼
        ┌──────────┐
        │   ●●●●   │ ← Bucket (capacity 100)
        │   ●●●●   │
        └──────────┘
              │
              ▼
        Request: take 1 token
        If empty: reject (or queue)
```

**Why it's brilliant:**
- **Allows bursts** — clients can use 100 quick requests if bucket is full
- **Smooths over time** — long-term rate is the refill rate
- **Predictable** — clients understand "I get 100 burst, 10/sec sustained"

```python
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.time()

    def allow(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```

### Algorithm 5: Leaky Bucket

**Mental model:** Bucket with a hole at the bottom.
- Requests fill the bucket
- Bucket "leaks" requests out at a fixed rate (processed)
- If bucket overflows → reject

**Difference from token bucket:**
- Leaky bucket = **smooths output** (requests processed at constant rate, queue absorbs bursts)
- Token bucket = **smooths input** (requests pass through immediately if tokens available)

**Use leaky bucket when:** You want strict, constant outflow (e.g., a downstream service that can only handle 100 RPS no matter what).

**Use token bucket when:** You want to allow bursts but cap long-term rate (more user-friendly).

### Distributed Rate Limiting

Single-server rate limiting is easy. Distributed rate limiting (5 LB instances all need to share counts) is hard.

**Approaches:**

1. **Centralized (Redis):** All LBs increment a Redis counter. Simple, but Redis becomes a bottleneck and SPOF.
   ```
   INCR rate_limit:user_123:60s
   EXPIRE rate_limit:user_123:60s 60
   ```

2. **Probabilistic / approximate:** Each LB tracks locally, syncs to central periodically. Tolerates slight overshoot.

3. **Sticky routing:** Route same user to same LB. No coordination needed but breaks failover.

4. **Cell-based:** Partition users by hash; each cell has its own LB+limiter.

### Rate Limit Response Best Practices

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1714850400
Retry-After: 60
Content-Type: application/json

{
  "error": "rate_limit_exceeded",
  "message": "Limit of 100 requests per minute exceeded.",
  "retry_after": 60
}
```

Always include `Retry-After` so well-behaved clients can back off intelligently.

---

# Part 3: Scenario-Based Interview Questions

## Scenario 1: Designing a Live Cricket Score System (CricketIQ-style)

**The question:** "Design a backend that delivers live ball-by-ball updates to 10 million concurrent users during an IPL final."

### Walk-through Answer

**Step 1: Clarify requirements**
- 10M concurrent users
- Sub-second latency
- ~6 events per over × 20 overs × 2 innings ≈ 240 events per match
- Read-heavy (millions of viewers, ~10 score-data publishers)

**Step 2: Pick the right protocol**

| Protocol | Verdict |
|----------|---------|
| Polling REST | ❌ 10M × poll/2s = 5M RPS, prohibitive |
| WebSockets | ⚠️ Works but expensive (10M open connections) |
| **SSE** | ✅ Perfect — server → client only, auto-reconnect |
| HTTP/2 push | ⚠️ Falling out of favor |

**Choose SSE** — unidirectional fits perfectly, simpler than WS, plays well with CDN.

**Step 3: Architecture**

```
                     ┌─────────────┐
   Score Operator ──►│ Ingest API  │
                     └──────┬──────┘
                            │
                            ▼
                     ┌─────────────┐
                     │   Kafka     │ (event log)
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         ┌─────────┐   ┌─────────┐   ┌─────────┐
         │SSE Edge1│   │SSE Edge2│   │SSE Edge3│ (geographically distributed)
         └────┬────┘   └────┬────┘   └────┬────┘
              │             │             │
       (anycast IP / GeoDNS — clients hit nearest edge)
              │             │             │
         ┌────▼─────────────▼─────────────▼────┐
         │              Clients                 │
         └──────────────────────────────────────┘
```

**Step 4: Load balancing**
- **Anycast** at the edge → users hit nearest data center
- **L4 LB** in each data center → spreads SSE connections across edge servers
- **Least connections** algorithm (since SSE = long-lived)
- Connection limit per server: ~50K (so 10M ÷ 50K = 200 servers globally)

**Step 5: Rate limiting**
- Per-IP token bucket (50 connection attempts/min) at the edge
- WAF for DDoS protection

**Step 6: Why this works**
- SSE = lightweight, browser-native, auto-reconnect
- Anycast = automatic geo-routing
- Kafka = decouples ingest from delivery, allows replay if needed

---

## Scenario 2: Designing a Distributed Cache (Memcached-like)

**The question:** "Design a sharded distributed cache for 1 PB of data across 100 nodes."

### Walk-through Answer

**Step 1: Distribution strategy**

Naive `hash(key) % 100`:
- Problem: adding/removing one node remaps ~99% of keys
- Cache stampede on backend DB
- **Unacceptable**

**Use consistent hashing:**
- Each node placed on a hash ring with ~150 vnodes
- Adding a node moves only ~1% of keys
- Removing a node moves only ~1% of keys

**Step 2: Replication for availability**

Each key replicated to next N nodes clockwise on the ring (N=3 for high availability).
- Reads can hit any replica (latency optimization)
- Writes must propagate to all (or use quorum)

**Step 3: Hot key problem**

What if a celebrity's profile is the hot key?
- All requests slam one node (and its 2 replicas)
- Solution: **client-side caching** of frequently-accessed keys
- Or: **request coalescing** at the cache layer (only one DB lookup for concurrent misses)
- Or: **bounded loads consistent hashing** (cap each node's load)

**Step 4: Client library**

Each client maintains a copy of the ring. When servers change, ring is updated (via gossip or central config).

**Step 5: Eviction**

LRU per node when memory fills.

**Why senior engineers care:** Knowing the failure modes (hot keys, ring rebalancing during deploys, replication lag) shows you've thought beyond the textbook.

---

## Scenario 3: Designing a Webhook Delivery System

**The question:** "Design a service like Stripe's webhook delivery — guaranteed delivery to millions of subscribers."

### Walk-through Answer

**Requirements:**
- At-least-once delivery
- Retries with exponential backoff (up to 3 days)
- Subscriber endpoint may be slow / down
- Don't let one slow subscriber affect others
- Order preservation per subscriber (best-effort)

**Architecture:**

```
   Event Source ──┐
                  ▼
            ┌──────────┐
            │  Kafka   │ (durable event log, partition by subscriber_id)
            └────┬─────┘
                 │
      ┌──────────┼──────────┐
      ▼          ▼          ▼
   ┌─────┐   ┌─────┐    ┌─────┐
   │Worker│  │Worker│   │Worker│
   │Pool 1│  │Pool 2│   │Pool 3│
   └──┬──┘   └──┬──┘    └──┬──┘
      │          │          │
      ▼          ▼          ▼
   POST to subscriber URLs
   ↓
   On 5xx or timeout → exponential backoff queue
```

**Key design choices:**

1. **Partition by subscriber_id** in Kafka — preserves order per subscriber, lets one slow subscriber not block others
2. **Worker isolation** — slow subscribers get isolated to their own worker pool
3. **Retry queue with backoff schedule:**
   - Attempt 1 → 1 min later
   - Attempt 2 → 5 min later
   - Attempt 3 → 30 min
   - Attempt 4 → 2 hours
   - ... up to 3 days, then dead-letter queue
4. **Idempotency:** Each event has a unique ID; subscriber dedupes
5. **HMAC signatures** for authenticity
6. **Per-subscriber rate limiting** (token bucket) — don't overwhelm subscribers

**Failure modes to discuss:**
- What if subscriber returns 200 but actually failed? → That's their bug; we did our part
- What if Kafka loses an event? → Use Kafka with replication factor 3, acks=all
- What if our delivery worker crashes mid-send? → Kafka offset only commits after successful POST

---

## Scenario 4: Rate Limiting at API Gateway (Multi-tenant SaaS)

**The question:** "Your SaaS has 100K customers, each with different rate limit tiers (Free: 100 RPM, Pro: 1000 RPM, Enterprise: 10000 RPM). Design the rate limiter."

### Walk-through Answer

**Step 1: Algorithm choice**

Token bucket is ideal:
- Allows bursts (Free user can briefly burst to 100 then sustain ~1.6/sec)
- Per-customer limits easy to configure (capacity + refill rate)

**Step 2: Where to enforce**

- API Gateway level → centralized, easier to update
- Distributed across N gateway nodes → need shared state

**Step 3: Storage strategy**

**Option A: Redis (centralized)**
```
Key: rl:{customer_id}
Value: { tokens: 95, last_refill: 1714850400 }
TTL: window size
```

Use **Redis Lua script** for atomicity (read tokens, refill, decrement, write — all in one round trip):

```lua
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1]) or capacity
local last_refill = tonumber(bucket[2]) or now

local elapsed = now - last_refill
tokens = math.min(capacity, tokens + elapsed * refill_rate)

if tokens >= 1 then
  tokens = tokens - 1
  redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
  redis.call('EXPIRE', key, 3600)
  return 1
else
  redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
  return 0
end
```

**Option B: Local + sync (eventual consistency)**

Each gateway node tracks locally, syncs to Redis every 100ms.
- Pro: Lower Redis load, lower latency
- Con: Possible 5-10% overshoot during sync delays
- Acceptable for non-billing-critical limits

**Step 4: Multi-tier limits**

Limits are **multi-dimensional**:
- Per-customer: 1000 RPM (tier-based)
- Per-customer per-endpoint: 100 RPM on `/expensive-endpoint`
- Per-IP: 1000 RPM (anti-abuse)
- Global: 1M RPS (capacity protection)

**All four checked in order**, fail at first violation.

**Step 5: Response design**

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset-Sec: 35
X-RateLimit-Tier: pro
Retry-After: 35
```

---

## Scenario 5: Choosing Between Protocols for a Microservice

**The question:** "You're building a multi-agent AI system with 8 services (orchestrator, retriever, analyzer, etc.). Each call has small JSON payloads but high frequency. Latency budget is tight. What protocol?"

### Walk-through Answer

**Analysis:**
- Internal traffic only (not browser)
- High frequency = parsing overhead matters
- Small payloads = JSON's verbosity = bandwidth waste
- Multi-language? Probably Python + maybe Go workers
- Need streaming for partial results (LLM tokens)?

**Verdict: gRPC**

**Why:**
- HTTP/2 multiplexing → no head-of-line blocking even with many calls
- Protobuf → 2-5x smaller, 10x faster parsing than JSON
- Strong typing → IDE autocomplete, compile-time checks
- Streaming primitives → built-in support for LLM token streams (`stream Response`)
- Polyglot support → Python, Go, JS clients from one .proto

**Tradeoffs accepted:**
- Harder to debug than REST (no curl)
- Need protobuf compiler in CI/CD
- gRPC-Web shim if you ever need browser access

**This maps to your real LangGraph multi-agent setup — gRPC between Python LangGraph nodes is a perfect fit.**

---

## Scenario 6: A/B Testing via Load Balancer

**The question:** "Roll out a new feature to 5% of users gradually, with ability to roll back instantly."

### Walk-through Answer

**Solution: L7 LB with header-based routing**

```
                     ┌──────────────┐
   Client ─────────► │   L7 LB     │
                     │ (NGINX/Envoy)│
                     └──────┬───────┘
                            │
                   ┌────────┴────────┐
                   │ Routing logic:  │
                   │ if cookie.exp_v=B → backend_v2
                   │ else if hash(user_id) % 100 < 5 → backend_v2
                   │ else → backend_v1
                   └────────┬────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
            ┌──────────┐        ┌──────────┐
            │ Backend  │        │ Backend  │
            │   v1     │        │   v2     │
            │  (95%)   │        │  (5%)    │
            └──────────┘        └──────────┘
```

**Why L7 is essential:**
- Need to inspect headers/cookies (L4 can't)
- Hash on stable user_id → same user always sees same version (consistent UX)
- Rollback = change config → instant

**Bonus:** Add metrics tagging by version → easy to compare error rates.

---

# Part 4: Cheat Sheet

## Decision Trees

### "Which load balancing algorithm?"

```
Are servers identical (same CPU/RAM)?
├── Yes
│   ├── Are requests uniform cost?
│   │   ├── Yes → Round Robin
│   │   └── No → Least Connections
│   └── Need stickiness?
│       └── IP Hash or Cookie
└── No
    └── Weighted Round Robin / Weighted Least Connections
```

### "Which API protocol?"

```
Is it server-to-server internal?
├── Yes
│   ├── Need streaming or high RPS?
│   │   └── gRPC
│   └── Just CRUD?
│       └── REST is fine
└── No (client-facing)
    ├── Browser-based real-time?
    │   ├── Bidirectional? → WebSockets
    │   └── Server → client only? → SSE
    ├── Mobile/web app?
    │   ├── Need flexible queries? → GraphQL
    │   └── Standard CRUD? → REST
    └── External service event notifications? → Webhooks
```

### "Which rate limit algorithm?"

```
Need bursts allowed?
├── Yes → Token Bucket
└── No
    ├── Strict constant rate? → Leaky Bucket
    ├── Approximate is fine? → Sliding Window Counter
    └── Need exact accounting? → Sliding Window Log
```

## Quick Reference Tables

### Load Balancing Algorithms

| Algorithm | Time Complexity | Memory | Best For |
|-----------|----------------|--------|----------|
| Round Robin | O(1) | O(1) | Stateless, uniform |
| Weighted RR | O(1) | O(N) | Heterogeneous fleet |
| Least Connections | O(N) or O(log N) | O(N) | Long-lived conns |
| IP Hash | O(1) | O(1) | Cheap stickiness |
| Consistent Hash | O(log N) | O(N×V) | Caches, sharding |
| Power of 2 Choices | O(1) | O(N) | General purpose |

### Rate Limiting Algorithms

| Algorithm | Memory/User | Accuracy | Bursts |
|-----------|-------------|----------|--------|
| Fixed Window | O(1) | Poor (boundary) | No control |
| Sliding Window Log | O(N requests) | Perfect | No control |
| Sliding Window Counter | O(1) | Good | Smooth |
| Token Bucket | O(1) | Good | Allowed |
| Leaky Bucket | O(1) | Good | Smoothed out |

### Protocol Comparison

| Protocol | Direction | Connection | Best For | Overhead |
|----------|-----------|------------|----------|----------|
| REST | Req/Resp | Per-request | Public APIs | Medium |
| GraphQL | Req/Resp | Per-request | Flexible queries | Medium |
| gRPC | Req/Resp + Streams | Multiplexed | Internal services | Low |
| WebSocket | Bidirectional | Persistent | Real-time chat | Very low |
| SSE | Server → Client | Persistent | Live feeds | Low |
| Webhooks | Server → Subscriber | Per-event | Async events | Per-call |

## Common Interview Trap Questions

**Q: "REST is stateless — what does that really mean for load balancing?"**
A: Each request contains all info to be processed. Any LB algorithm works because no server has session state. Contrast with WebSockets (stateful, requires sticky LB).

**Q: "Why doesn't TCP-level load balancing work for HTTP/2 microservices well?"**
A: HTTP/2 multiplexes many requests over one connection. L4 LB pins the whole connection to one backend, defeating multiplexing's load distribution. L7 LB can spread individual streams.

**Q: "Why do we say consistent hashing minimizes reshuffling?"**
A: When N servers become N+1, naive `hash % N` remaps ~(N/(N+1)) of keys (most). Consistent hashing remaps only ~1/(N+1) (the new server's share).

**Q: "Token bucket allows bursts. Isn't that bad?"**
A: Bursts are normal user behavior (page load fires 50 requests at once). Penalizing bursts hurts UX. Token bucket = friendly to humans, strict on long-term abuse.

**Q: "Why are webhooks fundamentally at-least-once, not exactly-once?"**
A: Network can fail between subscriber processing and ack. Provider can't tell if subscriber processed but ack was lost — must retry. Subscribers must dedupe.

---

## Final Mental Models to Internalize

1. **Load balancing is about allocation under uncertainty.** You don't know future load, server health, or request cost. Algorithms are heuristics that bias toward fairness, locality, or adaptability.

2. **Consistent hashing is a substrate, not just an algorithm.** Caches, queues, sharded DBs, CDNs — all use it. Master it once, recognize it everywhere.

3. **Protocol choice = constraint trade-off.** Each protocol optimizes one axis (latency, simplicity, type safety, real-time) at the cost of others. There's no "best" — only "best for this constraint set."

4. **Rate limiting is product policy, not just engineering.** Choosing limits affects user behavior. Burst tolerance is a UX decision. Per-tier limits are pricing strategy.

5. **The L4 vs L7 split is fundamental.** L4 = fast, dumb, protocol-agnostic. L7 = smart, slower, HTTP-aware. Most modern stacks use both: L4 at edge for raw throughput, L7 at gateway for routing intelligence.

---

## Where to Go Deeper

- **"Designing Data-Intensive Applications"** by Martin Kleppmann — chapters on partitioning and consistency
- **"Site Reliability Engineering"** (Google book) — load balancing chapters are gold
- **High Scalability blog** — real architectures from Twitter, Netflix, Discord
- **gRPC documentation** — the "Concepts" section is excellent
- **Cloudflare blog** — frequent deep-dives on rate limiting, anycast, edge LB

---

*Good luck with your interview, Abhishek. Your background with LangGraph multi-agent systems and distributed cricket data processing already gives you a strong foundation here — these aren't abstract concepts to you, they're the patterns you've been building with.* 🚀
