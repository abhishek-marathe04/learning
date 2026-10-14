# Senior Engineer Interview – Answer Key

Oct 14, 2026 · @Abhishek

## How to use this answer key

Score each section 1–4 and judge reasoning, not vocabulary. A strong lead explains trade-offs and failure modes in their own words; the answers below are what "4" sounds like.

| Score | What you hear |
| --- | --- |
| 4 | Clarifies the problem first, names trade-offs unprompted, tells a real story with numbers, adapts when you change constraints |
| 3 | Correct and practical, needs one nudge on edge cases |
| 2 | Knows the terms, struggles to explain why or what breaks |
| 1 | Buzzwords, generic answers, freezes on follow-ups |

Signals of real experience vs recitation:

- Real: specific numbers ("lag hit 2M messages"), mistakes they made, what they would do differently.
- Real: they say "it depends" and then name what it depends on.
- Recited: textbook lists in perfect order, no failures, "we" with no clear personal role.
- AI-assisted: long pauses then polished paragraphs, eyes moving off-screen, falls apart when you change one constraint.

## 1. Coding: idempotent event processor

**Question:** Process a stream of payment events. Events may arrive more than once. Process each exactly once in effect, at most 10 at a time, retry failures up to 3 times.

**Model answer, in plain terms:**

1. **Clarify first.** Good candidates ask: Where do events come from? What is the unique key? What happens if the process crashes mid-event? Is order important?
2. **Idempotency.** Every event needs a unique ID. Before processing, check a durable store (DB table with a unique constraint, or Redis with expiry). Record "processed" in the same transaction as the business effect, so a crash can't leave one without the other. An in-memory set alone is wrong: it is lost on restart.
3. **Bounded concurrency.** A worker pool of 10, or a semaphore with 10 permits. Unbounded threads or goroutines exhaust memory, DB connections and downstream services.
4. **Retries with backoff.** Retry only transient errors (timeouts, 503s), not validation failures. Use exponential backoff with jitter (e.g. 200ms, 400ms, 800ms, plus randomness) so retries don't hit the system all at once.
5. **Dead-letter path.** After 3 failures, move the event to a dead-letter queue with the error, and alert. Never drop silently.
6. **Acknowledge last.** Only acknowledge the message to the queue after processing succeeds, so a crash means redelivery, not loss.

**Twist: two instances run in parallel.** In-memory dedup and in-memory locks no longer work. Expected answers:

- Rely on the DB unique constraint: the second insert fails, and that instance skips the event.
- Or partition the stream by key (e.g. Kafka partitions) so one instance owns each payment.
- Distributed locks are possible but a weaker answer: they add failure modes like expired locks.

**Red flags:** "exactly once is guaranteed by the queue", no crash scenario, retrying every error including bad input.

## 2. Pseudocode review: the dual-write problem

```
begin transaction
  save order to DB
  publish "OrderCreated" to message queue
  update in-memory counter
commit
```

**Problem 1: DB and queue can disagree (dual write).** The queue is not part of the DB transaction. Two failure cases:

- Message published, then commit fails: consumers act on an order that doesn't exist.
- Commit succeeds, publish fails or the app crashes between them: the order exists but nobody hears about it.

**Fix: transactional outbox.** In the same DB transaction, insert the event into an `outbox` table. A separate relay reads the outbox and publishes to the queue, marking rows as sent. The DB commit is now the single source of truth. Duplicates can still happen (relay crashes after publish), so consumers must be idempotent.

**Alternative: Change Data Capture (CDC).** A tool like Debezium reads the DB's transaction log and publishes changes. No app code for publishing, but more infrastructure to run.

**Problem 2: in-memory counter.** Wrong with multiple instances (each has its own count), lost on restart, and updated even if the commit fails. Use the DB, a metrics system, or a shared store.

**Problem 3: async publish.** Many clients send asynchronously. Without checking the result or a callback, failures vanish silently.

**What a 4 sounds like:** "You can't atomically write to two systems. Make the DB the source of truth with an outbox, accept at-least-once delivery, and make consumers idempotent."

**Red flags:** proposing a distributed transaction (two-phase commit) as the default, or not seeing any problem at all.

## 3. Whiteboard design: zero-downtime migration

**Question:** Move a live order system from a monolith to event-driven services behind a load balancer. No downtime, rollback possible at every step.

**Model answer: the strangler-fig approach, in phases.**

1. **Prepare.** Define success metrics (error rate, p99 latency, order counts). Add dashboards and alerts on the monolith first, so there is a baseline to compare against.
2. **Put a routing layer in front.** The load balancer or API gateway (e.g. Nginx) routes all traffic to the monolith. Now any route can be shifted later without client changes.
3. **Start emitting events from the monolith.** Use outbox or CDC so the new service can build its own data from real events.
4. **Migrate data with expand/contract.** Expand: create the new schema alongside the old. Backfill historical data in batches. Keep it in sync via events. Verify with reconciliation jobs (row counts, checksums, sample comparisons). Contract: remove old tables only weeks after cutover.
5. **Shadow traffic (optional, strong signal).** Send copies of real requests to the new service, discard its responses, and compare them with the monolith's.
6. **Canary the cutover.** Shift reads first, then writes: 1% → 5% → 25% → 100%, by percentage or by low-risk customers. Hold at each step and watch the metrics.
7. **Rollback at every step.** A defined trigger (e.g. error rate up 0.5% or p99 doubles) routes traffic back. Keep the old path warm and in sync until the new one has run cleanly for a set period.
8. **Decommission.** Remove the old code and tables last.

**High availability points to listen for:** at least 2 LB instances (or a managed LB), services across multiple availability zones, a replicated queue (e.g. Kafka replication factor 3), DB with a standby replica and tested failover, no single point of failure.

**Follow-up: queue lag keeps growing during cutover.**

- First, check whether consumers are slow or producers suddenly faster (new service double-publishing?).
- Pause the traffic shift. Don't push forward while lag is growing.
- Scale consumers, but only up to the number of partitions; beyond that, extra consumers sit idle.
- Look for a slow dependency (DB, external API) or a poison message being retried in a loop.
- If it's not resolved quickly, roll the traffic back.

**Follow-up: 5% of requests fail with gateway errors (502/504).**

- 502 = the LB got a bad or no response from the backend; 504 = backend too slow.
- Check whether failures cluster on one backend instance: a bad node, or one failing health checks.
- Compare timeouts: LB timeout shorter than app timeout, or keep-alive mismatches causing reset connections.
- Check deploys: instances killed mid-request because connections weren't drained.
- Check resources: connection pool exhaustion, thread exhaustion, containers being OOM-killed.
- Use the LB access logs plus trace IDs to follow a failed request end to end.

**Red flags:** big-bang cutover over a weekend, no reconciliation, rollback plan = "restore from backup".

## 4. Concept checks

### Scalability: 1k → 20k requests/sec

Strong answers measure before changing anything: load test to find the first bottleneck. Usual suspects, in order:

- **Database:** too many queries per request, missing indexes, connection pool limits. Fixes: indexes, caching, read replicas, then sharding as a last resort.
- **Stateful app servers:** sessions or caches in memory prevent adding instances. Make services stateless so they scale horizontally behind the LB.
- **Synchronous calls to slow dependencies:** add timeouts, move non-urgent work to a queue.
- **Shared locks or hot keys:** one row or cache key everyone updates.

A 4 also mentions cost, autoscaling based on the right metric, and protecting downstream systems with rate limits.

### Event-driven vs direct API calls

Use events when the caller doesn't need an immediate answer, several services react to one fact ("order placed"), or you want services decoupled and resilient to each other being down. Use direct calls when the user needs the result now (e.g. "is this payment approved?").

What you give up with events:

- **Eventual consistency:** data is briefly out of sync across services.
- **Duplicates and ordering:** consumers must be idempotent; ordering is only guaranteed within a partition or key.
- **Harder debugging:** you need trace IDs carried through messages.
- **Schema evolution:** event formats must stay backward compatible.

### Load balancers and rolling deploys

- **Readiness vs liveness:** readiness = "send me traffic"; liveness = "restart me if I'm stuck". A new instance should get traffic only when ready.
- **Connection draining:** before an old instance stops, the LB stops sending new requests and lets in-flight ones finish (graceful shutdown).
- **L4 vs L7:** L4 routes by IP/port, fast and simple. L7 understands HTTP, so it can route by path or header, terminate TLS, and do canaries.
- **Sticky sessions:** a workaround for stateful apps; better to keep services stateless.

### Containers killed while the app looks healthy

The container has a memory limit (e.g. 1 GB). If the process uses more than that, the kernel kills it (OOMKilled), with no app error. Common causes:

- The runtime sizes itself to the host machine, not the container limit (older JVMs did this).
- Memory outside the main heap: threads, buffers, native libraries, caches.
- Limit set too close to normal usage, with no headroom for spikes.

A strong answer mentions checking the container's exit reason and setting runtime memory as a percentage of the container limit.

### CI/CD for 15 services

Good pipeline stages: build → unit tests → static analysis and security/dependency scan → build an immutable image tagged with the commit → deploy to test env → integration/contract tests → deploy to production with canary or blue-green → automatic checks on metrics.

Fast rollback: redeploy the previous image tag (seconds to minutes) or flip blue-green traffic back. Don't fix forward under pressure. Database changes must be backward compatible so old code still works after rollback.

At 15-service scale: shared pipeline templates, per-service pipelines so one team doesn't block others, and contract tests between services.

### Observability: "it's slow"

- **Metrics** tell you something is wrong: latency, error rate, traffic, saturation (the four golden signals).
- **Traces** tell you where: which service or call took the time.
- **Logs** tell you why: the specific error or slow query.

Across async hops, a correlation or trace ID must be passed in message headers so one request can be followed end to end. An alert worth waking up for is based on user impact (error rate or latency against an SLO), not on CPU usage alone.

## 5. AI in delivery: agents, skills, MCP

### Where has AI helped, and where has it hurt?

Strong answers are specific and balanced:

- **Real gains:** test generation, boilerplate, migrations across many files, explaining unfamiliar code, writing docs and runbooks, first drafts of designs.
- **Real problems:** confident but wrong code, subtle security issues, larger PRs that are harder to review, juniors not learning fundamentals.
- **Guardrails they set:** human review on every change, tests required, smaller PRs, rules for what data can go into which tools, team-shared prompts or skills.

A 4 gives a measured result ("test coverage went from X to Y", "migration took 2 weeks instead of 6").

### Draw an agent or MCP integration you built

What a real builder draws:

- **The loop:** the model receives a task, chooses a tool, gets the result, decides the next step, until done or a limit is hit.
- **Tools via MCP:** MCP is a standard protocol for exposing tools and data to AI apps, so one integration works across clients.
- **Skills:** packaged instructions plus scripts that teach an agent a repeatable workflow.
- **Access control:** least privilege (read-only by default), scoped credentials per tool, human approval for destructive actions, input validation against prompt injection, audit logs.
- **Limits:** max steps, timeouts, cost budgets.

### How do you know an agent is reliable enough to roll out?

- **An evaluation set:** real tasks with known good outcomes, run on every change to prompts, tools or model.
- **Metrics:** task success rate, cost and latency per task, tool error rates.
- **Failure review:** they read failed traces and categorize them, not just look at averages.
- **Staged rollout:** small pilot group, feedback, then wider.
- **Production monitoring:** traces of every run, alerts on failure spikes.

**Red flags:** "we just tested it manually and it looked good", no answer on permissions, treating the model's output as trusted.

## 5b. GenAI, MCP and prompt engineering deep-dive

The candidate's CV leans heavily on this area, so go deep here. Start with: "Pick the GenAI or MCP project on your CV you're proudest of and draw it." Then use the questions below as follow-ups.

### Whiteboard: design an MCP server for an internal system

**Question:** "Design an MCP server that lets engineers' AI assistants query Jira tickets and trigger deployments."

What a 4 covers:

- **Tool design:** a few clear, task-shaped tools (`search_tickets`, `get_ticket`, `trigger_deploy`) rather than one tool per raw API endpoint. Tool names and descriptions are written for the model, because that is how it chooses tools.
- **Read vs write:** read tools are safe; `trigger_deploy` needs human confirmation, environment restrictions (no prod without approval), and an audit log.
- **Auth:** acts as the user, not a shared super-account (OAuth or token passthrough), so permissions match what the person could do themselves.
- **Transport:** local stdio for a single developer's machine; remote HTTP for a shared, centrally run service.
- **Large outputs:** pagination, filtering, and summarizing results so one call doesn't flood the context window.
- **Errors:** clear, model-readable error messages so the agent can recover ("ticket not found, try search\_tickets").
- **Scale:** stateless server instances behind a load balancer, rate limits per user, and monitoring of tool latency and error rates.

**Red flags:** exposing every API endpoint as a tool, one shared admin credential, no confirmation on write actions.

### MCP fundamentals

**Question:** What are the main building blocks MCP exposes, and when do you use each?

- **Tools:** actions the model decides to call (search, create, deploy).
- **Resources:** data the app or user attaches as context (a file, a schema, a doc).
- **Prompts:** reusable templates the user picks (e.g. a "review this PR" workflow).

Good candidates also explain why MCP exists: build an integration once and use it from any compatible client, instead of custom glue per app.

### The model keeps calling the wrong tool. How do you fix it?

- Improve tool names and descriptions: say what it does, when to use it, and when not to.
- Reduce overlap: merge or remove similar tools. Too many tools confuses the model.
- Tighten input schemas: enums, required fields, examples.
- Add examples in the system prompt.
- Measure with an eval set of tasks and expected tool calls, so fixes are verified, not guessed.

**Red flag:** "switch to a bigger model" as the first and only answer.

### Prompt engineering in production

**Question:** How do you write, version and test prompts for a production feature?

- **Structure:** clear role and task, context separated from instructions (e.g. with tags or sections), explicit output format, a few good examples, and instructions on what to do when unsure.
- **Versioning:** prompts live in source control and go through review like code, not edited live in a console.
- **Testing:** a regression set of real inputs with expected behavior, run on every prompt or model change. Comparing old vs new outputs side by side.
- **Model upgrades:** re-run the eval set; prompts tuned for one model can behave differently on another.

**Follow-up:** "Your prompt works 90% of the time. How do you get the last 10%?" Good: collect and categorize the failures first, fix the biggest category, re-measure. Bad: keep adding "IMPORTANT!!" lines.

### Prompt injection

**Question:** Your agent reads emails or web pages and can call tools. How do you stop a malicious page from taking control?

- Treat all retrieved content as untrusted data, never as instructions.
- Least-privilege tools; the agent can't do more damage than its tools allow.
- Human confirmation for sensitive actions (sending, deleting, paying, deploying).
- Separate what the model reads from what it can do; don't let untrusted content flow straight into a write action.
- Don't send secrets or unrelated private data into the context.
- Log and monitor tool calls for unusual patterns.

A 4 admits there's no complete fix today and talks about limiting damage. **Red flag:** "we tell the model to ignore injected instructions" as the whole defense.

### RAG quality

**Question:** Users say your RAG assistant gives wrong answers. How do you debug it?

- **First split the problem:** was the right document retrieved? If not, it's a retrieval problem. If yes, it's a generation problem.
- **Retrieval fixes:** better chunking (by section, with overlap and titles), hybrid search (keyword + vector), re-ranking, metadata filters, fresher indexes.
- **Generation fixes:** instruct the model to answer only from the provided context, cite sources, and say "I don't know" when context is missing.
- **Measure:** a labeled question set with retrieval metrics (did the right chunk appear in the top results?) and answer quality checks.

### Evaluating LLM output

**Question:** How do you measure quality when there's no single right answer?

- Code checks where possible: valid JSON, required fields, correct tool called.
- LLM-as-judge with a clear rubric, checked against human ratings on a sample, because judges have biases (prefer longer answers, prefer their own style).
- Human review on a regular sample, plus user feedback signals in production.

### Cost, latency and reliability

**Question:** Your GenAI feature is too slow and too expensive. What do you do?

- Measure tokens and latency per step first.
- Trim the context: shorter prompts, better retrieval, summarize long history.
- Use prompt caching for repeated large prefixes.
- Route simple tasks to smaller, cheaper models; reserve large ones for hard tasks.
- Stream responses so users see output sooner.
- Run independent steps in parallel.

**Follow-up:** "Your model provider has an outage." Good: timeouts and retries with backoff, a fallback model or provider, graceful degradation in the UI, and alerts.

### Structured output

**Question:** The model must return JSON your code parses, and it sometimes breaks. How do you handle it?

- Use the provider's structured output or tool-calling features with a schema, rather than asking nicely in the prompt.
- Validate against the schema in code; on failure, retry once with the error message.
- Keep schemas simple and flat where possible.

### Long-running agents and context

**Question:** An agent runs for 50 steps and gets worse over time. Why, and what do you do?

- The context fills with old tool outputs, so important instructions get diluted or dropped.
- Fixes: summarize or trim old steps, store working notes outside the context, give tools compact outputs, split work across sub-agents with their own clean context, and set step limits.

### Probing their CV claims

For any GenAI project they list, ask:

- What was the success rate before and after your changes, and how did you measure it?
- What was the worst failure in production, and what did you change because of it?
- What did you decide **not** to use an LLM for, and why?

The last question is a strong filter: experienced engineers know where LLMs are the wrong tool, such as deterministic rules, calculations, and strict compliance logic.

## 6. Leadership and hands-on attitude

### Last production code you wrote yourself, and why that piece?

- **Good:** recent (weeks, not years), and chosen deliberately: the risky core piece, a prototype to unblock the team, or a tricky bug nobody else could find. They still review code and stay on call.
- **Red flag:** "I mostly do meetings now", or they take all the interesting work and leave the team the rest.

### A senior engineer disagrees with your design in front of the team

- **Good:** treats it as useful, not a threat. Acknowledges the point, asks for specifics, and either agrees to explore it or moves the deeper discussion to a follow-up. Decides based on data (a spike, a prototype), then commits as a team. May mention writing decisions down (an ADR) so the reasoning is clear.
- **Red flag:** "I'm the lead, so it's my call", or avoids conflict and lets the loudest person win.

### A go-live that went badly

- **Good:** a clear personal role, what went wrong, how they stabilized it (rollback, comms to stakeholders), and a blameless post-mortem with concrete changes afterwards (e.g. added canary releases, a runbook, better alerts).
- **Red flag:** blames others, no lessons, or claims nothing ever went wrong.

### Keeping a team motivated during a long, stressful migration

- **Good:** breaks the work into visible milestones and celebrates them, protects the team from scope creep and unrealistic dates, shares on-call load fairly, works alongside them during hard cutovers, recognizes people publicly, and watches for burnout.
- **Red flag:** "pizza and pep talks", or measuring motivation only by hours worked.

### Good connections and high spirits

Listen for how they talk about past teammates and other teams. Warm, specific credit to others and relationships beyond their own team (product, ops, security) are good signs. Constant complaints about "bad" colleagues or managers are a warning.
