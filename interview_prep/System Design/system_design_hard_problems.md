# Hard system design problems — behavioral anomaly detection pattern

A collection of 6 senior-level system design problems that share a common architectural pattern:
**sliding window tracking + behavioral fingerprinting + multi-tier thresholds + real-time circuit breakers.**

---

## Problem 1 — API token trust classification (the original)

### The problem

Your platform issues API tokens to 400,000 developers. A stolen token just ran $140,000 worth of inference overnight. Your logs show a valid token. Your auth system sees nothing wrong.

The token could belong to a developer, a CI pipeline, or an autonomous AI agent. Your gateway has no way to tell which one it is talking to and applies the same trust level to all three.

How do you redesign your token system to distinguish between a human, a pipeline, and an AI agent — without breaking 400,000 existing integrations?

### Core insight

You don't need to know *who* is calling. You need to know *how* they are calling. Every caller type leaves a distinct behavioral fingerprint in the request stream. The fix is not better auth — it is behavioral classification baked into the request path, combined with per-class spend limits and a real-time anomaly engine.

### The three caller fingerprints

| Signal | Human developer | CI pipeline | Autonomous agent |
|---|---|---|---|
| Request cadence | Irregular, bursty | Metronomic, cron-like | High RPM, no natural gaps |
| Prompt entropy | High — exploratory | Low — fixed templates | High — self-generated, looping |
| Time of day | Daytime, work hours | Any time, scheduled | 24/7, no sleep pattern |
| Model + params | Varies | Fixed at pipeline creation | Varies dynamically |
| Cost per request | Low to medium | Predictable, consistent | Can spike unpredictably |

### The token scope system

At issuance, every token is stamped with a client type and gets limits appropriate to that type:

- **Interactive scope** (human) — $50/day soft cap, human-in-the-loop alert on anomaly, session-bound tokens
- **Pipeline scope** (CI) — $500/day hard cap, allowlisted model + parameter set, signed CI assertions required
- **Agent scope** (autonomous) — $X/hour velocity cap, hard circuit breaker, immediate freeze + owner alert on breach

### Redis data model

A sorted set per token, per metric. Score is the Unix timestamp of each request. Member encodes the request ID and cost.

```
spend:tok_{token_id}:cost      → sorted set, sliding window of costs
spend:user_{user_id}:cost      → sorted set, rollup across all user tokens
spend:org_{org_id}:cost        → sorted set, rollup across all org tokens

token:{token_id}:meta          → hash: client_type, user_id, org_id, limits, frozen flag
user:{user_id}:meta            → hash: user profile, per-user limits
org:{org_id}:meta              → hash: org profile, per-org limits
```

**Why timestamp as score?** Redis sorted sets are ordered by score. A range query (`ZRANGEBYSCORE`) between `now - 15 minutes` and `now` jumps directly to the right slice — it does not scan the whole set. This makes sliding window queries sub-millisecond at scale.

**Example — the attack:**

Normal state of `spend:tok_agent_x:cost`:
```
11:00 PM  →  req_001 : $0.04
11:08 PM  →  req_002 : $0.03
11:21 PM  →  req_003 : $0.05
11:47 PM  →  req_004 : $0.04
```

After attack starts at midnight:
```
12:00:01 AM  →  req_005 : $0.89
12:00:03 AM  →  req_006 : $0.91
12:00:05 AM  →  req_007 : $0.88
... 200 more entries in 10 minutes ...
12:10:44 AM  →  req_210 : $0.87
```

At 12:05 AM your system queries the last 15 minutes, gets $89, compares to the $50/hour agent limit — token is frozen. Total damage: $89 instead of $140,000.

### The three checks on every request

1. **Count** — how many requests in the last 15 minutes. Catches volumetric attacks even if each request is cheap.
2. **Sum** — total cost in the window. Primary circuit breaker.
3. **Z-score** — is this sum unusual relative to this token's own 30-day history. Catches attacks that stay under the hard limit but are still wildly abnormal for that specific token.

### The edge case — behavioral mimicry

A sophisticated attacker who steals a pipeline token could replay requests at the exact same cadence and template structure, evading behavioral detection entirely.

**Defense 1 — Signed CI assertions.** Pipeline tokens require a signed header proving the request originates from a real CI run (a JWT signed by GitHub Actions, GitLab, etc.). An attacker with the stolen token cannot produce this header without access to the CI infrastructure.

**Defense 2 — Parameter allowlisting.** Pipeline tokens are locked at issuance to a specific model and max token count. Even a perfect behavioral impersonation cannot retarget the token to a larger model or run batch jobs — the token is structurally incapable of it.

### Zero-disruption migration path for 400K existing tokens

- **Phase 1 — Observe (2 weeks):** Shadow-classify all existing tokens, no enforcement. Build ground truth on behavioral fingerprints.
- **Phase 2 — Enforce:** New scoped tokens available and strictly enforced. Legacy tokens get behavioral classification applied with limits set 10x higher than normal — catches egregious cases without breaking anything.
- **Phase 3 — Deprecate (6-month window):** Sunset legacy scope. Provide a `/token/classify` endpoint so developers can see how their token looks to the system.

---

## Problem 2 — Financial trading platform: detecting wash trading across accounts

### The problem

Your stock trading platform processes 2 million trades per day. Your compliance team flags that certain accounts may be engaged in wash trading — buying and selling the same asset between accounts they control to artificially inflate volume and manipulate prices.

Each individual trade looks completely legitimate. No single account is doing anything obviously wrong. Your existing rules-based system catches nothing because it looks at accounts in isolation.

How do you detect coordinated wash trading across accounts that appear unrelated, in real time, without flagging legitimate high-frequency trading?

### Why this is hard

Wash traders deliberately use multiple accounts that appear unrelated — different names, different banks, different IPs. The signal is not in any single account's behavior. It is in the *relationship* between accounts — who trades with whom, at what price, at what timing, and whether the net position of the group stays flat.

A legitimate market maker also does high-frequency round trips. You cannot use trade frequency alone as a signal.

### Core insight

The same sliding window + behavioral fingerprinting approach, applied at the **graph level** rather than the account level. You track not just what each account does, but who each account trades *with*, and whether groups of accounts are net-flat on a position (the defining signature of wash trading).

### The behavioral fingerprints

**Wash trading signature:**
- Account A buys 1000 shares of X from Account B at 10:00:01 AM
- Account B buys 1000 shares of X from Account A at 10:00:04 AM
- Both accounts end the window with the same position they started — net flat
- The trades inflated the reported volume by 2000 shares

**Legitimate HFT signature:**
- High frequency, but net position changes over time
- Counterparties are diverse — not the same small cluster repeatedly
- Price is at or near the market bid/ask, not at artificial prices

### Redis data model

```
trades:{account_id}:counterparties    → sorted set: score=timestamp, member=counterparty_id
trades:{account_id}:{asset}:volume    → sorted set: score=timestamp, member="trade_id:volume:direction"
graph:edge:{account_a}:{account_b}    → hash: trade count, total volume, last seen timestamp
cluster:{cluster_id}:accounts         → set: all account IDs in this suspected cluster
```

**The key addition — a graph edge key.** Every time Account A trades with Account B, you increment the edge between them. Over a sliding 24-hour window, if the same pair of accounts repeatedly appear as counterparties to each other, that edge gets heavy. A heavy edge is a red flag.

**The net-flat check.** For every account in a suspected cluster, sum the signed volume over the window (buys are positive, sells are negative). If the cluster's aggregate net position is near zero despite high gross volume — that is wash trading. Legitimate trading leaves net positions.

### How the sliding window works here

On every trade between Account A and Account B for asset X:

1. Add to `trades:account_a:counterparties` with score=timestamp, member=account_b
2. Add to `trades:account_b:counterparties` with score=timestamp, member=account_a
3. Query last 60 minutes of counterparties for account_a — is account_b appearing more than N times?
4. If yes, pull both accounts' signed volume for asset X over the window
5. If gross volume is high but net volume is near zero — flag for compliance review

### The edge case

Two legitimate market makers who happen to frequently trade the same asset against each other will generate heavy edges. The distinguishing signal is **price variance**. Wash traders often trade at prices slightly away from market to avoid exchange matching engines — their price variance within the cluster is low. Legitimate market makers trade at market prices with normal variance.

---

## Problem 3 — Ride sharing: detecting GPS spoofing by drivers

### The problem

Your ride-sharing platform pays drivers based on distance driven. You are losing $8 million per month to GPS spoofing — drivers using apps that fake their location to inflate trip distances and collect higher fares.

Your current system trusts the GPS coordinates sent by the driver app completely. A spoofed trip looks identical to a legitimate one in your database. You cannot tell them apart after the fact.

How do you detect GPS spoofing in real time during an active trip, without punishing legitimate drivers who happen to be in areas with poor GPS signal?

### Why this is hard

GPS signal quality varies legitimately — tunnels, urban canyons, bad weather. A driver in downtown Mumbai will have noisier GPS than one on a highway. You cannot simply flag noisy GPS as fraud.

GPS spoofing tools are also increasingly sophisticated — they generate realistic-looking coordinate streams with appropriate noise and smooth trajectories. Simple sanity checks (is the speed physically possible?) no longer work.

### Core insight

A real driver moving through the physical world leaves correlated signals across multiple independent sensors. A GPS spoofer only controls one of them. You cross-correlate GPS against signals the spoofer cannot easily fake — accelerometer, gyroscope, network cell tower triangulation, and map-matching against known road geometry.

### The behavioral fingerprints

**Legitimate driver:**
- GPS coordinates match known road segments (map-matching score high)
- Accelerometer shows vibration consistent with road texture and speed
- Cell tower handoffs happen at the right times for the route being claimed
- Speed changes are smooth and physically plausible given road type
- GPS signal quality degrades predictably near tall buildings, not randomly

**Spoofed trip:**
- GPS coordinates may claim to be on roads but with suspiciously perfect smoothness — no real driver takes a perfectly straight line through a curve
- Accelerometer shows the phone is stationary or moving in a pattern inconsistent with the claimed route
- Cell tower handoffs either do not happen (phone is sitting still) or happen in a pattern inconsistent with the claimed geography
- GPS accuracy reported by the device is suspiciously high — real GPS in urban areas fluctuates

### Redis data model

```
trip:{trip_id}:gps_points          → sorted set: score=timestamp, member="lat:lng:accuracy:speed"
trip:{trip_id}:accelerometer       → sorted set: score=timestamp, member="x:y:z"
trip:{trip_id}:cell_towers         → sorted set: score=timestamp, member="tower_id:signal_strength"
driver:{driver_id}:anomaly_score   → sorted set: score=timestamp, member="trip_id:score"
driver:{driver_id}:meta            → hash: historical legit trip count, fraud flags, account status
```

### The sliding window check

Every 30 seconds during an active trip:

1. Pull last 30 seconds of GPS points from the sorted set
2. Pull last 30 seconds of accelerometer readings
3. Compute map-matching score — do the GPS points follow a plausible road geometry?
4. Compute motion consistency score — does accelerometer variance match the speed and road type claimed by GPS?
5. Compute cell tower consistency score — are tower handoffs happening as expected for this route?
6. Combine into a composite fraud score for this 30-second window
7. Accumulate fraud scores into `driver:{driver_id}:anomaly_score`

If a driver accumulates high fraud scores across multiple windows in a single trip — trip gets flagged, fare held pending review.

### The edge case — poor GPS areas

A legitimate driver in a tunnel will fail the map-matching check and the GPS accuracy check. The defense is **contextual thresholds**. You maintain a map of known poor-signal zones (tunnels, underground car parks, dense urban canyons). When a trip enters a known poor-signal zone, you relax the GPS-based checks and weight the non-GPS signals more heavily. The accelerometer and cell tower checks still work underground — GPS spoofing does not help a fraudster there.

---

## Problem 4 — Content platform: detecting coordinated inauthentic behavior at scale

### The problem

Your social platform has 50 million users. Your data science team has identified that trending topics are being manipulated — a small number of coordinated accounts are artificially amplifying content to make fringe ideas appear mainstream.

Each individual account looks legitimate — real profile photos, posting history going back years, normal follower counts. Your existing bot detection catches only the obvious cases. The sophisticated networks evade it completely.

How do you detect coordinated inauthentic behavior among accounts that individually appear genuine, at a scale of 50 million users and hundreds of millions of daily actions?

### Why this is hard

The operators of these networks have studied your bot detection and deliberately designed accounts to pass it. Accounts have aged for months or years before activation. They post genuine-looking content between coordinated campaigns. They follow each other in patterns that mimic organic social graphs.

You cannot look at any single account and call it inauthentic. The signal only exists at the **network level** — in the timing, coordination, and content similarity across a group.

### Core insight

Inauthentic coordination leaves a timing signature that organic behavior cannot replicate. When 500 real people all decide independently to share the same article, they do so over hours or days with a natural distribution. When 500 coordinated accounts share the same article, they do so within a narrow time window — often within seconds or minutes of each other — because they are responding to the same external trigger (a signal from whoever operates the network).

### The behavioral fingerprints

**Coordinated network:**
- Many accounts perform the same action (like, share, follow) within a narrow time window
- Content similarity is high across the group — same phrases, same hashtags, often copy-pasted
- The accounts were created in batches (similar creation timestamps even if spread over time)
- Interaction graph is suspiciously dense — everyone in the network follows everyone else

**Organic behavior:**
- Actions on the same content spread over hours or days with a natural long-tail distribution
- Content variation is high — people reword, add commentary, express individual reactions
- Follower graphs have the irregular, clustered structure of real social networks

### Redis data model

```
action:{content_id}:shares         → sorted set: score=timestamp, member=account_id
action:{content_id}:likes          → sorted set: score=timestamp, member=account_id
similarity:{account_a}:{account_b} → hash: shared actions count, content overlap score, last computed
cluster:{cluster_id}:members       → set: account IDs in this suspected coordination cluster
account:{account_id}:meta          → hash: creation date, anomaly score, cluster membership
```

### The sliding window check

On every share or like action:

1. Add account_id to `action:{content_id}:shares` with score=current timestamp
2. Query how many accounts performed the same action in the last 60 seconds
3. If count exceeds threshold — pull those account IDs and compare their action histories
4. Compute pairwise content similarity scores across the group
5. If timing is tight AND content similarity is high — flag as potential coordinated cluster
6. Add suspected accounts to a cluster and increase their anomaly scores

The key metric is **burst coefficient** — the ratio of actions in the last 60 seconds to actions in the last 24 hours. Organic content gets shared in a smooth curve. Coordinated amplification produces a sharp spike at the moment the network activates.

### The edge case — breaking news

A genuine news event causes thousands of real people to share the same article simultaneously. This looks exactly like coordinated behavior in the timing dimension.

The distinguishing signal is **content diversity**. Real people sharing breaking news add different commentary, use different hashtags, and engage with the content in different ways. Coordinated accounts copy-paste or minimally vary the same text. You compute a content diversity score alongside the timing burst — high burst + low diversity = coordination. High burst + high diversity = organic viral spread.

---

## Problem 5 — Cloud storage: detecting data exfiltration before it completes

### The problem

Your enterprise cloud storage platform holds sensitive documents for 200,000 companies. Your security team has identified a pattern: when an employee is planning to leave a company and take data with them, they typically exfiltrate over days or weeks before their last day.

Your current system logs every file access. You have petabytes of logs. But you only discover the exfiltration after the employee has left and the company notices files are missing — by then the data is gone.

How do you detect data exfiltration in progress, before it completes, without generating so many false positives that your security team cannot act on alerts?

### Why this is hard

Employees legitimately access large volumes of files — for projects, for reviews, for end-of-quarter reporting. An employee downloading 500 files before a business trip looks identical to an employee exfiltrating 500 files before their last day.

You also cannot alert on every anomaly. If you send 10,000 alerts per day, your security team will ignore all of them. You need high precision — alerts that are almost always real — even if that means accepting some false negatives.

### Core insight

Exfiltration has a behavioral arc that spans days or weeks. Individual days look explainable. The arc across time does not. A legitimate heavy-usage day is an outlier in an otherwise normal pattern. An exfiltration arc is a sustained, directional shift — access volume climbing steadily, breadth of files accessed expanding beyond normal work patterns, new file types appearing (HR files, financial files, IP files that the employee normally never touches).

### The behavioral fingerprints

**Normal employee:**
- File access volume varies day to day but stays within a personal baseline range
- File types accessed are consistent with job function — an engineer accesses code repos, not HR files
- Access patterns follow work hours and correlate with calendar events (big project = high access)
- Downloads are occasional and targeted — specific files for specific purposes

**Exfiltration arc:**
- Access volume shows a sustained upward trend over days or weeks, not a single spike
- File type breadth expands — employee starts accessing categories they have never touched before
- Download-to-view ratio increases — they are not just reading, they are taking
- Access happens outside normal work hours — early morning, late night, weekends
- The same files are accessed multiple times in a short window — they are making sure they have everything

### Redis data model

```
access:{user_id}:volume           → sorted set: score=timestamp, member="session_id:file_count:bytes"
access:{user_id}:file_types       → sorted set: score=timestamp, member="file_type:count"
access:{user_id}:downloads        → sorted set: score=timestamp, member="file_id:bytes"
baseline:{user_id}:stats          → hash: mean daily volume, std, normal file types, normal hours
alert:{user_id}:arc_score         → sorted set: score=timestamp, member="day:score"
```

### The multi-window check

Unlike the previous problems where a 15-minute window catches the attack, exfiltration requires **multiple window sizes running simultaneously**:

- **15-minute window** — catches bulk download tools (someone running a script to pull everything at once)
- **24-hour window** — catches daily volume anomalies
- **7-day window** — detects the sustained arc (volume trend, file type expansion, download ratio trend)
- **30-day window** — computes the baseline that all the above are compared against

On every file access event:

1. Update all three sliding windows for this user
2. Compute daily arc score — is today's behavior more anomalous than yesterday's?
3. Add arc score to `alert:{user_id}:arc_score` sorted set with today's timestamp
4. Query the 7-day arc score trend — is the score climbing day over day?
5. If score is climbing AND file type breadth is expanding AND download ratio is up — escalate alert

The alert fires not on a single bad day but on a **trend of worsening scores** — this is what makes it high precision. A single anomalous day does not trigger it. Three consecutive anomalous days with an expanding file type footprint does.

### The edge case — the employee who is legitimately offboarding a project

An employee handed a massive data migration project will generate an exfiltration-like signal for legitimate reasons.

The defense is **calendar and HR system integration**. You pull context from HR — is this employee on a PIP, have they submitted a resignation, are they interviewing elsewhere (sometimes detectable from LinkedIn profile changes)? You pull context from the project management system — is there a sanctioned data migration project assigned to this person?

High behavioral anomaly + no business context = high confidence alert. High behavioral anomaly + documented business context = low confidence, monitor only.

---

## Problem 6 — Healthcare: detecting prescription fraud across a distributed pharmacy network

### The problem

Your healthcare platform connects 15,000 pharmacies and processes 5 million prescriptions per month. Your fraud team has identified a pattern called prescription shopping — patients visiting multiple doctors to get the same controlled substance prescribed multiple times, then filling all prescriptions at different pharmacies.

Each individual prescription is valid. Each individual pharmacy fill is legitimate. No single doctor or pharmacy is doing anything wrong. The fraud only becomes visible when you look across the entire network.

How do you detect prescription shopping in real time — at the moment a pharmacist is deciding whether to fill a prescription — across 15,000 independent pharmacy systems?

### Why this is hard

Patient privacy laws severely restrict what data you can centralize and how you can use it. You cannot build a simple database of "patient X got prescription Y filled at pharmacy Z" without complex compliance controls.

You also have a hard latency constraint — the pharmacist is standing at the counter with a patient waiting. You have maybe 2 seconds to return a risk score before the delay becomes clinically disruptive.

And your false positive cost is very high — if your system incorrectly flags a cancer patient's legitimate opioid prescription as fraud, you are causing real harm to a vulnerable person.

### Core insight

You do not need to store the sensitive content of prescriptions to detect the pattern. You only need to store a **behavioral fingerprint** — the frequency, timing, and geographic spread of prescription fills for a patient, without storing which specific drugs or diagnoses are involved. The fraud pattern is visible in the metadata alone.

### The behavioral fingerprints

**Legitimate patient:**
- Fills prescriptions at one or two regular pharmacies
- Refill timing is consistent with the days supply dispensed (a 30-day supply is refilled in ~28-32 days)
- Doctor visits are from a consistent set of providers
- Geographic spread of pharmacy visits matches the patient's residential area

**Prescription shopper:**
- Fills at many different pharmacies within a short time window — often the same week
- Multiple prescriptions for the same drug class filled within a window shorter than the days supply
- Doctors are geographically spread — visiting multiple cities to get prescriptions
- Pharmacy visits span a much wider geographic area than the patient's home address

### Redis data model

```
patient:{patient_hash}:fills         → sorted set: score=timestamp, member="pharmacy_id:drug_class:days_supply"
patient:{patient_hash}:prescribers   → sorted set: score=timestamp, member="prescriber_id:location"
patient:{patient_hash}:geo_spread    → hash: pharmacy location hashes seen in last 30 days
patient:{patient_hash}:meta          → hash: risk score, flag count, last updated
drug_class:{class_id}:{patient_hash} → sorted set: score=timestamp, member="fill_id:days_supply"
```

Note: `patient_hash` is a one-way hash of the patient identifier — you can look up a specific patient but cannot reverse-engineer the identity from the key alone. This satisfies many privacy requirements while enabling the lookup.

### The sliding window check

When a pharmacist submits a prescription fill request, before confirming the fill:

1. Query `patient:{hash}:fills` for the last 30 days — how many fills, at how many distinct pharmacies?
2. Query `drug_class:{class}:{hash}` for the last 30 days — how many fills of this drug class, and do the days supplies overlap?
3. Compute geographic spread — how many distinct geographic zones have fills occurred in?
4. Query prescribers in the last 90 days — how many distinct prescribers, across how many geographic zones?
5. Combine into a risk score with weights on each dimension
6. Return risk score to the pharmacist within 2 seconds

The critical check is **days supply overlap**. If a patient filled a 30-day supply on January 1st and is trying to fill another 30-day supply on January 10th, they have 20 days of medication remaining. That is a strong fraud signal regardless of which pharmacy they are at.

### The latency requirement

5 million prescriptions per month is roughly 2 per second on average, but with heavy peaks (Monday mornings, end of month). You need sub-2-second response at peak load.

This is exactly why Redis is the right store for the hot path. All five queries above are sorted set range queries on small datasets (a patient's 30-day history is at most a few hundred entries). Total query time is under 10ms. The risk scoring computation is in-memory arithmetic. The pharmacist gets a response in under 200ms — well within the 2-second window.

The authoritative record of what was prescribed and why lives in the regulated healthcare database. Redis holds only the behavioral metadata — timing, counts, geographic hashes — which carries far lower compliance risk than clinical content.

### The edge case — the cancer patient on a complex pain management regimen

A legitimate patient with complex chronic pain may fill multiple controlled substance prescriptions from multiple specialists — an oncologist, a pain management specialist, a palliative care physician — within a short window. This looks identical to prescription shopping in the metadata.

The defense is **prescriber relationship tagging**. When a patient's care team is established (oncologist, specialist referral), those prescriber relationships are registered in the system. Fills from registered care team members are weighted differently in the risk score — they reduce the fraud signal rather than contributing to it. Unknown prescribers from distant geographic areas carry full weight.

This is the same principle as the signed CI assertions in Problem 1 — you cannot rely on behavioral signals alone for high-stakes decisions. You need a layer of verified, out-of-band context that a fraudster cannot easily fake.

---

## The common pattern across all six problems

Every problem above shares the same underlying architecture:

| Layer | What it does | Redis structure |
|---|---|---|
| Event ingestion | Record every action with a timestamp | Sorted set, score = timestamp |
| Sliding window query | Sum or count events in a time range | ZRANGEBYSCORE on the sorted set |
| Behavioral baseline | What does normal look like for this entity | Hash with mean, std, historical stats |
| Anomaly detection | Is current behavior unusual vs baseline | Z-score on the sliding window result |
| Multi-tier thresholds | Different limits for different entity types | Per-entity-type config in metadata hash |
| Circuit breaker | Freeze or flag when threshold is breached | Frozen flag in metadata hash + alert |
| Context layer | Is there a legitimate explanation | HR system, calendar, project data, care team |

The reason this pattern recurs is that fraud and abuse across all domains share a common structure — **individually legitimate actions that become illegitimate in aggregate, over time, across a network.** The sliding window makes the aggregate visible. The behavioral baseline makes the anomaly detectable. The context layer keeps the false positive rate acceptable.

The implementation details differ by domain — healthcare has privacy constraints, financial trading has microsecond latency requirements, content platforms have graph-level coordination. But the architectural skeleton is identical across all six.
