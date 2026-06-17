# Learning

A personal knowledge base covering AI engineering, ML internals, and system design interview preparation.

---

## Structure

```
learning/
├── ai_engineering/          # AI/ML study notes, talk materials, and a live demo
└── interview_prep/          # System design and database interview prep
```

---

## AI Engineering

### Talk: "AI Engineering — A Practical Introduction"

A 21-slide talk aimed at software engineers who are new to AI. Covers the full stack from LLM basics to autonomous agents.

| File | Description |
|------|-------------|
| [AI_Engineering_Intro.pptx](ai_engineering/AI_Engineering_Intro.pptx) | Slide deck |
| [AI_Engineering_Intro_Notes.md](ai_engineering/AI_Engineering_Intro_Notes.md) | Speaker notes and reference guide for all 21 slides |

### Demo: Release Notes Generator

A live Streamlit app that runs a ReAct agent against a GitLab repository and generates release notes in real time. Designed for mid-talk before/after demos using sidebar toggles (Tools, Chain of Thought, Memory).

See [ai_engineering/demo/README.md](ai_engineering/demo/README.md) for setup and usage.

### ML Internals Notes

| File | Description |
|------|-------------|
| [transformer-internals.md](ai_engineering/transformer-internals.md) | Feed-forward networks, attention, multi-head attention, and residual connections — built from first principles |
| [backprop_ninja_notes.md](ai_engineering/backprop_ninja_notes.md) | Full forward pass of a 2-layer MLP with BatchNorm, following Karpathy's Zero to Hero series |
| [backprop-ninja-variables.md](ai_engineering/backprop-ninja-variables.md) | Variable reference for the Makemore Part 4 backprop exercise |
| [finetuning_lora_qlora_guide.md](ai_engineering/finetuning_lora_qlora_guide.md) | LoRA and QLoRA from first principles — math, hyperparameters, adapter merging, and a Confluence fine-tuning pipeline |
| [ml-research-engineer-path.md](ai_engineering/ml-research-engineer-path.md) | Personal 24–30 month training path from senior SWE to ML research engineer (5–8 hrs/week, <$200 budget) |

---

## Interview Prep

### System Design

| File | Description |
|------|-------------|
| [system-design.md](interview_prep/System%20Design/system-design.md) | Index and quick cheatsheet |
| [01-foundations.md](interview_prep/System%20Design/01-foundations.md) | DB selection, distributed failure modes, latency, 9 DB principles, 10 interview mistakes |
| [02-lb-and-apis.md](interview_prep/System%20Design/02-lb-and-apis.md) | Load balancing (L4/L7, algorithms), API protocols (REST/gRPC/WebSocket/SSE), rate limiting |
| [03-fraud-detection.md](interview_prep/System%20Design/03-fraud-detection.md) | Wash trading, GPS spoofing, bot detection — sliding window and fingerprinting patterns |
| [04-queues-and-qa.md](interview_prep/System%20Design/04-queues-and-qa.md) | Cache, locking, dedup Q&As + message queue scenarios (DLQs, fan-out, ordering) |
| [whatsapp_system_design.md](interview_prep/System%20Design/whatsapp_system_design.md) | WhatsApp — XMPP stack, delivery states, queue flush sequence |
| [realtime_editor_system_design.md](interview_prep/System%20Design/realtime_editor_system_design.md) | Real-time collaborative text editor |
| [ticket-queue-system-design.md](interview_prep/System%20Design/ticket-queue-system-design.md) | High-concurrency ticket queue (inventory race conditions, fairness) |
| [swiggy-ipl-system-design.md](interview_prep/System%20Design/swiggy-ipl-system-design.md) | Swiggy IPL traffic spike — Kafka, CDN, Redis, WebSocket at scale |

### Databases

| File | Description |
|------|-------------|
| [database-systems-interview-prep.md](interview_prep/Database/database-systems-interview-prep.md) | Indexing, sharding, replication, CAP theorem, NoSQL trade-offs — scenario-based with interview-ready answers |
