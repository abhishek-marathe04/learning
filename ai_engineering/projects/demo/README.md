# Release Notes Generator — Agent Demo

A live presentation demo for **"AI Engineering: A Practical Introduction"**. A Streamlit app that runs a ReAct agent against a GitLab repository (real or mocked) and generates release notes. The audience watches the agent think, call tools, observe results, and produce output — in real time, in a browser.

---

## What It Demonstrates

| Concept | Slide | How it shows up |
|---|---|---|
| ReAct pattern | 11 | THOUGHT → ACTION → OBSERVATION loop visible in the trace |
| Chain of Thought | 10 | Toggle on to see step-by-step reasoning before each action |
| Memory | 15 | Toggle on to inject past release context into the agent's prompt |

The sidebar toggles are designed for **before/after comparisons mid-talk** — flip a toggle, hit Run, and the audience sees the difference immediately.

---

## Project Structure

```
demo/
├── app.py                   # Streamlit entry point
├── config.py                # All config read from env vars — one place, no magic
├── agent_memory.json        # Pre-seeded memory file (team preferences, past release ref)
├── .env.example             # Template — copy to .env and fill in your values
│
├── agent/
│   ├── loop.py              # ReAct agent loop — the core of the demo
│   ├── tools.py             # Tool definitions (OpenAI format) + dispatch registry
│   ├── llm.py               # Raw HTTP calls to LiteLLM — no SDK, plain httpx
│   ├── prompts.py           # System prompt segments and builder
│   └── memory.py            # Load/save agent_memory.json
│
├── gitlab/
│   ├── client.py            # Real GitLab REST calls (used when USE_MOCK=false)
│   └── mock.py              # Realistic mock data — same interface as client.py
│
└── ui/
    ├── sidebar.py           # Presenter controls (all the toggles)
    ├── inspector.py         # System prompt + tool definitions panel
    └── trace.py             # THOUGHT / ACTION / OBSERVATION event renderer
```

---

## Setup

### Prerequisites

- Python 3.9+
- A LiteLLM gateway URL and API key (or any OpenAI-compatible endpoint)
- GitLab token + project ID (only needed when `USE_MOCK=false`)

### Install

Create and activate a virtual environment, then install pinned dependencies from `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `requirements.txt` pins exact versions (`streamlit==1.50.0`, `httpx==0.28.1`, `python-dotenv==1.2.1`).
> Do not `pip install` packages globally — always activate the venv first.

### Configure

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
# Leave this true for the demo — no GitLab connection needed
USE_MOCK=true

# Your LiteLLM (or OpenAI-compatible) gateway
LLM_BASE_URL=https://your-litellm-gateway
LLM_API_KEY=your-team-token
MODEL_NAME=claude-sonnet-4-6

# Only needed when USE_MOCK=false
GITLAB_BASE_URL=https://gitlab.example.com
GITLAB_TOKEN=glpat-xxxxxxxxxxxx
GITLAB_PROJECT_ID=123
GITLAB_DEFAULT_BRANCH=main

# How many loop iterations before giving up
AGENT_MAX_ITERATIONS=10
```

### Run

Make sure the venv is active, then:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## The Toggles

All toggles live in the sidebar. They take effect on the next **Run Agent** click — each run starts fresh.

### USE MOCK DATA
Routes all GitLab tool calls to hardcoded mock data instead of making real API requests. **Leave this on for any presentation** — no network, no tokens, no surprises.

Mock scenario: `release/v2.2.0` vs `release/v2.1.0`, with 4 merged PRs:
- !341 — Remove deprecated `/v1/auth` endpoint *(breaking change)*
- !338 — Fix memory leak in session handler *(bug fix)*
- !335 — Add OAuth2 PKCE support *(new feature)*
- !332 — Update CI pipeline to Node 22 *(internal)*

### TOOLS ENABLED
Includes the three tool definitions in the LLM call, giving the agent access to GitLab data.

- **OFF:** The model answers from training knowledge only — expect hallucinated commits or a refusal.
- **ON:** The model calls tools to fetch real (or mock) data before writing.

The three tools:

| Tool | What it does |
|---|---|
| `get_branch_diff` | Returns commits between two branches — establishes the scope of a release |
| `get_pr_details` | Fetches title, description, labels, and breaking-change flag for a merge request |
| `get_past_release_notes` | Retrieves release notes from a past version for consistency and context |

### CHAIN OF THOUGHT
Appends a step-by-step reasoning instruction to the system prompt.

- **OFF:** The model jumps straight to output — fast but opaque.
- **ON:** The model writes its reasoning in THOUGHT cards before each action. Useful for showing the audience *why* it makes decisions, and for debugging when things go wrong.

### MEMORY
Loads `agent_memory.json` and injects its contents into the system prompt before the agent runs.

- **OFF:** The agent starts fresh with no knowledge of team conventions or past releases.
- **ON:** The agent receives pre-seeded context:
  - *team_preferences* — formatting rules (breaking changes first, emoji headers, one line per item)
  - *last_release_version* — `v2.1.0`
  - *known_issues* — a note that the `/v1/auth` removal was announced in v2.1.0

  The agent uses this to format output consistently and reference the prior release.

---

## The UI

```
┌─────────────────────────┬───────────────────────────────────────────┐
│  SIDEBAR                │  MAIN PANEL                               │
│                         │                                           │
│  Presenter Controls     │  ┌─ Inspector ──────────────────────────┐ │
│                         │  │  System Prompt (rendered)            │ │
│  [x] Use Mock Data      │  │  Tool Definitions (name + schema)    │ │
│                         │  │  Current Messages                    │ │
│  Demo Mode              │  └──────────────────────────────────────┘ │
│  [ ] Tools Enabled      │                                           │
│  [ ] Chain of Thought   │  ┌─ Agent Trace ────────────────────────┐ │
│  [ ] Memory             │  │  🤔 THOUGHT  (streams token by token) │ │
│                         │  │  ⚡ ACTION   tool name + JSON inputs  │ │
│  Task                   │  │  👁️  OBSERVATION  tool result         │ │
│  Current branch input   │  │  🤔 THOUGHT  ...                     │ │
│  Previous branch input  │  │  ✅ FINAL OUTPUT                      │ │
│  [Run Agent]            │  └──────────────────────────────────────┘ │
└─────────────────────────┴───────────────────────────────────────────┘
```

**Inspector tab** — shows what the agent knows before it starts. Walk through this with the audience before clicking Run.

**Agent Trace tab** — populates live as the agent runs. Each event type has its own colour-coded card. THOUGHT content streams in token by token.

---

## Presentation Flow

| What you say | Toggle state |
|---|---|
| "Without tools, the model guesses..." | Tools=OFF, CoT=OFF, Memory=OFF |
| "Now let's give it tools..." | Tools=ON, CoT=OFF |
| "It works, but it's a black box..." | Tools=ON, CoT=OFF |
| "Chain of thought makes reasoning visible..." | Tools=ON, CoT=ON |
| "No memory of past releases yet..." | Tools=ON, CoT=ON, Memory=OFF |
| "Now with memory loaded..." | Tools=ON, CoT=ON, Memory=ON |

Each state change: clear the trace, click **Run Agent**, watch the difference.

---

## Real GitLab (Optional)

To run against a real GitLab project, set `USE_MOCK=false` in `.env` and fill in:
- `GITLAB_BASE_URL` — your GitLab instance URL
- `GITLAB_TOKEN` — a personal access token with `read_api` scope
- `GITLAB_PROJECT_ID` — numeric project ID (found in project settings)

The `get_past_release_notes` tool will look for a `RELEASE_NOTES.md` file at the given version tag ref.

---

## Editing the Memory

`agent_memory.json` is plain JSON — edit it directly to change what the agent "remembers":

```json
{
  "team_preferences": "Always put breaking changes first. Use emoji section headers. One line per item.",
  "last_release_version": "v2.1.0",
  "known_issues": "The /v1/auth removal was announced in v2.1.0 — reference this when it appears."
}
```

Changes take effect on the next run (no restart needed).
