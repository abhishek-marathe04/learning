# AI Engineering Demo — Specification

**Purpose:** Live presentation demo for "AI Engineering: A Practical Introduction"
**Audience:** ~250 software engineers, mostly unfamiliar with AI
**Primary teaching goals:** ReAct pattern (Slide 11) and Chain of Thought (Slide 10)
**Secondary goal:** Memory component (Slide 15)
**Date:** 25 May 2026

---

## Overview

A Streamlit web app that runs a Release Notes Generator agent against a GitLab repository (real or mocked). The audience watches the agent think, call tools, observe results, and produce output — live, in a browser, with the ReAct loop visible at every step.

The presenter controls the experience through toggles on the sidebar, enabling before/after comparisons mid-talk.

---

## Project Structure

```
demo/
├── SPEC.md                  # this file
├── app.py                   # Streamlit entry point
├── config.py                # all configuration, env vars
├── agent/
│   ├── __init__.py
│   ├── loop.py              # the agent loop (ReAct)
│   ├── tools.py             # tool definitions + dispatch
│   ├── llm.py               # raw HTTP calls to LiteLLM — no SDK
│   └── memory.py            # memory load/save
├── gitlab/
│   ├── __init__.py
│   ├── client.py            # real GitLab REST calls
│   └── mock.py              # mock data, same interface as client.py
├── ui/
│   ├── sidebar.py           # presenter controls
│   ├── inspector.py         # prompt + tool definitions panel
│   └── trace.py             # THOUGHT / ACTION / OBSERVATION stream
└── .env.example
```

---

## Configuration (`config.py`)

All config is read from environment variables (or a `.env` file). One file, no magic.

```python
# .env.example

# --- Mock mode ---
USE_MOCK=true                       # true = all GitLab calls use mock data

# --- GitLab (used only when USE_MOCK=false) ---
GITLAB_BASE_URL=https://gitlab.example.com
GITLAB_TOKEN=glpat-xxxxxxxxxxxx
GITLAB_PROJECT_ID=123
GITLAB_DEFAULT_BRANCH=main

# --- LiteLLM gateway ---
LLM_BASE_URL=https://your-litellm-gateway
LLM_API_KEY=your-team-token
MODEL_NAME=claude-sonnet-4-6     # any model your gateway supports

# --- Agent behaviour ---
AGENT_MAX_ITERATIONS=10
```

`config.py` exposes a single `Config` dataclass. No config is hardcoded anywhere else.

---

## LLM Layer (`agent/llm.py`)

**No SDK. No wrappers. Plain HTTP.**

The LiteLLM gateway exposes an OpenAI-compatible `/chat/completions` endpoint. We call it directly with `httpx` (or `requests`).

```python
# agent/llm.py — interface contract

def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    stream: bool = False,
) -> dict | Generator:
    """
    POST to LLM_BASE_URL/chat/completions
    Returns the raw JSON response (or a streaming generator of chunks).
    Caller inspects response["choices"][0]["message"].
    """
```

**Streaming is required.** The frontend must receive tokens as they arrive so the audience sees the model "thinking" in real time, not a long pause followed by a wall of text.

The function must handle:
- Non-streaming: return full response dict
- Streaming: yield each delta chunk as it arrives
- Tool calls in the response (parsed from `tool_calls` field)
- HTTP errors (surface cleanly to the UI, don't crash)

No retry logic. No fallbacks. Keep it simple.

---

## Agent Layer

### Tools (`agent/tools.py`)

Three tools, designed to make the ReAct loop visible with 3–4 iterations.

#### Tool 1 — `get_branch_diff`
```
name:        get_branch_diff
description: Returns commits that are in the current release branch but not
             in the previous release branch. Use this first to establish the
             exact scope of a release — what changed between two named branches.
input:
  current_branch:  string  # e.g. "release/v2.2.0"
  previous_branch: string  # e.g. "release/v2.1.0"
returns:
  list of { sha, message, author, date, pr_number | null }
```

GitLab API: `GET /projects/:id/repository/compare?from=<previous_branch>&to=<current_branch>`
The response `commits` array is the diff. PR numbers are extracted from commit messages (GitLab convention: "Merge branch ... into ... See merge request !341").

#### Tool 2 — `get_pr_details`
```
name:        get_pr_details
description: Retrieves full details of a merge request including title,
             description, labels, and linked issue. Use this to understand
             the intent and impact of a specific change.
input:
  pr_number: integer
returns:
  { number, title, description, labels[], author, merged_at, breaking_change: bool }
```

#### Tool 3 — `get_past_release_notes`
```
name:        get_past_release_notes
description: Retrieves release notes from a previous version. Use this to
             maintain consistent format and to reference what was shipped
             before so you don't repeat items.
input:
  version: string  # e.g. "v2.1.0"
returns:
  { version, date, content: string }
```

Tool 3 is what demonstrates memory — it gives the agent "institutional knowledge" of past releases.

### Tool Dispatch (`agent/tools.py`)

```python
TOOL_REGISTRY = {
    "get_branch_diff":       get_branch_diff,
    "get_pr_details":        get_pr_details,
    "get_past_release_notes": get_past_release_notes,
}

def execute_tool(name: str, inputs: dict) -> dict:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    return fn(**inputs)
```

Each tool function routes to either `gitlab.client` or `gitlab.mock` based on `Config.use_mock`. The tool itself doesn't know which.

### Agent Loop (`agent/loop.py`)

```python
def run(
    task: str,
    tools_enabled: bool,
    cot_enabled: bool,
    memory_enabled: bool,
    on_event: Callable[[AgentEvent], None],   # callback for UI streaming
) -> str:
    """
    Runs the ReAct loop. Calls on_event() for each THOUGHT, ACTION,
    OBSERVATION, and FINAL step so the UI can render them as they happen.
    Returns the final text response.
    """
```

**AgentEvent** is a simple dataclass:
```python
@dataclass
class AgentEvent:
    type: Literal["thought", "action", "observation", "final", "error"]
    content: str
    tool_name: str | None = None
    tool_input: dict | None = None
```

**Loop logic:**

```
1. Build system prompt (with or without CoT instruction, based on flag)
2. If memory_enabled: load memory file, inject into system prompt
3. If tools_enabled: include tool definitions in the LLM call
4. Loop (max AGENT_MAX_ITERATIONS):
   a. Call LLM (streaming)
   b. If response is text → emit THOUGHT event, stream tokens to UI
   c. If response has tool_calls → emit ACTION event for each
   d. Execute each tool → emit OBSERVATION event with result
   e. Append tool results to messages, continue loop
   f. If stop_reason == "end_turn" → emit FINAL event, break
5. Return final content
```

---

## System Prompts

### Base prompt (always included)
```
You are a software release engineer. Your job is to generate clear,
well-structured release notes from GitLab merge request data.

You have access to tools to fetch commit history and MR details.
Use them — do not guess or invent changes.

Format the release notes with these sections:
- ⚠️ Breaking Changes (if any)
- ✨ New Features
- 🐛 Bug Fixes
- 🔧 Internal Changes

Keep each item to one line. Be specific about what changed and why it matters.
```

### CoT addition (appended when `cot_enabled=True`)
```
Before writing the release notes, reason step by step:
1. What commits exist and which PRs do they belong to?
2. Which PRs are breaking changes?
3. How should I categorise the rest?
4. Is there anything from past releases I should reference?

Write your reasoning before producing the final output.
```

### Memory addition (appended when `memory_enabled=True`)
```
You have access to past release notes via the get_past_release_notes tool.
Check the previous release before writing to:
- Maintain consistent format and tone
- Reference continuations of previous work where relevant
- Avoid re-announcing features already shipped
```

---

## Mock Data (`gitlab/mock.py`)

Realistic mock data so the demo never depends on network or tokens.

### Mock branch diff
Keyed by `(current_branch, previous_branch)` tuple so the mock can serve
different scenarios if needed. The default pair used in the demo:
`("release/v2.2.0", "release/v2.1.0")`

```python
MOCK_BRANCH_DIFF = {
    ("release/v2.2.0", "release/v2.1.0"): [
        {"sha": "a1b2c3", "message": "Merge branch 'feature/remove-v1-auth' into 'release/v2.2.0' See merge request !341", "author": "alice", "date": "2026-05-12", "pr_number": 341},
        {"sha": "d4e5f6", "message": "Merge branch 'fix/session-memory-leak' into 'release/v2.2.0' See merge request !338",  "author": "bob",   "date": "2026-05-11", "pr_number": 338},
        {"sha": "g7h8i9", "message": "Merge branch 'feature/oauth2-pkce' into 'release/v2.2.0' See merge request !335",      "author": "alice", "date": "2026-05-10", "pr_number": 335},
        {"sha": "j1k2l3", "message": "Merge branch 'chore/ci-node22' into 'release/v2.2.0' See merge request !332",          "author": "carol", "date": "2026-05-09", "pr_number": 332},
    ]
}
```

### Mock PRs
```python
MOCK_PRS = {
    341: {"number": 341, "title": "Remove deprecated /v1/auth endpoint", "description": "The /v1/auth endpoint was deprecated in v2.0.0. This MR removes it entirely. Clients must migrate to /v2/auth.", "labels": ["breaking-change", "auth"], "author": "alice", "merged_at": "2026-05-12", "breaking_change": True},
    338: {"number": 338, "title": "Fix memory leak in session handler",   "description": "Sessions were not being cleaned up on timeout. Fixed by adding explicit cleanup in the session expiry handler.", "labels": ["bug", "performance"], "author": "bob",   "merged_at": "2026-05-11", "breaking_change": False},
    335: {"number": 335, "title": "Add OAuth2 PKCE support",              "description": "Implements PKCE flow for public clients as per RFC 7636. Required for mobile app support.", "labels": ["feature", "auth"], "author": "alice", "merged_at": "2026-05-10", "breaking_change": False},
    332: {"number": 332, "title": "Update CI pipeline to Node 22",        "description": "Bumps CI base image to Node 22 LTS. All tests passing.", "labels": ["internal", "ci"], "author": "carol", "merged_at": "2026-05-09", "breaking_change": False},
}
```

### Mock past release notes
```python
MOCK_PAST_RELEASES = {
    "v2.1.0": {
        "version": "v2.1.0",
        "date": "2026-04-15",
        "content": """## v2.1.0 — 2026-04-15
### ✨ New Features
- Introduced /v2/auth endpoint with improved token handling
### 🐛 Bug Fixes
- Fixed CSRF token validation on login flow
### ⚠️ Deprecations
- /v1/auth endpoint deprecated — will be removed in v2.2.0
"""
    }
}
```

---

## Frontend — Streamlit App

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  SIDEBAR                │  MAIN PANEL                        │
│                         │                                     │
│  ── Presenter Controls  │  ┌─ Inspector Tab ──────────────┐  │
│                         │  │  System Prompt               │  │
│  [ ] Use Mock Data      │  │  Tool Definitions            │  │
│                         │  │  Messages so far             │  │
│  ── Demo Toggles        │  └──────────────────────────────┘  │
│                         │                                     │
│  [ ] Tools Enabled      │  ┌─ Agent Trace Tab ─────────────┐ │
│  [ ] Chain of Thought   │  │  🤔 THOUGHT  (streamed)       │ │
│  [ ] Memory             │  │  ⚡ ACTION   tool + inputs    │ │
│                         │  │  👁 OBSERVATION  result       │ │
│  ── Task               │  │  🤔 THOUGHT  ...              │ │
│  [Release date input]   │  │  ✅ FINAL OUTPUT              │ │
│  [Run Agent] button     │  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Tabs

**Tab 1 — Inspector** (shown before firing the agent)

Shows the audience what the agent "knows" before it starts:
- System prompt (rendered, not raw — syntax highlighted)
- Tool definitions: name, description, input schema for each enabled tool
- Current messages list (starts with just the user task)

Purpose: presenter walks through this before clicking Run so the audience understands the setup.

**Tab 2 — Agent Trace** (populates as agent runs)

Each AgentEvent renders as a distinct visual block:

| Event | Visual |
|-------|--------|
| `thought` | Blue card, 🤔 label, streamed text token by token |
| `action` | Yellow card, ⚡ label, tool name + formatted JSON inputs |
| `observation` | Grey card, 👁 label, tool result (collapsible if long) |
| `final` | Green card, ✅ label, formatted markdown output |
| `error` | Red card, ❌ label, error message |

Thought content streams in real time. Action and Observation appear instantly (they are discrete events).

### Sidebar Controls

```
USE MOCK DATA       [toggle]   # routes GitLab calls to mock or real API

── Demo Mode ──────────────────

TOOLS ENABLED       [toggle]   # include tool definitions in LLM call
  → OFF: model answers from training knowledge only (shows hallucination)
  → ON:  model uses tools to fetch real (or mock) data

CHAIN OF THOUGHT    [toggle]   # append CoT instruction to system prompt
  → OFF: model outputs answer directly (black box)
  → ON:  model reasons step by step before answering

MEMORY              [toggle]   # load past release notes into context
  → OFF: agent starts fresh, no knowledge of previous releases
  → ON:  agent loads v2.1.0 notes, references them in output

── Task ────────────────────────

Current branch:   [text input]   # e.g. release/v2.2.0
Previous branch:  [text input]   # e.g. release/v2.1.0
Run Agent         [button]
```

### Before/After Presentation Flow

The toggles are designed so the presenter can show contrasts mid-talk:

| Presenter says | Toggle state |
|---------------|-------------|
| "Without tools, the model guesses..." | Tools=OFF, CoT=OFF |
| "Now let's give it tools..." | Tools=ON, CoT=OFF |
| "You can see it works but it's a black box..." | Tools=ON, CoT=OFF |
| "Chain of thought makes reasoning visible..." | Tools=ON, CoT=ON |
| "It has no memory of past releases yet..." | Tools=ON, CoT=ON, Memory=OFF |
| "Now with memory loaded..." | Tools=ON, CoT=ON, Memory=ON |

Each state change: clear the trace, click Run, audience watches the difference.

---

## Memory Component (`agent/memory.py`)

Simple file-based memory for the demo. No vector DB, no embeddings — the point is to show the concept, not the infrastructure.

```python
MEMORY_FILE = "agent_memory.json"

def load_memory() -> str:
    """Returns a formatted string ready to inject into the system prompt."""

def save_memory(key: str, value: str) -> None:
    """Persists a key-value pair to the memory file."""
```

The memory file is pre-seeded with:
```json
{
  "team_preferences": "Always put breaking changes first. Use emoji section headers. One line per item.",
  "last_release_version": "v2.1.0",
  "known_issues": "The /v1/auth removal was announced in v2.1.0 — reference this when it appears."
}
```

When memory is enabled, this is injected into the system prompt before the agent runs. The audience sees the agent use this knowledge to format output consistently and reference the prior release.

---

## Demo Script (Presenter Guidance)

### Beat 1 — Setup (~2 min)
- Open the Inspector tab
- Walk through the system prompt: "This is the job description we gave the model"
- Walk through tool definitions: "These are the three things it can call"
- Point out the task: "Generate release notes for everything in release/v2.2.0 that wasn't in release/v2.1.0"

### Beat 2 — No tools (1 min)
- Toggle: Tools=OFF, CoT=OFF, Memory=OFF
- Click Run
- Model either refuses or hallucinates commits
- Say: "It's making things up — it has no way to fetch real data"

### Beat 3 — Tools ON (2 min)
- Toggle: Tools=ON, CoT=OFF
- Click Run
- Show the ACTION/OBSERVATION cards appearing
- Say: "Now we can see it calling our tools — but we can't see why it makes decisions"

### Beat 4 — Chain of Thought (2 min)
- Toggle: CoT=ON (tools still ON)
- Click Run
- THOUGHT cards now show reasoning before each action
- Say: "Now we can see it thinking — it noticed the breaking change first, it's reasoning about order"
- Say: "This is chain of thought — and it's also how you debug agents"

### Beat 5 — Memory (2 min)
- Toggle: Memory=ON (tools and CoT still ON)
- Click Run
- Agent calls `get_past_release_notes("v2.1.0")`
- Output references v2.1.0 and notes the /v1/auth removal was announced then
- Say: "It remembered what shipped before — this is external memory injected at startup"

---

## Key Implementation Notes

- **Streaming first.** The demo lives or dies on the audience seeing tokens arrive in real time. Implement streaming in `llm.py` before anything else and test it early.
- **Mock data must look realistic.** The PRs and commits should feel like a real codebase — the audience will read them.
- **Keep the UI calm.** Don't animate everything. The trace cards appearing one at a time is enough movement. Avoid spinners, progress bars, or anything that distracts from the content.
- **Presenter mode.** Add a `?presenter=true` URL param (or sidebar toggle) that hides the raw JSON in Observation cards and shows only a summary — cleaner for the big screen.
- **Test the "Run" button cold.** Before the talk, restart the app and click Run once with each toggle state to confirm all paths work.

---

## Out of Scope

- Authentication for the Streamlit app
- Persistent conversation history across page reloads
- Multi-agent orchestration
- Fine-tuning or embeddings
- Any framework (LangChain, LangGraph, etc.) — raw Python only

---

*Spec version: 1.0 — May 2026*
