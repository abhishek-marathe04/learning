"""ReAct agent loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Callable, Optional

from config import config
from agent.llm import chat
from agent.tools import TOOL_DEFINITIONS, execute_tool
from agent.memory import load_memory
from agent.prompts import build_system_prompt


@dataclass
class AgentEvent:
    type: Literal["thought", "action", "observation", "final", "error"]
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None


def run(
    task: str,
    tools_enabled: bool,
    cot_enabled: bool,
    memory_enabled: bool,
    on_event: Callable[[AgentEvent], None],
) -> str:
    """
    Runs the ReAct agent loop.

    Emits AgentEvent objects via on_event callback for the UI to render.
    Returns the final text content produced by the agent.
    """
    # 1. Build system prompt
    system_prompt = build_system_prompt(cot_enabled=cot_enabled, memory_enabled=memory_enabled)

    # 2. Inject memory into system prompt if enabled
    if memory_enabled:
        memory_text = load_memory()
        system_prompt = system_prompt + f"\n\n{memory_text}"

    # 3. Initialise message history
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    tools = TOOL_DEFINITIONS if tools_enabled else None
    final_content = ""

    print(f"\n[AGENT] Starting run  tools_enabled={tools_enabled}  cot={cot_enabled}  memory={memory_enabled}")
    print(f"[AGENT] Task: {task}")

    # 4. Agent loop
    for iteration in range(config.agent_max_iterations):
        print(f"\n[AGENT] --- Iteration {iteration + 1} ---")
        # Call LLM with streaming
        stream = chat(messages=messages, tools=tools, stream=True)

        # Accumulate the streamed response
        accumulated_text = ""
        accumulated_tool_calls: dict[int, dict] = {}  # index -> tool call accumulator
        finish_reason = None

        for chunk in stream:
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason") or finish_reason

            # Accumulate text content
            text_delta = delta.get("content")
            if text_delta:
                accumulated_text += text_delta
                # Stream thought tokens to UI in real time
                on_event(AgentEvent(type="thought", content=accumulated_text))

            # Accumulate tool calls (may arrive in fragments)
            tool_calls_delta = delta.get("tool_calls", [])
            for tc_delta in tool_calls_delta:
                idx = tc_delta.get("index", 0)
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                tc = accumulated_tool_calls[idx]
                if tc_delta.get("id"):
                    tc["id"] += tc_delta["id"]
                func_delta = tc_delta.get("function", {})
                if func_delta.get("name"):
                    tc["function"]["name"] += func_delta["name"]
                if func_delta.get("arguments"):
                    tc["function"]["arguments"] += func_delta["arguments"]

        # Convert accumulated tool calls dict to list
        tool_calls = [accumulated_tool_calls[i] for i in sorted(accumulated_tool_calls.keys())]

        # Decide what happened
        has_tool_calls = len(tool_calls) > 0
        print(f"[AGENT] finish_reason={finish_reason}  tool_calls={len(tool_calls)}  text_len={len(accumulated_text)}")

        if has_tool_calls:
            # Append the assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": accumulated_text or "",
                "tool_calls": tool_calls,
            })

            # Execute each tool call
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    tool_input = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    tool_input = {}

                # Emit ACTION event
                on_event(AgentEvent(
                    type="action",
                    content=f"Calling **{tool_name}** with: `{json.dumps(tool_input)}`",
                    tool_name=tool_name,
                    tool_input=tool_input,
                ))

                # Execute the tool
                print(f"[AGENT] Tool call: {tool_name}({tool_input})")
                result = execute_tool(tool_name, tool_input)
                print(f"[AGENT] Tool result keys: {list(result.keys()) if isinstance(result, dict) else type(result).__name__}")

                # Emit OBSERVATION event
                on_event(AgentEvent(
                    type="observation",
                    content=json.dumps(result, indent=2),
                    tool_name=tool_name,
                ))

                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                })

            # Continue the loop to let the agent respond to tool results
            continue

        else:
            # No tool calls — this is the final response
            final_content = accumulated_text
            print(f"[AGENT] Final response  len={len(final_content)}")
            on_event(AgentEvent(type="final", content=final_content))
            break

    else:
        # Exceeded max iterations
        on_event(AgentEvent(
            type="error",
            content=f"Agent reached max iterations ({config.agent_max_iterations}) without finishing.",
        ))

    return final_content
