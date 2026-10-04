"""Raw HTTP calls to LiteLLM — no SDK."""

from __future__ import annotations

import json
import time
from typing import Generator

import httpx

from config import config


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    stream: bool = False,
) -> dict | Generator:
    """
    POST to LLM_BASE_URL/chat/completions.
    Returns the raw JSON response (non-streaming) or a generator of chunks (streaming).
    """
    url = f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict = {
        "model": config.MODEL_NAME,
        "messages": messages,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools

    print(f"\n[LLM] POST {url}")
    print(f"[LLM] model={config.MODEL_NAME}  messages={len(messages)}  tools={len(tools) if tools else 0}  stream={stream}")

    if not stream:
        t0 = time.monotonic()
        response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        elapsed = time.monotonic() - t0
        print(f"[LLM] status={response.status_code}  elapsed={elapsed:.2f}s")
        if response.status_code != 200:
            print(f"[LLM] ERROR body: {response.text}")
            raise RuntimeError(
                f"LiteLLM request failed: {response.status_code} {response.text}"
            )
        return response.json()

    # Streaming: return a generator
    return _stream_chunks(url, headers, payload)


def _stream_chunks(url: str, headers: dict, payload: dict) -> Generator:
    """Yields parsed SSE chunks from the streaming response."""
    t0 = time.monotonic()
    with httpx.stream("POST", url, headers=headers, json=payload, timeout=120.0) as response:
        print(f"[LLM] status={response.status_code} (streaming)")
        if response.status_code != 200:
            body = response.read().decode()
            print(f"[LLM] ERROR body: {body}")
            raise RuntimeError(
                f"LiteLLM streaming request failed: {response.status_code} {body}"
            )
        is_error_event = False
        for line in response.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line == "event: error":
                is_error_event = True
                print(f"[LLM] SSE error event received")
                continue
            if not line.startswith("data: "):
                print(f"[LLM] raw non-data line: {line!r}")
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                print(f"[LLM] stream done  elapsed={time.monotonic() - t0:.2f}s")
                break
            try:
                parsed = json.loads(data)
                if is_error_event:
                    print(f"[LLM] ERROR detail: {data}")
                    is_error_event = False
                    raise RuntimeError(f"Groq stream error: {data}")
                yield parsed
            except json.JSONDecodeError:
                print(f"[LLM] unparseable chunk: {data!r}")
                continue
