"""Agent memory: load and save key-value pairs from a JSON file."""

import json
import os

MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_memory.json")


def load_memory() -> str:
    """Returns a formatted string ready to inject into the system prompt."""
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "Team Memory:\n(no memory available)"

    lines = ["Team Memory:"]
    for key, value in data.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def save_memory(key: str, value: str) -> None:
    """Persists a key-value pair to the memory file."""
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    data[key] = value

    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)
