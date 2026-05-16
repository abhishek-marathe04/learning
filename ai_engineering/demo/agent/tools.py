"""Tool definitions and dispatch. Routes to mock or real client based on Config."""

import json
from config import config

if config.use_mock:
    from gitlab.mock import get_branch_diff as _get_branch_diff
    from gitlab.mock import get_pr_details as _get_pr_details
    from gitlab.mock import get_past_release_notes as _get_past_release_notes
else:
    from gitlab.client import get_branch_diff as _get_branch_diff
    from gitlab.client import get_pr_details as _get_pr_details
    from gitlab.client import get_past_release_notes as _get_past_release_notes


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_branch_diff",
            "description": (
                "Returns commits that are in the current release branch but not in the "
                "previous release branch. Use this first to establish the exact scope of "
                "a release — what changed between two named branches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "current_branch": {"type": "string"},
                    "previous_branch": {"type": "string"},
                },
                "required": ["current_branch", "previous_branch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pr_details",
            "description": (
                "Retrieves full details of a merge request including title, description, "
                "labels, and linked issue. Use this to understand the intent and impact "
                "of a specific change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_number": {"type": "integer"},
                },
                "required": ["pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_past_release_notes",
            "description": (
                "Retrieves release notes from a previous version. Use this to maintain "
                "consistent format and to reference what was shipped before so you don't "
                "repeat items."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "version": {"type": "string"},
                },
                "required": ["version"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "get_branch_diff": _get_branch_diff,
    "get_pr_details": _get_pr_details,
    "get_past_release_notes": _get_past_release_notes,
}


def execute_tool(name: str, inputs: dict) -> dict:
    """Looks up the tool by name and calls it with the provided inputs."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    func = TOOL_REGISTRY[name]
    result = func(**inputs)
    return result
