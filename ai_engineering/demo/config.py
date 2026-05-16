from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    use_mock: bool
    gitlab_base_url: str
    gitlab_token: str
    gitlab_project_id: str
    gitlab_default_branch: str
    LLM_BASE_URL: str
    LLM_API_KEY: str
    MODEL_NAME: str
    agent_max_iterations: int


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes")


config = Config(
    use_mock=_parse_bool(os.getenv("USE_MOCK"), default=True),
    gitlab_base_url=os.getenv("GITLAB_BASE_URL", "https://gitlab.example.com"),
    gitlab_token=os.getenv("GITLAB_TOKEN", ""),
    gitlab_project_id=os.getenv("GITLAB_PROJECT_ID", "123"),
    gitlab_default_branch=os.getenv("GITLAB_DEFAULT_BRANCH", "main"),
    LLM_BASE_URL=os.getenv("LLM_BASE_URL", "https://your-litellm-gateway"),
    LLM_API_KEY=os.getenv("LLM_API_KEY", ""),
    MODEL_NAME=os.getenv("MODEL_NAME", "claude-sonnet-4-6"),
    agent_max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "10")),
)
