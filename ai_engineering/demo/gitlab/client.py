"""Real GitLab REST client using httpx."""

import re
import base64
import httpx

from config import config


def _http_client() -> httpx.Client:
    return httpx.Client(
        base_url=config.gitlab_base_url,
        headers={"PRIVATE-TOKEN": config.gitlab_token},
        timeout=30.0,
    )


def get_branch_diff(current_branch: str, previous_branch: str) -> list[dict]:
    """Returns commits in current_branch that are not in previous_branch."""
    with _http_client() as client:
        response = client.get(
            f"/api/v4/projects/{config.gitlab_project_id}/repository/compare",
            params={"from": previous_branch, "to": current_branch},
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"GitLab compare failed: {response.status_code} {response.text}"
            )
        data = response.json()

    commits = data.get("commits", [])
    result = []
    for commit in commits:
        message = commit.get("message", "")
        pr_match = re.search(r"!(\d+)", message)
        pr_number = int(pr_match.group(1)) if pr_match else None
        result.append({
            "sha": commit.get("short_id", commit.get("id", "")[:7]),
            "message": message,
            "author": commit.get("author_name", ""),
            "date": commit.get("created_at", "")[:10],
            "pr_number": pr_number,
        })
    return result


def get_pr_details(pr_number: int) -> dict:
    """Returns full details for a merge request."""
    with _http_client() as client:
        response = client.get(
            f"/api/v4/projects/{config.gitlab_project_id}/merge_requests/{pr_number}"
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"GitLab MR fetch failed: {response.status_code} {response.text}"
            )
        mr = response.json()

    labels = [label["name"] if isinstance(label, dict) else label for label in mr.get("labels", [])]
    breaking_change = any("breaking-change" in label.lower() for label in labels)

    return {
        "number": mr.get("iid"),
        "title": mr.get("title", ""),
        "description": mr.get("description", ""),
        "labels": labels,
        "author": mr.get("author", {}).get("username", ""),
        "merged_at": (mr.get("merged_at") or "")[:10],
        "breaking_change": breaking_change,
    }


def get_past_release_notes(version: str) -> dict:
    """Returns release notes for a past version by reading RELEASE_NOTES.md at that tag."""
    with _http_client() as client:
        response = client.get(
            f"/api/v4/projects/{config.gitlab_project_id}/repository/files/RELEASE_NOTES.md",
            params={"ref": f"refs/tags/{version}"},
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"GitLab file fetch failed: {response.status_code} {response.text}"
            )
        data = response.json()

    raw_content = base64.b64decode(data.get("content", "")).decode("utf-8")
    return {
        "version": version,
        "date": data.get("last_commit_id", "")[:10],
        "content": raw_content,
    }
