"""Mock GitLab data for demo purposes. Same interface as client.py."""

MOCK_BRANCH_DIFF = {
    ("release/v2.2.0", "release/v2.1.0"): [
        {"sha": "a1b2c3", "message": "Merge branch 'feature/remove-v1-auth' into 'release/v2.2.0' See merge request !341", "author": "alice", "date": "2026-05-12", "pr_number": 341},
        {"sha": "d4e5f6", "message": "Merge branch 'fix/session-memory-leak' into 'release/v2.2.0' See merge request !338",  "author": "bob",   "date": "2026-05-11", "pr_number": 338},
        {"sha": "g7h8i9", "message": "Merge branch 'feature/oauth2-pkce' into 'release/v2.2.0' See merge request !335",      "author": "alice", "date": "2026-05-10", "pr_number": 335},
        {"sha": "j1k2l3", "message": "Merge branch 'chore/ci-node22' into 'release/v2.2.0' See merge request !332",          "author": "carol", "date": "2026-05-09", "pr_number": 332},
    ]
}

MOCK_PRS = {
    341: {"number": 341, "title": "Remove deprecated /v1/auth endpoint", "description": "The /v1/auth endpoint was deprecated in v2.0.0. This MR removes it entirely. Clients must migrate to /v2/auth.", "labels": ["breaking-change", "auth"], "author": "alice", "merged_at": "2026-05-12", "breaking_change": True},
    338: {"number": 338, "title": "Fix memory leak in session handler",   "description": "Sessions were not being cleaned up on timeout. Fixed by adding explicit cleanup in the session expiry handler.", "labels": ["bug", "performance"], "author": "bob",   "merged_at": "2026-05-11", "breaking_change": False},
    335: {"number": 335, "title": "Add OAuth2 PKCE support",              "description": "Implements PKCE flow for public clients as per RFC 7636. Required for mobile app support.", "labels": ["feature", "auth"], "author": "alice", "merged_at": "2026-05-10", "breaking_change": False},
    332: {"number": 332, "title": "Update CI pipeline to Node 22",        "description": "Bumps CI base image to Node 22 LTS. All tests passing.", "labels": ["internal", "ci"], "author": "carol", "merged_at": "2026-05-09", "breaking_change": False},
}

MOCK_PAST_RELEASES = {
    "v2.1.0": {
        "version": "v2.1.0",
        "date": "2026-04-15",
        "content": """## v2.1.0 — 2026-04-15\n### ✨ New Features\n- Introduced /v2/auth endpoint with improved token handling\n### 🐛 Bug Fixes\n- Fixed CSRF token validation on login flow\n### ⚠️ Deprecations\n- /v1/auth endpoint deprecated — will be removed in v2.2.0\n"""
    }
}


def get_branch_diff(current_branch: str, previous_branch: str) -> list[dict]:
    """Returns commits in current_branch that are not in previous_branch."""
    key = (current_branch, previous_branch)
    if key in MOCK_BRANCH_DIFF:
        return MOCK_BRANCH_DIFF[key]
    return []


def get_pr_details(pr_number: int) -> dict:
    """Returns full details for a merge request."""
    if pr_number in MOCK_PRS:
        return MOCK_PRS[pr_number]
    return {"error": f"PR {pr_number} not found in mock data"}


def get_past_release_notes(version: str) -> dict:
    """Returns release notes for a past version."""
    if version in MOCK_PAST_RELEASES:
        return MOCK_PAST_RELEASES[version]
    return {"error": f"Version {version} not found in mock data"}
