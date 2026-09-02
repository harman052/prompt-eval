"""Prompt provenance.

A score is only meaningful next to the prompt revision that produced it, so
every report records the git commit of the tree plus a ``-dirty`` marker when
``prompts/`` has uncommitted edits.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache

UNKNOWN_VERSION = "unknown"
GIT_TIMEOUT_SECONDS = 5


@lru_cache(maxsize=1)
def get_prompt_version() -> str:
    """Return ``<short-sha>`` or ``<short-sha>-dirty``.

    Falls back to ``"unknown"`` outside a git checkout (installed wheel, docker
    build, source tarball) instead of raising: provenance is metadata, and
    losing it must never fail an evaluation run.
    """
    commit = _git("rev-parse", "--short", "HEAD")
    if commit is None:
        return UNKNOWN_VERSION

    uncommitted_prompt_changes = _git("status", "--porcelain", "prompts/")
    return f"{commit}-dirty" if uncommitted_prompt_changes else commit


def _git(*args: str) -> str | None:
    """Run a git command, returning stripped stdout or ``None`` on failure."""
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()
