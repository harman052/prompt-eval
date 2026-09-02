"""Prompt loading and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from prompt_eval.errors import PromptError
from prompt_eval.prompts import (
    DATASET_PROMPT,
    JUDGE_PROMPT,
    SOLUTION_PROMPT,
    load_prompt,
    render_prompt,
)


@pytest.mark.parametrize("name", [SOLUTION_PROMPT, JUDGE_PROMPT, DATASET_PROMPT])
def test_every_referenced_template_exists(name: str) -> None:
    assert load_prompt(name).strip()


def test_render_substitutes_placeholders() -> None:
    rendered = render_prompt(
        JUDGE_PROMPT, task="T", solution="S", solution_criteria="C"
    )
    assert "<task>\nT\n</task>" in rendered
    assert "$" not in rendered


def test_render_reports_a_missing_placeholder() -> None:
    with pytest.raises(PromptError, match="requires placeholder"):
        render_prompt(JUDGE_PROMPT, task="T")


def test_braces_in_values_are_not_treated_as_placeholders() -> None:
    """Solutions are full of JSON and regex braces; str.format would choke."""
    rendered = render_prompt(
        JUDGE_PROMPT,
        task="T",
        solution='{"a": {"b": 1}}',
        solution_criteria=r"^\d{3}$",
    )
    assert '{"a": {"b": 1}}' in rendered
    assert r"^\d{3}$" in rendered


def test_missing_template_is_reported_clearly(workspace: Path) -> None:
    load_prompt.cache_clear()
    (workspace / "prompts" / "judge_prompt.txt").unlink()
    with pytest.raises(PromptError, match="not found"):
        load_prompt(JUDGE_PROMPT)
    load_prompt.cache_clear()


def test_malformed_template_is_reported_clearly(workspace: Path) -> None:
    """A stray `$` is a template error, not a crash mid-run."""
    load_prompt.cache_clear()
    (workspace / "prompts" / "broken.txt").write_text("100$ and $")
    with pytest.raises(PromptError, match="malformed"):
        render_prompt("broken")
    load_prompt.cache_clear()
