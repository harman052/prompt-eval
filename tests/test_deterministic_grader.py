"""Deterministic grader: validity is binary, and fences must not be penalised."""

from __future__ import annotations

import pytest

from prompt_eval.constants import MAX_SCORE, MIN_SCORE
from prompt_eval.graders.deterministic import (
    SYNTAX_VALIDATORS,
    grade_syntax,
    strip_code_fence,
)
from prompt_eval.models import SolutionFormat


@pytest.mark.parametrize(
    ("solution", "solution_format", "expected"),
    [
        ("x = 1", SolutionFormat.PYTHON, MAX_SCORE),
        ("def f(:", SolutionFormat.PYTHON, MIN_SCORE),
        ('{"a": [1, 2]}', SolutionFormat.JSON, MAX_SCORE),
        ("{'a': 1}", SolutionFormat.JSON, MIN_SCORE),
        (r"^\d{3}$", SolutionFormat.REGEX, MAX_SCORE),
        ("[unclosed", SolutionFormat.REGEX, MIN_SCORE),
        ("", SolutionFormat.JSON, MIN_SCORE),
        ("   ", SolutionFormat.PYTHON, MAX_SCORE),  # empty module is valid python
    ],
)
def test_grade_syntax(
    solution: str, solution_format: SolutionFormat, expected: float
) -> None:
    assert grade_syntax(solution, solution_format) == expected


def test_every_format_has_a_validator() -> None:
    """A new SolutionFormat must not silently fall through to a KeyError."""
    assert set(SYNTAX_VALIDATORS) == set(SolutionFormat)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("```python\nx = 1\n```", "x = 1"),
        ("```\nx = 1\n```", "x = 1"),
        ("```json\n{}\n```", "{}"),
        ("x = 1", "x = 1"),
        ("`x = 1`", "`x = 1`"),  # inline code is not a fence
        ("```python\nx = 1", "```python\nx = 1"),  # unterminated fence is left alone
    ],
)
def test_strip_code_fence(text: str, expected: str) -> None:
    assert strip_code_fence(text) == expected


def test_fenced_solution_is_still_graded_valid() -> None:
    assert grade_syntax("```python\nx = 1\n```", SolutionFormat.PYTHON) == MAX_SCORE


def test_python_null_byte_is_invalid_not_crashing() -> None:
    """ast.parse raises ValueError (not SyntaxError) on null bytes."""
    assert grade_syntax("x = 1\x00", SolutionFormat.PYTHON) == MIN_SCORE
