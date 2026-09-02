from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Mapping

from prompt_eval.constants import MAX_SCORE, MIN_SCORE
from prompt_eval.models import SolutionFormat

CODE_FENCE_PATTERN = re.compile(r"^\s*```[^\n]*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown code fence, if present."""
    match = CODE_FENCE_PATTERN.match(text)
    return (match.group("body") if match else text).strip()


def _is_valid_python(text: str) -> bool:
    try:
        ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    return True


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return False
    return True


def _is_valid_regex(text: str) -> bool:
    try:
        re.compile(text)
    except re.error:
        return False
    return True


SYNTAX_VALIDATORS: Mapping[SolutionFormat, Callable[[str], bool]] = {
    SolutionFormat.PYTHON: _is_valid_python,
    SolutionFormat.JSON: _is_valid_json,
    SolutionFormat.REGEX: _is_valid_regex,
}


def grade_syntax(solution: str, solution_format: SolutionFormat) -> float:
    is_valid = SYNTAX_VALIDATORS[solution_format]
    return MAX_SCORE if is_valid(strip_code_fence(solution)) else MIN_SCORE
