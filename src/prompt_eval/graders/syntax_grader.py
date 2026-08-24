import ast
import json
import re

from ..models import TestCase


def validate_json(text: str) -> float:
  try:
    json.loads(text.strip())
    return 10.0
  except json.JSONDecodeError:
    return 0.0


def validate_python(text: str) -> float:
  try:
    ast.parse(text.strip())
    return 10.0
  except SyntaxError:
    return 0.0


def validate_regex(text: str) -> float:
  try:
    re.compile(text.strip())
    return 10.0
  except re.error:
    return 0.0


def code_grader(test_case: TestCase, response: str) -> float:
  validators = {
    "python": validate_python,
    "json": validate_json,
    "regex": validate_regex,
  }

  return validators[test_case.format](response)
