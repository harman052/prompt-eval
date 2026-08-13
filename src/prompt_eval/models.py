from typing import Literal

from pydantic import BaseModel


class TestCase(BaseModel):
  task: str
  format: Literal["python", "json", "regex"]
  solution_criteria: str


class ModelGrade(BaseModel):
  strengths: list[str]
  weaknesses: list[str]
  reasoning: str
  score: float


class EvaluationResult(BaseModel):
  test_case: TestCase
  output: str
  model_score: float
  validity_score: float
  score: float
  reasoning: str
