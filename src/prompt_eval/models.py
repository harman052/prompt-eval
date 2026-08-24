from typing import Literal

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    task: str
    format: Literal["python", "json", "regex"]
    solution_criteria: str


class Dataset(BaseModel):
    test_cases: list[TestCase]


class ModelGrade(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    score: int = Field(ge=1, le=10)


class EvaluationResult(BaseModel):
    test_case: TestCase
    output: str
    model_score: float
    validity_score: float
    final_score: float
    reasoning: str
