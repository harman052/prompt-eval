from typing import Literal

from pydantic import BaseModel, Field, RootModel


class GeneratedTestCase(BaseModel):
    task: str
    format: Literal["python", "json", "regex"]
    solution_criteria: str


class GeneratedDataset(BaseModel):
    test_cases: list[GeneratedTestCase]


class TestCase(BaseModel):
    id: str
    task: str
    format: Literal["python", "json", "regex"]
    solution_criteria: str


class Dataset(RootModel[list[TestCase]]):
    pass


class Solution(BaseModel):
    test_case_id: str
    task: str
    solution: str


class Solutions(RootModel[list[Solution]]):
    pass


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
