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


class DeterministicGraderResult(BaseModel):
    test_case_id: str
    task: str
    format: str
    score: float


class DeterministicGraderResults(RootModel[list[DeterministicGraderResult]]):
    pass


class ModelGraderResult(BaseModel):
    test_case_id: str
    task: str
    format: str
    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    score: float


class ModelGraderResults(RootModel[list[ModelGraderResult]]):
    pass


class CombinedResult(BaseModel):
    task: str
    format: str
    strengths: list[str] | None
    weaknesses: list[str] | None
    reasoning: str | None
    deterministic_score: float
    llm_judge_score: float
    final_score: float


class CombinedResults(RootModel[list[CombinedResult]]):
    pass


class ComparisonResult(BaseModel):
    task: str
    baseline_score: float
    current_score: float
    delta: float


class ComparisonResults(RootModel[list[ComparisonResult]]):
    pass
