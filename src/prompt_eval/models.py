from typing import Literal

from pydantic import BaseModel, Field, RootModel


class GeneratedTestCase(BaseModel):
    task: str
    format: Literal["python", "json", "regex"]
    solution_criteria: str


class GeneratedDataset(BaseModel):
    test_cases: list[GeneratedTestCase]


class TestCase(BaseModel):
    test_case_id: str
    task: str
    format: Literal["python", "json", "regex"]
    solution_criteria: str


class Dataset(RootModel[list[TestCase]]):
    pass


class Solution(BaseModel):
    test_case_id: str
    task: str
    solution: str


class PromptMetadata(BaseModel):
    prompt_version: str
    model: str
    run_at: str


class Solutions(BaseModel):
    metadata: PromptMetadata
    solutions: list[Solution]


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


class DeterministicGraderResults(BaseModel):
    metadata: PromptMetadata
    results: list[DeterministicGraderResult]


class ModelGraderResult(BaseModel):
    test_case_id: str
    task: str
    format: str
    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    score: float


class ModelGraderResults(BaseModel):
    metadata: PromptMetadata
    results: list[ModelGraderResult]


class CombinedResult(BaseModel):
    test_case_id: str
    task: str
    format: str
    strengths: list[str] | None
    weaknesses: list[str] | None
    reasoning: str | None
    deterministic_score: float
    llm_judge_score: float
    final_score: float


class CombinedResults(BaseModel):
    metadata: PromptMetadata
    results: list[CombinedResult]


class ComparisonResult(BaseModel):
    test_case_id: str
    task: str
    baseline_score: float
    current_score: float
    delta: float


class ComparisonResults(BaseModel):
    metadata: PromptMetadata
    results: list[ComparisonResult]
