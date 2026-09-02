"""Pydantic models for datasets, solutions and grading reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    computed_field,
    model_validator,
)

from prompt_eval.constants import MAX_SCORE, MIN_SCORE

Score = Annotated[float, Field(ge=MIN_SCORE, le=MAX_SCORE)]
"""A grader score on 0-10 scale."""


class SolutionFormat(StrEnum):
    """Solution format a test case expects"""

    PYTHON = "python"
    JSON = "json"
    REGEX = "regex"


class StrictModel(BaseModel):
    """Base model for prompt-eval app"""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)


class TestCaseSpec(StrictModel):
    """Test case specifications.

    Field descriptions, in addition to documentation for developers, are
    included in the JSON schema sent to LLM, so they also serve as instructions
    for dataset generation.
    """

    task: str = Field(
        min_length=1, description="A single, self-contained AWS-related task."
    )
    format: SolutionFormat = Field(
        description="The format the solution must be written in."
    )
    solution_criteria: str = Field(
        min_length=1,
        description="Objective criteria a reviewer can use to judge a solution.",
    )


class DatasetSpec(StrictModel):
    """Dataset specifications for an LLM."""

    test_cases: list[TestCaseSpec] = Field(min_length=1)


class SolutionSpec(StrictModel):
    """Solution specifications for an LLM."""

    code: str = Field(
        min_length=1,
        description="The complete solution source, with no markdown fences or prose.",
    )


class TestCase(TestCaseSpec):
    """A stored test case: an LLM-generated spec plus its stable identifier."""

    test_case_id: str = Field(min_length=1, pattern=r"^\d{3,}$")


class Dataset(RootModel[list[TestCase]]):
    """A non-empty list of test cases with unique ids."""

    root: list[TestCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> Self:
        counts = Counter(case.test_case_id for case in self.root)
        duplicates = sorted(id_ for id_, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate test case ids: {', '.join(duplicates)}")
        return self

    def __iter__(self) -> Iterator[TestCase]:  # type: ignore[override]
        """Iterate over test cases rather than Pydantic model fields."""
        return iter(self.root)

    def __len__(self) -> int:
        """Return the number of test cases."""
        return len(self.root)


class RunMetadata(StrictModel):
    """Metadata for a report: which prompts and which model produced it."""

    prompt_version: str
    model: str
    run_at: datetime


class TestCaseRef(StrictModel):
    """Common test case fields included in report results."""

    test_case_id: str
    task: str
    format: SolutionFormat


class Report[ResultT: TestCaseRef](BaseModel):
    """Generic ``metadata + results`` wrapper for every persisted artifact."""

    model_config = ConfigDict(frozen=True)

    metadata: RunMetadata
    results: list[ResultT]

    @property
    def by_test_case_id(self) -> dict[str, ResultT]:
        """Index the results by test case id for joining reports together."""
        return {result.test_case_id: result for result in self.results}


class GeneratedSolution(TestCaseRef):
    """An LLM-generated solution for a test case."""

    solution: str


class ModelGrade(BaseModel):
    """Structured output returned by the LLM-as-judge grader."""

    model_config = ConfigDict(extra="ignore")

    strengths: list[str] = Field(
        description="Concrete things the solution does well.",
    )
    weaknesses: list[str] = Field(
        description="Concrete defects or omissions in the solution.",
    )
    reasoning: str = Field(
        min_length=1,
        description="Short justification for the score, referencing the criteria.",
    )
    score: Score = Field(
        description=(
            f"Overall quality from {MIN_SCORE:.0f} (unusable) to "
            f"{MAX_SCORE:.0f} (fully satisfies every criterion)."
        )
    )


class DeterministicResult(TestCaseRef):
    """Syntax-validation score for a test case."""

    score: Score


class ModelGraderResult(TestCaseRef):
    """An LLM-as-judge grade attached to its test case."""

    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    score: Score

    @classmethod
    def from_model_grader(
        cls, test_case: TestCase, grade: ModelGrade
    ) -> ModelGraderResult:
        return cls(
            test_case_id=test_case.test_case_id,
            task=test_case.task,
            format=test_case.format,
            **grade.model_dump(),
        )


class CombinedResult(TestCaseRef):
    """Both grader scores for a test case, and their mean."""

    # ``final_score`` is computed, so it is *written* to JSON but cannot be
    # *read* back as a field. Extras are ignored (rather than forbidden) so a
    # saved report can be loaded again - which `compare` and `set-baseline`
    # both rely on.
    model_config = ConfigDict(frozen=True, extra="ignore")

    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    deterministic_score: Score
    llm_judge_score: Score

    @computed_field  # type: ignore[prop-decorator]
    @property
    def final_score(self) -> float:
        """Mean (average) of the two graders."""
        return (self.deterministic_score + self.llm_judge_score) / 2


class ComparisonResult(TestCaseRef):
    """A test case's score movement against the baseline."""

    model_config = ConfigDict(frozen=True, extra="ignore")  # see CombinedResult

    baseline_score: Score
    current_score: Score

    @computed_field  # type: ignore[prop-decorator]
    @property
    def delta(self) -> float:
        """Positive value means the current run improved on the baseline."""
        return self.current_score - self.baseline_score

    @property
    def is_regression(self) -> bool:
        return self.delta < 0


SolutionReport = Report[GeneratedSolution]
DeterministicReport = Report[DeterministicResult]
ModelGraderReport = Report[ModelGraderResult]
CombinedReport = Report[CombinedResult]
ComparisonReport = Report[ComparisonResult]
