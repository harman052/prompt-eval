"""Model invariants: constrained scores, computed aggregates, immutability."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from prompt_eval.models import (
    CombinedReport,
    CombinedResult,
    ComparisonResult,
    Dataset,
    GeneratedSolution,
    ModelGrade,
    ModelGraderResult,
    Report,
    RunMetadata,
    SolutionFormat,
    TestCase,
)


@pytest.mark.parametrize("score", [-0.1, 10.1, 100.0])
def test_scores_outside_the_scale_are_rejected(score: float) -> None:
    """A grader bug cannot produce an out-of-range score, in memory or on disk."""
    with pytest.raises(ValidationError):
        ModelGrade(strengths=[], weaknesses=[], reasoning="r", score=score)


def test_final_score_is_computed_not_stored(metadata: RunMetadata) -> None:
    result = CombinedResult(
        test_case_id="001",
        task="t",
        format=SolutionFormat.PYTHON,
        strengths=[],
        weaknesses=[],
        reasoning="r",
        deterministic_score=10.0,
        llm_judge_score=5.0,
    )
    assert result.final_score == 7.5
    assert json.loads(result.model_dump_json())["final_score"] == 7.5


def test_a_saved_report_can_be_loaded_again(metadata: RunMetadata) -> None:
    """Computed fields are serialised but not accepted as input, so the
    round-trip only works because these models ignore extras."""
    report = CombinedReport(
        metadata=metadata,
        results=[
            CombinedResult(
                test_case_id="001",
                task="t",
                format=SolutionFormat.JSON,
                strengths=["a"],
                weaknesses=[],
                reasoning="r",
                deterministic_score=10.0,
                llm_judge_score=9.0,
            )
        ],
    )
    reloaded = CombinedReport.model_validate_json(report.model_dump_json())
    assert reloaded == report


def test_comparison_delta_is_derived_from_the_two_scores() -> None:
    result = ComparisonResult(
        test_case_id="001",
        task="t",
        format=SolutionFormat.JSON,
        baseline_score=8.0,
        current_score=6.5,
    )
    assert result.delta == pytest.approx(-1.5)
    assert result.is_regression


def test_models_are_immutable() -> None:
    solution = GeneratedSolution(
        test_case_id="001", task="t", format=SolutionFormat.JSON, solution="{}"
    )
    with pytest.raises(ValidationError):
        # setattr, because a frozen model's fields are read-only to the type
        # checker too - which is the point of the config.
        setattr(solution, "solution", "tampered")  # noqa: B010


def test_unknown_fields_are_rejected_on_models_we_author() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TestCase.model_validate(
            {
                "test_case_id": "001",
                "task": "t",
                "format": "json",
                "solution_criteria": "c",
                "typo_field": 1,
            }
        )


def test_dataset_rejects_duplicate_ids(make_test_case) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError, match="duplicate test case ids"):
        Dataset(root=[make_test_case("001"), make_test_case("001")])


def test_dataset_supports_len_and_iteration(dataset: Dataset) -> None:
    assert len(dataset) == 2
    assert [case.test_case_id for case in dataset] == ["001", "002"]


@pytest.mark.parametrize("test_case_id", ["", "1", "abc", "01"])
def test_test_case_ids_must_be_zero_padded_numbers(test_case_id: str) -> None:
    with pytest.raises(ValidationError):
        TestCase(
            test_case_id=test_case_id,
            task="t",
            format=SolutionFormat.JSON,
            solution_criteria="c",
        )


def test_unknown_solution_format_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TestCase(
            test_case_id="001",
            task="t",
            format="yaml",
            solution_criteria="c",
        )


def test_report_indexes_rows_by_test_case_id(metadata: RunMetadata) -> None:
    report = Report[GeneratedSolution](
        metadata=metadata,
        results=[
            GeneratedSolution(
                test_case_id="007",
                task="t",
                format=SolutionFormat.REGEX,
                solution=".*",
            )
        ],
    )
    assert list(report.by_test_case_id) == ["007"]


def test_model_grader_result_is_built_from_a_grade(make_test_case) -> None:  # type: ignore[no-untyped-def]
    test_case = make_test_case("042")
    grade = ModelGrade(
        strengths=["s"], weaknesses=["w"], reasoning="because", score=6.0
    )

    result = ModelGraderResult.from_model_grader(test_case, grade)

    assert result.test_case_id == "042"
    assert result.task == test_case.task
    assert (result.strengths, result.weaknesses, result.score) == (["s"], ["w"], 6.0)


def test_model_grade_schema_documents_the_scale() -> None:
    """The judge sees these descriptions as its output schema, so they matter."""
    schema = ModelGrade.model_json_schema()
    assert "0" in schema["properties"]["score"]["description"]
    assert "10" in schema["properties"]["score"]["description"]
