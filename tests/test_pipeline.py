"""Pipeline stages: joining, aggregation, and comparison semantics."""

from __future__ import annotations

import pytest

from prompt_eval.errors import DatasetError, LLMError
from prompt_eval.models import (
    CombinedReport,
    CombinedResult,
    Dataset,
    DatasetSpec,
    SolutionSpec,
    ModelGrade,
    RunMetadata,
    SolutionFormat,
    SolutionReport,
    TestCaseSpec,
)
from prompt_eval.pipeline import (
    average_score,
    combine_reports,
    compare_to_baseline,
    count_regressions,
    generate_dataset,
    generate_solutions,
    index_solutions,
    score_deterministic,
    score_with_judge,
    unmatched_test_case_ids,
)
from tests.conftest import FakeLLMClient


def combined(
    metadata: RunMetadata, scores: dict[str, tuple[float, float]]
) -> CombinedReport:
    return CombinedReport(
        metadata=metadata,
        results=[
            CombinedResult(
                test_case_id=test_case_id,
                task=f"task {test_case_id}",
                format=SolutionFormat.PYTHON,
                strengths=[],
                weaknesses=[],
                reasoning="because",
                deterministic_score=deterministic,
                llm_judge_score=judge,
            )
            for test_case_id, (deterministic, judge) in scores.items()
        ],
    )


# --------------------------------------------------------------------------- #
# average_score
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("scores", "expected"),
    [([], 0.0), ([5.0], 5.0), ([0.0, 10.0], 5.0), ([1.0, 2.0, 3.0], 2.0)],
)
def test_average_score(scores: list[float], expected: float) -> None:
    assert average_score(scores) == expected


def test_average_score_consumes_a_generator_once() -> None:
    assert average_score(x for x in [2.0, 4.0]) == 3.0


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


async def test_generate_dataset_assigns_sequential_ids() -> None:
    specs = [
        TestCaseSpec(
            task=f"task {i}", format=SolutionFormat.JSON, solution_criteria="c"
        )
        for i in range(3)
    ]
    llm = FakeLLMClient(responses={DatasetSpec: DatasetSpec(test_cases=specs)})

    dataset = await generate_dataset(llm, 3)  # type: ignore[arg-type]

    assert [case.test_case_id for case in dataset.root] == ["001", "002", "003"]
    assert "exactly 3" in llm.prompts[0]


async def test_generate_dataset_rejects_a_short_response() -> None:
    """A model that returns fewer cases than asked must not silently pass."""
    llm = FakeLLMClient(
        responses={
            DatasetSpec: DatasetSpec(
                test_cases=[
                    TestCaseSpec(
                        task="t", format=SolutionFormat.JSON, solution_criteria="c"
                    )
                ]
            )
        }
    )
    with pytest.raises(DatasetError, match="Requested 3 test case"):
        await generate_dataset(llm, 3)  # type: ignore[arg-type]


async def test_generate_solutions_covers_every_test_case(
    dataset: Dataset, fake_llm: FakeLLMClient
) -> None:
    report = await generate_solutions(fake_llm, dataset)  # type: ignore[arg-type]

    assert [row.test_case_id for row in report.results] == ["001", "002"]
    assert all(row.solution == "x = 1" for row in report.results)


async def test_generate_solutions_renders_the_format_into_the_prompt(
    dataset: Dataset, fake_llm: FakeLLMClient
) -> None:
    await generate_solutions(fake_llm, dataset)  # type: ignore[arg-type]
    assert any("valid json source" in prompt for prompt in fake_llm.prompts)
    assert any("valid python source" in prompt for prompt in fake_llm.prompts)


async def test_generate_solutions_propagates_llm_failures(dataset: Dataset) -> None:
    """A partial solution set would silently shrink every later average."""
    llm = FakeLLMClient(responses={SolutionSpec: LLMError("upstream down")})
    with pytest.raises(LLMError, match="upstream down"):
        await generate_solutions(llm, dataset)  # type: ignore[arg-type]


async def test_generate_solutions_bounds_concurrency(
    dataset: Dataset, fake_llm: FakeLLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    big = Dataset(
        root=[
            case.model_copy(update={"test_case_id": f"{index:03d}"})
            for index, case in enumerate(dataset.root * 10, start=1)
        ]
    )
    await generate_solutions(fake_llm, big)  # type: ignore[arg-type]
    assert fake_llm.max_in_flight <= 4  # MAX_CONCURRENCY from the settings fixture


# --------------------------------------------------------------------------- #
# index_solutions
# --------------------------------------------------------------------------- #


def test_index_solutions_maps_ids_to_bodies(
    dataset: Dataset, solutions: SolutionReport
) -> None:
    assert index_solutions(solutions, dataset) == {"001": "x = 1", "002": '{"a": 1}'}


def test_index_solutions_rejects_duplicates(
    dataset: Dataset, solutions: SolutionReport
) -> None:
    duplicated = solutions.model_copy(
        update={"results": [*solutions.results, solutions.results[0]]}
    )
    with pytest.raises(DatasetError, match="Duplicate solution"):
        index_solutions(duplicated, dataset)


def test_index_solutions_rejects_a_missing_solution(
    dataset: Dataset, solutions: SolutionReport
) -> None:
    partial = solutions.model_copy(update={"results": solutions.results[:1]})
    with pytest.raises(DatasetError, match="No solution for test case"):
        index_solutions(partial, dataset)


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def test_score_deterministic_scores_each_format(
    dataset: Dataset, solutions: SolutionReport
) -> None:
    report = score_deterministic(dataset, solutions)
    assert [row.score for row in report.results] == [10.0, 10.0]


def test_score_deterministic_penalises_invalid_syntax(
    dataset: Dataset, solutions: SolutionReport
) -> None:
    broken = solutions.model_copy(
        update={
            "results": [
                solutions.results[0].model_copy(update={"solution": "def f(:"}),
                solutions.results[1],
            ]
        }
    )
    report = score_deterministic(dataset, broken)
    assert [row.score for row in report.results] == [0.0, 10.0]


async def test_score_with_judge_attaches_grades_to_test_cases(
    dataset: Dataset, solutions: SolutionReport, fake_llm: FakeLLMClient
) -> None:
    report = await score_with_judge(fake_llm, dataset, solutions)  # type: ignore[arg-type]

    assert [row.test_case_id for row in report.results] == ["001", "002"]
    assert all(row.score == 8.0 for row in report.results)
    assert all("It must run" in prompt for prompt in fake_llm.prompts)


async def test_score_with_judge_propagates_failures(
    dataset: Dataset, solutions: SolutionReport
) -> None:
    llm = FakeLLMClient(responses={ModelGrade: LLMError("judge unavailable")})
    with pytest.raises(LLMError, match="judge unavailable"):
        await score_with_judge(llm, dataset, solutions)  # type: ignore[arg-type]


async def test_combine_reports_averages_both_graders(
    dataset: Dataset, solutions: SolutionReport, fake_llm: FakeLLMClient
) -> None:
    deterministic = score_deterministic(dataset, solutions)
    judge = await score_with_judge(fake_llm, dataset, solutions)  # type: ignore[arg-type]

    report = combine_reports(dataset, deterministic, judge)

    assert [row.final_score for row in report.results] == [9.0, 9.0]


async def test_combine_reports_rejects_a_missing_row(
    dataset: Dataset, solutions: SolutionReport, fake_llm: FakeLLMClient
) -> None:
    deterministic = score_deterministic(dataset, solutions)
    judge = await score_with_judge(fake_llm, dataset, solutions)  # type: ignore[arg-type]
    truncated = judge.model_copy(update={"results": judge.results[:1]})

    with pytest.raises(DatasetError, match="Missing grader result"):
        combine_reports(dataset, deterministic, truncated)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def test_compare_joins_by_test_case_id_not_position(metadata: RunMetadata) -> None:
    """Reordering the dataset must not manufacture a regression."""
    baseline = combined(metadata, {"001": (10.0, 10.0), "002": (0.0, 0.0)})
    reordered = combined(metadata, {"002": (0.0, 0.0), "001": (10.0, 10.0)})

    report = compare_to_baseline(baseline, reordered)

    assert {row.test_case_id: row.delta for row in report.results} == {
        "001": 0.0,
        "002": 0.0,
    }


def test_compare_ignores_test_cases_missing_from_either_side(
    metadata: RunMetadata,
) -> None:
    baseline = combined(metadata, {"001": (10.0, 10.0), "002": (10.0, 10.0)})
    current = combined(metadata, {"002": (10.0, 8.0), "003": (2.0, 2.0)})

    report = compare_to_baseline(baseline, current)

    assert [row.test_case_id for row in report.results] == ["002"]
    assert unmatched_test_case_ids(baseline, current) == (["001"], ["003"])


def test_compare_rejects_a_completely_disjoint_baseline(metadata: RunMetadata) -> None:
    baseline = combined(metadata, {"001": (10.0, 10.0)})
    current = combined(metadata, {"009": (10.0, 10.0)})
    with pytest.raises(DatasetError, match="no test cases in common"):
        compare_to_baseline(baseline, current)


def test_delta_sign_follows_the_direction_of_change(metadata: RunMetadata) -> None:
    baseline = combined(metadata, {"001": (10.0, 4.0)})  # final 7.0
    current = combined(metadata, {"001": (10.0, 8.0)})  # final 9.0
    (row,) = compare_to_baseline(baseline, current).results
    assert row.delta == pytest.approx(2.0)
    assert not row.is_regression


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.0, 1), (1.0, 1), (2.0, 0), (5.0, 0)],
)
def test_count_regressions_is_exclusive_of_the_threshold(
    metadata: RunMetadata, threshold: float, expected: int
) -> None:
    """A 2.0 drop is not a regression at a 2.0 threshold - only worse ones are."""
    baseline = combined(metadata, {"001": (10.0, 10.0)})
    current = combined(metadata, {"001": (10.0, 6.0)})  # final drops by 2.0
    report = compare_to_baseline(baseline, current)
    assert count_regressions(report, threshold) == expected
