"""Pipeline stages: dataset generation, solution generation, grading, comparison."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from prompt_eval.concurrency import map_concurrently
from prompt_eval.config import get_settings
from prompt_eval.errors import DatasetError
from prompt_eval.graders.deterministic import grade_syntax
from prompt_eval.graders.judge import grade_with_judge
from prompt_eval.llm import LLMClient
from prompt_eval.models import (
    CombinedReport,
    CombinedResult,
    ComparisonReport,
    ComparisonResult,
    Dataset,
    DatasetSpec,
    DeterministicReport,
    DeterministicResult,
    GeneratedSolution,
    ModelGraderReport,
    ModelGraderResult,
    RunMetadata,
    SolutionReport,
    SolutionSpec,
    TestCase,
)
from prompt_eval.prompts import DATASET_PROMPT, SOLUTION_PROMPT, render_prompt
from prompt_eval.versioning import get_prompt_version


def build_metadata() -> RunMetadata:
    """Stamp the current prompt revision, model and time onto a report."""
    return RunMetadata(
        prompt_version=get_prompt_version(),
        model=get_settings().claude_model,
        run_at=datetime.now(UTC),
    )


def average_score(scores: Iterable[float]) -> float:
    """Mean of ``scores``; ``0.0`` for an empty sequence."""
    values = list(scores)
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


async def generate_dataset(llm: LLMClient, test_case_count: int) -> Dataset:
    """Ask the model for ``test_case_count`` test cases and assign stable ids."""
    prompt = render_prompt(DATASET_PROMPT, test_case_count=test_case_count)
    generated = await llm.parse(prompt, DatasetSpec)

    if len(generated.test_cases) != test_case_count:
        raise DatasetError(
            f"Requested {test_case_count} test case(s) but the model returned "
            f"{len(generated.test_cases)}. Re-run to try again."
        )

    return Dataset(
        root=[
            TestCase(test_case_id=f"{index:03d}", **spec.model_dump())
            for index, spec in enumerate(generated.test_cases, start=1)
        ]
    )


async def generate_solutions(llm: LLMClient, dataset: Dataset) -> SolutionReport:
    """Generate one solution per test case, concurrently."""

    async def solve(test_case: TestCase) -> GeneratedSolution:
        prompt = render_prompt(
            SOLUTION_PROMPT, task=test_case.task, format=test_case.format.value
        )
        generated = await llm.parse(prompt, SolutionSpec)
        return GeneratedSolution(
            test_case_id=test_case.test_case_id,
            task=test_case.task,
            format=test_case.format,
            solution=generated.code,
        )

    solutions = await map_concurrently(
        dataset.root, solve, limit=get_settings().max_concurrency
    )
    return SolutionReport(metadata=build_metadata(), results=solutions)


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def index_solutions(report: SolutionReport, dataset: Dataset) -> dict[str, str]:
    """Map test case id -> solution text, validating it covers the dataset.

    Both graders need this, and both need the same failure modes: a duplicate
    id would silently drop a score, and a missing one would silently shrink the
    average.
    """
    solutions_by_id: dict[str, str] = {}
    for solution in report.results:
        if solution.test_case_id in solutions_by_id:
            raise DatasetError(
                f"Duplicate solution for test case '{solution.test_case_id}'."
            )
        solutions_by_id[solution.test_case_id] = solution.solution

    missing = [
        case.test_case_id
        for case in dataset.root
        if case.test_case_id not in solutions_by_id
    ]
    if missing:
        raise DatasetError(
            f"No solution for test case(s) {', '.join(missing)}. "
            "Re-run `prompt-eval generate` so the solutions match the dataset."
        )
    return solutions_by_id


def score_deterministic(
    dataset: Dataset, solutions: SolutionReport
) -> DeterministicReport:
    """Syntax-check every solution. Pure and offline - no model calls."""
    solutions_by_id = index_solutions(solutions, dataset)
    return DeterministicReport(
        metadata=build_metadata(),
        results=[
            DeterministicResult(
                test_case_id=case.test_case_id,
                task=case.task,
                format=case.format,
                score=grade_syntax(solutions_by_id[case.test_case_id], case.format),
            )
            for case in dataset.root
        ],
    )


async def score_with_judge(
    llm: LLMClient, dataset: Dataset, solutions: SolutionReport
) -> ModelGraderReport:
    """Grade every solution with the LLM judge, concurrently."""
    solutions_by_id = index_solutions(solutions, dataset)

    async def judge(test_case: TestCase) -> ModelGraderResult:
        grade = await grade_with_judge(
            llm, test_case, solutions_by_id[test_case.test_case_id]
        )
        return ModelGraderResult.from_model_grader(test_case, grade)

    results = await map_concurrently(
        dataset.root, judge, limit=get_settings().max_concurrency
    )
    return ModelGraderReport(metadata=build_metadata(), results=results)


def combine_reports(
    dataset: Dataset,
    deterministic: DeterministicReport,
    judge: ModelGraderReport,
) -> CombinedReport:
    """Join both graders' rows per test case; ``final_score`` is their mean."""
    deterministic_by_id = deterministic.by_test_case_id
    judge_by_id = judge.by_test_case_id

    results: list[CombinedResult] = []
    for case in dataset.root:
        deterministic_result = deterministic_by_id.get(case.test_case_id)
        judge_result = judge_by_id.get(case.test_case_id)
        if deterministic_result is None or judge_result is None:
            raise DatasetError(
                f"Missing grader result for test case '{case.test_case_id}'."
            )

        results.append(
            CombinedResult(
                test_case_id=case.test_case_id,
                task=case.task,
                format=case.format,
                strengths=judge_result.strengths,
                weaknesses=judge_result.weaknesses,
                reasoning=judge_result.reasoning,
                deterministic_score=deterministic_result.score,
                llm_judge_score=judge_result.score,
            )
        )

    return CombinedReport(metadata=build_metadata(), results=results)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def compare_to_baseline(
    baseline: CombinedReport, current: CombinedReport
) -> ComparisonReport:
    """Diff current scores against the baseline, joined by test case id.

    Joining by id rather than by list position matters: reordering the dataset,
    or baselining a run with a different number of cases, would otherwise
    compare unrelated test cases (or abort on a length mismatch) and report a
    regression that never happened.
    """
    baseline_by_id = baseline.by_test_case_id
    current_by_id = current.by_test_case_id

    shared_ids = [id_ for id_ in current_by_id if id_ in baseline_by_id]
    if not shared_ids:
        raise DatasetError(
            "The baseline and the current results have no test cases in common. "
            "Re-run `prompt-eval set-baseline` after regenerating the dataset."
        )

    return ComparisonReport(
        metadata=build_metadata(),
        results=[
            ComparisonResult(
                test_case_id=id_,
                task=current_by_id[id_].task,
                format=current_by_id[id_].format,
                baseline_score=baseline_by_id[id_].final_score,
                current_score=current_by_id[id_].final_score,
            )
            for id_ in shared_ids
        ],
    )


def count_regressions(report: ComparisonReport, threshold: float) -> int:
    """Number of test cases whose score dropped by more than ``threshold``."""
    return sum(1 for result in report.results if result.delta < -threshold)


def unmatched_test_case_ids(
    baseline: CombinedReport, current: CombinedReport
) -> tuple[list[str], list[str]]:
    """Ids only in the baseline, and ids only in the current run."""
    baseline_ids = set(baseline.by_test_case_id)
    current_ids = set(current.by_test_case_id)
    return sorted(baseline_ids - current_ids), sorted(current_ids - baseline_ids)
