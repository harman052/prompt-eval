"""``prompt-eval evaluate`` - grade existing solutions."""

from __future__ import annotations

import typer

from prompt_eval.cli.options import (
    DatasetOption,
    FailUnderOption,
    VerboseOption,
    handle_errors,
    require_file,
    run_async,
    status,
)
from prompt_eval.cli.types import GraderChoice
from prompt_eval.constants import EXIT_GATE_FAILED
from prompt_eval.llm import LLMClient
from prompt_eval.models import (
    CombinedReport,
    Dataset,
    ModelGraderReport,
    SolutionReport,
)
from prompt_eval.paths import (
    COMBINED_RESULTS_FILE,
    DEFAULT_DATASET_FILE,
    DEFAULT_SOLUTIONS_FILE,
    DETERMINISTIC_RESULTS_FILE,
    MODEL_RESULTS_FILE,
)
from prompt_eval.pipeline import (
    average_score,
    combine_reports,
    score_deterministic,
    score_with_judge,
)
from prompt_eval.reporting import (
    err_console,
    print_average,
    print_combined_report,
    print_deterministic_report,
    print_judge_report,
)
from prompt_eval.storage import load_model, save_model


@handle_errors
def evaluate(
    dataset: DatasetOption = DEFAULT_DATASET_FILE,
    grader: GraderChoice = GraderChoice.BOTH,
    fail_under: FailUnderOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Grade existing solutions using one or more graders."""
    require_file(dataset, hint="Create one with: [bold]prompt-eval init-dataset[/bold]")
    require_file(
        DEFAULT_SOLUTIONS_FILE,
        hint="Generate solutions with: [bold]prompt-eval generate[/bold]",
    )

    test_cases = load_model(Dataset, dataset)
    solutions = load_model(SolutionReport, DEFAULT_SOLUTIONS_FILE)

    match grader:
        case GraderChoice.DETERMINISTIC:
            average = _run_deterministic(test_cases, solutions)
        case GraderChoice.LLM_JUDGE:
            average = _run_judge(test_cases, solutions)
        case GraderChoice.BOTH:
            average = _run_both(test_cases, solutions, verbose=verbose)

    _enforce_gate(average, fail_under)


def _run_deterministic(test_cases: Dataset, solutions: SolutionReport) -> float:
    report = score_deterministic(test_cases, solutions)
    save_model(report, DETERMINISTIC_RESULTS_FILE)
    print_deterministic_report(report)
    average = average_score(result.score for result in report.results)
    print_average("Average score", average)
    return average


def _run_judge(test_cases: Dataset, solutions: SolutionReport) -> float:
    with status(f"Grading {len(test_cases)} solution(s) with the LLM judge"):
        report = run_async(_judge(test_cases, solutions))
    save_model(report, MODEL_RESULTS_FILE)
    print_judge_report(report)
    average = average_score(result.score for result in report.results)
    print_average("Average score", average)
    return average


def _run_both(
    test_cases: Dataset, solutions: SolutionReport, *, verbose: bool
) -> float:
    with status("Scoring solution syntax"):
        deterministic_report = score_deterministic(test_cases, solutions)
    save_model(deterministic_report, DETERMINISTIC_RESULTS_FILE)

    with status(f"Grading {len(test_cases)} solution(s) with the LLM judge"):
        judge_report = run_async(_judge(test_cases, solutions))
    save_model(judge_report, MODEL_RESULTS_FILE)

    combined = combine_reports(test_cases, deterministic_report, judge_report)
    save_model(combined, COMBINED_RESULTS_FILE)

    print_combined_report(combined, verbose=verbose)
    average = average_score(result.final_score for result in combined.results)
    print_average("Average final score", average)
    return average


async def _judge(test_cases: Dataset, solutions: SolutionReport) -> ModelGraderReport:
    async with LLMClient() as llm:
        return await score_with_judge(llm, test_cases, solutions)


def _enforce_gate(average: float, fail_under: float | None) -> None:
    """Fail the run if the average score is below the configured floor.

    Check for `None` so 0 is still accepted as a threshold.
    """
    if fail_under is None or average >= fail_under:
        return
    err_console.print(
        f"[bold red]Average score {average:.2f} is below the "
        f"--fail-under threshold of {fail_under:.2f}.[/bold red]"
    )
    raise typer.Exit(code=EXIT_GATE_FAILED)


__all__ = ["CombinedReport", "evaluate"]
