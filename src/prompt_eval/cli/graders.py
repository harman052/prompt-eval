import typer
from rich.console import Console
from rich.table import Table

from prompt_eval.cli.constants import (
    COMBINED_RESULTS_FILE,
    DETERMINISTIC_RESULTS_FILE,
    MODEL_RESULTS_FILE,
)
from prompt_eval.cli.utils import (
    err_console,
    generate_numbered_list,
    print_table,
    save_file,
)
from prompt_eval.graders.deterministic_grader import deterministic_grader
from prompt_eval.graders.model_grader import ModelGrader
from prompt_eval.llm import LLMClient
from prompt_eval.models import (
    CombinedResult,
    CombinedResults,
    Dataset,
    DeterministicGraderResult,
    DeterministicGraderResults,
    ModelGraderResult,
    ModelGraderResults,
    Solutions,
)

console = Console()


def _get_solutions_by_id(solutions: Solutions) -> dict[str, str]:
    """Build a lookup table for solutions keyed by test case ID."""
    solutions_by_id: dict[str, str] = {}

    for solution in solutions.root:
        if solution.test_case_id in solutions_by_id:
            raise ValueError(
                f"Duplicate solution found for test case '{solution.test_case_id}'."
            )

        solutions_by_id[solution.test_case_id] = solution.solution

    return solutions_by_id


def _get_solution(
    solutions_by_id: dict[str, str],
    test_case_id: str,
) -> str:
    """Return a solution for a test case or raise a descriptive error."""
    try:
        return solutions_by_id[test_case_id]
    except KeyError as exc:
        raise ValueError(
            f"No solution found for test case '{test_case_id}'. "
            "Make sure the solutions file matches the dataset."
        ) from exc


def _print_deterministic_results(
    results: DeterministicGraderResults,
) -> None:
    table = Table(
        "Test Case ID",
        "Test Case",
        "Format",
        "Deterministic",
        title="Deterministic Grader Scores",
        width=100,
    )

    for result in results.root:
        table.add_row(
            result.test_case_id,
            result.task,
            result.format,
            str(result.score),
        )

    print_table(table)


def _print_model_results(results: ModelGraderResults) -> None:
    table = Table(
        "Test Case ID",
        "Test Case",
        "Format",
        "Strengths",
        "Weaknesses",
        "Reasoning",
        "LLM-Judge",
    )

    for result in results.root:
        table.add_row(
            result.test_case_id,
            result.task,
            result.format,
            generate_numbered_list(result.strengths),
            generate_numbered_list(result.weaknesses),
            result.reasoning,
            str(result.score),
        )

    print_table(table)


def _build_combined_results(
    dataset: Dataset,
    deterministic_results: DeterministicGraderResults,
    model_results: ModelGraderResults,
) -> CombinedResults:
    deterministic_by_id = {
        result.test_case_id: result for result in deterministic_results.root
    }
    model_by_id = {result.test_case_id: result for result in model_results.root}

    results = CombinedResults(root=[])

    for test_case in dataset.root:
        try:
            deterministic_result = deterministic_by_id[test_case.id]
            model_result = model_by_id[test_case.id]
        except KeyError as exc:
            raise ValueError(
                f"Missing grader result for test case '{test_case.id}'."
            ) from exc

        final_score = (deterministic_result.score + model_result.score) / 2

        results.root.append(
            CombinedResult(
                test_case_id=test_case.id,
                task=test_case.task,
                format=test_case.format,
                strengths=model_result.strengths,
                weaknesses=model_result.weaknesses,
                reasoning=model_result.reasoning,
                deterministic_score=deterministic_result.score,
                llm_judge_score=model_result.score,
                final_score=final_score,
            )
        )

    return results


def _print_combined_results(
    results: CombinedResults,
    verbose: bool,
) -> None:
    if verbose:
        table = Table(
            "Test Case ID",
            "Test Case",
            "Format",
            "Strengths",
            "Weaknesses",
            "Reasoning",
            "Deterministic",
            "LLM-Judge",
            "Final Score",
            title="Combined Scores (Deterministic and LLM as Judge)",
        )

        for result in results.root:
            table.add_row(
                result.test_case_id,
                result.task,
                result.format,
                generate_numbered_list(result.strengths or []),
                generate_numbered_list(result.weaknesses or []),
                result.reasoning or "",
                str(result.deterministic_score),
                str(result.llm_judge_score),
                str(result.final_score),
            )
    else:
        table = Table(
            "Test Case ID",
            "Test Case",
            "Format",
            "Deterministic",
            "LLM-Judge",
            "Final Score",
            title="Combined Scores (Deterministic and LLM as Judge)",
            width=100,
        )

        for result in results.root:
            table.add_row(
                result.test_case_id,
                result.task,
                result.format,
                str(result.deterministic_score),
                str(result.llm_judge_score),
                str(result.final_score),
            )

    print_table(table)


def calculate_average_score(scores: list[float]) -> float:
    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def exit_with_non_zero_code(threshold: float, average: float):
    err_console.print(f"Threshhold: {threshold}")
    err_console.print("\n[bold red]Exit code: 1[/bold red]\n")
    raise typer.Exit(code=1)


def deterministic(
    dataset: Dataset,
    solutions: Solutions,
    display_results: bool = True,
    fail_under: float | None = None,
) -> DeterministicGraderResults:
    solutions_by_id = _get_solutions_by_id(solutions)
    results = DeterministicGraderResults(root=[])

    for test_case in dataset.root:
        solution = _get_solution(solutions_by_id, test_case.id)

        results.root.append(
            DeterministicGraderResult(
                test_case_id=test_case.id,
                task=test_case.task,
                format=test_case.format,
                score=deterministic_grader(test_case, solution),
            )
        )

    average = calculate_average_score([result.score for result in results.root])

    if display_results:
        _print_deterministic_results(results)
        console.print(f"Average score: {average:.2f}")

    if fail_under != None and average < fail_under:
        exit_with_non_zero_code(fail_under, average)

    save_file(results, DETERMINISTIC_RESULTS_FILE)
    return results


def llm_judge(
    dataset: Dataset,
    solutions: Solutions,
    display_results: bool = True,
    fail_under: float | None = None,
) -> ModelGraderResults:
    model_grader = ModelGrader(LLMClient())
    solutions_by_id = _get_solutions_by_id(solutions)
    results = ModelGraderResults(root=[])

    for test_case in dataset.root:
        solution = _get_solution(solutions_by_id, test_case.id)
        response = model_grader.grade(test_case, solution)

        results.root.append(
            ModelGraderResult(
                test_case_id=test_case.id,
                task=test_case.task,
                format=test_case.format,
                strengths=response.strengths,
                weaknesses=response.weaknesses,
                reasoning=response.reasoning,
                score=response.score,
            )
        )

    average = calculate_average_score([result.score for result in results.root])

    if display_results:
        _print_model_results(results)
        console.print(f"Average score: {average:.2f}")

    if fail_under != None and average < fail_under:
        exit_with_non_zero_code(fail_under, average)

    save_file(results, MODEL_RESULTS_FILE)
    return results


def both(
    dataset: Dataset,
    solutions: Solutions,
    fail_under: float | None,
    verbose: bool,
) -> CombinedResults:
    with console.status("Getting Deterministic Scores..."):
        deterministic_results = deterministic(dataset, solutions, False)

    console.print("✓ Getting Deterministic Scores")

    with console.status("Getting LLM-Judge Scores..."):
        model_results = llm_judge(dataset, solutions, False)

    console.print("✓ Getting LLM-Judge Scores")

    results = _build_combined_results(
        dataset,
        deterministic_results,
        model_results,
    )

    average = calculate_average_score([result.final_score for result in results.root])

    _print_combined_results(results, verbose)
    console.print(f"Average final score: {average:.2f}")

    if fail_under != None and average < fail_under:
        exit_with_non_zero_code(fail_under, average)

    save_file(results, COMBINED_RESULTS_FILE)

    return results
