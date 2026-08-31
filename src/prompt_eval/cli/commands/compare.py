from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from prompt_eval.cli.constants import (
    COMPARISON_RESULTS_DIR,
    DEFAULT_BASELINE_FILE,
    DEFAULT_COMBINED_RESULTS_FILE,
)
from prompt_eval.cli.utils import (
    calcuate_delta,
    format_delta,
    get_prompt_metadata,
    load_file,
    print_table,
    save_file,
)
from prompt_eval.models import CombinedResults, ComparisonResult, ComparisonResults

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()


def count_regressions(results: ComparisonResults, threshold: float):
    count: list[ComparisonResult] = [r for r in results.results if r.delta < -threshold]
    return len(count)


def display_regression_summary(regression_count: int, threshold: float):
    message = "[bold green]✓ No regressions found.[/bold green]\n"
    if regression_count > 0:
        message = f"[bold red]{regression_count} {'regression' if regression_count >= 1 else 'regressions'} detected[/bold red] (threshold: {threshold:.2f})\n"
        err_console.print(message)
    else:
        console.print(message)


def print_comparison_results(results: ComparisonResults) -> None:

    table = Table(
        "Test Case ID",
        "Test Case",
        "Baseline",
        "Current",
        "Delta",
        title="Baseline Comparison",
        width=100,
    )

    for result in results.results:
        table.add_row(
            result.test_case_id,
            result.task,
            str(result.baseline_score),
            str(result.current_score),
            str(format_delta(result.delta)),
        )

    print_table(table)


def persist_comparison_results(results: ComparisonResults):
    try:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        file_name = Path(f"{timestamp}.json")
        path = Path(COMPARISON_RESULTS_DIR / file_name)
        save_file(results, path)
        console.print(f"[bold]Comparison written to {path}[bold]\n")
    except (OSError, ValueError) as exc:
        err_console.print(
            f"[bold red]failed to save comparison results[bold red]: {exc}"
        )


@app.command()
def compare(
    regression_threshold: Annotated[
        float | None,
        typer.Option(
            help=(
                "Exit with a non-zero status if any test case's score drops by "
                "more than this amount compared to the baseline. If unset, "
                "regressions are still reported but never cause a non-zero exit."
            ),
        ),
    ] = None,
):
    """
    Diffs baseline vs. current
    """
    results: ComparisonResults = ComparisonResults(
        metadata=get_prompt_metadata(),
        results=[],
    )
    try:
        baseline = load_file(CombinedResults, DEFAULT_BASELINE_FILE)
        current = load_file(CombinedResults, DEFAULT_COMBINED_RESULTS_FILE)

        for baseline_results, current_results in zip(
            baseline.results, current.results, strict=True
        ):
            results.results.append(
                ComparisonResult(
                    test_case_id=current_results.test_case_id,
                    task=baseline_results.task,
                    baseline_score=baseline_results.final_score,
                    current_score=current_results.final_score,
                    delta=calcuate_delta(
                        baseline_results.final_score, current_results.final_score
                    ),
                )
            )

        print_comparison_results(results)

        if regression_threshold:
            regression_count = count_regressions(results, regression_threshold)
            display_regression_summary(regression_count, regression_threshold)
            if regression_count > 0:
                raise typer.Exit(code=1)

        persist_comparison_results(results)

    except (OSError, ValueError) as exc:
        err_console.print(f"[bold red]Comparison failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
