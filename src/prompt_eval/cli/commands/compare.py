from datetime import UTC, datetime
from pathlib import Path

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
    load_file,
    print_table,
    save_file,
)
from prompt_eval.models import CombinedResults, ComparisonResult, ComparisonResults

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()


def count_regressions(results: ComparisonResults):
    count: list[ComparisonResult] = [r for r in results.root if r.delta < 0]
    return len(count)


def display_regression_summary(regression_count: int):
    message = f"[bold]{regression_count} {'regression' if regression_count >= 1 else 'regressions'} detected.[/bold]\n"
    console.print(message)


def print_comparison_results(
    results: ComparisonResults,
) -> None:

    table = Table(
        "Test Case",
        "Baseline",
        "Current",
        "Delta",
        title="Baseline Comparison",
        width=100,
    )

    for result in results.root:
        table.add_row(
            result.task,
            str(result.baseline_score),
            str(result.current_score),
            str(format_delta(result.delta)),
        )

    print_table(table)
    count = count_regressions(results)
    display_regression_summary(count)


def persist_comparison_results(results: ComparisonResults):
    try:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        file_name = Path(f"{timestamp}.json")
        path = Path(COMPARISON_RESULTS_DIR / file_name)
        save_file(results, path)
        console.print(f"Comparison written to {path}\n")
    except (OSError, ValueError) as exc:
        err_console.print(
            f"[bold red]failed to save comparison results[bold red]: {exc}"
        )


@app.command()
def compare():
    """
    Diffs baseline vs. current
    """
    results = ComparisonResults(root=[])
    try:
        baseline = load_file(CombinedResults, DEFAULT_BASELINE_FILE)
        current = load_file(CombinedResults, DEFAULT_COMBINED_RESULTS_FILE)

        for baseline_scores, current_score in zip(
            baseline.root, current.root, strict=True
        ):
            results.root.append(
                ComparisonResult(
                    task=baseline_scores.task,
                    baseline_score=baseline_scores.final_score,
                    current_score=current_score.final_score,
                    delta=calcuate_delta(
                        baseline_scores.final_score, current_score.final_score
                    ),
                )
            )

        print_comparison_results(results)
        persist_comparison_results(results)

    except Exception as exc:
        err_console.print(f"[bold red]Comparison failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
