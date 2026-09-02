"""Terminal rendering."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from prompt_eval.models import (
    CombinedReport,
    ComparisonReport,
    DeterministicReport,
    ModelGraderReport,
)

console = Console()
err_console = Console(stderr=True)

TABLE_WIDTH = 100


def numbered_list(items: list[str]) -> str:
    """Render list items as a numbered block for a table cell."""
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def format_delta(delta: float) -> str:
    """Colour a score delta: green for improvement, red for regression."""
    if delta > 0:
        return f"[bold green]+{delta:.2f}[/bold green]"
    if delta < 0:
        return f"[bold red]{delta:.2f}[/bold red]"
    return "[dim]0.00[/dim]"


def _print(table: Table) -> None:
    console.print()
    console.print(table)
    console.print()


def print_average(label: str, average: float) -> None:
    console.print(f"{label}: [bold]{average:.2f}[/bold]")


def print_deterministic_report(report: DeterministicReport) -> None:
    table = Table(
        "Test Case ID",
        "Test Case",
        "Format",
        "Deterministic",
        title="Deterministic Grader Scores",
        width=TABLE_WIDTH,
    )
    for result in report.results:
        table.add_row(
            result.test_case_id, result.task, result.format, f"{result.score:.2f}"
        )
    _print(table)


def print_judge_report(report: ModelGraderReport) -> None:
    table = Table(
        "Test Case ID",
        "Test Case",
        "Format",
        "Strengths",
        "Weaknesses",
        "Reasoning",
        "LLM-Judge",
        title="LLM-Judge Scores",
    )
    for result in report.results:
        table.add_row(
            result.test_case_id,
            result.task,
            result.format,
            numbered_list(result.strengths),
            numbered_list(result.weaknesses),
            result.reasoning,
            f"{result.score:.2f}",
        )
    _print(table)


def print_combined_report(report: CombinedReport, *, verbose: bool) -> None:
    """Print combined scores, optionally including the judge's rationale."""
    title = "Combined Scores (Deterministic and LLM as Judge)"
    detail_columns = ("Strengths", "Weaknesses", "Reasoning") if verbose else ()
    table = Table(
        "Test Case ID",
        "Test Case",
        "Format",
        *detail_columns,
        "Deterministic",
        "LLM-Judge",
        "Final Score",
        title=title,
        width=None if verbose else TABLE_WIDTH,
    )

    for result in report.results:
        details = (
            (
                numbered_list(result.strengths),
                numbered_list(result.weaknesses),
                result.reasoning,
            )
            if verbose
            else ()
        )
        table.add_row(
            result.test_case_id,
            result.task,
            result.format,
            *details,
            f"{result.deterministic_score:.2f}",
            f"{result.llm_judge_score:.2f}",
            f"{result.final_score:.2f}",
        )
    _print(table)


def print_comparison_report(report: ComparisonReport) -> None:
    table = Table(
        "Test Case ID",
        "Test Case",
        "Baseline",
        "Current",
        "Delta",
        title="Baseline Comparison",
        width=TABLE_WIDTH,
    )
    for result in report.results:
        table.add_row(
            result.test_case_id,
            result.task,
            f"{result.baseline_score:.2f}",
            f"{result.current_score:.2f}",
            format_delta(result.delta),
        )
    _print(table)


def print_regression_summary(regression_count: int, threshold: float) -> None:
    if regression_count == 0:
        console.print("[bold green]✓ No regressions found.[/bold green]")
        return
    noun = "regression" if regression_count == 1 else "regressions"
    err_console.print(
        f"[bold red]{regression_count} {noun} detected[/bold red] "
        f"(threshold: {threshold:.2f})"
    )


def print_error(message: str) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str) -> None:
    console.print(f"[bold green]✓ {message}[/bold green]")
