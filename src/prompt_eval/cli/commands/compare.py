"""``prompt-eval compare`` - diff the current results against the baseline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

import typer

from prompt_eval.cli.options import handle_errors, require_file
from prompt_eval.constants import EXIT_GATE_FAILED
from prompt_eval.models import CombinedReport
from prompt_eval.paths import (
    COMBINED_RESULTS_FILE,
    COMPARISON_RESULTS_DIR,
    DEFAULT_BASELINE_FILE,
)
from prompt_eval.pipeline import (
    average_score,
    compare_to_baseline,
    count_regressions,
    unmatched_test_case_ids,
)
from prompt_eval.reporting import (
    err_console,
    print_average,
    print_comparison_report,
    print_regression_summary,
    print_success,
)
from prompt_eval.storage import load_model, save_model

RegressionThresholdOption = Annotated[
    float | None,
    typer.Option(
        "--regression-threshold",
        min=0.0,
        help=(
            "Exit non-zero if any test case's score drops by more than this "
            "amount versus the baseline. If unset, regressions are reported "
            "but never fail the run."
        ),
    ),
]


@handle_errors
def compare(regression_threshold: RegressionThresholdOption = None) -> None:
    """Diff baseline scores against the current run."""
    require_file(
        DEFAULT_BASELINE_FILE,
        hint="Create one with: [bold]prompt-eval set-baseline[/bold]",
    )
    require_file(
        COMBINED_RESULTS_FILE,
        hint="Produce results first with: [bold]prompt-eval evaluate[/bold]",
    )

    baseline = load_model(CombinedReport, DEFAULT_BASELINE_FILE)
    current = load_model(CombinedReport, COMBINED_RESULTS_FILE)

    report = compare_to_baseline(baseline, current)
    _warn_about_unmatched(baseline, current)

    print_comparison_report(report)
    print_average("Average delta", average_score(r.delta for r in report.results))

    # Persist before the gate: a run that fails CI is exactly the run whose
    # artifact we want to inspect afterwards.
    path = save_model(report, COMPARISON_RESULTS_DIR / _timestamped_filename())
    print_success(f"Comparison written to {path}")

    if regression_threshold is None:
        return

    regressions = count_regressions(report, regression_threshold)
    print_regression_summary(regressions, regression_threshold)
    if regressions:
        raise typer.Exit(code=EXIT_GATE_FAILED)


def _warn_about_unmatched(baseline: CombinedReport, current: CombinedReport) -> None:
    """Report test cases present on only one side of the comparison."""
    dropped, added = unmatched_test_case_ids(baseline, current)
    if dropped:
        err_console.print(
            f"[yellow]⚠ Not in the current run (skipped): {', '.join(dropped)}[/yellow]"
        )
    if added:
        err_console.print(
            f"[yellow]⚠ Not in the baseline (skipped): {', '.join(added)}[/yellow]"
        )


def _timestamped_filename() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}.json"
