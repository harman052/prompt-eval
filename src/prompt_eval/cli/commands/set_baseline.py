"""``prompt-eval set-baseline`` - promote the current results to the baseline."""

from __future__ import annotations

from prompt_eval.cli.options import handle_errors, require_file
from prompt_eval.models import CombinedReport
from prompt_eval.paths import COMBINED_RESULTS_FILE, DEFAULT_BASELINE_FILE
from prompt_eval.reporting import print_success
from prompt_eval.storage import load_model, save_model


@handle_errors
def set_baseline() -> None:
    """Set the current combined results as the new comparison baseline."""
    require_file(
        COMBINED_RESULTS_FILE,
        hint="Produce results first with: [bold]prompt-eval evaluate[/bold]",
    )

    # Load-then-save rather than a file copy: it validates the artifact before
    # it becomes the thing every future run is judged against, so a corrupt
    # results file cannot poison the baseline.
    report = load_model(CombinedReport, COMBINED_RESULTS_FILE)
    save_model(report, DEFAULT_BASELINE_FILE)

    print_success(
        f"Baseline set from {len(report.results)} test case(s) "
        f"({DEFAULT_BASELINE_FILE})"
    )
