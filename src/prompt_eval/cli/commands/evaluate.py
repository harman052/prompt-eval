from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from prompt_eval.cli.constants import (
    DEFAULT_DATASET_FILE,
    DEFAULT_OUTPUTS_FILE,
)
from prompt_eval.cli.graders import both, deterministic, llm_judge
from prompt_eval.cli.types import Grader
from prompt_eval.cli.utils import load_file, print_dataset_error
from prompt_eval.models import Dataset, Solutions

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()


@app.command()
def evaluate(
    dataset: Annotated[
        Path, typer.Option(help="Path where the test dataset is loaded from.")
    ] = Path(DEFAULT_DATASET_FILE),
    grader: Annotated[
        Grader,
        typer.Option(
            help=("Specify the grader to run"),
        ),
    ] = Grader.BOTH,
    fail_under: Annotated[
        float | None,
        typer.Option(
            "--fail-under",
            help=(
                "Exit with a non-zero status if the average score across all "
                "test cases falls below this value. Uses the Final (blended) "
                "score when both graders ran, or the single grader's score "
                "when --grader restricts to one."
            ),
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Display detailed LLM-Judge reasoning.",
        ),
    ] = False,
) -> None:
    """Grade existing outputs using one or more graders."""
    if not dataset.is_file():
        print_dataset_error(dataset)
        raise typer.Exit(code=2)

    try:
        solutions = load_file(Solutions, DEFAULT_OUTPUTS_FILE)
        test_cases = load_file(Dataset, dataset)
    except Exception as exc:
        err_console.print(f"[bold red]Failed to load evaluation data:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        if grader == Grader.DETERMINISTIC:
            deterministic(test_cases, solutions, True, fail_under)

        elif grader == Grader.LLM_JUDGE:
            llm_judge(test_cases, solutions, True, fail_under)

        else:
            both(test_cases, solutions, fail_under, verbose)

    except ValueError as exc:
        err_console.print(f"[bold red]Evaluation failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
