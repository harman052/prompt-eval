from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from prompt_eval.cli.constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUTS_PATH,
    DEFAULT_TEST_CASES,
    MIN_TEST_CASES,
)
from prompt_eval.cli.graders import both, deterministic, llm_judge
from prompt_eval.cli.types import Grader
from prompt_eval.cli.utils import load_file
from prompt_eval.models import Dataset, Solutions

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()


@app.command()
def evaluate(
    dataset: Annotated[
        Path, typer.Option(help="Path where the test dataset is loaded from.")
    ] = Path(DEFAULT_DATASET_PATH),
    grader: Annotated[
        Grader,
        typer.Option(
            help=("Specify the grader to run"),
        ),
    ] = Grader.BOTH,
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
        _print_dataset_error(dataset)
        raise typer.Exit(code=2)

    try:
        solutions = load_file(Solutions, DEFAULT_OUTPUTS_PATH)
        test_cases = load_file(Dataset, dataset)
    except Exception as exc:
        err_console.print(f"[bold red]Failed to load evaluation data:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        if grader == Grader.DETERMINISTIC:
            deterministic(test_cases, solutions, display_results=True)

        elif grader == Grader.LLM_JUDGE:
            llm_judge(test_cases, solutions, display_results=True)

        else:
            both(test_cases, solutions, verbose)

    except ValueError as exc:
        err_console.print(f"[bold red]Evaluation failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


def _print_dataset_error(dataset: Path) -> None:
    err_console.print(f"\n[bold red]Test dataset not found:[/bold red] {dataset}\n")
    err_console.print(
        "Generate a new dataset with:\n"
        "[bold]prompt-eval init-dataset "
        f"--num-cases {MIN_TEST_CASES}[/bold]"
    )
    err_console.print(
        f"\nThe minimum number of test cases is {MIN_TEST_CASES}; "
        f"the default is {DEFAULT_TEST_CASES}."
    )
    err_console.print(
        "\nFor detailed help, use: [bold]prompt-eval evaluate --help[/bold]"
    )
