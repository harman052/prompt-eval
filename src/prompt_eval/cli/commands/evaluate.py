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
            help="Runs only the deterministic grader, only the LLM judge, or both side by side. Useful to a reviewer specifically because it lets them see the two grading strategies independently and compare them."
        ),
    ] = Grader.BOTH,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Displays the detailed reasoning behind the LLM-Judge grading",
        ),
    ] = False,
):
    """
    Grades existing outputs with different graders
    """
    if dataset.exists() and dataset.is_file():
        solutions = load_file(Solutions, DEFAULT_OUTPUTS_PATH)
        test_cases = load_file(Dataset, DEFAULT_DATASET_PATH)

        if grader == Grader.DETERMINISTIC:
            deterministic(test_cases, solutions, grader)
        elif grader == Grader.LLM_JUDGE:
            llm_judge(test_cases, solutions, grader)
        else:
            both(test_cases, solutions, verbose)
    else:
        err_console.print(
            "\n[bold]Test dataset is not found or path is invalid[/bold]\n"
        )
        err_console.print(
            f"Generate new dataset with command: [bold]prompt-eval init-dataset[/bold]. Use flag [bold]--num-cases[/bold] to define the number of test cases to generate (min: {MIN_TEST_CASES},  default: {DEFAULT_TEST_CASES}).\n"
        )
        err_console.print(
            "For example: [bold]prompt-eval init-dataset --num-cases 5[/bold]\n"
        )
        err_console.print(
            "For detailed help, use: [bold]prompt-eval run --help[/bold]\n"
        )
        raise typer.Exit(code=2)
