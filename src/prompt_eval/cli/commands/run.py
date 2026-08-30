from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from prompt_eval.cli.graders import both, deterministic, llm_judge
from prompt_eval.cli.prompt import run_prompt
from prompt_eval.cli.types import Grader
from prompt_eval.dataset import load_dataset
from prompt_eval.models import Dataset, TestCase

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()

DEFAULT_TEST_CASES = 3
MIN_TEST_CASES = 1


@app.command()
def run(
    prompt: Annotated[
        bool,
        typer.Option(
            "--prompt",
            help="Run a prompt through a LLM to generate solutions against test cases for evaluation",
        ),
    ] = False,
    set_baseline: Annotated[
        bool,
        typer.Option(
            "--set-baseline",
            help="Explicitly promote the latest grader results as a new baseline to compare against subsequent runs",
        ),
    ] = False,
    dataset: Annotated[
        Path, typer.Option(help="Path where the test case dataset is loaded from.")
    ] = Path("data/dataset.json"),
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
    test_cases: Dataset
    if prompt:
        if dataset.exists() and dataset.is_file():
            test_cases = load_dataset(dataset)
            run_prompt(test_cases)
        raise typer.Exit()

    if dataset.exists() and dataset.is_file():
        test_cases = load_dataset(dataset)

        if grader == Grader.DETERMINISTIC:
            deterministic(test_cases, grader)
        elif grader == Grader.LLM_JUDGE:
            llm_judge(test_cases, grader)
        else:
            both(test_cases, verbose)
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
