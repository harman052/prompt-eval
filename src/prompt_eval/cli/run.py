from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from prompt_eval.cli.graders import both, deterministic, llm_judge
from prompt_eval.cli.types import Grader
from prompt_eval.dataset import generate_dataset, load_test_cases
from prompt_eval.models import TestCase

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()


@app.command()
def run(
    dataset: Annotated[
        Path, typer.Option(help="Path where the test case dataset is loaded from.")
    ] = Path("data/dataset.json"),
    grader: Annotated[
        Grader,
        typer.Option(
            help="Runs only the deterministic grader, only the LLM judge, or both side by side. Useful to a reviewer specifically because it lets them see the two grading strategies independently and compare them."
        ),
    ] = Grader.BOTH,
    regenerate: Annotated[
        bool,
        typer.Option(
            "--regenerate",
            help="Forces dataset regeneration even if data/dataset.json already exists. Use --num-cases to specify the number of test cases to generate.",
        ),
    ] = False,
    num_cases: Annotated[
        int,
        typer.Option(
            "--num-cases",
            help="Number of test cases to generate. Use it with --regenerate flag. Minimum value: 1",
            min=1,
        ),
    ] = 3,
):
    test_cases: list[TestCase]
    if regenerate:
        with console.status(f"Generating new dataset with {num_cases} test cases..."):
            test_cases = generate_dataset(num_cases)

            console.print(
                f"\n[bold green]✓ Dataset generated with {num_cases} test cases at `data/dataset.json`.[/bold green]\n"
            )
        raise typer.Exit()

    if dataset.exists() and dataset.is_file():
        test_cases = load_test_cases(dataset)

        if grader == Grader.DETERMINISTIC:
            deterministic(test_cases, grader)
        elif grader == Grader.LLM_JUDGE:
            llm_judge(test_cases, grader)
        else:
            both(test_cases)
    else:
        err_console.print(
            "\n[bold]Test dataset is not found or path is invalid[/bold]\n"
        )
        err_console.print(
            "Generate new dataset with [bold]--regenerate[/bold] flag (default test cases: 5).\nUse [bold]--num-cases[/bold] along with [bold]--regenerate[/bold] to generate arbirary number of test cases\n"
        )
        err_console.print(
            "For detailed help, use: [bold]prompt-eval run --help[/bold]\n"
        )
        raise typer.Exit(code=2)
