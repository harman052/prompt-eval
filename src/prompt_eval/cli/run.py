from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from prompt_eval.cli.graders import both, deterministic, llmJudge
from prompt_eval.cli.types import Grader
from prompt_eval.dataset import load_test_cases

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()


@app.command()
def run(
    dataset: Annotated[
        Path, typer.Option(help="Path where the test case dataset is loaded from")
    ] = Path("data/dataset.json"),
    grader: Annotated[
        Grader,
        typer.Option(
            help="Runs only the deterministic grader, only the LLM judge, or both side by side. Useful to a reviewer specifically because it lets them see the two grading strategies independently and compare them"
        ),
    ] = Grader.BOTH,
):
    if dataset.exists() and dataset.is_file():
        test_cases = load_test_cases(dataset)

        if grader == Grader.DETERMINISTIC:
            deterministic(test_cases, grader)
        elif grader == Grader.LLM_JUDGE:
            llmJudge(test_cases, grader)
        else:
            both(test_cases)
    else:
        err_console.print("\n[bold red]Test dataset path is invalid[/bold red]\n")
        raise typer.Exit(code=2)
