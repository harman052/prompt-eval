from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from prompt_eval.cli.graders import deterministic
from prompt_eval.dataset import load_test_cases

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()


class Grader(str, Enum):
    deterministic = "deterministic"
    llm_judge = "llm-judge"
    both = "both"


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
    ] = Grader.both,
):
    if dataset.exists() and dataset.is_file():
        test_cases = load_test_cases(dataset)

        if grader == Grader.deterministic:
            deterministic(test_cases)
        elif grader == Grader.llm_judge:
            print("Selected grader is LLM")
        else:
            print("Selected grader is both")
    else:
        err_console.print("[red]Test dataset path is invalid[/red]")
        raise typer.Exit(code=2)
