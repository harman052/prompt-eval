from enum import Enum
from typing import Annotated

import typer
from rich.console import Console

from prompt_eval.cli.graders import deterministic

console = Console()

app = typer.Typer()


class Grader(str, Enum):
    deterministic = "deterministic"
    llm_judge = "llm-judge"
    both = "both"


@app.command()
def run(
    grader: Annotated[
        Grader,
        typer.Option(
            help="Runs only the deterministic grader, only the LLM judge, or both side by side. Useful to a reviewer specifically because it lets them see the two grading strategies independently and compare them"
        ),
    ] = Grader.both,
):
    if grader == Grader.deterministic:
        deterministic()
    elif grader == Grader.llm_judge:
        print("Selected grader is LLM")
    else:
        print("Selected grader is both")
