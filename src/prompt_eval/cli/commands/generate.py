from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from prompt_eval.cli.constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUTS_PATH,
)
from prompt_eval.cli.prompt import run_prompt
from prompt_eval.cli.utils import load_file

# from prompt_eval.dataset import load_dataset
from prompt_eval.models import Dataset

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()


@app.command()
def generate(
    dataset: Annotated[
        Path, typer.Option(help="Path where the test dataset is loaded from.")
    ] = Path(DEFAULT_DATASET_PATH),
):
    """
    Generate solution per test case using a LLM
    """
    if dataset.exists() and dataset.is_file():
        with console.status("Generating solutions to test cases..."):
            test_cases = load_file(Dataset, dataset)
            run_prompt(test_cases)
            console.print(
                f"\n[green]✓ Solution per test case are saved in {DEFAULT_OUTPUTS_PATH}.[/green]\n"
            )
    else:
        err_console.print(
            f"\n[red]Dataset at path [bold]{dataset}[/bold] is not found.[/red]\n"
        )
        raise typer.Exit(code=2)
