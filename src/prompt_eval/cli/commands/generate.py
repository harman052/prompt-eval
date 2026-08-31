from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from prompt_eval.cli.constants import (
    DEFAULT_DATASET_FILE,
    DEFAULT_OUTPUTS_FILE,
)
from prompt_eval.cli.prompt import generate_prompt_output
from prompt_eval.cli.utils import load_file, print_dataset_error

# from prompt_eval.dataset import load_dataset
from prompt_eval.models import Dataset

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()


@app.command()
def generate(
    dataset: Annotated[
        Path, typer.Option(help="Path where the test dataset is loaded from.")
    ] = Path(DEFAULT_DATASET_FILE),
):
    """
    Generate solution per test case using a LLM
    """
    if not dataset.is_file():
        print_dataset_error(dataset)
        raise typer.Exit(code=2)

    with console.status("Generating solutions to test cases..."):
        test_cases = load_file(Dataset, dataset)
        generate_prompt_output(test_cases)
        console.print(
            f"\n[green bold]✓ Solution per test case are saved in {DEFAULT_OUTPUTS_FILE}.[/green bold]\n"
        )
