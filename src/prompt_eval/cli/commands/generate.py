from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from prompt_eval.cli.constants import (
    DEFAULT_DATASET_PATH,
)
from prompt_eval.cli.prompt import run_prompt
from prompt_eval.dataset import load_dataset

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
        test_cases = load_dataset(dataset)
        run_prompt(test_cases)
