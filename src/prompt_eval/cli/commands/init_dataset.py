from typing import Annotated

import typer
from rich.console import Console

from prompt_eval.cli.constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_TEST_CASES,
    MIN_TEST_CASES,
)
from prompt_eval.dataset import generate_dataset

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()


@app.command()
def init_dataset(
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
            help=f"Number of test cases to generate. Use it with --regenerate flag. Minimum value: {MIN_TEST_CASES}",
            min=MIN_TEST_CASES,
        ),
    ] = DEFAULT_TEST_CASES,
):
    if regenerate:
        with console.status(f"Generating new dataset with {num_cases} test cases..."):
            generate_dataset(num_cases)

            console.print(
                f"\n[green]✓ Dataset generated with {num_cases} test cases at data/dataset.json.[/green]\n"
            )
        raise typer.Exit()

    if DEFAULT_DATASET_PATH.exists() and DEFAULT_DATASET_PATH.is_file():
        err_console.print(
            "\n[red]⚠︎ Dataset already exists at data/dataset.json. Use [code]--regenerate[/code] flag to override exisiting dataset.[/red]\n"
        )
    else:
        generate_dataset(num_cases)
