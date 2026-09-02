"""``prompt-eval init-dataset`` - create a test dataset with the LLM."""

from __future__ import annotations

import typer

from prompt_eval.cli.options import (
    NumCasesOption,
    handle_errors,
    run_async,
    status,
)
from prompt_eval.constants import DEFAULT_TEST_CASES, EXIT_USAGE
from prompt_eval.llm import LLMClient
from prompt_eval.models import Dataset
from prompt_eval.paths import DEFAULT_DATASET_FILE
from prompt_eval.pipeline import generate_dataset
from prompt_eval.reporting import err_console, print_success
from prompt_eval.storage import save_model


@handle_errors
def init_dataset(
    num_cases: NumCasesOption = DEFAULT_TEST_CASES,
    regenerate: bool = typer.Option(
        False,
        "--regenerate",
        help="Overwrite an existing dataset.",
    ),
) -> None:
    """Create a test dataset at data/dataset.json via the LLM."""
    if DEFAULT_DATASET_FILE.is_file() and not regenerate:
        err_console.print(
            f"[bold yellow]⚠ A dataset already exists at {DEFAULT_DATASET_FILE}."
            "[/bold yellow]\nPass [bold]--regenerate[/bold] to overwrite it."
        )
        raise typer.Exit(code=EXIT_USAGE)

    noun = "case" if num_cases == 1 else "cases"
    with status(f"Generating a dataset with {num_cases} test {noun}"):
        dataset = run_async(_generate(num_cases))

    save_model(dataset, DEFAULT_DATASET_FILE)
    print_success(f"{len(dataset)} test {noun} written to {DEFAULT_DATASET_FILE}")


async def _generate(num_cases: int) -> Dataset:
    async with LLMClient() as llm:
        return await generate_dataset(llm, num_cases)
