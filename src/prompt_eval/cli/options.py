"""Shared CLI option types and the error boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Annotated, Any

import typer

from prompt_eval.cli.types import GraderChoice
from prompt_eval.constants import EXIT_ERROR, EXIT_USAGE, MIN_TEST_CASES
from prompt_eval.errors import PromptEvalError
from prompt_eval.paths import DEFAULT_DATASET_FILE
from prompt_eval.reporting import console, err_console, print_error

DatasetOption = Annotated[
    Path,
    typer.Option(
        "--dataset",
        help="Path the test dataset is loaded from.",
        show_default=str(DEFAULT_DATASET_FILE),
    ),
]

FailUnderOption = Annotated[
    float | None,
    typer.Option(
        "--fail-under",
        min=0.0,
        help=(
            "Exit non-zero if the average score across all test cases falls "
            "below this value. If unset, no gate is applied."
        ),
    ),
]

GraderOption = Annotated[
    GraderChoice,
    typer.Option(
        "--grader",
        help="Which grader(s) to run: syntax only, LLM-judge only, or both.",
    ),
]

VerboseOption = Annotated[
    bool,
    typer.Option("--verbose", help="Include the LLM-Judge rationale in the output."),
]

NumCasesOption = Annotated[
    int,
    typer.Option(
        "--num-cases",
        min=MIN_TEST_CASES,
        help="Number of test cases to generate.",
    ),
]


def handle_errors[**P, R](command: Callable[P, R]) -> Callable[P, R]:
    """Turn any :class:`PromptEvalError` into a clean non-zero exit."""

    @wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return command(*args, **kwargs)
        except PromptEvalError as exc:
            print_error(str(exc))
            raise typer.Exit(code=EXIT_ERROR) from exc

    return wrapper


def run_async[R](coroutine: Coroutine[Any, Any, R]) -> R:
    """Run an async pipeline stage from a synchronous Typer command.

    Typer commands must be synchronous, so exactly one ``asyncio.run`` per
    command invocation is the event-loop boundary. ``KeyboardInterrupt`` is
    translated here so ^C during a long run exits quietly rather than dumping a
    traceback from deep inside the task group.
    """
    try:
        return asyncio.run(coroutine)
    except KeyboardInterrupt as exc:
        err_console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=EXIT_ERROR) from exc


def require_file(path: Path, *, hint: str) -> None:
    """Exit with a usage error if ``path`` is not an existing file."""
    if path.is_file():
        return
    print_error(f"{path} not found.")
    err_console.print(hint)
    raise typer.Exit(code=EXIT_USAGE)


@contextmanager
def status(message: str) -> Any:
    """Show a spinner for a long-running stage, then confirm it."""
    with console.status(message):
        yield
    console.print(f"[green]✓[/green] {message}")
