from pathlib import Path

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from prompt_eval.cli.constants import DEFAULT_TEST_CASES, MIN_TEST_CASES

console = Console()
err_console = Console(stderr=True)


def print_table(table: Table) -> None:
    console.print()
    console.print(table)
    console.print()


def generate_numbered_list(items: list[str]) -> str:
    result = "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))
    return result


def save_file(
    model: BaseModel,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2))


def load_file[ModelT: BaseModel](
    model: type[ModelT],
    path: Path,
) -> ModelT:
    return model.model_validate_json(path.read_text())


def print_dataset_error(dataset: Path) -> None:
    err_console.print(f"\n[bold red]Test dataset not found:[/bold red] {dataset}\n")
    err_console.print(
        "Generate a new dataset with:\n"
        "[bold]prompt-eval init-dataset "
        f"--num-cases {MIN_TEST_CASES}[/bold]"
    )
    err_console.print(
        f"\nThe minimum number of test cases is {MIN_TEST_CASES}; "
        f"the default is {DEFAULT_TEST_CASES}."
    )
    err_console.print(
        "\nFor detailed help, use: [bold]prompt-eval evaluate --help[/bold]"
    )
