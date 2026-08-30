from pathlib import Path

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

console = Console()


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
