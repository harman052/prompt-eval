import json
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


def save_results[ModelT: BaseModel](results: list[ModelT], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(
            [result.model_dump() for result in results],
            file,
            indent=2,
        )
