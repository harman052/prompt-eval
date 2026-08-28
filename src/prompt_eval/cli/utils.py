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
