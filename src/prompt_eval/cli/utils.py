from rich.console import Console
from rich.table import Table

console = Console()


def print_table(table: Table) -> None:
    console.print()
    console.print(table)
    console.print()
