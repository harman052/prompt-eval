from pathlib import Path
import shutil

import typer
from rich.console import Console

from prompt_eval.cli.constants import (
    BASELINE_DIR,
    DEFAULT_BASELINE_FILE,
    DEFAULT_COMBINED_RESULTS_FILE,
)

console = Console()
err_console = Console(stderr=True)

app = typer.Typer()


@app.command()
def set_baseline():
    """
    Explicitly sets current results as a new baseline for comparisons
    """
    try:
        destination = Path(DEFAULT_BASELINE_FILE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DEFAULT_COMBINED_RESULTS_FILE, DEFAULT_BASELINE_FILE)
        console.print("[bold green]✓ A new baseline is set.[/bold green]")
    except Exception as exc:
        err_console.print(f"[bold red]Failed to set new baseline:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
