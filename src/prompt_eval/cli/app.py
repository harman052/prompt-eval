"""Typer application wiring."""

from __future__ import annotations

import typer

from prompt_eval.cli.commands.compare import compare
from prompt_eval.cli.commands.evaluate import evaluate
from prompt_eval.cli.commands.generate import generate
from prompt_eval.cli.commands.init_dataset import init_dataset
from prompt_eval.cli.commands.set_baseline import set_baseline

app = typer.Typer(
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
    help="Prompt Evaluation CLI-Tool\n\nGitHub: https://github.com/harman052/prompt-eval",
)

for command in (init_dataset, generate, evaluate, set_baseline, compare):
    app.command()(command)


if __name__ == "__main__":
    app()
