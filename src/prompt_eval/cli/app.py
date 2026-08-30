import typer

from prompt_eval.cli.commands.generate import generate
from prompt_eval.cli.commands.init_dataset import init_dataset
from prompt_eval.cli.commands.run import run

app = typer.Typer()


@app.callback()
def callback():
    """
    Prompt Evaluation CLI-Tool

    GitHub: https://github.com/harman052/prompt-eval
    """


app.command()(run)
app.command()(init_dataset)
app.command()(generate)
