import typer

from prompt_eval.cli.commands.evaluate import evaluate
from prompt_eval.cli.commands.generate import generate
from prompt_eval.cli.commands.init_dataset import init_dataset
from prompt_eval.cli.commands.set_baseline import set_baseline

app = typer.Typer()


@app.callback()
def callback():
    """
    Prompt Evaluation CLI-Tool

    GitHub: https://github.com/harman052/prompt-eval
    """


app.command()(init_dataset)
app.command()(generate)
app.command()(evaluate)
app.command()(set_baseline)
