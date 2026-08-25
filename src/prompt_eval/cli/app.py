import typer

from prompt_eval.cli.run import run

app = typer.Typer()


@app.callback()
def callback():
    """
    Prompt Evaluation CLI-Tool

    GitHub: https://github.com/harman052/prompt-eval
    """


app.command()(run)
