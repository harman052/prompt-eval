"""Prompt template loading and rendering.

Templates live in ``prompts/`` as plain text so that they are reviewable in a
diff and versioned by git
"""

from __future__ import annotations

from functools import cache
from string import Template

from prompt_eval.errors import PromptError
from prompt_eval.paths import prompts_dir

SOLUTION_PROMPT = "solution_prompt"
JUDGE_PROMPT = "judge_prompt"
DATASET_PROMPT = "dataset_prompt"


@cache
def load_prompt(name: str) -> str:
    """Return the raw text of the ``prompts/<name>.txt`` template."""
    path = prompts_dir() / f"{name}.txt"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptError(f"Prompt template not found: {path}") from exc
    except OSError as exc:
        raise PromptError(f"Could not read prompt template {path}: {exc}") from exc


def render_prompt(name: str, /, **values: object) -> str:
    """Render a template, substituting ``$placeholder`` values.

    ``string.Template`` is used instead of ``str.format`` because prompts
    routinely contain JSON and regex braces, which ``format`` would try to
    interpret as fields.`
    """
    template = Template(load_prompt(name))
    try:
        return template.substitute(values)
    except KeyError as exc:
        raise PromptError(
            f"Prompt '{name}' requires placeholder {exc} which was not provided."
        ) from exc
    except ValueError as exc:
        raise PromptError(f"Prompt '{name}' is malformed: {exc}") from exc
