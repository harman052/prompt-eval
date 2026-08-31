from datetime import UTC, datetime
from pathlib import Path

from prompt_eval.cli.constants import DEFAULT_OUTPUTS_FILE
from prompt_eval.cli.utils import (
    err_console,
    get_prompt_metadata,
    load_prompt,
    save_file,
)
from prompt_eval.llm import LLMClient
from prompt_eval.models import Dataset, Solution, Solutions


def generate_prompt_output(dataset: Dataset) -> Solutions:
    llm = LLMClient()
    messages: list[dict[str, str]] = []
    outputs: Solutions = Solutions(
        metadata=get_prompt_metadata(),
        solutions=[],
    )

    try:
        for test_case in dataset.root:
            prompt = load_prompt("solution_prompt").format(
                task=test_case.task, format=test_case.format
            )

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "```code"},
            ]
            outputs.solutions.append(
                Solution(
                    test_case_id=test_case.test_case_id,
                    task=test_case.task,
                    solution=llm.chat(messages, stop_sequences=["```"]),
                )
            )

        save_file(outputs, Path(DEFAULT_OUTPUTS_FILE))
    except (OSError, ValueError) as exc:
        err_console.print(f"Error occurred during solution generation: {exc}")

    return outputs
