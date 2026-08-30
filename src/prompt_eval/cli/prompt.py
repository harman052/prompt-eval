from pathlib import Path

from prompt_eval.cli.constants import DEFAULT_OUTPUTS_PATH
from prompt_eval.cli.utils import save_file
from prompt_eval.llm import LLMClient
from prompt_eval.models import Dataset, Solution, Solutions, TestCase


def get_prompt(test_case: TestCase):
    return f"""Complete the following task:

        {test_case.task}

        * The entire response must be valid {test_case.format} source.
        * Do not add any comments or commentary or explanation
        """


def run_prompt(dataset: Dataset) -> Solutions:
    llm = LLMClient()
    messages: list[dict[str, str]] = []
    outputs = Solutions(root=[])

    for test_case in dataset.root:
        prompt = get_prompt(test_case)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "```code"},
        ]
        outputs.root.append(
            Solution(
                test_case_id=test_case.id,
                task=test_case.task,
                solution=llm.chat(messages, stop_sequences=["```"]),
            )
        )

    save_file(outputs, Path(DEFAULT_OUTPUTS_PATH))

    return outputs
