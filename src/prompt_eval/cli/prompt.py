from pathlib import Path

from pydantic import BaseModel

from prompt_eval.cli.utils import save_results
from prompt_eval.llm import LLMClient
from prompt_eval.models import TestCase


class PromptOutput(BaseModel):
    test_case_id: str
    task: str
    solution: str


def get_prompt(test_case: TestCase):
    return f"""Complete the following task:

        {test_case.task}

        * The entire response must be valid {test_case.format} source.
        * Do not add any comments or commentary or explanation
        """


def run_prompt(test_cases: list[TestCase]) -> list[PromptOutput]:
    llm = LLMClient()
    messages: list[dict[str, str]] = []
    outputs: list[PromptOutput] = []

    for test_case in test_cases:
        prompt = get_prompt(test_case)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "```code"},
        ]
        outputs.append(
            PromptOutput(
                test_case_id=test_case.id,
                task=test_case.task,
                solution=llm.chat(messages, stop_sequences=["```"]),
            )
        )
    save_results(outputs, Path("output/outputs.json"))
    return outputs
