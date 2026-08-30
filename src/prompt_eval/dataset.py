from pathlib import Path

from prompt_eval.cli.constants import DEFAULT_DATASET_PATH
from prompt_eval.cli.utils import save_results
from prompt_eval.llm import LLMClient
from prompt_eval.models import Dataset, GeneratedDataset, TestCase


def load_dataset(path: Path) -> Dataset:
    return Dataset.model_validate_json(path.read_text())


def generate_dataset(test_cases_count: int = 3) -> Dataset:
    prompt = f"""
        Generate exactly {test_cases_count} AWS-related test case.

        Each test case should:
        - require a Python, JSON, or Regex solution
        - be solvable without a large codebase
        - have clear solution criteria
  """

    llm = LLMClient()
    messages = [{"role": "user", "content": prompt}]

    generated_dataset = llm.parse(messages, GeneratedDataset)

    test_cases = [
        TestCase(
            id=f"{index:03d}",
            task=test_case.task,
            format=test_case.format,
            solution_criteria=test_case.solution_criteria,
        )
        for index, test_case in enumerate(generated_dataset.test_cases, start=1)
    ]

    save_results(test_cases, DEFAULT_DATASET_PATH)
    return load_dataset(DEFAULT_DATASET_PATH)
