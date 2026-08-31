from pathlib import Path

from prompt_eval.cli.constants import DEFAULT_DATASET_FILE
from prompt_eval.cli.utils import load_file, save_file
from prompt_eval.llm import LLMClient
from prompt_eval.models import Dataset, GeneratedDataset, TestCase


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

    test_cases = Dataset(root=[])

    for index, test_case in enumerate(generated_dataset.test_cases, start=1):
        test_cases.root.append(
            TestCase(
                test_case_id=f"{index:03d}",
                task=test_case.task,
                format=test_case.format,
                solution_criteria=test_case.solution_criteria,
            )
        )

    save_file(test_cases, Path(DEFAULT_DATASET_FILE))
    return load_file(Dataset, DEFAULT_DATASET_FILE)
