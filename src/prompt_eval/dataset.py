import json
from pathlib import Path

from prompt_eval.llm import LLMClient
from prompt_eval.models import Dataset, GeneratedDataset, TestCase


def load_test_cases(path: Path) -> list[TestCase]:
    data = json.loads(path.read_text())
    dataset = Dataset.model_validate(data)
    return dataset.test_cases


def generate_dataset(test_cases_count: int = 3) -> list[TestCase]:
    prompt = f"""
        Generate exactly {test_cases_count} AWS-related test case.

        Each test case should:
        - require a Python, JSON, or Regex solution
        - be solvable without a large codebase
        - have clear solution criteria
  """

    llm = LLMClient()
    path = Path("data/dataset.json")
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

    dataset = Dataset(test_cases=test_cases)

    with open(path, "w") as f:
        json.dump(dataset.model_dump(), f, indent=2)
    return load_test_cases(path)
