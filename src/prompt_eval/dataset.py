import json
from pathlib import Path

from prompt_eval.llm import LLMClient
from prompt_eval.models import Dataset, TestCase


def load_dataset(path: Path) -> list[TestCase]:
    data = json.loads(path.read_text())
    dataset = Dataset.model_validate(data)
    return dataset.test_cases


def generate_dataset():
    prompt = """
        Generate exactly 1 AWS-related test case.

        Each test case should:
        - require a Python, JSON, or Regex solution
        - be solvable without a large codebase
        - have clear solution criteria
  """

    llm = LLMClient()
    path = Path("data/dataset.json")
    messages = [{"role": "user", "content": prompt}]

    dataset = llm.parse(messages, Dataset)

    with open(path, "w") as f:
        json.dump(dataset.model_dump(), f, indent=2)
    return load_dataset(path)
