import json
from pathlib import Path

from prompt_eval.dataset import generate_dataset, load_dataset
from prompt_eval.evaluator import Evaluator
from prompt_eval.graders.model_grader import ModelGrader
from prompt_eval.llm import LLMClient
from prompt_eval.models import EvaluationResult, TestCase

file_path = Path("data/dataset.json")

llm = LLMClient()
model_grader = ModelGrader(llm)

evaluator = Evaluator(llm, model_grader)


def main():
    test_cases: list[TestCase]
    results: list[EvaluationResult] = []
    if file_path.exists() and file_path.is_file():
        test_cases = load_dataset(file_path)
    else:
        test_cases = generate_dataset()

    for test_case in test_cases:
        result = evaluator.evaluate(test_case)
        results.append(result)

    json_ready_data = [result.model_dump() for result in results]
    with open("results/result.json", "w") as f:
        json.dump(json_ready_data, f, indent=4)


main()
