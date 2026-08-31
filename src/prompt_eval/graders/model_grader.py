from prompt_eval.cli.utils import load_prompt
from prompt_eval.llm import LLMClient
from prompt_eval.models import ModelGrade, TestCase


class ModelGrader:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def grade(
        self,
        test_case: TestCase,
        solution: str,
    ) -> ModelGrade:
        prompt = load_prompt("judge_prompt").format(
            task=test_case.task,
            solution=solution,
            solution_criteria=test_case.solution_criteria,
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.parse(messages, ModelGrade)
        return ModelGrade.model_validate(response)
