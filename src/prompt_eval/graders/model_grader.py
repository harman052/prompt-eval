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
        prompt = f"""
You are an expert code reviewer.

Original Task:
<task>
{test_case.task}
</task>

Solution to evaluate:
<solution>
{solution}
</solution>

Criteria you should use to evaluate the solution:
<criteria>
{test_case.solution_criteria}
</criteria>
"""

        messages = [{"role": "user", "content": prompt}]
        response = self.llm.parse(messages, ModelGrade)
        return ModelGrade.model_validate(response)
