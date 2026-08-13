from ..llm import LLMClient
from ..models import ModelGrade, TestCase


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
    text = self.llm.chat(messages, None, ModelGrade)

    return ModelGrade.model_validate_json(text)
