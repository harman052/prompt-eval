from __future__ import annotations

from prompt_eval.llm import LLMClient
from prompt_eval.models import ModelGrade, TestCase
from prompt_eval.prompts import JUDGE_PROMPT, render_prompt


async def grade_with_judge(
    llm: LLMClient, test_case: TestCase, solution: str
) -> ModelGrade:
    """Score ``solution`` against the test case's criteria."""
    prompt = render_prompt(
        JUDGE_PROMPT,
        task=test_case.task,
        solution=solution,
        solution_criteria=test_case.solution_criteria,
    )
    return await llm.parse(prompt, ModelGrade)
