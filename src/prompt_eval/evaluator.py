from prompt_eval.cli.prompt import get_prompt
from prompt_eval.graders.deterministic_grader import deterministic_grader
from prompt_eval.graders.model_grader import ModelGrader
from prompt_eval.llm import LLMClient
from prompt_eval.models import EvaluationResult, TestCase


class Evaluator:
    def __init__(
        self,
        llm: LLMClient,
        model_grader: ModelGrader,
    ):
        self.llm = llm
        self.model_grader = model_grader

    def run_prompt(self, test_case: TestCase) -> str:
        prompt = get_prompt(test_case)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "```code"},
        ]
        return self.llm.chat(messages, stop_sequences=["```"])

    def evaluate(self, test_case: TestCase) -> EvaluationResult:
        output = self.run_prompt(test_case)

        model_grade = self.model_grader.grade(
            test_case,
            output,
        )

        validity_score = deterministic_grader(
            test_case,
            output,
        )

        final_score = (model_grade.score + validity_score) / 2

        return EvaluationResult(
            test_case=test_case,
            output=output,
            model_score=model_grade.score,
            validity_score=validity_score,
            final_score=final_score,
            reasoning=model_grade.reasoning,
        )
