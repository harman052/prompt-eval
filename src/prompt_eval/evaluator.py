from prompt_eval.graders.model_grader import ModelGrader
from prompt_eval.graders.syntax_grader import grade_syntax
from prompt_eval.llm import LLMClient, chat
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
    prompt = f"""Complete the following task:

    {test_case.task}

    * Respond only with Python, JSON, or a plain Regex
    * Do not add any comments or commentary or explanation
    """

    messages = [{"role": "user", "content": prompt}]
    return self.llm.chat(messages)

  def evaluate(self, test_case: TestCase) -> EvaluationResult:
    output = self.run_prompt(test_case)

    model_grade = self.model_grader.grade(
      test_case,
      output,
    )

    syntax_score = grade_syntax(
      test_case,
      output,
    )

    final_score = (model_grade.score + syntax_score) / 2

    return EvaluationResult(
      output=output,
      test_case=test_case,
      model_score=model_grade.score,
      syntax_score=syntax_score,
      score=final_score,
      reasoning=model_grade.reasoning,
    )
