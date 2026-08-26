import typer
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from prompt_eval.cli.types import Grader
from prompt_eval.cli.utils import print_table
from prompt_eval.graders.deterministic_grader import deterministic_grader
from prompt_eval.graders.model_grader import ModelGrader
from prompt_eval.llm import LLMClient
from prompt_eval.models import TestCase

console = Console()

app = typer.Typer()


class Result(BaseModel):
    score: float
    task: str
    format: str


class CombinedResults(BaseModel):
    task: str
    format: str
    deterministic_score: float
    llm_judge_score: float
    final_score: float


def deterministic(
    test_cases: list[TestCase], grader: Grader | None = None
) -> list[Result]:
    results: list[Result] = []
    for test_case in test_cases:
        score = deterministic_grader(
            test_case,
            "\nimport json\nimport sys\nfrom typing import List\n\ndef extract_resources_with_depends_on(template: dict) -> List[str]:\n    \"\"\"Extract all resource logical IDs that have a DependsOn property.\"\"\"\n    resources_with_depends_on = []\n    \n    if 'Resources' not in template:\n        return resources_with_depends_on\n    \n    resources = template['Resources']\n    \n    for logical_id, resource_config in resources.items():\n        if isinstance(resource_config, dict) and 'DependsOn' in resource_config:\n            resources_with_depends_on.append(logical_id)\n    \n    return resources_with_depends_on\n\ndef main():\n    if len(sys.argv) > 1:\n        template_file = sys.argv[1]\n        with open(template_file, 'r') as f:\n            template = json.load(f)\n    else:\n        template = json.load(sys.stdin)\n    \n    result = extract_resources_with_depends_on(template)\n    print(json.dumps(result, indent=2))\n\nif __name__ == '__main__':\n    main()\n",
        )

        results.append(
            Result(
                score=score,
                task=test_case.task,
                format=test_case.format,
            )
        )
    if grader == Grader.DETERMINISTIC:
        table = Table(
            "Test Case",
            "Format",
            "Deterministic",
            title="Deterministic Grader Scores",
            width=100,
        )
        for result in results:
            table.add_row(
                result.task,
                result.format,
                str(result.score),
            )
        print_table(table)
    return results


def llm_judge(test_cases: list[TestCase], grader: Grader | None = None) -> list[Result]:
    llm = LLMClient()
    model_grader = ModelGrader(llm)

    results: list[Result] = []
    for test_case in test_cases:
        response = model_grader.grade(
            test_case,
            "\nimport json\nimport sys\nfrom typing import List\n\ndef extract_resources_with_depends_on(template: dict) -> List[str]:\n    \"\"\"Extract all resource logical IDs that have a DependsOn property.\"\"\"\n    resources_with_depends_on = []\n    \n    if 'Resources' not in template:\n        return resources_with_depends_on\n    \n    resources = template['Resources']\n    \n    for logical_id, resource_config in resources.items():\n        if isinstance(resource_config, dict) and 'DependsOn' in resource_config:\n            resources_with_depends_on.append(logical_id)\n    \n    return resources_with_depends_on\n\ndef main():\n    if len(sys.argv) > 1:\n        template_file = sys.argv[1]\n        with open(template_file, 'r') as f:\n            template = json.load(f)\n    else:\n        template = json.load(sys.stdin)\n    \n    result = extract_resources_with_depends_on(template)\n    print(json.dumps(result, indent=2))\n\nif __name__ == '__main__':\n    main()\n",
        )

        results.append(
            Result(
                score=response.score,
                task=test_case.task,
                format=test_case.format,
            )
        )
    if grader == Grader.LLM_JUDGE:
        table = Table(
            "Test Case",
            "Format",
            "Deterministic",
            title="LLM-Judge Scores",
            width=100,
        )
        for result in results:
            table.add_row(
                result.task,
                result.format,
                str(result.score),
            )
        print_table(table)
    return results


def both(test_cases: list[TestCase]) -> None:
    results = []

    with console.status("Getting Deterministic Scores..."):
        deterministic_scores = deterministic(test_cases)

    console.print("✓ Getting Deterministic Scores")

    with console.status("Getting LLM-Judge Scores..."):
        llmJudge_scores = llm_judge(test_cases)

    console.print("✓ Getting LLM-Judge Scores")

    for i, test_case in enumerate(test_cases):
        deterministic_score = deterministic_scores[i].score
        llm_judge_score = llmJudge_scores[i].score

        results.append(
            CombinedResults(
                task=test_case.task,
                format=test_case.format,
                deterministic_score=deterministic_score,
                llm_judge_score=llm_judge_score,
                final_score=(deterministic_score + llm_judge_score) / 2,
            )
        )

    table = Table(
        "Test Case",
        "Format",
        "Deterministic",
        "LLM-Judge",
        "Final Score",
        title="Combined scores (Deterministic and LLM as Judge)",
        width=100,
    )

    for result in results:
        table.add_row(
            result.task,
            result.format,
            str(result.deterministic_score),
            str(result.llm_judge_score),
            str(result.final_score),
        )

    print_table(table)
