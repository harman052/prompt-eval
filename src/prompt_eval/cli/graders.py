import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from prompt_eval.cli.types import Grader
from prompt_eval.cli.utils import generate_numbered_list, print_table
from prompt_eval.graders.deterministic_grader import deterministic_grader
from prompt_eval.graders.model_grader import ModelGrader
from prompt_eval.llm import LLMClient
from prompt_eval.models import TestCase

console = Console()

RESULTS_DIR = Path("eval_results")
DETERMINISTIC_RESULTS_FILE = RESULTS_DIR / "deterministic_grader_results.json"
MODEL_RESULTS_FILE = RESULTS_DIR / "model_grader_results.json"

TEST_SOLUTION = """
import json
import sys
from typing import List


def extract_resources_with_depends_on(template: dict) -> List[str]:
    \"\"\"Extract all resource logical IDs that have a DependsOn property.\"\"\"
    resources_with_depends_on = []

    if "Resources" not in template:
        return resources_with_depends_on

    resources = template["Resources"]

    for logical_id, resource_config in resources.items():
        if isinstance(resource_config, dict) and "DependsOn" in resource_config:
            resources_with_depends_on.append(logical_id)

    return resources_with_depends_on


def main():
    if len(sys.argv) > 1:
        template_file = sys.argv[1]
        with open(template_file, "r") as f:
            template = json.load(f)
    else:
        template = json.load(sys.stdin)

    result = extract_resources_with_depends_on(template)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
"""


class DeterministicGraderResults(BaseModel):
    test_case_id: str
    task: str
    format: str
    score: float


class ModelGraderResults(BaseModel):
    test_case_id: str
    task: str
    format: str
    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    score: float


class CombinedResults(BaseModel):
    task: str
    format: str
    strengths: list[str] | None
    weaknesses: list[str] | None
    reasoning: str | None
    deterministic_score: float
    llm_judge_score: float
    final_score: float


def _save_results[ModelT: BaseModel](results: list[ModelT], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(
            [result.model_dump() for result in results],
            file,
            indent=2,
        )


def _print_deterministic_results(
    results: list[DeterministicGraderResults],
) -> None:
    table = Table(
        "Test Case ID",
        "Test Case",
        "Format",
        "Deterministic",
        title="Deterministic Grader Scores",
        width=100,
    )

    for result in results:
        table.add_row(
            result.test_case_id,
            result.task,
            result.format,
            str(result.score),
        )

    print_table(table)


def _print_model_results(results: list[ModelGraderResults]) -> None:
    table = Table(
        "Test Case ID",
        "Test Case",
        "Format",
        "Strengths",
        "Weaknesses",
        "Reasoning",
        "LLM-Judge",
        title="LLM-Judge Scores",
    )

    for result in results:
        table.add_row(
            result.test_case_id,
            result.task,
            result.format,
            generate_numbered_list(result.strengths),
            generate_numbered_list(result.weaknesses),
            result.reasoning,
            str(result.score),
        )

    print_table(table)


def _build_combined_results(
    test_cases: list[TestCase],
    deterministic_results: list[DeterministicGraderResults],
    model_results: list[ModelGraderResults],
    verbose: bool,
) -> list[CombinedResults]:
    results: list[CombinedResults] = []

    for test_case, deterministic_result, model_result in zip(
        test_cases,
        deterministic_results,
        model_results,
        strict=True,
    ):
        deterministic_score = deterministic_result.score
        llm_judge_score = model_result.score

        results.append(
            CombinedResults(
                task=test_case.task,
                format=test_case.format,
                strengths=model_result.strengths if verbose else None,
                weaknesses=model_result.weaknesses if verbose else None,
                reasoning=model_result.reasoning if verbose else None,
                deterministic_score=deterministic_score,
                llm_judge_score=llm_judge_score,
                final_score=(deterministic_score + llm_judge_score) / 2,
            )
        )

    return results


def _print_combined_results(
    results: list[CombinedResults],
    verbose: bool,
) -> None:
    if verbose:
        table = Table(
            "Test Case",
            "Format",
            "Strengths",
            "Weaknesses",
            "Reasoning",
            "Deterministic",
            "LLM-Judge",
            "Final Score",
            title="Combined Scores (Deterministic and LLM as Judge)",
        )

        for result in results:
            table.add_row(
                result.task,
                result.format,
                generate_numbered_list(result.strengths or []),
                generate_numbered_list(result.weaknesses or []),
                result.reasoning or "",
                str(result.deterministic_score),
                str(result.llm_judge_score),
                str(result.final_score),
            )
    else:
        table = Table(
            "Test Case",
            "Format",
            "Deterministic",
            "LLM-Judge",
            "Final Score",
            title="Combined Scores (Deterministic and LLM as Judge)",
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


def deterministic(
    test_cases: list[TestCase],
    grader: Grader | None = None,
) -> list[DeterministicGraderResults]:
    results = [
        DeterministicGraderResults(
            test_case_id=test_case.id,
            task=test_case.task,
            format=test_case.format,
            score=deterministic_grader(test_case, TEST_SOLUTION),
        )
        for test_case in test_cases
    ]

    _save_results(results, DETERMINISTIC_RESULTS_FILE)

    if grader == Grader.DETERMINISTIC:
        _print_deterministic_results(results)

    return results


def llm_judge(
    test_cases: list[TestCase],
    grader: Grader | None = None,
) -> list[ModelGraderResults]:
    model_grader = ModelGrader(LLMClient())

    results: list[ModelGraderResults] = []

    for test_case in test_cases:
        response = model_grader.grade(test_case, TEST_SOLUTION)

        results.append(
            ModelGraderResults(
                test_case_id=test_case.id,
                task=test_case.task,
                format=test_case.format,
                strengths=response.strengths,
                weaknesses=response.weaknesses,
                reasoning=response.reasoning,
                score=response.score,
            )
        )

    _save_results(results, MODEL_RESULTS_FILE)

    if grader == Grader.LLM_JUDGE:
        _print_model_results(results)

    return results


def both(test_cases: list[TestCase], verbose: bool) -> None:
    with console.status("Getting Deterministic Scores..."):
        deterministic_results = deterministic(test_cases)

    console.print("✓ Getting Deterministic Scores")

    with console.status("Getting LLM-Judge Scores..."):
        model_results = llm_judge(test_cases)

    console.print("✓ Getting LLM-Judge Scores")

    combined_results = _build_combined_results(
        test_cases,
        deterministic_results,
        model_results,
        verbose,
    )

    _print_combined_results(combined_results, verbose)
