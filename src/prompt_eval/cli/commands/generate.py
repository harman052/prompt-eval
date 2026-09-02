"""``prompt-eval generate`` - produce one solution per test case."""

from __future__ import annotations

from prompt_eval.cli.options import (
    DatasetOption,
    handle_errors,
    require_file,
    run_async,
    status,
)
from prompt_eval.llm import LLMClient
from prompt_eval.models import Dataset, SolutionReport
from prompt_eval.paths import DEFAULT_DATASET_FILE, DEFAULT_SOLUTIONS_FILE
from prompt_eval.pipeline import generate_solutions
from prompt_eval.reporting import print_success
from prompt_eval.storage import load_model, save_model


@handle_errors
def generate(dataset: DatasetOption = DEFAULT_DATASET_FILE) -> None:
    """Generate a solution per test case using the LLM."""
    require_file(
        dataset,
        hint="Create one with: [bold]prompt-eval init-dataset[/bold]",
    )
    test_cases = load_model(Dataset, dataset)

    with status(f"Generating solutions for {len(test_cases)} test case(s)"):
        report = run_async(_generate(test_cases))

    save_model(report, DEFAULT_SOLUTIONS_FILE)
    print_success(
        f"{len(report.results)} solution(s) written to {DEFAULT_SOLUTIONS_FILE}"
    )


async def _generate(test_cases: Dataset) -> SolutionReport:
    async with LLMClient() as llm:
        return await generate_solutions(llm, test_cases)
