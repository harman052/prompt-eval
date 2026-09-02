"""CLI behaviour: exit codes, guard rails and artifact side effects.

Exit codes are the tool's real contract with CI, so they are asserted
explicitly: 0 success, 1 error, 2 usage, 3 quality gate failed.
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from prompt_eval.cli.app import app
from prompt_eval.constants import EXIT_ERROR, EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE
from prompt_eval.errors import LLMError
from prompt_eval.models import (
    CombinedReport,
    Dataset,
    DatasetSpec,
    SolutionSpec,
    ModelGrade,
    RunMetadata,
    SolutionFormat,
    SolutionReport,
    TestCaseSpec,
)
from prompt_eval.paths import (
    COMBINED_RESULTS_FILE,
    COMPARISON_RESULTS_DIR,
    DEFAULT_BASELINE_FILE,
    DEFAULT_DATASET_FILE,
    DEFAULT_SOLUTIONS_FILE,
    DETERMINISTIC_RESULTS_FILE,
    MODEL_RESULTS_FILE,
)
from prompt_eval.storage import load_model, save_model
from tests.conftest import FakeLLMClient

runner = CliRunner()


@pytest.fixture
def patch_llm(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace ``LLMClient`` everywhere the commands construct one."""

    def install(llm: FakeLLMClient) -> FakeLLMClient:
        for module in (
            "prompt_eval.cli.commands.init_dataset",
            "prompt_eval.cli.commands.generate",
            "prompt_eval.cli.commands.evaluate",
        ):
            monkeypatch.setattr(f"{module}.LLMClient", lambda **_: llm)
        return llm

    return install


def graded_llm() -> FakeLLMClient:
    return FakeLLMClient(
        responses={
            SolutionSpec: SolutionSpec(code="x = 1"),
            ModelGrade: ModelGrade(
                strengths=["works"], weaknesses=[], reasoning="good", score=8.0
            ),
            DatasetSpec: DatasetSpec(
                test_cases=[
                    TestCaseSpec(
                        task="Write python",
                        format=SolutionFormat.PYTHON,
                        solution_criteria="it runs",
                    )
                ]
            ),
        }
    )


# --------------------------------------------------------------------------- #
# init-dataset
# --------------------------------------------------------------------------- #


def test_init_dataset_writes_a_dataset(patch_llm: Any) -> None:
    patch_llm(graded_llm())

    result = runner.invoke(app, ["init-dataset", "--num-cases", "1"])

    assert result.exit_code == EXIT_OK, result.output
    assert len(load_model(Dataset, DEFAULT_DATASET_FILE)) == 1


def test_init_dataset_refuses_to_clobber_an_existing_dataset(
    dataset: Dataset, patch_llm: Any
) -> None:
    patch_llm(graded_llm())
    save_model(dataset, DEFAULT_DATASET_FILE)

    result = runner.invoke(app, ["init-dataset"])

    assert result.exit_code == EXIT_USAGE
    assert "--regenerate" in result.output
    assert load_model(Dataset, DEFAULT_DATASET_FILE) == dataset  # untouched


def test_init_dataset_overwrites_with_regenerate(
    dataset: Dataset, patch_llm: Any
) -> None:
    patch_llm(graded_llm())
    save_model(dataset, DEFAULT_DATASET_FILE)

    result = runner.invoke(app, ["init-dataset", "--regenerate", "--num-cases", "1"])

    assert result.exit_code == EXIT_OK, result.output
    assert len(load_model(Dataset, DEFAULT_DATASET_FILE)) == 1


def test_init_dataset_rejects_zero_cases() -> None:
    result = runner.invoke(app, ["init-dataset", "--num-cases", "0"])
    assert result.exit_code != EXIT_OK
    assert not DEFAULT_DATASET_FILE.exists()


def test_init_dataset_reports_an_llm_failure(patch_llm: Any) -> None:
    patch_llm(FakeLLMClient(responses={DatasetSpec: LLMError("api is down")}))

    result = runner.invoke(app, ["init-dataset"])

    assert result.exit_code == EXIT_ERROR
    assert "api is down" in result.output
    assert not DEFAULT_DATASET_FILE.exists()  # no partial artifact


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def test_generate_writes_one_solution_per_test_case(
    dataset: Dataset, patch_llm: Any
) -> None:
    patch_llm(graded_llm())
    save_model(dataset, DEFAULT_DATASET_FILE)

    result = runner.invoke(app, ["generate"])

    assert result.exit_code == EXIT_OK, result.output
    report = load_model(SolutionReport, DEFAULT_SOLUTIONS_FILE)
    assert [row.test_case_id for row in report.results] == ["001", "002"]


def test_generate_reports_a_missing_dataset(patch_llm: Any) -> None:
    patch_llm(graded_llm())
    result = runner.invoke(app, ["generate"])
    assert result.exit_code == EXIT_USAGE
    assert "init-dataset" in result.output


def test_generate_leaves_no_artifact_when_the_llm_fails(
    dataset: Dataset, patch_llm: Any
) -> None:
    """The old implementation caught the error and still reported success."""
    patch_llm(FakeLLMClient(responses={SolutionSpec: LLMError("timeout")}))
    save_model(dataset, DEFAULT_DATASET_FILE)

    result = runner.invoke(app, ["generate"])

    assert result.exit_code == EXIT_ERROR
    assert not DEFAULT_SOLUTIONS_FILE.exists()


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #


def prepare(dataset: Dataset, solutions: SolutionReport) -> None:
    save_model(dataset, DEFAULT_DATASET_FILE)
    save_model(solutions, DEFAULT_SOLUTIONS_FILE)


def test_evaluate_both_graders_writes_every_report(
    dataset: Dataset, solutions: SolutionReport, patch_llm: Any
) -> None:
    patch_llm(graded_llm())
    prepare(dataset, solutions)

    result = runner.invoke(app, ["evaluate"])

    assert result.exit_code == EXIT_OK, result.output
    assert DETERMINISTIC_RESULTS_FILE.is_file()
    assert MODEL_RESULTS_FILE.is_file()
    combined = load_model(CombinedReport, COMBINED_RESULTS_FILE)
    assert [row.final_score for row in combined.results] == [9.0, 9.0]


def test_evaluate_deterministic_only_makes_no_model_calls(
    dataset: Dataset, solutions: SolutionReport, patch_llm: Any
) -> None:
    llm = patch_llm(graded_llm())
    prepare(dataset, solutions)

    result = runner.invoke(app, ["evaluate", "--grader", "deterministic"])

    assert result.exit_code == EXIT_OK, result.output
    assert llm.prompts == []
    assert not MODEL_RESULTS_FILE.exists()


def test_evaluate_llm_judge_only(
    dataset: Dataset, solutions: SolutionReport, patch_llm: Any
) -> None:
    patch_llm(graded_llm())
    prepare(dataset, solutions)

    result = runner.invoke(app, ["evaluate", "--grader", "llm-judge"])

    assert result.exit_code == EXIT_OK, result.output
    assert MODEL_RESULTS_FILE.is_file()
    assert not COMBINED_RESULTS_FILE.exists()


def test_evaluate_verbose_shows_the_judge_rationale(
    dataset: Dataset, solutions: SolutionReport, patch_llm: Any
) -> None:
    patch_llm(graded_llm())
    prepare(dataset, solutions)

    quiet = runner.invoke(app, ["evaluate"])
    verbose = runner.invoke(app, ["evaluate", "--verbose"])

    assert "Reasoning" not in quiet.output
    assert "Reasoning" in verbose.output


def test_evaluate_fails_the_gate_below_the_threshold(
    dataset: Dataset, solutions: SolutionReport, patch_llm: Any
) -> None:
    patch_llm(graded_llm())
    prepare(dataset, solutions)

    result = runner.invoke(app, ["evaluate", "--fail-under", "9.5"])

    assert result.exit_code == EXIT_GATE_FAILED
    assert COMBINED_RESULTS_FILE.is_file()  # results still saved for inspection


def test_evaluate_passes_the_gate_at_the_threshold(
    dataset: Dataset, solutions: SolutionReport, patch_llm: Any
) -> None:
    """The gate is `average < threshold`, so exactly meeting it passes."""
    patch_llm(graded_llm())
    prepare(dataset, solutions)

    result = runner.invoke(app, ["evaluate", "--fail-under", "9.0"])

    assert result.exit_code == EXIT_OK, result.output


def test_evaluate_honours_a_zero_threshold(
    dataset: Dataset, solutions: SolutionReport, patch_llm: Any
) -> None:
    """`--fail-under 0` is an explicit gate, not an absent one."""
    patch_llm(graded_llm())
    prepare(dataset, solutions)

    result = runner.invoke(
        app, ["evaluate", "--grader", "deterministic", "--fail-under", "0"]
    )

    assert result.exit_code == EXIT_OK, result.output


def test_evaluate_reports_missing_solutions(dataset: Dataset, patch_llm: Any) -> None:
    patch_llm(graded_llm())
    save_model(dataset, DEFAULT_DATASET_FILE)

    result = runner.invoke(app, ["evaluate"])

    assert result.exit_code == EXIT_USAGE
    assert "prompt-eval generate" in result.output


def test_evaluate_reports_a_solution_set_that_does_not_match_the_dataset(
    dataset: Dataset, solutions: SolutionReport, patch_llm: Any
) -> None:
    patch_llm(graded_llm())
    partial = solutions.model_copy(update={"results": solutions.results[:1]})
    prepare(dataset, partial)

    result = runner.invoke(app, ["evaluate"])

    assert result.exit_code == EXIT_ERROR
    assert "No solution for test case" in result.output


def test_evaluate_rejects_an_unknown_grader(
    dataset: Dataset, solutions: SolutionReport
) -> None:
    prepare(dataset, solutions)
    result = runner.invoke(app, ["evaluate", "--grader", "vibes"])
    assert result.exit_code != EXIT_OK


# --------------------------------------------------------------------------- #
# set-baseline and compare
# --------------------------------------------------------------------------- #


def combined_report(metadata: RunMetadata, scores: dict[str, float]) -> CombinedReport:
    from prompt_eval.models import CombinedResult

    return CombinedReport(
        metadata=metadata,
        results=[
            CombinedResult(
                test_case_id=test_case_id,
                task=f"task {test_case_id}",
                format=SolutionFormat.PYTHON,
                strengths=[],
                weaknesses=[],
                reasoning="r",
                deterministic_score=score,
                llm_judge_score=score,
            )
            for test_case_id, score in scores.items()
        ],
    )


def test_set_baseline_promotes_the_current_results(metadata: RunMetadata) -> None:
    report = combined_report(metadata, {"001": 8.0})
    save_model(report, COMBINED_RESULTS_FILE)

    result = runner.invoke(app, ["set-baseline"])

    assert result.exit_code == EXIT_OK, result.output
    assert load_model(CombinedReport, DEFAULT_BASELINE_FILE) == report


def test_set_baseline_requires_results() -> None:
    result = runner.invoke(app, ["set-baseline"])
    assert result.exit_code == EXIT_USAGE
    assert "prompt-eval evaluate" in result.output


def test_set_baseline_refuses_a_corrupt_results_file() -> None:
    COMBINED_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    COMBINED_RESULTS_FILE.write_text("{}")

    result = runner.invoke(app, ["set-baseline"])

    assert result.exit_code == EXIT_ERROR
    assert not DEFAULT_BASELINE_FILE.exists()


def test_compare_reports_deltas_and_persists_them(metadata: RunMetadata) -> None:
    save_model(combined_report(metadata, {"001": 8.0}), DEFAULT_BASELINE_FILE)
    save_model(combined_report(metadata, {"001": 9.0}), COMBINED_RESULTS_FILE)

    result = runner.invoke(app, ["compare"])

    assert result.exit_code == EXIT_OK, result.output
    assert "+1.00" in result.output
    assert list(COMPARISON_RESULTS_DIR.glob("*.json"))


def test_compare_fails_on_a_regression_beyond_the_threshold(
    metadata: RunMetadata,
) -> None:
    save_model(combined_report(metadata, {"001": 9.0}), DEFAULT_BASELINE_FILE)
    save_model(combined_report(metadata, {"001": 5.0}), COMBINED_RESULTS_FILE)

    result = runner.invoke(app, ["compare", "--regression-threshold", "1.0"])

    assert result.exit_code == EXIT_GATE_FAILED
    assert list(COMPARISON_RESULTS_DIR.glob("*.json"))  # artifact kept for triage


def test_compare_tolerates_a_regression_within_the_threshold(
    metadata: RunMetadata,
) -> None:
    save_found = combined_report(metadata, {"001": 9.0})
    save_model(save_found, DEFAULT_BASELINE_FILE)
    save_model(combined_report(metadata, {"001": 8.5}), COMBINED_RESULTS_FILE)

    result = runner.invoke(app, ["compare", "--regression-threshold", "1.0"])

    assert result.exit_code == EXIT_OK, result.output
    assert "No regressions" in result.output


def test_compare_never_fails_without_a_threshold(metadata: RunMetadata) -> None:
    save_model(combined_report(metadata, {"001": 10.0}), DEFAULT_BASELINE_FILE)
    save_model(combined_report(metadata, {"001": 0.0}), COMBINED_RESULTS_FILE)

    result = runner.invoke(app, ["compare"])

    assert result.exit_code == EXIT_OK, result.output


def test_compare_warns_about_test_cases_on_only_one_side(
    metadata: RunMetadata,
) -> None:
    save_model(
        combined_report(metadata, {"001": 8.0, "002": 8.0}), DEFAULT_BASELINE_FILE
    )
    save_model(
        combined_report(metadata, {"002": 8.0, "003": 8.0}), COMBINED_RESULTS_FILE
    )

    result = runner.invoke(app, ["compare"])

    assert result.exit_code == EXIT_OK, result.output
    assert "001" in result.output and "003" in result.output


def test_compare_requires_a_baseline(metadata: RunMetadata) -> None:
    save_model(combined_report(metadata, {"001": 8.0}), COMBINED_RESULTS_FILE)
    result = runner.invoke(app, ["compare"])
    assert result.exit_code == EXIT_USAGE
    assert "set-baseline" in result.output


def test_compare_reports_a_disjoint_baseline(metadata: RunMetadata) -> None:
    save_model(combined_report(metadata, {"001": 8.0}), DEFAULT_BASELINE_FILE)
    save_model(combined_report(metadata, {"999": 8.0}), COMBINED_RESULTS_FILE)

    result = runner.invoke(app, ["compare"])

    assert result.exit_code == EXIT_ERROR
    assert "no test cases in common" in result.output


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "command",
    ["init-dataset", "generate", "evaluate", "set-baseline", "compare"],
)
def test_every_command_documents_itself(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == EXIT_OK
    assert result.output.strip()


def test_bare_invocation_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Commands" in result.output


def test_a_missing_api_key_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset
) -> None:
    """The whole point of lazy settings: a clean message, not a traceback."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from prompt_eval.config import get_settings

    get_settings.cache_clear()
    save_model(dataset, DEFAULT_DATASET_FILE)

    result = runner.invoke(app, ["generate"])

    assert result.exit_code == EXIT_ERROR
    assert "ANTHROPIC_API_KEY" in result.output


def test_interrupting_a_run_exits_quietly(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset
) -> None:
    """^C during a long generation must not dump a task-group traceback."""
    save_model(dataset, DEFAULT_DATASET_FILE)

    async def interrupt(*_: object, **__: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "prompt_eval.cli.commands.generate.generate_solutions", interrupt
    )

    result = runner.invoke(app, ["generate"])

    assert result.exit_code == EXIT_ERROR
    assert "Interrupted" in result.output
    assert not DEFAULT_SOLUTIONS_FILE.exists()
