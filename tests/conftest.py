from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from prompt_eval.config import Settings, get_settings
from prompt_eval.models import (
    Dataset,
    GeneratedSolution,
    ModelGrade,
    RunMetadata,
    SolutionFormat,
    SolutionReport,
    SolutionSpec,
    TestCase,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Provide deterministic settings and stop ``.env`` from leaking in."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CLAUDE_MODEL", "test-model")
    monkeypatch.setenv("MAX_TOKENS", "1024")
    monkeypatch.setenv("MAX_CONCURRENCY", "4")
    # Rich falls back to an 80-column terminal when stdout is captured, which
    # truncates table cells and makes output assertions flaky.
    monkeypatch.setenv("COLUMNS", "300")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run each test in an isolated working directory with real prompts."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts").mkdir()
    for template in (REPO_ROOT / "prompts").glob("*.txt"):
        (tmp_path / "prompts" / template.name).write_text(template.read_text())

    from prompt_eval.prompts import load_prompt
    from prompt_eval.versioning import get_prompt_version

    load_prompt.cache_clear()
    get_prompt_version.cache_clear()
    return tmp_path


class FakeLLMClient:
    """Stand-in for :class:`prompt_eval.llm.LLMClient`.

    Records every prompt it receives so tests can assert on prompt rendering,
    and lets each test decide what ``parse`` returns per schema - including
    raising, to exercise the failure paths.
    """

    def __init__(
        self,
        *,
        responses: dict[type[BaseModel], Any] | None = None,
        text: str = "fake text",
    ) -> None:
        self.responses = responses or {}
        self.text = text
        self.prompts: list[str] = []
        self.closed = False
        self.max_in_flight = 0
        self._in_flight = 0

    @property
    def model(self) -> str:
        return "test-model"

    async def __aenter__(self) -> FakeLLMClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self.closed = True

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.text

    async def parse[SchemaT: BaseModel](
        self, prompt: str, schema: type[SchemaT], *, system: str | None = None
    ) -> SchemaT:
        self.prompts.append(prompt)
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(0)  # yield, so concurrency is observable
            response = self.responses.get(schema)
            if isinstance(response, Exception):
                raise response
            if callable(response):
                response = response(prompt)
            if response is None:
                raise AssertionError(f"FakeLLMClient has no response for {schema}")
            return cast(SchemaT, response)
        finally:
            self._in_flight -= 1


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient(
        responses={
            ModelGrade: ModelGrade(
                strengths=["correct"],
                weaknesses=[],
                reasoning="Meets every criterion.",
                score=8.0,
            ),
            SolutionSpec: SolutionSpec(code="x = 1"),
        }
    )


@pytest.fixture
def make_test_case() -> Callable[..., TestCase]:
    def factory(
        test_case_id: str = "001",
        solution_format: SolutionFormat = SolutionFormat.PYTHON,
        task: str = "Write a boto3 snippet",
    ) -> TestCase:
        return TestCase(
            test_case_id=test_case_id,
            task=task,
            format=solution_format,
            solution_criteria="It must run",
        )

    return factory


@pytest.fixture
def metadata() -> RunMetadata:
    return RunMetadata(
        prompt_version="abc1234", model="test-model", run_at="2026-01-01T00:00:00Z"
    )


@pytest.fixture
def dataset(make_test_case: Callable[..., TestCase]) -> Dataset:
    return Dataset(
        root=[
            make_test_case("001", SolutionFormat.PYTHON, "Write python"),
            make_test_case("002", SolutionFormat.JSON, "Write json"),
        ]
    )


@pytest.fixture
def solutions(dataset: Dataset, metadata: RunMetadata) -> SolutionReport:
    bodies = {SolutionFormat.PYTHON: "x = 1", SolutionFormat.JSON: '{"a": 1}'}
    return SolutionReport(
        metadata=metadata,
        results=[
            GeneratedSolution(
                test_case_id=case.test_case_id,
                task=case.task,
                format=case.format,
                solution=bodies[case.format],
            )
            for case in dataset.root
        ],
    )
