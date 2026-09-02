"""Prompt version stamping."""

from __future__ import annotations

import subprocess

import pytest

from prompt_eval.versioning import UNKNOWN_VERSION, get_prompt_version


def test_falls_back_when_git_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provenance is metadata: losing it must never fail an evaluation."""

    def explode(*_: object, **__: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", explode)
    get_prompt_version.cache_clear()
    assert get_prompt_version() == UNKNOWN_VERSION
    get_prompt_version.cache_clear()


def test_falls_back_outside_a_git_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **__: object) -> None:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(subprocess, "run", fail)
    get_prompt_version.cache_clear()
    assert get_prompt_version() == UNKNOWN_VERSION
    get_prompt_version.cache_clear()


def test_marks_uncommitted_prompt_edits_as_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(["abc1234", " M prompts/judge_prompt.txt"])

    class Completed:
        stdout = ""

    def fake_run(*_: object, **__: object) -> Completed:
        completed = Completed()
        completed.stdout = next(outputs)
        return completed

    monkeypatch.setattr(subprocess, "run", fake_run)
    get_prompt_version.cache_clear()
    assert get_prompt_version() == "abc1234-dirty"
    get_prompt_version.cache_clear()


def test_reports_a_clean_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter(["abc1234", ""])

    class Completed:
        stdout = ""

    def fake_run(*_: object, **__: object) -> Completed:
        completed = Completed()
        completed.stdout = next(outputs)
        return completed

    monkeypatch.setattr(subprocess, "run", fake_run)
    get_prompt_version.cache_clear()
    assert get_prompt_version() == "abc1234"
    get_prompt_version.cache_clear()
