"""Settings validation and failure reporting."""

from __future__ import annotations

import pytest

from prompt_eval.config import get_settings
from prompt_eval.errors import ConfigurationError


def test_defaults_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CLAUDE_MODEL", "MAX_TOKENS", "MAX_CONCURRENCY"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.claude_model.startswith("claude-")
    assert settings.max_tokens > 0
    assert settings.max_concurrency >= 1


def test_a_missing_api_key_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reported as a clean message naming the fix, not a raw ValidationError.

    The autouse fixture runs each test in a temporary directory, so no local
    ``.env`` can supply the key behind our back.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()

    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        get_settings()


def test_the_api_key_is_not_leaked_by_repr() -> None:
    settings = get_settings()
    assert "test-key" not in repr(settings)
    assert settings.anthropic_api_key.get_secret_value() == "test-key"


def test_settings_are_frozen() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        get_settings().max_tokens = 1  # type: ignore[misc]


def test_settings_are_cached() -> None:
    assert get_settings() is get_settings()


@pytest.mark.parametrize(
    ("name", "value"),
    [("MAX_TOKENS", "0"), ("MAX_CONCURRENCY", "0"), ("MAX_RETRIES", "-1")],
)
def test_invalid_values_are_rejected(
    name: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    with pytest.raises(ConfigurationError):
        get_settings()
