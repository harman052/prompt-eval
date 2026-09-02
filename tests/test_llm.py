"""LLMClient: response extraction and SDK error translation.

The client is faked at the SDK boundary (``AsyncAnthropic``) rather than at our
own boundary, so these tests cover the code that actually reads Anthropic's
response shape - the part most likely to break on an SDK or model change.
"""

from __future__ import annotations

from typing import Any

import anthropic
import httpx
import pytest

from prompt_eval.config import Settings
from prompt_eval.errors import LLMError
from prompt_eval.llm import LLMClient, translate_sdk_error
from prompt_eval.models import ModelGrade


class Block:
    def __init__(self, block_type: str, text: str = "") -> None:
        self.type = block_type
        self.text = text


class Response:
    def __init__(
        self,
        content: list[Block] | None = None,
        stop_reason: str = "end_turn",
        parsed_output: Any = None,
    ) -> None:
        self.content = content or []
        self.stop_reason = stop_reason
        self.parsed_output = parsed_output


class StubMessages:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def _respond(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    create = _respond
    parse = _respond


class StubAnthropic:
    def __init__(self, response: Any) -> None:
        self.messages = StubMessages(response)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def client_for(response: Any, settings: Settings) -> tuple[LLMClient, StubAnthropic]:
    stub = StubAnthropic(response)
    return LLMClient(client=stub, settings=settings), stub  # type: ignore[arg-type]


async def test_complete_joins_text_blocks_and_skips_others(
    settings: Settings,
) -> None:
    """``content[0].text`` is unsafe - thinking blocks have no text."""
    response = Response(
        [Block("thinking"), Block("text", "line one"), Block("text", "line two")]
    )
    client, _ = client_for(response, settings)
    assert await client.complete("hi") == "line one\nline two"


async def test_complete_sends_configured_model_and_limits(
    settings: Settings,
) -> None:
    client, stub = client_for(Response([Block("text", "ok")]), settings)
    await client.complete("hi", system="be terse")
    (call,) = stub.messages.calls
    assert call["model"] == settings.claude_model
    assert call["max_tokens"] == settings.max_tokens
    assert call["system"] == "be terse"
    assert call["messages"] == [{"role": "user", "content": "hi"}]


async def test_complete_omits_system_when_unset(settings: Settings) -> None:
    client, stub = client_for(Response([Block("text", "ok")]), settings)
    await client.complete("hi")
    assert "system" not in stub.messages.calls[0]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (Response([]), "no text content"),
        (Response([Block("text", "   ")]), "no text content"),
        (Response([Block("text", "x")], stop_reason="refusal"), "declined"),
        (Response([Block("text", "x")], stop_reason="max_tokens"), "truncated"),
    ],
)
async def test_complete_rejects_unusable_responses(
    response: Response, message: str, settings: Settings
) -> None:
    client, _ = client_for(response, settings)
    with pytest.raises(LLMError, match=message):
        await client.complete("hi")


async def test_parse_returns_the_validated_model(settings: Settings) -> None:
    grade = ModelGrade(strengths=[], weaknesses=[], reasoning="fine", score=7)
    client, stub = client_for(Response(parsed_output=grade), settings)
    assert await client.parse("hi", ModelGrade) is grade
    assert stub.messages.calls[0]["output_format"] is ModelGrade


async def test_parse_rejects_a_missing_parsed_output(settings: Settings) -> None:
    client, _ = client_for(Response(parsed_output=None), settings)
    with pytest.raises(LLMError, match="no ModelGrade output"):
        await client.parse("hi", ModelGrade)


async def test_parse_reports_a_refusal(settings: Settings) -> None:
    client, _ = client_for(Response(stop_reason="refusal"), settings)
    with pytest.raises(LLMError, match="declined"):
        await client.parse("hi", ModelGrade)


async def test_sdk_errors_become_llm_errors(settings: Settings) -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    failure = anthropic.RateLimitError(
        "slow down",
        response=httpx.Response(429, request=request),
        body=None,
    )
    client, _ = client_for(failure, settings)
    with pytest.raises(LLMError, match="Rate limited") as exc_info:
        await client.complete("hi")
    assert isinstance(exc_info.value.__cause__, anthropic.RateLimitError)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, "rejected the API key"),
        (403, "lacks permission"),
        (404, "Unknown model"),
        (429, "Rate limited"),
        (400, "rejected the request"),
        (503, "API error 503"),
    ],
)
def test_translate_sdk_error_is_specific_per_status(
    status_code: int, expected: str
) -> None:
    """Most-specific-first matching: each status gets actionable advice."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    error = anthropic.APIStatusError.__new__(
        {
            401: anthropic.AuthenticationError,
            403: anthropic.PermissionDeniedError,
            404: anthropic.NotFoundError,
            429: anthropic.RateLimitError,
            400: anthropic.BadRequestError,
            503: anthropic.InternalServerError,
        }[status_code]
    )
    anthropic.APIStatusError.__init__(error, "boom", response=response, body=None)
    assert expected in str(translate_sdk_error(error))


def test_translate_sdk_error_handles_connection_failures() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    assert "timed out" in str(translate_sdk_error(anthropic.APITimeoutError(request)))
    assert (
        "could not reach"
        in str(
            translate_sdk_error(anthropic.APIConnectionError(request=request))
        ).lower()
    )


async def test_aclose_releases_the_connection_pool(settings: Settings) -> None:
    stub = StubAnthropic(Response([Block("text", "ok")]))
    async with LLMClient(client=stub, settings=settings):  # type: ignore[arg-type]
        pass
    assert stub.closed
