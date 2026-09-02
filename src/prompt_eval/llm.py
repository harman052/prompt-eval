from __future__ import annotations

from types import TracebackType
from typing import Any, Self

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import Message
from pydantic import BaseModel

from prompt_eval.config import Settings, get_settings
from prompt_eval.errors import LLMError

MAX_ERROR_DETAIL = 300


class LLMClient:
    """Async Anthropic client.

    The SDK already implements exponential-backoff retries for connection
    errors, 408/409/429 and 5xx, so ``max_retries`` is configured here instead
    of hand-rolling a retry loop.
    """

    def __init__(
        self,
        *,
        client: AsyncAnthropic | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or AsyncAnthropic(
            api_key=self._settings.anthropic_api_key.get_secret_value(),
            timeout=self._settings.request_timeout_seconds,
            max_retries=self._settings.max_retries,
        )

    @property
    def model(self) -> str:
        return self._settings.claude_model

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.close()

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Send ``prompt`` and return the response text."""
        try:
            response = await self._client.messages.create(
                **self._request_kwargs(prompt, system)
            )
        except anthropic.AnthropicError as exc:
            raise translate_sdk_error(exc) from exc
        return self._extract_text(response)

    async def parse[SchemaT: BaseModel](
        self, prompt: str, schema: type[SchemaT], *, system: str | None = None
    ) -> SchemaT:
        """Send ``prompt`` and validate the response against ``schema``.

        Uses the API's structured-output support, which constrains generation to
        the schema.
        """
        try:
            response = await self._client.messages.parse(
                **self._request_kwargs(prompt, system),
                output_format=schema,
            )
        except anthropic.AnthropicError as exc:
            raise translate_sdk_error(exc) from exc

        self._check_stop_reason(response)
        if response.parsed_output is None:
            raise LLMError(
                f"Model returned no {schema.__name__} output "
                f"(stop_reason={response.stop_reason!r})."
            )
        return response.parsed_output

    def _request_kwargs(self, prompt: str, system: str | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._settings.claude_model,
            "max_tokens": self._settings.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system
        return kwargs

    def _extract_text(self, response: Message) -> str:
        """Concatenate the text blocks of a response.

        ``response.content[0].text`` is unsafe: the first block can be a
        thinking or tool-use block, which has no ``.text`` at all.
        """
        self._check_stop_reason(response)
        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            raise LLMError(
                f"Model returned no text content (stop_reason={response.stop_reason!r})."
            )
        return text

    @staticmethod
    def _check_stop_reason(response: Message) -> None:
        if response.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request.")
        if response.stop_reason == "max_tokens":
            raise LLMError(
                "The response was truncated at max_tokens. "
                "Increase MAX_TOKENS and re-run."
            )


def translate_sdk_error(exc: anthropic.AnthropicError) -> LLMError:
    """Map an SDK exception onto an :class:`LLMError` with actionable advice."""
    match exc:
        case anthropic.AuthenticationError():
            return LLMError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.")
        case anthropic.PermissionDeniedError():
            return LLMError("The API key lacks permission for this model.")
        case anthropic.NotFoundError():
            return LLMError("Unknown model or endpoint. Check CLAUDE_MODEL.")
        case anthropic.RateLimitError():
            return LLMError(
                "Rate limited after exhausting retries. "
                "Lower MAX_CONCURRENCY or retry later."
            )
        case anthropic.BadRequestError():
            return LLMError(f"Anthropic rejected the request: {_detail(exc)}")
        case anthropic.APIStatusError():
            return LLMError(f"Anthropic API error {exc.status_code}: {_detail(exc)}")
        case anthropic.APITimeoutError():
            return LLMError(
                "The request timed out after exhausting retries. "
                "Increase REQUEST_TIMEOUT_SECONDS."
            )
        case anthropic.APIConnectionError():
            return LLMError(f"Could not reach the Anthropic API: {exc}")
        case _:
            return LLMError(f"Unexpected Anthropic SDK failure: {exc}")


def _detail(exc: anthropic.APIStatusError) -> str:
    return str(getattr(exc, "message", exc))[:MAX_ERROR_DETAIL]
