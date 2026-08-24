from typing import TypeVar

from anthropic import Anthropic
from pydantic import BaseModel

from .config import settings

T = TypeVar("T", bound=BaseModel)
Messages = list[dict[str, str]]


class LLMClient:
    def __init__(self) -> None:
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def chat(self, messages: Messages, stop_sequences: list[str] | None = None) -> str:
        response = self.client.messages.create(
            **self._request_config(messages, stop_sequences),
        )

        return response.content[0].text

    def parse(self, messages: Messages, output_format: type[T]) -> T:
        response = self.client.messages.parse(
            **self._request_config(messages),
            output_format=output_format,
        )

        if response.parsed_output is None:
            raise ValueError("LLM returned no parsed output")

        return response.parsed_output

    def _request_config(
        self, messages: Messages, stop_sequences: list[str] | None = None
    ) -> dict:
        return {
            "model": settings.claude_model,
            "max_tokens": settings.max_tokens,
            "messages": messages,
            "stop_sequences": stop_sequences,
        }
