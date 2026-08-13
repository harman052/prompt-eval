from typing import TypeVar

from anthropic import Anthropic
from pydantic import BaseModel

from .config import settings

T = TypeVar("T", bound=BaseModel)


class LLMClient:
  def __init__(self) -> None:
    self.client = Anthropic(api_key=settings.anthropic_api_key)

  def chat(
    self,
    messages: list[dict[str, str]],
  ) -> str:
    response = self.client.messages.create(
      **self._request_config(messages),
    )

    return response.content[0].text

  def parse(
    self,
    messages: list[dict[str, str]],
    output_format: type[T],
  ) -> T:
    response = self.client.messages.parse(
      **self._request_config(messages),
      output_format=output_format,
    )

    return response.parsed_output

  def _request_config(
    self,
    messages: list[dict[str, str]],
  ) -> dict:
    return {
      "model": settings.claude_model,
      "max_tokens": settings.max_tokens,
      "messages": messages,
    }
