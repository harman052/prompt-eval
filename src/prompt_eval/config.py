from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from prompt_eval.errors import ConfigurationError


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    anthropic_api_key: SecretStr = Field(
        description="Anthropic API key used for generation and LLM-as-judge grading."
    )
    claude_model: str = Field(
        default="claude-haiku-4-5",
        min_length=1,
        description="Model id used for both solution generation and grading.",
    )
    max_tokens: int = Field(
        default=1000, ge=1, description="Output token ceiling per request."
    )
    max_concurrency: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Maximum number of in-flight model requests.",
    )
    request_timeout_seconds: float = Field(
        default=120.0, gt=0, description="Per-request timeout."
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Retries for transient failures (429/5xx/network), "
        "performed by the Anthropic SDK with exponential backoff.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings."""
    try:
        return Settings()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        missing = ", ".join(str(error["loc"][0]) for error in exc.errors())
        raise ConfigurationError(
            f"Invalid configuration ({missing}). "
            "Set ANTHROPIC_API_KEY in your environment or in a .env file."
        ) from exc
