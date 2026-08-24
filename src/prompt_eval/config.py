from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  anthropic_api_key: str
  claude_model: str
  max_tokens: int

  model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()  # pyright: ignore[reportCallIssue]
