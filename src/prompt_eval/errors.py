class PromptEvalError(Exception):
    """Base class for every expected (i.e. reportable) failure."""


class ConfigurationError(PromptEvalError):
    """Settings are missing or invalid (e.g. no API key)."""


class ArtifactError(PromptEvalError):
    """A JSON artifact could not be read, written or validated."""


class DatasetError(PromptEvalError):
    """The dataset is missing, empty or inconsistent with the solutions."""


class PromptError(PromptEvalError):
    """A prompt template is missing or has unfilled placeholders."""


class LLMError(PromptEvalError):
    """The model call failed or returned something unusable."""
