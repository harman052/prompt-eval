from enum import StrEnum


class GraderChoice(StrEnum):
    """Which grader(s) ``evaluate`` should run."""

    DETERMINISTIC = "deterministic"
    LLM_JUDGE = "llm-judge"
    BOTH = "both"
