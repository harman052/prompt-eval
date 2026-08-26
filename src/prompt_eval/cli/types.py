from enum import Enum


class Grader(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM_JUDGE = "llm-judge"
    BOTH = "both"
