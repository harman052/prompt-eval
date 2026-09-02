"""Graders: one scoring strategy per module, all on the shared 0-10 scale."""

from prompt_eval.graders.deterministic import grade_syntax
from prompt_eval.graders.judge import grade_with_judge

__all__ = ["grade_syntax", "grade_with_judge"]
