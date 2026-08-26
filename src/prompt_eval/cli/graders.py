from enum import Enum

import typer
from rich.console import Console
from rich.table import Table

from prompt_eval.graders.deterministic_grader import deterministic_grader
from prompt_eval.models import TestCase

console = Console()

app = typer.Typer()


class Grader(str, Enum):
    deterministic = "deterministic"
    llm_judge = "llm-judge"
    both = "both"


def deterministic(test_cases: list[TestCase]) -> None:
    results = []
    for test_case in test_cases:
        score = deterministic_grader(
            test_case,
            "\nimport json\nimport sys\nfrom typing import List\n\ndef extract_resources_with_depends_on(template: dict) -> List[str]:\n    \"\"\"Extract all resource logical IDs that have a DependsOn property.\"\"\"\n    resources_with_depends_on = []\n    \n    if 'Resources' not in template:\n        return resources_with_depends_on\n    \n    resources = template['Resources']\n    \n    for logical_id, resource_config in resources.items():\n        if isinstance(resource_config, dict) and 'DependsOn' in resource_config:\n            resources_with_depends_on.append(logical_id)\n    \n    return resources_with_depends_on\n\ndef main():\n    if len(sys.argv) > 1:\n        template_file = sys.argv[1]\n        with open(template_file, 'r') as f:\n            template = json.load(f)\n    else:\n        template = json.load(sys.stdin)\n    \n    result = extract_resources_with_depends_on(template)\n    print(json.dumps(result, indent=2))\n\nif __name__ == '__main__':\n    main()\n",
        )

        results.append(
            {
                "score": score,
                "task": test_case.task,
                "format": test_case.format,
            }
        )
    table = Table(
        "Test Case",
        "Format",
        "Deterministic",
        title="Deterministic Grader Scores",
        width=100,
    )
    for result in results:
        table.add_row(
            result["task"],
            result["format"],
            str(result["score"]),
        )
    console.print(table)
