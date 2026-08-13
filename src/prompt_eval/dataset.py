import json
from pathlib import Path

from .models import TestCase


def load_dataset(path: Path) -> list[TestCase]:
    data = json.loads(path.read_text())

    return [TestCase.model_validate(item) for item in data]
