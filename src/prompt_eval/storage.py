"""Reading and writing JSON artifacts."""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, ValidationError

from prompt_eval.errors import ArtifactError


def save_model(model: BaseModel, path: Path) -> Path:
    """Serialise ``model`` to ``path`` atomically.

    The write goes to a temporary file in the destination directory and is then
    renamed, so a crash (or a cancelled CI job) can never leave a half-written
    artifact that a later ``compare`` would fail to parse.
    """
    payload = model.model_dump_json(indent=2)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(path)
    except OSError as exc:
        raise ArtifactError(f"Could not write {path}: {exc}") from exc
    return path


def load_model[ModelT: BaseModel](model: type[ModelT], path: Path) -> ModelT:
    """Load and validate ``path`` as ``model``."""
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ArtifactError(f"{path} does not exist.") from exc
    except OSError as exc:
        raise ArtifactError(f"Could not read {path}: {exc}") from exc

    try:
        return model.model_validate_json(payload)
    except ValidationError as exc:
        raise ArtifactError(
            f"{path} is not a valid {model.__name__} "
            f"({exc.error_count()} validation error(s)): {exc.errors()[0]['msg']}"
        ) from exc
