"""Artifact persistence: atomicity and actionable load failures."""

from __future__ import annotations

from pathlib import Path

import pytest

from prompt_eval.errors import ArtifactError
from prompt_eval.models import Dataset, RunMetadata, SolutionReport
from prompt_eval.storage import load_model, save_model


def test_save_then_load_round_trips(dataset: Dataset, tmp_path: Path) -> None:
    path = save_model(dataset, tmp_path / "nested" / "dataset.json")
    assert load_model(Dataset, path) == dataset


def test_save_creates_missing_parent_directories(
    dataset: Dataset, tmp_path: Path
) -> None:
    path = save_model(dataset, tmp_path / "a" / "b" / "c.json")
    assert path.is_file()


def test_save_overwrites_an_existing_artifact(
    dataset: Dataset, solutions: SolutionReport, tmp_path: Path
) -> None:
    path = tmp_path / "out.json"
    save_model(dataset, path)
    save_model(solutions, path)
    assert load_model(SolutionReport, path) == solutions


def test_save_leaves_no_temporary_files_behind(
    dataset: Dataset, tmp_path: Path
) -> None:
    destination = tmp_path / "artifacts"
    save_model(dataset, destination / "dataset.json")
    assert [path.name for path in destination.iterdir()] == ["dataset.json"]


def test_load_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="does not exist"):
        load_model(Dataset, tmp_path / "absent.json")


def test_load_reports_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(ArtifactError, match="not a valid Dataset"):
        load_model(Dataset, path)


def test_load_reports_a_schema_mismatch(tmp_path: Path) -> None:
    """A results file from an older layout must fail loudly, not partially load."""
    path = tmp_path / "old.json"
    path.write_text('{"metadata": {}, "solutions": []}')
    with pytest.raises(ArtifactError, match="not a valid Report"):
        load_model(SolutionReport, path)


def test_save_reports_an_unwritable_destination(
    dataset: Dataset, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    with pytest.raises(ArtifactError, match="Could not write"):
        save_model(dataset, blocker / "dataset.json")


def test_metadata_datetimes_survive_the_round_trip(
    solutions: SolutionReport, tmp_path: Path
) -> None:
    path = save_model(solutions, tmp_path / "solutions.json")
    loaded = load_model(SolutionReport, path)
    assert isinstance(loaded.metadata, RunMetadata)
    assert loaded.metadata.run_at == solutions.metadata.run_at
