from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import SimpleNamespace

import pytest

import scenario_data_factory.persistence.run_repository as run_repository
import scenario_data_factory.persistence.scenario_repository as scenario_repository
from scenario_data_factory.blueprints.registry import get_blueprint
from scenario_data_factory.exceptions import ScenarioDataFactoryError, ScenarioRevisionConflict
from scenario_data_factory.persistence.run_repository import RunRepository
from scenario_data_factory.persistence.scenario_repository import ScenarioRepository


class _VolumeFiles:
    def __init__(self) -> None:
        self.directories: list[str] = []
        self.contents: dict[str, bytes] = {}

    def create_directory(self, path: str) -> None:
        self.directories.append(path)

    def upload(self, path: str, contents, *, overwrite: bool) -> None:
        assert overwrite
        self.contents[path] = contents.read()

    def download(self, path: str):
        return SimpleNamespace(contents=BytesIO(self.contents[path]))

    def list_directory_contents(self, _path: str):
        return [
            SimpleNamespace(
                name=path.rsplit("/", 1)[-1], path=path, is_directory=False, last_modified=1
            )
            for path in self.contents
        ]


def test_repository_rejects_stale_revision(tmp_path) -> None:
    repo = ScenarioRepository(tmp_path)
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    repo.save(spec)
    repo.patch(spec.scenario_id, spec.revision, {"name": "changed"})
    with pytest.raises(ScenarioRevisionConflict):
        repo.patch(spec.scenario_id, spec.revision, {"name": "stale"})


@pytest.mark.parametrize("identifier", ["../secret", "scn_../secret", "scn_bad/name"])
def test_scenario_repository_rejects_path_traversal(tmp_path, identifier: str) -> None:
    repository = ScenarioRepository(tmp_path / "scenarios")
    with pytest.raises(ScenarioDataFactoryError, match="INVALID_SCENARIO_ID"):
        repository.get(identifier)


def test_run_repository_rejects_path_traversal(tmp_path) -> None:
    repository = RunRepository(tmp_path / "runs")
    with pytest.raises(ScenarioDataFactoryError, match="INVALID_RUN_ID"):
        repository.get("../secret")


def test_concurrent_scenario_patch_has_one_winner(tmp_path) -> None:
    repository = ScenarioRepository(tmp_path / "scenarios")
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    repository.save(spec)

    def patch(name: str) -> str:
        try:
            return repository.patch(spec.scenario_id, 1, {"name": name}).name
        except ScenarioRevisionConflict:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(patch, ["first", "second"]))
    assert results.count("stale") == 1
    assert repository.get(spec.scenario_id).revision == 2


def test_repositories_use_databricks_files_api_for_volume_roots(monkeypatch) -> None:
    files = _VolumeFiles()
    monkeypatch.setattr(scenario_repository, "_workspace_files", lambda: files)
    monkeypatch.setattr(run_repository, "_workspace_files", lambda: files)
    scenarios = ScenarioRepository("/Volumes/sdf/schema/control/drafts")
    runs = RunRepository("/Volumes/sdf/schema/control/runs")
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")

    scenarios.save(spec)
    assert scenarios.get(spec.scenario_id).spec_hash() == spec.spec_hash()
    run = runs.create(spec.scenario_id, "generation", spec.spec_hash())
    assert runs.get(str(run["run_id"]))["status"] == "prepared"
    assert files.directories == [
        "/Volumes/sdf/schema/control/drafts",
        "/Volumes/sdf/schema/control/runs",
    ]
