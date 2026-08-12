from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from scenario_data_factory.blueprints.registry import get_blueprint
from scenario_data_factory.exceptions import ScenarioDataFactoryError, ScenarioRevisionConflict
from scenario_data_factory.persistence.run_repository import RunRepository
from scenario_data_factory.persistence.scenario_repository import ScenarioRepository


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
