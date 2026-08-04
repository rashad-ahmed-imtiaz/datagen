from __future__ import annotations

import pytest

from scenario_data_factory.blueprints.registry import get_blueprint
from scenario_data_factory.exceptions import ScenarioRevisionConflict
from scenario_data_factory.persistence.scenario_repository import ScenarioRepository


def test_repository_rejects_stale_revision(tmp_path) -> None:
    repo = ScenarioRepository(tmp_path)
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    repo.save(spec)
    repo.patch(spec.scenario_id, spec.revision, {"name": "changed"})
    with pytest.raises(ScenarioRevisionConflict):
        repo.patch(spec.scenario_id, spec.revision, {"name": "stale"})
