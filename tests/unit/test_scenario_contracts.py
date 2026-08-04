from __future__ import annotations

import pytest

from scenario_data_factory.blueprints.registry import get_blueprint
from scenario_data_factory.models.scenario import LifecycleState, ScenarioSpec


def test_insurance_blueprint_has_four_related_tables() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    assert [t.name for t in spec.tables] == ["customers", "policies", "claims", "payments"]
    assert {r.name for r in spec.relationships} == {
        "customers_policies",
        "policies_claims",
        "claims_payments",
    }


def test_canonical_hash_is_stable_across_yaml_round_trip() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    loaded = ScenarioSpec.from_yaml(spec.to_yaml())
    assert loaded.canonical_json() == spec.canonical_json()
    assert loaded.spec_hash() == spec.spec_hash()


def test_invalid_relationship_column_is_rejected() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    data = spec.model_dump(mode="json")
    data["relationships"][0]["parent_column"] = "missing"
    with pytest.raises(ValueError, match="missing parent column"):
        ScenarioSpec.model_validate(data)


def test_lifecycle_rejects_invalid_transition() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        spec.transition(LifecycleState.SUCCEEDED)


def test_lifecycle_accepts_validate_transition_and_bumps_revision() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    updated = spec.transition(LifecycleState.VALIDATED)
    assert updated.lifecycle_state == LifecycleState.VALIDATED
    assert updated.revision == spec.revision + 1
