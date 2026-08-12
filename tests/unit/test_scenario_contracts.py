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


def test_relationship_key_type_mismatch_is_rejected() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    data = spec.model_dump(mode="json")
    for column in data["tables"][1]["columns"]:
        if column["name"] == "customer_id":
            column["type"] = "string"
    with pytest.raises(ValueError, match="key column types must match"):
        ScenarioSpec.model_validate(data)


def test_duplicate_issue_ids_and_oversized_counts_are_rejected() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    duplicate_ids = spec.model_dump(mode="json")
    duplicate_ids["issues"][1]["issue_id"] = duplicate_ids["issues"][0]["issue_id"]
    with pytest.raises(ValueError, match="duplicate issue IDs"):
        ScenarioSpec.model_validate(duplicate_ids)

    oversized_count = spec.model_dump(mode="json")
    oversized_count["issues"][0]["exact_count"] = 101
    with pytest.raises(ValueError, match="exact_count exceeds table row count"):
        ScenarioSpec.model_validate(oversized_count)


def test_unsafe_delta_identifiers_are_rejected() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    data = spec.model_dump(mode="json")
    data["outputs"]["catalog"] = "sdf`; DROP TABLE x; --"
    with pytest.raises(ValueError, match="valid identifiers"):
        ScenarioSpec.model_validate(data)


def test_weighted_values_cannot_be_silently_dropped() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    data = spec.model_dump(mode="json")
    for column in data["tables"][0]["columns"]:
        if column["name"] == "province":
            column["weights"] = [1]
    with pytest.raises(ValueError, match="weights must align"):
        ScenarioSpec.model_validate(data)


def test_relationships_cannot_overwrite_the_same_child_column() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    data = spec.model_dump(mode="json")
    data["relationships"].append(
        {
            "name": "duplicate_claim_policy",
            "parent_table": "policies",
            "parent_column": "policy_id",
            "child_table": "claims",
            "child_column": "policy_id",
        }
    )
    with pytest.raises(ValueError, match="multiple relationships"):
        ScenarioSpec.model_validate(data)


def test_scenario_id_and_name_are_bounded() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    data = spec.model_dump(mode="json")
    data["scenario_id"] = "../unsafe"
    with pytest.raises(ValueError, match="scenario_id"):
        ScenarioSpec.model_validate(data)

    data = spec.model_dump(mode="json")
    data["name"] = "x" * 121
    with pytest.raises(ValueError, match="scenario name"):
        ScenarioSpec.model_validate(data)


def test_relationship_filter_and_constraints_must_be_executable() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    data = spec.model_dump(mode="json")
    data["relationships"][0]["parent_filter"] = {
        "column": "province",
        "values": ["not_a_real_province"],
    }
    with pytest.raises(ValueError, match="cannot match parent values"):
        ScenarioSpec.model_validate(data)

    data = spec.model_dump(mode="json")
    data["relationships"][2]["constraints"] = {
        "aggregate_caps": [
            {
                "child_amount_column": "amount",
                "parent_amount_column": "claim_amount",
                "maximum_fraction": 1.5,
            }
        ]
    }
    with pytest.raises(ValueError, match="fraction"):
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
