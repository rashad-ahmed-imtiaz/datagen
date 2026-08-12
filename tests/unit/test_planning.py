from __future__ import annotations

import pytest

from scenario_data_factory.blueprints.registry import get_blueprint
from scenario_data_factory.compiler.datagen_compiler import compile_table
from scenario_data_factory.compiler.dependency_graph import dependency_order
from scenario_data_factory.compiler.validation import validate_scenario
from scenario_data_factory.exceptions import ScenarioValidationError
from scenario_data_factory.issues.planner import build_issue_plan, estimate_issues
from scenario_data_factory.issues.registry import ISSUE_REGISTRY
from scenario_data_factory.models.scenario import IssueType, RelationshipSpec


def test_all_required_issue_types_are_registered() -> None:
    assert set(ISSUE_REGISTRY) == set(IssueType)


def test_issue_counts_resolve_exact_and_rates() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    counts = estimate_issues(spec)
    assert counts["iss_duplicate_claims"] == 9
    assert counts["iss_missing_adjusters"] == 15
    assert counts["iss_invalid_customer_provinces"] == 1


def test_issue_targeting_is_deterministic() -> None:
    left = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    right = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    right = right.model_copy(update={"scenario_id": left.scenario_id})
    assert build_issue_plan(left) == build_issue_plan(right)


def test_dependency_order_parents_before_children() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    order = dependency_order(spec.tables, spec.relationships)
    assert order.index("customers") < order.index("policies") < order.index("claims")
    assert order.index("claims") < order.index("payments")


def test_cycle_detection() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    relationships = [
        *spec.relationships,
        RelationshipSpec(
            name="cycle",
            parent_table="payments",
            parent_column="claim_id",
            child_table="customers",
            child_column="customer_id",
        ),
    ]
    with pytest.raises(ScenarioValidationError, match="RELATIONSHIP_CYCLE"):
        dependency_order(spec.tables, relationships)


def test_raw_output_warning_for_physical_issues() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    data = spec.model_dump(mode="json")
    data["outputs"]["mode"] = "delta"
    warnings = validate_scenario(type(spec).model_validate(data))
    assert any("physical/raw-file issue" in warning for warning in warnings)


def test_default_string_compilation_avoids_photon_format_path() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    compiled = compile_table(spec.table("customers"))
    first_name = next(column for column in compiled.columns if column.name == "first_name")
    assert first_name.options["faker"] == "first_name"
    assert "format" not in first_name.options
