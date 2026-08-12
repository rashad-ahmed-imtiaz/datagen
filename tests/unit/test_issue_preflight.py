from __future__ import annotations

import pytest
from pydantic import ValidationError

from scenario_data_factory.blueprints.registry import get_blueprint
from scenario_data_factory.compiler.validation import validate_scenario
from scenario_data_factory.exceptions import IssuePlanningError
from scenario_data_factory.models.scenario import IssueSpec, IssueType, ScenarioSpec


@pytest.mark.parametrize(
    "issue_type",
    [
        IssueType.NULL_VALUE,
        IssueType.BLANK_VALUE,
        IssueType.INVALID_FORMAT,
        IssueType.INVALID_VALUE,
        IssueType.REFERENTIAL_ORPHAN,
        IssueType.DATE_RULE_VIOLATION,
        IssueType.LATE_ARRIVAL,
        IssueType.CORRELATED_MISSINGNESS,
    ],
)
def test_column_targeted_issue_requires_column(issue_type: IssueType) -> None:
    with pytest.raises(ValidationError, match="requires a target column"):
        IssueSpec(type=issue_type, table="claims", exact_count=1)


def test_registered_issue_types_have_an_executable_blueprint_example() -> None:
    spec = get_blueprint("insurance_claims").build(name="issue matrix", seed=42, scale="small")
    assert {IssueType(issue.type) for issue in spec.issues} == {
        IssueType.INVALID_VALUE,
        IssueType.NULL_VALUE,
        IssueType.DATE_RULE_VIOLATION,
        IssueType.DUPLICATE_RECORD,
        IssueType.REFERENTIAL_ORPHAN,
        IssueType.LATE_ARRIVAL,
        IssueType.FILE_REPLAY,
        IssueType.SCHEMA_DRIFT,
        IssueType.CORRELATED_MISSINGNESS,
    }
    validate_scenario(spec)


@pytest.mark.parametrize(
    ("issue_type", "column"),
    [
        (IssueType.BLANK_VALUE, "claim_amount"),
        (IssueType.INVALID_FORMAT, "claim_amount"),
        (IssueType.REFERENTIAL_ORPHAN, "adjuster_id"),
    ],
)
def test_preflight_rejects_unsafe_column_type(issue_type: IssueType, column: str) -> None:
    data = get_blueprint("insurance_claims").build(
        name="invalid issue", seed=42, scale="small"
    ).model_dump(mode="json")
    data["issues"] = [
        {"type": issue_type, "table": "claims", "column": column, "exact_count": 1}
    ]
    with pytest.raises(IssuePlanningError, match="ISSUE_COLUMN_TYPE_INVALID"):
        validate_scenario(ScenarioSpec.model_validate(data))


@pytest.mark.parametrize(
    "parameters",
    [
        {"source_batch": 12, "target_batch": 10},
        {"source_batch": 31, "target_batch": 32},
    ],
)
def test_preflight_rejects_invalid_replay_batches(parameters: dict[str, int]) -> None:
    data = get_blueprint("insurance_claims").build(
        name="invalid replay", seed=42, scale="small"
    ).model_dump(mode="json")
    data["issues"] = [
        {
            "type": IssueType.FILE_REPLAY,
            "table": "claims",
            "exact_count": 1,
            "parameters": parameters,
        }
    ]
    with pytest.raises(IssuePlanningError, match="ISSUE_BATCH_INVALID"):
        validate_scenario(ScenarioSpec.model_validate(data))


def test_preflight_rejects_late_arrival_on_non_temporal_column() -> None:
    data = get_blueprint("insurance_claims").build(
        name="invalid late", seed=42, scale="small"
    ).model_dump(mode="json")
    data["issues"] = [
        {
            "type": IssueType.LATE_ARRIVAL,
            "table": "payments",
            "column": "amount",
            "exact_count": 1,
        }
    ]
    with pytest.raises(IssuePlanningError, match="ISSUE_PARAMETER_INVALID"):
        validate_scenario(ScenarioSpec.model_validate(data))
