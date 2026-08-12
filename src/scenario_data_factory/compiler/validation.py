from __future__ import annotations

from scenario_data_factory.compiler.dependency_graph import dependency_order
from scenario_data_factory.exceptions import ScenarioValidationError, UnsupportedIssueError
from scenario_data_factory.issues.registry import ISSUE_REGISTRY
from scenario_data_factory.models.scenario import IssueType, ScenarioSpec

RAW_REQUIRED = {IssueType.FILE_REPLAY, IssueType.SCHEMA_DRIFT, IssueType.OUT_OF_ORDER}


def validate_scenario(spec: ScenarioSpec) -> list[str]:
    warnings: list[str] = []
    try:
        ScenarioSpec.model_validate(spec.model_dump(mode="json"))
        dependency_order(spec.tables, spec.relationships)
    except Exception as exc:
        if isinstance(exc, ScenarioValidationError):
            raise
        raise ScenarioValidationError(
            "INVALID_SCENARIO",
            "ScenarioSpec failed validation.",
            technical_detail=str(exc),
            scenario_id=spec.scenario_id,
        ) from exc

    for issue in spec.issues:
        if issue.type not in ISSUE_REGISTRY:
            raise UnsupportedIssueError(
                "UNSUPPORTED_ISSUE",
                f"Unsupported issue type: {issue.type}",
                scenario_id=spec.scenario_id,
            )
        plugin = ISSUE_REGISTRY[IssueType(issue.type)]
        plugin.validate(spec, issue)
        issue_type = IssueType(issue.type)
        if issue_type in RAW_REQUIRED and spec.outputs.mode == "delta":
            raise ScenarioValidationError(
                "RAW_OUTPUT_REQUIRED",
                f"{issue.type} requires raw output; output mode cannot be delta only.",
                scenario_id=spec.scenario_id,
            )
        if issue_type == IssueType.SCHEMA_DRIFT and spec.outputs.raw_format != "json":
            raise ScenarioValidationError(
                "JSON_RAW_OUTPUT_REQUIRED",
                "schema_drift requires JSON raw output so batches can have different fields.",
                scenario_id=spec.scenario_id,
            )
    return warnings
