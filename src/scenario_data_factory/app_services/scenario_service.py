from __future__ import annotations

import ast
import json
import os
import re
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from scenario_data_factory.blueprints.registry import get_blueprint
from scenario_data_factory.compiler.validation import validate_scenario
from scenario_data_factory.jobs.preview import preview_scenario
from scenario_data_factory.models.scenario import (
    ColumnSpec,
    ColumnType,
    IssueSpec,
    IssueType,
    OutputMode,
    OutputSpec,
    RelationshipSpec,
    ScenarioSpec,
    TableSpec,
    TimelineSpec,
)
from scenario_data_factory.persistence.run_repository import RunRepository
from scenario_data_factory.persistence.scenario_repository import ScenarioRepository


class AgentPlanningError(ValueError):
    """Raised only after the model planner exhausts its internal recovery attempts."""


_MAX_SCHEMA_DESIGN_ATTEMPTS = 2
_MAX_SCHEMA_REPAIR_ATTEMPTS = 2
_MAX_COLUMN_COMPLETION_ATTEMPTS = 2
_SCHEMA_DESIGN_MAX_TOKENS = 8000
_COMPLEX_SCHEMA_DESIGN_MAX_TOKENS = 12000
_BASIC_SCHEMA_DESIGN_MAX_TOKENS = 4500


class ScenarioService:
    def __init__(
        self,
        scenarios: ScenarioRepository | None = None,
        runs: RunRepository | None = None,
    ) -> None:
        control_root = os.getenv("SDF_CONTROL_VOLUME")
        self.scenarios = scenarios or ScenarioRepository(
            Path(control_root) / "drafts" if control_root else ".sdf/scenarios"
        )
        self.runs = runs or RunRepository(
            Path(control_root) / "runs" if control_root else ".sdf/runs"
        )

    def create_scenario_draft(self, input: dict[str, Any]) -> dict[str, object]:
        spec = get_blueprint(input.get("domain", "insurance_claims")).build(
            name=input["name"],
            seed=int(input.get("seed", 42)),
            scale=input.get("scale", "demo"),
        )
        spec = _ensure_databricks_outputs(spec)
        warnings = validate_scenario(spec)
        self.scenarios.save(spec)
        return _summary(spec, warnings)

    def create_scenario_from_prompt(self, prompt: str) -> dict[str, object]:
        spec, assumptions = _spec_from_prompt(prompt)
        warnings = [*assumptions, *validate_scenario(spec)]
        self.scenarios.save(spec)
        return _summary(spec, warnings)

    def get_scenario_draft(self, scenario_id: str) -> dict[str, object]:
        return _summary(self.scenarios.get(scenario_id), [])

    def patch_scenario_draft(
        self, scenario_id: str, expected_revision: int, patch: dict[str, Any]
    ) -> dict[str, object]:
        spec = self.scenarios.patch(scenario_id, expected_revision, patch)
        warnings = validate_scenario(spec)
        return _summary(spec, warnings)

    def validate_scenario_draft(self, scenario_id: str) -> dict[str, object]:
        spec = self.scenarios.get(scenario_id)
        return _summary(spec, validate_scenario(spec))

    def estimate_scenario(self, scenario_id: str) -> dict[str, object]:
        return preview_scenario(self.scenarios.get(scenario_id))

    def prepare_preview(self, scenario_id: str) -> dict[str, object]:
        spec = self.scenarios.get(scenario_id)
        run = self.runs.create(scenario_id, "preview", spec.spec_hash())
        run["summary"] = preview_scenario(spec)
        self.runs.save(run)
        return run

    def prepare_generation(self, scenario_id: str) -> dict[str, object]:
        spec = self.scenarios.get(scenario_id)
        run = self.runs.create(scenario_id, "generation", spec.spec_hash())
        run["requires_confirmation_hash"] = spec.spec_hash()
        self.runs.save(run)
        return run

    def confirm_generation(self, run_id: str, confirmation_hash: str) -> dict[str, object]:
        run = self.runs.get(run_id)
        if run["status"] in {"submitted", "running", "succeeded"}:
            return run
        if confirmation_hash != run["spec_hash"]:
            return self.runs.update_status(run_id, "rejected", reason="confirmation hash mismatch")
        return self.runs.update_status(run_id, "confirmed")

    def confirm_and_submit_generation(
        self, run_id: str, confirmation_hash: str
    ) -> dict[str, object]:
        run = self.confirm_generation(run_id, confirmation_hash)
        if run["status"] != "confirmed":
            return run
        spec = self.scenarios.get(str(run["scenario_id"]))
        if spec.spec_hash() != run["spec_hash"]:
            return self.runs.update_status(
                run_id,
                "rejected",
                reason=(
                    "scenario changed after generation was prepared; prepare a new generation run"
                ),
            )
        try:
            spec_path = _write_databricks_scenario_spec(spec)
        except Exception as exc:
            return self.runs.update_status(
                run_id,
                "submit_failed",
                reason=f"Could not write ScenarioSpec to control volume: {exc}",
            )
        job_id = os.getenv("SDF_GENERATION_JOB_ID")
        if not job_id:
            return self.runs.update_status(
                run_id,
                "confirmed",
                reason="SDF_GENERATION_JOB_ID is missing; run the bundle job manually.",
                scenario_spec_path=spec_path,
            )
        try:  # pragma: no cover - exercised in Databricks App runtime
            from databricks.sdk import WorkspaceClient

            waiter = WorkspaceClient().jobs.run_now(
                int(job_id),
                job_parameters={"scenario_path": spec_path, "output_root": _raw_runs_root()},
                idempotency_token=run_id,
            )
            databricks_run_id = _databricks_run_id(waiter)
            if databricks_run_id is None:
                raise RuntimeError("Databricks accepted submission without returning a run ID")
            return self.runs.update_status(
                run_id,
                "submitted",
                databricks_run_id=databricks_run_id,
                databricks_job_id=job_id,
                scenario_spec_path=spec_path,
            )
        except Exception as exc:
            return self.runs.update_status(
                run_id,
                "submit_failed",
                reason=str(exc),
                scenario_spec_path=spec_path,
            )

    def get_run_status(self, run_id: str) -> dict[str, object]:
        run = self._refresh_run_status(run_id)
        return {"run_id": run_id, "status": run["status"]}

    def get_run_summary(self, run_id: str) -> dict[str, object]:
        return self._refresh_run_status(run_id)

    def _refresh_run_status(self, run_id: str) -> dict[str, object]:
        """Synchronize a submitted Databricks run without treating a read error as failure."""
        run = self.runs.get(run_id)
        if run.get("status") not in {"submitted", "running"}:
            return run
        databricks_run_id = run.get("databricks_run_id")
        if not isinstance(databricks_run_id, int):
            return run
        state = _databricks_run_state(databricks_run_id)
        if state is None:
            return run
        lifecycle_state, result_state = state
        normalized_lifecycle = lifecycle_state.upper()
        normalized_result = (result_state or "").upper()
        if normalized_lifecycle in {"PENDING", "QUEUED"}:
            status = "submitted"
        elif normalized_lifecycle == "RUNNING":
            status = "running"
        elif normalized_lifecycle == "TERMINATED" and normalized_result == "SUCCESS":
            status = "succeeded"
        elif normalized_lifecycle in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}:
            status = "failed"
        else:
            return run
        if status == run["status"]:
            return run
        return self.runs.update_status(
            run_id,
            status,
            databricks_lifecycle_state=lifecycle_state,
            databricks_result_state=result_state,
        )

    def list_recent_scenarios(self, limit: int = 20) -> list[dict[str, object]]:
        return [_summary(spec, []) for spec in self.scenarios.list_recent(limit)]

    def clone_scenario(self, scenario_id: str, new_name: str) -> dict[str, object]:
        spec = self.scenarios.get(scenario_id)
        clone = ScenarioSpec.model_validate(
            {
                **spec.model_dump(mode="json"),
                "scenario_id": f"scn_{uuid4().hex[:12]}",
                "name": new_name,
                "revision": 1,
            }
        )
        self.scenarios.save(clone)
        return _summary(clone, [])


def _summary(spec: ScenarioSpec, warnings: list[str]) -> dict[str, object]:
    return {
        "scenario_id": spec.scenario_id,
        "revision": spec.revision,
        "spec_hash": spec.spec_hash(),
        "name": spec.name,
        "domain": spec.domain,
        "timeline": spec.timeline.model_dump(mode="json"),
        "metadata": spec.metadata,
        "tables": {table.name: table.row_count for table in spec.tables},
        "columns": {
            table.name: [
                {
                    "name": column.name,
                    "type": column.type,
                    "primary_key": column.primary_key,
                    "nullable": column.nullable,
                    "faker": column.faker,
                    "values": column.values,
                    "weights": column.weights,
                    "semantic": column.semantic,
                }
                for column in table.columns
            ]
            for table in spec.tables
        },
        "relationships": [
            {
                "name": relationship.name,
                "parent_table": relationship.parent_table,
                "parent_column": relationship.parent_column,
                "child_table": relationship.child_table,
                "child_column": relationship.child_column,
                "parent_filter": relationship.parent_filter,
                "constraints": relationship.constraints,
            }
            for relationship in spec.relationships
        ],
        "issues": [
            {
                "issue_id": issue.issue_id,
                "type": issue.type,
                "table": issue.table,
                "column": issue.column,
                "rate": issue.rate,
                "exact_count": issue.exact_count,
                "count": _issue_count(issue),
                "unit": _issue_unit(issue),
                "display_value": _issue_display_value(issue),
                "parameters": issue.parameters,
                "correlation": issue.correlation,
            }
            for issue in spec.issues
        ],
        "warnings": warnings,
    }


def _issue_count(issue: IssueSpec) -> int | None:
    if IssueType(issue.type) == IssueType.FILE_REPLAY and issue.parameters.get("file_count"):
        return int(issue.parameters["file_count"])
    return issue.exact_count


def _issue_unit(issue: IssueSpec) -> str | None:
    if IssueType(issue.type) == IssueType.FILE_REPLAY and issue.parameters.get("file_count"):
        return "file"
    if issue.exact_count is not None:
        return "row"
    if issue.rate is not None:
        return "rate"
    return None


def _issue_display_value(issue: IssueSpec) -> str:
    if IssueType(issue.type) == IssueType.FILE_REPLAY and issue.parameters.get("file_count"):
        count = int(issue.parameters["file_count"])
        return f"{count} file" if count == 1 else f"{count} files"
    if issue.exact_count is not None:
        return str(issue.exact_count)
    if issue.rate is not None:
        return str(issue.rate)
    return "-"


def _databricks_run_state(run_id: int) -> tuple[str, str | None] | None:
    """Return lifecycle/result state, or None when Databricks cannot be queried."""
    try:  # pragma: no cover - exercised in Databricks App runtime
        from databricks.sdk import WorkspaceClient

        state = WorkspaceClient().jobs.get_run(run_id).state
        lifecycle = _enum_text(getattr(state, "life_cycle_state", None))
        if not lifecycle:
            return None
        return lifecycle, _enum_text(getattr(state, "result_state", None))
    except Exception as exc:  # A temporary status-read failure must not falsify run state.
        print(f"Could not refresh Databricks run {run_id}: {exc}")
        return None


def _databricks_run_id(waiter: object) -> int | None:
    """Extract a run ID from either SDK response shape without waiting for completion."""
    response = getattr(waiter, "response", waiter)
    run_id = getattr(response, "run_id", None)
    return run_id if isinstance(run_id, int) else None


def _enum_text(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)


def _spec_from_prompt(prompt: str) -> tuple[ScenarioSpec, list[str]]:
    # Natural-language requests are model-designed. Blueprints remain available only
    # to the explicit YAML/JSON and legacy API paths; they are never selected here.
    return _custom_spec_from_agent_or_fallback(prompt)


def _custom_spec_from_agent_or_fallback(prompt: str) -> tuple[ScenarioSpec, list[str]]:
    failures: list[str] = []
    for _ in range(_MAX_SCHEMA_DESIGN_ATTEMPTS):
        intent = _custom_schema_intent_from_model(prompt)
        if not intent or not isinstance(intent.get("table_specs"), list):
            failures.append("planner returned no usable schema")
            continue
        intent = _normalize_agent_intent(intent)
        spec, assumptions, failure = _complete_agent_schema_contract(prompt, intent)
        if spec is not None:
            return spec, [
                "Schema-design agent designed and verified the custom ScenarioSpec.",
                *assumptions,
            ]
        failures.append(failure)

    print("Scenario agent exhausted planning attempts:", " | ".join(failures))
    raise AgentPlanningError(
        "The scenario planner could not complete an executable draft after internal recovery. "
        "No data or tables were created; please submit the request again."
    )


def _complete_agent_schema_contract(
    prompt: str, intent: dict[str, Any]
) -> tuple[ScenarioSpec | None, list[str], str]:
    """Validate an agent draft with one focused completion and bounded model repairs."""
    current_intent = intent
    try:
        spec, assumptions = _custom_spec_from_intent(current_intent, prompt)
        return spec, assumptions, ""
    except Exception as exc:
        current_error = str(exc)

    def complete_with(
        enrichment: dict[str, Any] | None, merge
    ) -> tuple[ScenarioSpec | None, list[str]]:
        nonlocal current_intent, current_error
        if not enrichment:
            return None, []
        current_intent = _normalize_agent_intent(merge(current_intent, enrichment))
        try:
            return _custom_spec_from_intent(current_intent, prompt)
        except Exception as exc:
            current_error = str(exc)
            return None, []

    # Prefer focused model completions to a full-schema retry. Each completion has a
    # narrow contract, so it preserves the original domain design while filling in
    # only the execution details that validation identified as incomplete.
    for _ in range(_MAX_COLUMN_COMPLETION_ATTEMPTS):
        if not _needs_column_enrichment(current_error):
            break
        spec, assumptions = complete_with(
            _enrich_column_strategies_with_model(prompt, current_intent, current_error),
            _merge_model_column_strategies,
        )
        if spec is not None:
            return spec, assumptions, ""

    if _needs_contextual_enrichment(current_error):
        spec, assumptions = complete_with(
            _enrich_contextual_issue_targets_with_model(prompt, current_intent, current_error),
            _merge_model_contextual_issue_targets,
        )
        if spec is not None:
            return spec, assumptions, ""

    if _needs_issue_enrichment(current_error):
        spec, assumptions = complete_with(
            _enrich_issue_parameters_with_model(prompt, current_intent, current_error),
            _merge_model_issue_parameters,
        )
        if spec is not None:
            return spec, assumptions, ""

    if _needs_operational_enrichment(current_error):
        spec, assumptions = complete_with(
            _enrich_operational_contracts_with_model(prompt, current_intent, current_error),
            _merge_model_operational_contracts,
        )
        if spec is not None:
            return spec, assumptions, ""

    if _needs_relationship_enrichment(current_error):
        spec, assumptions = complete_with(
            _enrich_relationship_contracts_with_model(prompt, current_intent, current_error),
            _merge_model_relationship_contracts,
        )
        if spec is not None:
            return spec, assumptions, ""

    # Bounded full repairs handle structural defects the focused completion cannot express.
    for _ in range(_MAX_SCHEMA_REPAIR_ATTEMPTS):
        repaired = _repair_custom_schema_intent_with_model(prompt, current_intent, current_error)
        if not repaired or not isinstance(repaired.get("table_specs"), list):
            continue
        current_intent = _normalize_agent_intent(
            _normalize_repaired_intent(repaired, current_intent)
        )
        try:
            spec, assumptions = _custom_spec_from_intent(current_intent, prompt)
            return spec, assumptions, ""
        except Exception as exc:
            current_error = str(exc)
    return None, [], current_error






def _custom_spec_from_intent(intent: dict[str, Any], prompt: str) -> tuple[ScenarioSpec, list[str]]:
    name = str(intent.get("name") or _extract_name(prompt, "custom_schema"))
    seed = int(intent.get("seed") or 42)
    timeline = _timeline_from_intent(intent)
    tables = _unique_table_specs_from_intent(intent["table_specs"])
    if not tables:
        raise ValueError("custom schema must contain at least one valid table")
    table_map = {table.name: table for table in tables}
    relationships: list[RelationshipSpec] = []
    for relationship in intent.get("relationships", []):
        if not isinstance(relationship, dict):
            raise ValueError("relationship entries must be objects")
        try:
            parsed = RelationshipSpec.model_validate(
                {
                    **relationship,
                    "name": _identifier(relationship.get("name"), "relationship"),
                    "parent_table": _identifier(relationship.get("parent_table"), "parent"),
                    "parent_column": _identifier(relationship.get("parent_column"), "parent_id"),
                    "child_table": _identifier(relationship.get("child_table"), "child"),
                    "child_column": _identifier(relationship.get("child_column"), "child_id"),
                }
            )
        except Exception as exc:
            raise ValueError(f"relationship could not be parsed: {relationship}") from exc
        if parsed.parent_table not in table_map or parsed.child_table not in table_map:
            raise ValueError(
                f"relationship {parsed.name} references missing table "
                f"{parsed.parent_table} or {parsed.child_table}"
            )
        if parsed.parent_column not in table_map[parsed.parent_table].column_names():
            raise ValueError(
                f"relationship {parsed.name} references missing parent column "
                f"{parsed.parent_table}.{parsed.parent_column}"
            )
        if parsed.child_column not in table_map[parsed.child_table].column_names():
            raise ValueError(
                f"relationship {parsed.name} references missing child column "
                f"{parsed.child_table}.{parsed.child_column}"
            )
        relationships.append(parsed)
    spec = ScenarioSpec(
        name=name,
        domain="custom_schema",
        seed=seed,
        locale=_normalized_locale(intent.get("locale")),
        timeline=timeline,
        tables=tables,
        relationships=relationships,
        outputs=OutputSpec(delta_namespace=_dataset_code_from_intent(intent, name)),
        metadata=_metadata_from_intent(intent),
    )
    spec = _ensure_databricks_outputs(spec)
    data = spec.model_dump(mode="json")
    base_spec = ScenarioSpec.model_validate(data)
    parsed_issues = _issues_from_intent(intent.get("issues"), base_spec, prompt)
    _ensure_agent_issue_parse_complete(intent.get("issues"), parsed_issues)
    data["issues"] = [
        issue.model_dump(mode="json")
        for issue in parsed_issues
    ]
    spec = ScenarioSpec.model_validate(data)
    semantic_gaps = [
        *_custom_semantic_gaps(spec, prompt),
        *_semantic_value_generation_gaps(spec),
        *_execution_rule_gaps(spec, prompt),
        *_insurance_execution_rule_gaps(spec, prompt),
        *_ai_model_ops_execution_rule_gaps(spec, prompt),
        *_custom_issue_gaps(spec, prompt),
        *_issue_parameter_reference_gaps(spec),
    ]
    if semantic_gaps:
        raise ValueError(f"custom ScenarioSpec missed requested intent: {', '.join(semantic_gaps)}")
    spec = _ensure_timeline_supports_issue_batches(spec)
    # The typed model validates references, but issue plugins enforce executable
    # injection parameters. Preflight here so an agent draft cannot be saved and
    # later fail only when previewing or generating.
    validate_scenario(spec)
    return spec, [
        "Agent created a custom schema because the request was not limited to a known blueprint.",
        (
            "Agent-provided tables, relationships, row counts, issue rules, and guardrails "
            "were normalized and validated."
        ),
        *_custom_issue_assumptions(prompt.lower(), spec),
        "Generation still requires hash confirmation before the Databricks job is submitted.",
    ]


def _table_spec_from_intent(raw_table: dict[str, Any]) -> TableSpec:
    columns: list[ColumnSpec] = []
    seen_columns: set[str] = set()
    primary_key_seen = False
    for column in raw_table.get("columns", []):
        if not isinstance(column, dict) or not column.get("name"):
            continue
        column_name = _identifier(column["name"], "column")
        if column_name in seen_columns:
            continue
        seen_columns.add(column_name)
        is_primary_key = bool(column.get("primary_key", False)) and not primary_key_seen
        primary_key_seen = primary_key_seen or is_primary_key
        columns.append(
            ColumnSpec(
                name=column_name,
                type=_column_type(column.get("type")),
                nullable=False if is_primary_key else bool(column.get("nullable", True)),
                primary_key=is_primary_key,
                faker=column.get("faker"),
                values=column.get("values"),
                weights=_column_weights_from_intent(column),
                semantic=_column_semantic_from_intent(column),
                min_value=column.get("min_value"),
                max_value=column.get("max_value"),
                precision=column.get("precision"),
                scale=column.get("scale"),
            )
        )
    if not columns:
        table_name = _identifier(raw_table.get("name"), "table")
        columns = [
            ColumnSpec(
                name=_default_pk_name(table_name),
                type=ColumnType.LONG,
                nullable=False,
                primary_key=True,
            )
        ]
    elif not any(column.primary_key for column in columns):
        columns[0] = columns[0].model_copy(update={"primary_key": True, "nullable": False})
    return TableSpec(
        name=_identifier(raw_table.get("name"), "table"),
        row_count=max(1, int(raw_table.get("row_count") or raw_table.get("record_count") or 1000)),
        columns=columns,
        source_systems=list(raw_table.get("source_systems") or []),
    )


def _column_semantic_from_intent(column: dict[str, Any]) -> dict[str, Any] | None:
    semantic = column.get("semantic")
    if isinstance(semantic, dict):
        if semantic.get("kind") == "date":
            return {**semantic, "kind": "timeline"}
        return semantic
    distribution = column.get("distribution")
    if isinstance(distribution, dict) and distribution.get("type") == "log_normal":
        return {
            "kind": "log_normal",
            "median": distribution.get("median"),
            "sigma": distribution.get("sigma", 1.0),
            "max": distribution.get("max"),
        }
    if isinstance(distribution, dict) and distribution.get("kind") in {"monthly", "timeline"}:
        return {"kind": "timeline"}
    return None


def _column_weights_from_intent(column: dict[str, Any]) -> object:
    weights = column.get("weights")
    if weights is not None:
        values = column.get("values")
        if (
            isinstance(values, list)
            and isinstance(weights, list)
            and values
            and weights
            and len(values) == len(weights)
            and all(isinstance(weight, (int, float)) and weight > 0 for weight in weights)
        ):
            return weights
        return None
    values = column.get("values")
    distribution = column.get("distribution")
    if not isinstance(values, list) or not isinstance(distribution, dict) or "type" in distribution:
        return None
    weights = [distribution.get(value) for value in values]
    if not all(isinstance(weight, (int, float)) and weight > 0 for weight in weights):
        return None
    return weights


def _needs_issue_enrichment(error: str) -> bool:
    return any(issue_type.value in error for issue_type in IssueType) or " correlation" in error


def _needs_column_enrichment(error: str) -> bool:
    return any(
        marker in error
        for marker in (
            "needs values",
            "needs a timeline",
            "needs a numeric",
            "needs a boolean",
            "unsupported Faker provider",
            "invalid semantic lookup",
        )
    )


def _needs_contextual_enrichment(error: str) -> bool:
    return "AI scenario" in error


def _needs_operational_enrichment(error: str) -> bool:
    return any(
        marker in error
        for marker in (
            "late_arrival",
            "file_replay",
            "schema_drift",
            "out_of_order",
            "arrival_column",
            "date_rule_violation",
        )
    )


def _needs_relationship_enrichment(error: str) -> bool:
    return any(
        marker in error
        for marker in ("relationship", "parent_filter", "child_date_ranges", "aggregate_caps")
    )


def _normalize_repaired_intent(
    repaired: dict[str, Any], previous: dict[str, Any]
) -> dict[str, Any]:
    """Normalize harmless model-format aliases without inventing scenario content."""
    normalized = dict(repaired)
    tables = normalized.get("table_specs")
    if isinstance(tables, list):
        for table in tables:
            if isinstance(table, dict) and not table.get("row_count") and table.get("record_count"):
                table["row_count"] = table["record_count"]
    relationships = normalized.get("relationships")
    if isinstance(relationships, list):
        for index, relationship in enumerate(relationships, start=1):
            if isinstance(relationship, dict) and not relationship.get("name"):
                relationship["name"] = (
                    f"{relationship.get('parent_table', 'parent')}_"
                    f"{relationship.get('child_table', 'child')}_{index}"
                )
    issues = normalized.get("issues")
    executable = isinstance(issues, list) and all(
        isinstance(issue, dict)
        and (
            issue.get("rate") is not None
            or issue.get("exact_count") is not None
            or (issue.get("parameters") or {}).get("file_count") is not None
        )
        for issue in issues
    )
    if not executable and isinstance(previous.get("issues"), list):
        normalized["issues"] = previous["issues"]
    return normalized


def _normalize_agent_intent(intent: dict[str, Any]) -> dict[str, Any]:
    """Accept lossless model JSON aliases before validating the executable contract."""
    normalized = json.loads(json.dumps(intent))
    locale = _normalized_locale(normalized.get("locale"))
    for relationship in normalized.get("relationships", []):
        if not isinstance(relationship, dict):
            continue
        if not relationship.get("parent_column") and isinstance(
            relationship.get("parent_key"), str
        ):
            relationship["parent_column"] = relationship["parent_key"]
        if not relationship.get("child_column") and isinstance(relationship.get("child_key"), str):
            relationship["child_column"] = relationship["child_key"]
        relationship.pop("parent_key", None)
        relationship.pop("child_key", None)
        constraints = relationship.setdefault("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
            relationship["constraints"] = constraints
        nested_filter = constraints.pop("parent_filter", None)
        if not relationship.get("parent_filter") and isinstance(nested_filter, dict):
            relationship["parent_filter"] = nested_filter
        for key in ("child_date_ranges", "aggregate_caps"):
            top_level_value = relationship.pop(key, None)
            if key not in constraints and isinstance(top_level_value, list):
                constraints[key] = top_level_value

    for table in normalized.get("table_specs", []):
        if not isinstance(table, dict):
            continue
        table_name = str(table.get("name") or "").lower()
        for column in table.get("columns", []):
            if not isinstance(column, dict):
                continue
            column_name = str(column.get("name") or "").lower()
            values = column.get("values")
            weights = column.get("weights")
            if weights is not None and not (
                isinstance(values, list)
                and values
                and isinstance(weights, list)
                and weights
                and len(values) == len(weights)
                and all(isinstance(weight, (int, float)) and weight > 0 for weight in weights)
            ):
                # Models often emit weights: [] as an empty optional field. It has
                # no distribution meaning and must not invalidate an otherwise
                # executable column. Real weighted anchors are still verified later.
                column.pop("weights", None)
            if (
                column.get("type") == ColumnType.STRING.value
                and not column.get("primary_key")
                and not column.get("faker")
                and not column.get("values")
                and not column.get("semantic")
                and column_name in {"name", "full_name", "customer_name", "investigator_name"}
            ):
                column["faker"] = (
                    "company"
                    if any(token in table_name for token in ("institution", "company", "merchant"))
                    else "name"
                )
            if (
                column.get("type") == ColumnType.STRING.value
                and not column.get("primary_key")
                and not column.get("faker")
                and not column.get("values")
                and not column.get("semantic")
                and column_name in {"kunnr", "bukrs", "belnr"}
            ):
                column["faker"] = "uuid4"
            if column.get("faker") in {"both", "bothify"}:
                column["faker"] = "sentence" if column_name.endswith("_name") else "uuid4"
            if column.get("faker") == "commerce":
                column["faker"] = "word"
            if (
                locale == "en_CA"
                and column_name in {"province", "province_code"}
                and column.get("faker") in {"state", "state_abbr", "province"}
            ):
                column.pop("faker", None)
                column["values"] = [
                    "ON",
                    "QC",
                    "BC",
                    "AB",
                    "MB",
                    "SK",
                    "NS",
                    "NB",
                    "NL",
                    "PE",
                ]
            semantic = column.get("semantic")
            if not isinstance(semantic, dict):
                continue
            distribution = semantic.pop("distribution", None)
            if distribution == "log_normal" and not semantic.get("kind"):
                semantic["kind"] = "log_normal"
            if semantic.get("kind") == "uniform":
                semantic["kind"] = "uniform_range"
            if semantic.get("kind") in {"lognormal", "log-normal"}:
                semantic["kind"] = "log_normal"
            if semantic.get("kind") == "log_normal":
                semantic.setdefault("sigma", 1.0)
                has_numeric_tail_max = isinstance(semantic.get("tail_max"), (int, float))
                if semantic.get("max") is None and has_numeric_tail_max:
                    semantic["max"] = semantic["tail_max"]

        if locale == "en_CA":
            columns_by_name = {
                str(column.get("name")).lower(): column
                for column in table.get("columns", [])
                if isinstance(column, dict) and isinstance(column.get("name"), str)
            }
            province = (
                columns_by_name.get("province")
                or columns_by_name.get("province_code")
                or columns_by_name.get("state")
                or columns_by_name.get("state_code")
            )
            city = columns_by_name.get("city")
            if isinstance(province, dict) and isinstance(city, dict):
                province_values = province.get("values")
                if not isinstance(province_values, list) or not province_values:
                    province_values = list(_CANADIAN_CITIES_BY_PROVINCE)
                    province["values"] = province_values
                supported_provinces = {
                    str(value): _CANADIAN_PROVINCE_CODES.get(
                        str(value).upper(), str(value).upper()
                    )
                    for value in province_values
                }
                supported_provinces = {
                    value: code
                    for value, code in supported_provinces.items()
                    if code in _CANADIAN_CITIES_BY_PROVINCE
                }
                if supported_provinces:
                    # Faker's en_CA city provider emits synthetic locality names.
                    # Use real cities and preserve province-to-city consistency.
                    city.pop("faker", None)
                    city.pop("values", None)
                    city.pop("weights", None)
                    city["semantic"] = {
                        "kind": "lookup",
                        "key_column": province["name"],
                        "values_by_key": {
                            value: _CANADIAN_CITIES_BY_PROVINCE[code]
                            for value, code in supported_provinces.items()
                        },
                    }

    for issue in normalized.get("issues", []):
        if not isinstance(issue, dict) or issue.get("type") != IssueType.LATE_ARRIVAL.value:
            continue
        parameters = issue.get("parameters")
        if not isinstance(parameters, dict):
            continue
        if not parameters.get("event_time_column") and isinstance(issue.get("column"), str):
            parameters["event_time_column"] = issue["column"]
        delay_range = parameters.pop("delay_range_days", None)
        if (
            isinstance(delay_range, list)
            and len(delay_range) == 2
            and all(isinstance(value, (int, float)) for value in delay_range)
        ):
            parameters.setdefault("delay_days_min", int(delay_range[0]))
            parameters.setdefault("delay_days_max", int(delay_range[1]))

    for issue in normalized.get("issues", []):
        if not isinstance(issue, dict) or issue.get("type") != IssueType.FILE_REPLAY.value:
            continue
        parameters = issue.get("parameters")
        if isinstance(parameters, dict) and parameters.get("file_count") is not None:
            issue["rate"] = None
            issue["exact_count"] = None

    tables_by_name = {
        str(table.get("name")): table
        for table in normalized.get("table_specs", [])
        if isinstance(table, dict) and isinstance(table.get("name"), str)
    }
    normalized_issues: list[Any] = []
    for issue in normalized.get("issues", []):
        if not isinstance(issue, dict) or issue.get("type") != IssueType.DATE_RULE_VIOLATION.value:
            normalized_issues.append(issue)
            continue
        parameters = issue.setdefault("parameters", {})
        if not isinstance(parameters, dict) or parameters.get("after_column"):
            normalized_issues.append(issue)
            continue
        table = tables_by_name.get(str(issue.get("table")))
        target_column = issue.get("column")
        candidates = [
            str(column.get("name"))
            for column in (table or {}).get("columns", [])
            if isinstance(column, dict)
            and column.get("name") != target_column
            and column.get("type") in {ColumnType.DATE.value, ColumnType.TIMESTAMP.value}
        ]
        if candidates:
            parameters["after_column"] = candidates[0]
            parameters.setdefault("days_after", -1)
            normalized_issues.append(issue)
    normalized["issues"] = normalized_issues
    return normalized


def _unique_table_specs_from_intent(raw_tables: object) -> list[TableSpec]:
    if not isinstance(raw_tables, list):
        return []
    tables: list[TableSpec] = []
    seen_tables: set[str] = set()
    for raw_table in raw_tables:
        if not isinstance(raw_table, dict):
            continue
        table_name = _identifier(raw_table.get("name"), "table")
        if table_name in seen_tables:
            continue
        table = _table_spec_from_intent(raw_table)
        seen_tables.add(table.name)
        tables.append(table)
    return tables


def _timeline_from_intent(intent: dict[str, Any]) -> TimelineSpec:
    raw_timeline = intent.get("timeline")
    if isinstance(raw_timeline, dict):
        start_value = str(raw_timeline.get("start_date") or "2026-01-01")
        try:
            start = date.fromisoformat(start_value)
        except ValueError:
            start = date(2026, 1, 1)
        frequency = str(raw_timeline.get("frequency") or "daily").lower()
        if frequency not in {"daily", "monthly"}:
            frequency = "daily"
        return TimelineSpec(
            start_date=start,
            batches=max(1, int(raw_timeline.get("batches") or 30)),
            frequency=frequency,  # type: ignore[arg-type]
        )
    return TimelineSpec(start_date=date(2026, 1, 1), batches=30)


def _normalized_locale(value: object) -> str:
    locale = str(value or "en_CA").replace("-", "_")
    aliases = {
        "ca": "en_CA",
        "canada": "en_CA",
        "ca_es": "en_CA",
    }
    return aliases.get(locale.lower(), locale)


def _metadata_from_intent(intent: dict[str, Any]) -> dict[str, Any]:
    metadata = {"synthetic_data": True}
    raw_metadata = intent.get("metadata")
    if isinstance(raw_metadata, dict):
        metadata.update(raw_metadata)
    for key in (
        "business_rules",
        "statistical_anchors",
        "distribution_rules",
        "seasonality",
        "guardrails",
    ):
        value = intent.get(key)
        if value:
            metadata[key] = value
    return metadata






def _ensure_timeline_supports_issue_batches(spec: ScenarioSpec) -> ScenarioSpec:
    required_batches = spec.timeline.batches
    for issue in spec.issues:
        parameters = issue.parameters
        for key in ("batch", "activation_batch", "source_batch", "target_batch"):
            value = parameters.get(key)
            if isinstance(value, int):
                required_batches = max(required_batches, value)
    if required_batches <= spec.timeline.batches:
        return spec
    data = spec.model_dump(mode="json")
    data["timeline"]["batches"] = required_batches
    return ScenarioSpec.model_validate(data)


def _custom_semantic_gaps(spec: ScenarioSpec, prompt: str) -> list[str]:
    text = prompt.lower()
    searchable = " ".join(
        [
            *(table.name for table in spec.tables),
            *(column.name for table in spec.tables for column in table.columns),
            *(relationship.name for relationship in spec.relationships),
        ]
    )
    concept_rules = {
        "cell tower": ("cell tower", "tower"),
        "network engineer": ("network engineer", "engineer"),
        "incident closure": ("incident closure", "closure"),
        "customer region": ("customer region", "region"),
        "provider": ("provider",),
        "patient": ("patient",),
        "merchant": ("merchant",),
        "account": ("account",),
        "settlement": ("settlement",),
        "promotion": ("promotion",),
        "coupon redemption": ("coupon redemption", "redemption"),
        "coupon": ("coupon",),
        "product": ("products", "product_id"),
        "customer": ("customer",),
        "return": ("returns", "return"),
        "postal code": ("postal code", "postal"),
    }
    gaps: list[str] = []
    mentioned_concepts = 0
    for concept, markers in concept_rules.items():
        if concept == "customer" and "customer or tenant" in text:
            continue
        if any(re.search(rf"\b{re.escape(marker)}\b", text) for marker in markers):
            mentioned_concepts += 1
            has_concept = any(
                marker.replace(" ", "_") in searchable or marker in searchable
                for marker in markers
            )
            if not has_concept:
                gaps.append(concept)
    if _explicit_table_section(text):
        mentioned_table_names = {
            table.name
            for table in spec.tables
            if table.name in text or table.name.replace("_", " ") in text
        }
        if mentioned_table_names and len(spec.tables) >= len(mentioned_table_names):
            return gaps
    if mentioned_concepts >= 3 and len(spec.tables) < 4:
        gaps.append("multi-entity relational design")
    if (
        "event" in text
        and not any("event" in table.name for table in spec.tables)
        and not any(
            token in table.name
            for table in spec.tables
            for token in ("inference", "transaction", "activity", "log")
        )
    ):
        gaps.append("event fact table")
    return gaps


_SUPPORTED_FAKER_PROVIDERS = {
    "address",
    "ascii_email",
    "city",
    "company",
    "country",
    "credit_card_number",
    "date_time",
    "domain_name",
    "email",
    "file_name",
    "first_name",
    "iban",
    "job",
    "last_name",
    "name",
    "paragraph",
    "phone_number",
    "postcode",
    "sentence",
    "state",
    "state_abbr",
    "street_address",
    "text",
    "url",
    "user_name",
    "uuid4",
    "word",
}

_CANADIAN_CITIES_BY_PROVINCE = {
    "ON": ["Toronto", "Ottawa", "Mississauga", "Hamilton", "London", "Kitchener"],
    "QC": ["Montreal", "Quebec City", "Laval", "Gatineau", "Sherbrooke"],
    "BC": ["Vancouver", "Surrey", "Burnaby", "Victoria", "Kelowna"],
    "AB": ["Calgary", "Edmonton", "Red Deer", "Lethbridge"],
    "MB": ["Winnipeg", "Brandon"],
    "SK": ["Saskatoon", "Regina"],
    "NS": ["Halifax", "Sydney"],
    "NB": ["Moncton", "Fredericton", "Saint John"],
    "NL": ["St. John's", "Corner Brook"],
    "PE": ["Charlottetown"],
    "YT": ["Whitehorse"],
    "NT": ["Yellowknife"],
    "NU": ["Iqaluit"],
}

_CANADIAN_PROVINCE_CODES = {
    "ONTARIO": "ON",
    "QUEBEC": "QC",
    "QUÉBEC": "QC",
    "BRITISH COLUMBIA": "BC",
    "ALBERTA": "AB",
    "MANITOBA": "MB",
    "SASKATCHEWAN": "SK",
    "NOVA SCOTIA": "NS",
    "NEW BRUNSWICK": "NB",
    "NEWFOUNDLAND AND LABRADOR": "NL",
    "PRINCE EDWARD ISLAND": "PE",
    "YUKON": "YT",
    "NORTHWEST TERRITORIES": "NT",
    "NUNAVUT": "NU",
}


def _semantic_value_generation_gaps(spec: ScenarioSpec) -> list[str]:
    """Reject plans that would fall through to generic compiler defaults."""
    gaps: list[str] = []
    placeholder = re.compile(r"^(?:city|name|place|value|text|record|item|category)_?\d+$", re.I)
    foreign_keys = {
        (relationship.child_table, relationship.child_column)
        for relationship in spec.relationships
    }
    for table in spec.tables:
        columns = table.column_names()
        for column in table.columns:
            if column.primary_key or (table.name, column.name) in foreign_keys:
                continue
            semantic = column.semantic or {}
            if column.type == ColumnType.STRING:
                if column.faker:
                    if not _faker_provider_available(column.faker, spec.locale):
                        gaps.append(
                            f"{table.name}.{column.name} uses unsupported Faker provider "
                            f"{column.faker}"
                        )
                    continue
                if column.values:
                    if any(
                        isinstance(value, str) and placeholder.fullmatch(value.strip())
                        for value in column.values
                    ):
                        gaps.append(f"{table.name}.{column.name} contains placeholder labels")
                    continue
                if semantic.get("kind") != "lookup":
                    gaps.append(
                        f"{table.name}.{column.name} needs values, a Faker provider, "
                        "or a semantic lookup"
                    )
                    continue
                key_column = semantic.get("key_column")
                values_by_key = semantic.get("values_by_key")
                if (
                    key_column not in columns
                    or not isinstance(values_by_key, dict)
                    or not values_by_key
                ):
                    gaps.append(f"{table.name}.{column.name} has an invalid semantic lookup")
                    continue
                if any(
                    not isinstance(values, list)
                    or not values
                    or any(not isinstance(value, str) or not value.strip() for value in values)
                    for values in values_by_key.values()
                ):
                    gaps.append(f"{table.name}.{column.name} lookup has no usable real values")
                continue
            if column.type in {ColumnType.DATE, ColumnType.TIMESTAMP}:
                if semantic.get("kind") not in {"timeline", "date_offset"}:
                    gaps.append(
                        f"{table.name}.{column.name} needs a timeline or date-offset rule"
                    )
                continue
            if column.type == ColumnType.DECIMAL:
                if semantic.get("kind") not in {"log_normal", "uniform_range", "normal"}:
                    gaps.append(
                        f"{table.name}.{column.name} needs a numeric distribution rule"
                    )
                continue
            if column.type in {ColumnType.INTEGER, ColumnType.LONG}:
                if not semantic and column.min_value is None and column.max_value is None:
                    gaps.append(
                        f"{table.name}.{column.name} needs a numeric range or semantic rule"
                    )
                continue
            if column.type == ColumnType.BOOLEAN and not semantic and not column.values:
                gaps.append(
                    f"{table.name}.{column.name} needs a boolean value strategy"
                )
    return gaps


def _faker_provider_available(provider: str, locale: str) -> bool:
    """Validate the model's provider against the installed locale-aware Faker surface."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", provider):
        return False
    try:
        from faker import Faker

        return callable(getattr(Faker(_normalized_locale(locale)), provider, None))
    except Exception:
        return False


def _execution_rule_gaps(spec: ScenarioSpec, prompt: str) -> list[str]:
    """Ensure material guardrails are executable by the generated data, not just documented."""
    text = prompt.lower()
    gaps: list[str] = []

    def column(table_name: str, column_name: str) -> ColumnSpec | None:
        table = next((item for item in spec.tables if item.name == table_name), None)
        if table is None:
            return None
        return next((item for item in table.columns if item.name == column_name), None)

    if "population" in text and "state" in text:
        state = column("customers", "state")
        if state is None or not state.values or not state.weights:
            gaps.append("population-weighted customers.state values")
    if "65%" in text and "35%" in text and "channel" in text:
        channel = column("orders", "channel")
        if channel is None or not channel.values or not channel.weights:
            gaps.append("weighted orders.channel values")
    if "log-normal" in text or "log normal" in text:
        amount = column("orders", "amount") or column("orders", "order_amount")
        if amount is None or (amount.semantic or {}).get("kind") != "log_normal":
            gaps.append("log-normal orders amount rule")
    if "ship_date must be on or after" in text:
        ship_date = column("orders", "ship_date")
        semantic = ship_date.semantic if ship_date else {}
        if (
            not isinstance(semantic, dict)
            or semantic.get("kind") != "date_offset"
            or semantic.get("base_column") != "order_date"
            or semantic.get("min_days", 0) < 0
        ):
            gaps.append("non-negative orders.ship_date offset from order_date")
    if "returns only exist for delivered orders" in text:
        relation = next(
            (
                item
                for item in spec.relationships
                if item.parent_table == "orders" and item.child_table == "returns"
            ),
            None,
        )
        parent_filter = relation.parent_filter if relation else None
        if not parent_filter or "delivered" not in parent_filter.get("values", []):
            gaps.append("orders-to-returns delivered-order relationship filter")
    if "appointment is scheduled before its encounter" in text or (
        "encounter dates before appointment dates" in text
    ):
        encounter_date = column("encounters", "encounter_date")
        local_appointment_date = column("encounters", "appointment_date")
        semantic = encounter_date.semantic if encounter_date else {}
        if (
            not local_appointment_date
            or (local_appointment_date.semantic or {}).get("kind")
            not in {"timeline", "date_offset"}
            or not isinstance(semantic, dict)
            or semantic.get("kind") != "date_offset"
            or semantic.get("base_column") != "appointment_date"
            or semantic.get("min_days", -1) < 0
        ):
            gaps.append("non-negative encounters.encounter_date offset from appointment_date")
        date_issue = next(
            (
                issue
                for issue in spec.issues
                if IssueType(issue.type) == IssueType.DATE_RULE_VIOLATION
                and issue.table == "encounters"
                and issue.column == "encounter_date"
            ),
            None,
        )
        if (
            not date_issue
            or date_issue.parameters.get("after_column") != "appointment_date"
            or date_issue.parameters.get("days_after") != -1
        ):
            gaps.append("encounters encounter-before-appointment date-rule violation")
    return gaps


def _insurance_execution_rule_gaps(spec: ScenarioSpec, prompt: str) -> list[str]:
    """Require the agent to encode explicit insurance rules as executable contracts."""
    text = prompt.lower()
    tables = {table.name: table for table in spec.tables}
    if not {"customers", "policies", "claims", "payments"}.issubset(tables):
        return []

    def column(table_name: str, column_name: str) -> ColumnSpec | None:
        return next(
            (item for item in tables[table_name].columns if item.name == column_name), None
        )

    def weights_match(column_spec: ColumnSpec | None, expected: dict[str, float]) -> bool:
        if not column_spec or not column_spec.values or not column_spec.weights:
            return False
        weights = {
            str(value): float(weight)
            for value, weight in zip(column_spec.values, column_spec.weights, strict=False)
        }
        total = sum(weights.values())
        return total > 0 and all(
            abs(weights.get(value, 0.0) / total - fraction) <= 0.02
            for value, fraction in expected.items()
        )

    def relationship(parent: str, child: str) -> RelationshipSpec | None:
        return next(
            (
                item
                for item in spec.relationships
                if item.parent_table == parent and item.child_table == child
            ),
            None,
        )

    gaps: list[str] = []
    if all(token in text for token in ("55% auto", "30% home", "15% tenant")):
        if not weights_match(
            column("policies", "policy_type"), {"auto": 0.55, "home": 0.30, "tenant": 0.15}
        ):
            gaps.append("weighted policies.policy_type distribution")
    if "70% of claims should be closed or settled" in text:
        status = column("claims", "claim_status")
        if not status or not status.values or not status.weights:
            gaps.append("weighted claims.claim_status distribution")
        else:
            weights = {
                str(value): float(weight)
                for value, weight in zip(status.values, status.weights, strict=False)
            }
            total = sum(weights.values())
            closed_or_settled = (weights.get("closed", 0.0) + weights.get("settled", 0.0)) / max(
                total, 1.0
            )
            if total <= 0 or abs(closed_or_settled - 0.70) > 0.02:
                gaps.append("70% closed-or-settled claims distribution")
    if "ontario and quebec should have the largest volume" in text:
        province = column("policies", "province")
        if not province or not province.values or not province.weights:
            gaps.append("weighted Canadian province volume")
        else:
            weights = {
                str(value): float(weight)
                for value, weight in zip(province.values, province.weights, strict=False)
            }
            if not {"ON", "QC", "AB", "BC"}.issubset(weights) or min(
                weights["ON"], weights["QC"]
            ) <= max(weights["AB"], weights["BC"]):
                gaps.append("Ontario-and-Quebec largest province volumes")
    if "long tail above $100,000" in text or "long tail above 100,000" in text:
        amount = column("claims", "claim_amount")
        semantic = amount.semantic if amount else {}
        if (
            not isinstance(semantic, dict)
            or semantic.get("kind") != "log_normal"
            or not isinstance(semantic.get("tail_share"), (int, float))
            or semantic.get("tail_share", 0) <= 0
            or not isinstance(semantic.get("tail_min"), (int, float))
            or semantic.get("tail_min", 0) < 100_000
        ):
            gaps.append("material claims.claim_amount tail above 100000")
    policy_claims = relationship("policies", "claims")
    if "active policy on the loss_date" in text:
        constraints = policy_claims.constraints if policy_claims else {}
        ranges = constraints.get("child_date_ranges", []) if isinstance(constraints, dict) else []
        valid_range = any(
            isinstance(rule, dict)
            and rule.get("child_column") == "loss_date"
            and rule.get("parent_start_column") == "effective_date"
            and rule.get("parent_end_column") == "expiry_date"
            for rule in ranges
        )
        if (
            not policy_claims
            or policy_claims.parent_filter != {"column": "status", "values": ["active"]}
            or not valid_range
        ):
            gaps.append("active-policy loss-date relationship constraint")
    settlement = column("claims", "settlement_date")
    if "settlement_date must be on or after loss_date" in text and (
        not settlement
        or (settlement.semantic or {}).get("kind") != "date_offset"
        or (settlement.semantic or {}).get("base_column") != "loss_date"
        or (settlement.semantic or {}).get("min_days", -1) < 0
    ):
        gaps.append("non-negative claims.settlement_date offset")
    claims_payments = relationship("claims", "payments")
    if "payments can only exist for approved or settled claims" in text:
        expected = {"column": "claim_status", "values": ["approved", "settled"]}
        if not claims_payments or claims_payments.parent_filter != expected:
            gaps.append("approved-or-settled claims-to-payments filter")
    if "total payments for a claim should normally not exceed claim_amount" in text:
        constraints = claims_payments.constraints if claims_payments else {}
        caps = constraints.get("aggregate_caps", []) if isinstance(constraints, dict) else []
        payment_amount_columns = {"amount", "payment_amount"} & tables["payments"].column_names()
        if not any(
            isinstance(rule, dict)
            and rule.get("child_amount_column") in payment_amount_columns
            and rule.get("parent_amount_column") == "claim_amount"
            and isinstance(rule.get("maximum_fraction"), (int, float))
            and rule["maximum_fraction"] <= 1
            for rule in caps
        ):
            gaps.append("aggregate payments.amount cap by claims.claim_amount")
    if "late_arrival on payments" in text:
        late = next(
            (
                issue
                for issue in spec.issues
                if IssueType(issue.type) == IssueType.LATE_ARRIVAL and issue.table == "payments"
            ),
            None,
        )
        parameters = late.parameters if late else {}
        arrival_column = parameters.get("arrival_column")
        if (
            not late
            or late.rate != 0.05
            or parameters.get("event_time_column") != "payment_date"
            or not isinstance(arrival_column, str)
            or arrival_column not in tables["payments"].column_names()
            or parameters.get("delay_days_min") != 1
            or parameters.get("delay_days_max") != 7
        ):
            gaps.append("5% late payments event-to-ingestion contract")
    if "half" in text and "missing adjuster" in text and "legacy_batch" in text:
        missing = next(
            (
                issue
                for issue in spec.issues
                if IssueType(issue.type) == IssueType.NULL_VALUE
                and issue.table == "claims"
                and issue.column == "adjuster_id"
            ),
            None,
        )
        correlation = (
            (missing.correlation if missing else None)
            or (missing.parameters.get("correlation") if missing else None)
            or {}
        )
        where = correlation.get("where") if isinstance(correlation, dict) else {}
        if (
            not missing
            or missing.rate != 0.04
            or correlation.get("share") != 0.5
            or where.get("source_system") != "legacy_batch"
            or where.get("after_batch") != 6
        ):
            gaps.append("half-correlated legacy-batch adjuster missingness")
    return gaps


def _ai_model_ops_execution_rule_gaps(spec: ScenarioSpec, prompt: str) -> list[str]:
    """Keep AI platform defects on the operational records they are meant to test."""
    text = prompt.lower()
    ai_markers = (
        "ai model",
        "model operations",
        "model inference",
        "prompt request",
        "evaluation result",
    )
    if not any(marker in text for marker in ai_markers):
        return []

    tables = {table.name: table for table in spec.tables}

    def table(*names: str) -> TableSpec | None:
        for name in names:
            if name in tables:
                return tables[name]
        return None

    def column_spec(target: TableSpec | None, column: str) -> ColumnSpec | None:
        if target is None:
            return None
        columns = getattr(target, "columns", None)
        if isinstance(columns, list):
            return next((item for item in columns if item.name == column), None)
        return None

    def has_column(target: TableSpec | None, column: str) -> bool:
        if column_spec(target, column) is not None:
            return True
        return target is not None and column in target.column_names()

    def issue(
        issue_type: IssueType, target: TableSpec | None, column: str | None = None
    ) -> IssueSpec | None:
        if target is None:
            return None
        return next(
            (
                item
                for item in spec.issues
                if IssueType(item.type) == issue_type
                and item.table == target.name
                and (column is None or item.column == column)
            ),
            None,
        )

    inference = table("model_inferences", "inferences")
    prompts = table("prompt_requests", "prompts")
    feedback = table("feedback_scores", "feedback")
    evaluations = table("evaluation_results", "evaluations")
    tenants = table("tenant_metadata", "tenants")
    gaps: list[str] = []

    def require_issue(
        marker: str,
        label: str,
        issue_type: IssueType,
        target: TableSpec | None,
        column: str,
    ) -> IssueSpec | None:
        if marker not in text:
            return None
        if not has_column(target, column):
            gaps.append(f"AI scenario needs {label} column")
            return None
        found = issue(issue_type, target, column)
        if found is None:
            gaps.append(f"AI scenario must map {label} to {target.name}.{column}")
        return found

    require_issue(
        "orphan model", "orphan model IDs", IssueType.REFERENTIAL_ORPHAN, inference, "model_id"
    )
    require_issue(
        "orphan user", "orphan user IDs", IssueType.REFERENTIAL_ORPHAN, inference, "user_id"
    )
    require_issue(
        "missing prompt categor",
        "missing prompt categories",
        IssueType.NULL_VALUE,
        prompts,
        "prompt_category",
    )
    require_issue(
        "missing evaluation label",
        "missing evaluation labels",
        IssueType.NULL_VALUE,
        evaluations,
        "evaluation_label",
    )
    require_issue(
        "missing inference latency",
        "missing inference latency",
        IssueType.NULL_VALUE,
        inference,
        "response_latency_ms",
    )
    if "replayed inference" in text and issue(IssueType.FILE_REPLAY, inference) is None:
        gaps.append("AI scenario must replay model_inferences ingestion files")
    replay_causes_duplicates = "replay" in text and (
        "duplicated inference" in text or "duplicate inference" in text
    ) and "caused by" in text
    if replay_causes_duplicates:
        if issue(IssueType.FILE_REPLAY, inference) is None:
            gaps.append("AI replay-caused inference duplicates need a file_replay rule")
        if issue(IssueType.DUPLICATE_RECORD, inference) is not None:
            gaps.append("AI replay-caused duplicates must not add duplicate_record")
    if "schema drift" in text:
        drift_issues = [
            item for item in spec.issues if IssueType(item.type) == IssueType.SCHEMA_DRIFT
        ]
        if issue(IssueType.SCHEMA_DRIFT, inference) is None:
            gaps.append("AI scenario must apply schema drift to model_inferences batches")
        elif any(item.table != inference.name for item in drift_issues):
            gaps.append("AI schema drift must not target an unrelated table")
    invalid_model_version_marker = (
        "invalid model version" if "invalid model version" in text else "invalid versions"
    )
    require_issue(
        invalid_model_version_marker,
        "invalid model versions",
        IssueType.INVALID_VALUE,
        inference,
        "model_version",
    )
    require_issue(
        "tenant region", "invalid tenant regions", IssueType.INVALID_VALUE, tenants, "region"
    )

    late_feedback = None
    if "late-arriving feedback" in text or "late feedback" in text:
        late_feedback = require_issue(
            "feedback",
            "late feedback events",
            IssueType.LATE_ARRIVAL,
            feedback,
            "created_at",
        )
    if late_feedback is not None:
        parameters = late_feedback.parameters
        if (
            parameters.get("event_time_column") != "created_at"
            or not has_column(feedback, str(parameters.get("arrival_column", "")))
        ):
            gaps.append("AI late feedback needs created_at and a feedback ingestion timestamp")

    if "feedback occurs before inference" in text or "feedback before inference" in text:
        feedback_before_inference = issue(IssueType.DATE_RULE_VIOLATION, feedback, "created_at")
        inference_created_at = column_spec(feedback, "inference_created_at")
        if (
            feedback_before_inference is None
            or feedback_before_inference.parameters.get("after_column") != "inference_created_at"
            or not has_column(feedback, "inference_created_at")
            or (
                inference_created_at is not None
                and (inference_created_at.semantic or {}).get("kind")
                not in {"timeline", "date_offset"}
            )
        ):
            gaps.append("AI feedback-before-inference needs local feedback timestamp contract")

    response_text_requested = "response_text" in text or "response text" in text
    if response_text_requested and ("null" in text or "malformed" in text):
        require_issue(
            "malformed response_text",
            "malformed response_text",
            IssueType.INVALID_FORMAT,
            inference,
            "response_text",
        )
        if "null" in text:
            require_issue(
                "null",
                "null response_text",
                IssueType.NULL_VALUE,
                inference,
                "response_text",
            )
    require_issue(
        "negative latency",
        "negative latency",
        IssueType.INVALID_VALUE,
        inference,
        "response_latency_ms",
    )
    if "empty prompt" in text:
        require_issue(
            "empty prompt", "empty prompt text", IssueType.BLANK_VALUE, prompts, "prompt_text"
        )
    return gaps


def _custom_issue_gaps(spec: ScenarioSpec, prompt: str) -> list[str]:
    text = prompt.lower()
    issue_types = {IssueType(issue.type) for issue in spec.issues}
    required: dict[IssueType, tuple[str, ...]] = {
        IssueType.DUPLICATE_RECORD: ("duplicate", "duplicated"),
        IssueType.REFERENTIAL_ORPHAN: ("orphan", "referential"),
        IssueType.NULL_VALUE: ("missing", "null"),
        IssueType.DATE_RULE_VIOLATION: (
            "date_rule_violation",
            "ship before order",
            "before inference",
            "mismatched timestamp",
            "mismatched timestamps",
        ),
        IssueType.LATE_ARRIVAL: ("late-arriving", "late arriving", "late_arrival"),
        IssueType.FILE_REPLAY: ("replayed", "replay", "file_replay"),
        IssueType.SCHEMA_DRIFT: ("schema drift", "renamed columns", "inconsistent data types"),
        IssueType.INVALID_VALUE: ("invalid", "impossible"),
        IssueType.INVALID_FORMAT: ("malformed", "invalid format"),
        IssueType.BLANK_VALUE: ("empty prompt", "blank"),
    }
    gaps = []
    for issue_type, markers in required.items():
        if (
            issue_type == IssueType.DUPLICATE_RECORD
            and "replay" in text
            and ("duplicated inference" in text or "duplicate inference" in text)
            and "caused by" in text
        ):
            continue
        if any(marker in text for marker in markers) and issue_type not in issue_types:
            gaps.append(f"{issue_type.value} issue")
    return gaps


def _ensure_agent_issue_parse_complete(raw_issues: object, parsed_issues: list[IssueSpec]) -> None:
    if raw_issues is None:
        return
    if not isinstance(raw_issues, list):
        raise ValueError("issues must be a list")
    issue_objects = [issue for issue in raw_issues if isinstance(issue, dict)]
    if len(issue_objects) != len(parsed_issues):
        raise ValueError(
            "one or more agent-provided issue rules were unsupported or referenced "
            "missing tables/columns"
        )


def _issue_parameter_reference_gaps(spec: ScenarioSpec) -> list[str]:
    gaps: list[str] = []
    for issue in spec.issues:
        table = spec.table(issue.table)
        columns = table.column_names()
        issue_type = IssueType(issue.type)
        if issue_type == IssueType.DATE_RULE_VIOLATION:
            after_column = issue.parameters.get("after_column")
            if not after_column or after_column not in columns:
                gaps.append(f"{issue.issue_id} missing valid after_column")
        elif issue_type == IssueType.LATE_ARRIVAL:
            for parameter in ("event_time_column", "arrival_column"):
                column = issue.parameters.get(parameter)
                if column and column not in columns:
                    gaps.append(f"{issue.issue_id} references missing {parameter} {column}")
        correlation = issue.correlation or issue.parameters.get("correlation") or {}
        where = correlation.get("where") if isinstance(correlation, dict) else None
        if isinstance(where, dict):
            source_column = where.get("source_column")
            if not isinstance(source_column, str) and "source_system" in where:
                source_column = "source_system"
            if isinstance(source_column, str) and source_column not in columns:
                gaps.append(f"{issue.issue_id} correlation references missing {source_column}")
        if issue_type == IssueType.SCHEMA_DRIFT:
            for rename in issue.parameters.get("rename_columns", []):
                if isinstance(rename, dict) and rename.get("from") not in columns:
                    gaps.append(
                        f"{issue.issue_id} renames missing column {rename.get('from')}"
                    )
            for change in issue.parameters.get("type_changes", []):
                if isinstance(change, dict) and change.get("column") not in columns:
                    gaps.append(
                        f"{issue.issue_id} changes missing column {change.get('column')}"
                    )
    return gaps


def _custom_issue_assumptions(text: str, spec: ScenarioSpec) -> list[str]:
    assumptions: list[str] = []
    if "duplicate" in text and not _keyword_has_explicit_quantity(text, "duplicate"):
        assumptions.append("Duplicate rate was not specified; defaulting to 1%.")
    if ("invalid" in text or "bad value" in text) and not _keyword_has_explicit_quantity(
        text, "invalid"
    ):
        invalid_issues = [
            issue for issue in spec.issues if IssueType(issue.type) == IssueType.INVALID_VALUE
        ]
        invalid_issue = invalid_issues[0] if invalid_issues else None
        if _is_ai_model_ops_spec(spec):
            assumptions.append(
                "Invalid model-version and tenant-region rates were not specified; "
                "defaulting each to 1%. Small status/impossible-value defects default "
                "to 100 records each."
            )
        elif invalid_issue and invalid_issue.exact_count == 1:
            table = spec.table(invalid_issue.table)
            assumptions.append(
                "Invalid-value rate was not specified. Because "
                f"{invalid_issue.table} has only {table.row_count} rows, defaulting to "
                f"one invalid {invalid_issue.column or 'value'} record rather than a percentage."
            )
        else:
            assumptions.append("Invalid-value rate was not specified; defaulting to 1%.")
    if "schema drift" in text or "new column" in text:
        drift_issue = next(
            (issue for issue in spec.issues if IssueType(issue.type) == IssueType.SCHEMA_DRIFT),
            None,
        )
        drift_column = drift_issue.parameters.get("column") if drift_issue else None
        if drift_issue and isinstance(drift_column, dict) and drift_column.get("name"):
            batch = drift_issue.parameters.get("batch") or drift_issue.parameters.get(
                "activation_batch", 3
            )
            assumptions.append(
                f"Schema drift will occur in batch {batch} by adding "
                f"{drift_issue.table}.{drift_column['name']} as "
                f"{drift_column.get('type', 'string')}."
            )
        else:
            assumptions.append("Schema drift will occur once in batch 3.")
    if "replay" in text and not _keyword_has_explicit_quantity(text, "replay"):
        replay_issue = next(
            (issue for issue in spec.issues if IssueType(issue.type) == IssueType.FILE_REPLAY),
            None,
        )
        parameters = replay_issue.parameters if replay_issue else {}
        source = parameters.get("source_batch_label", "batch_002")
        target = parameters.get("replay_batch", "batch_004")
        assumptions.append(
            "File replay rate was not specified. Defaulting to replaying one source file "
            f"from {source} in {target}."
        )
    return assumptions


def _keyword_has_explicit_quantity(text: str, keyword: str) -> bool:
    clause = _keyword_clause(text, keyword)
    has_percent = re.search(r"\b\d+(?:\.\d+)?\s*%", clause)
    has_count = re.search(r"\b\d[\d,]*\s*[km]?\b", clause)
    return bool(has_percent or has_count)






def _explicit_table_section(text: str) -> bool:
    return "tables:" in text or "tables -" in text or "tables such as" in text


























def _is_ai_model_ops_spec(spec: ScenarioSpec) -> bool:
    table_names = {table.name for table in spec.tables}
    return {
        "prompt_requests",
        "model_inferences",
        "feedback_scores",
        "evaluation_results",
        "model_registry",
        "tenant_metadata",
    }.issubset(table_names)














































def _column_type(value: object) -> ColumnType:
    normalized = str(value or "string").strip().lower()
    aliases = {
        "int": ColumnType.INTEGER,
        "integer": ColumnType.INTEGER,
        "bigint": ColumnType.LONG,
        "long": ColumnType.LONG,
        "double": ColumnType.DECIMAL,
        "float": ColumnType.DECIMAL,
        "decimal": ColumnType.DECIMAL,
        "date": ColumnType.DATE,
        "datetime": ColumnType.TIMESTAMP,
        "timestamp": ColumnType.TIMESTAMP,
        "bool": ColumnType.BOOLEAN,
        "boolean": ColumnType.BOOLEAN,
        "string": ColumnType.STRING,
        "str": ColumnType.STRING,
    }
    return aliases.get(normalized, ColumnType.STRING)


def _identifier(value: object, default: str) -> str:
    raw = str(value or default).strip().lower()
    identifier = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    if not identifier:
        identifier = default
    if identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier


def _dataset_code_from_intent(intent: dict[str, Any], scenario_name: str) -> str:
    """Use the agent's compact dataset code, with a stable readable-name fallback."""
    raw_code = intent.get("dataset_code")
    if raw_code:
        return _identifier(raw_code, "scenario")[:32]
    return _identifier(scenario_name, "scenario")[:32]


def _default_pk_name(table_name: str) -> str:
    return f"{_singular(table_name)}_id"


def _singular(value: str) -> str:
    if value.endswith("ies"):
        return f"{value[:-3]}y"
    if value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value








def _issues_from_intent(
    raw_issues: object, spec: ScenarioSpec, prompt_context: str = ""
) -> list[IssueSpec]:
    if not isinstance(raw_issues, list):
        return []
    tables = {table.name: table for table in spec.tables}
    result: list[IssueSpec] = []
    for index, raw_issue in enumerate(raw_issues, start=1):
        if not isinstance(raw_issue, dict):
            continue
        issue_type = raw_issue.get("type")
        table = raw_issue.get("table")
        column = raw_issue.get("column")
        if issue_type not in {kind.value for kind in IssueType}:
            continue
        table, column, parameters, count, rate = _normalize_issue_intent(
            str(issue_type), table, column, raw_issue, spec, prompt_context
        )
        if table not in tables:
            continue
        if column and column not in tables[str(table)].column_names():
            continue
        issue = IssueSpec(
            issue_id=str(raw_issue.get("issue_id") or f"iss_agent_{index:02d}_{issue_type}"),
            type=IssueType(str(issue_type)),
            table=str(table),
            column=str(column) if column else None,
            exact_count=int(count) if count is not None else None,
            rate=float(rate) if count is None and rate is not None else None,
            parameters=parameters,
            correlation=raw_issue.get("correlation"),
        )
        result.append(issue)
    return result


def _normalize_issue_intent(
    issue_type: str,
    table: object,
    column: object,
    raw_issue: dict[str, Any],
    spec: ScenarioSpec,
    prompt_context: str = "",
) -> tuple[str | None, str | None, dict[str, Any], object, object]:
    table_names = {table_spec.name for table_spec in spec.tables}
    parameters = dict(raw_issue.get("parameters") or {})
    count = raw_issue.get("exact_count")
    rate = raw_issue.get("rate")
    table_name = str(table) if table else None
    column_name = str(column) if column else None
    raw_issue_text = f"{prompt_context}\n{json.dumps(raw_issue, sort_keys=True)}".lower()
    if spec.domain == "insurance_claims":
        if issue_type in {IssueType.SCHEMA_DRIFT.value, IssueType.FILE_REPLAY.value}:
            table_name = table_name if table_name in table_names else "claims"
            column_name = None
        elif issue_type == IssueType.LATE_ARRIVAL.value:
            table_name = table_name if table_name in table_names else "payments"
            if table_name == "payments" and not column_name:
                column_name = "ingestion_ts"
        elif issue_type == IssueType.REFERENTIAL_ORPHAN.value and not column_name:
            table_name = table_name if table_name in table_names else "claims"
            column_name = "policy_id"
        elif (
            issue_type == IssueType.NULL_VALUE.value
            and not column_name
            and table_name == "claims"
        ):
            column_name = "adjuster_id"
    if issue_type == IssueType.SCHEMA_DRIFT.value:
        count = 1
        rate = None
        parameters.setdefault("activation_batch", 20)
        parameters.setdefault(
            "add_columns",
            [{"name": "fraud_score", "type": "decimal"}],
        )
    if issue_type == IssueType.FILE_REPLAY.value:
        parameters.setdefault("source_batch", 10)
        parameters.setdefault("target_batch", 12)
        if count is None and rate is None and parameters.get("file_count") is None:
            rate = 0.01
    if (
        issue_type == IssueType.DATE_RULE_VIOLATION.value
        and parameters.get("after_column")
        and "days_after" not in parameters
        and "before" in raw_issue_text
    ):
        parameters["days_after"] = -1
    if (
        count is None
        and rate is None
        and not (issue_type == IssueType.FILE_REPLAY.value and parameters.get("file_count"))
    ):
        rate = _default_issue_rate(issue_type)
    return table_name, column_name, parameters, count, rate


def _default_issue_rate(issue_type: str) -> float:
    defaults = {
        IssueType.DUPLICATE_RECORD.value: 0.01,
        IssueType.INVALID_VALUE.value: 0.01,
        IssueType.INVALID_FORMAT.value: 0.01,
        IssueType.BLANK_VALUE.value: 0.01,
        IssueType.NULL_VALUE.value: 0.01,
        IssueType.REFERENTIAL_ORPHAN.value: 0.01,
        IssueType.DATE_RULE_VIOLATION.value: 0.01,
        IssueType.LATE_ARRIVAL.value: 0.01,
        IssueType.OUT_OF_ORDER.value: 0.01,
        IssueType.FILE_REPLAY.value: 0.01,
        IssueType.CORRELATED_MISSINGNESS.value: 0.01,
    }
    return defaults.get(issue_type, 0.01)




def _schema_design_token_budget(prompt: str) -> int:
    """Reserve the larger JSON budget only for explicitly broad enterprise designs."""
    if _is_basic_schema_request(prompt):
        return _BASIC_SCHEMA_DESIGN_MAX_TOKENS
    text = prompt.lower()
    enterprise_signals = (
        "sap",
        "loyalty",
        "campaign",
        "advertising",
        "credit card",
    )
    if sum(signal in text for signal in enterprise_signals) >= 3:
        return _COMPLEX_SCHEMA_DESIGN_MAX_TOKENS
    return _SCHEMA_DESIGN_MAX_TOKENS


def _is_basic_schema_request(prompt: str) -> bool:
    """Identify terse requests that need a compact agent-designed operating model."""
    text = prompt.lower()
    if len(prompt) > 420 or "tables:" in text or "business rules:" in text:
        return False
    enterprise_signals = ("sap", "loyalty", "campaign", "advertising", "credit card")
    return sum(signal in text for signal in enterprise_signals) < 2


def _schema_design_system_prompt(prompt: str) -> str:
    if _is_basic_schema_request(prompt):
        return _BASIC_SCHEMA_SYSTEM_PROMPT
    return _CUSTOM_SCHEMA_SYSTEM_PROMPT


def _custom_schema_intent_from_model(prompt: str) -> dict[str, Any] | None:
    endpoint = os.getenv("SDF_MODEL_ENDPOINT")
    if not endpoint:
        return None
    try:  # pragma: no cover - exercised in Databricks App runtime
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        response = WorkspaceClient().serving_endpoints.query(
            endpoint,
            messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM,
                    content=_schema_design_system_prompt(prompt),
                ),
                ChatMessage(role=ChatMessageRole.USER, content=prompt),
            ],
            temperature=0.0,
            max_tokens=_schema_design_token_budget(prompt),
        )
        return _json_from_text(_response_text(response))
    except Exception:
        return None


def _repair_custom_schema_intent_with_model(
    prompt: str, invalid_intent: dict[str, Any], error: str
) -> dict[str, Any] | None:
    endpoint = os.getenv("SDF_MODEL_ENDPOINT")
    if not endpoint:
        return None
    try:  # pragma: no cover - exercised in Databricks App runtime
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        payload = {
            "user_prompt": prompt,
            "validation_error": error,
            "invalid_intent": invalid_intent,
        }
        response = WorkspaceClient().serving_endpoints.query(
            endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=_CUSTOM_SCHEMA_REPAIR_PROMPT),
                ChatMessage(role=ChatMessageRole.USER, content=json.dumps(payload)),
            ],
            temperature=0.0,
            max_tokens=_schema_design_token_budget(prompt),
        )
        return _json_from_text(_response_text(response))
    except Exception:
        return None


def _enrich_column_strategies_with_model(
    prompt: str, intent: dict[str, Any], validation_error: str
) -> dict[str, Any] | None:
    endpoint = os.getenv("SDF_MODEL_ENDPOINT")
    if not endpoint:
        return None
    try:  # pragma: no cover - exercised in Databricks App runtime
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        payload = {
            "user_prompt": prompt,
            "validation_error": validation_error,
            "schema_intent": intent,
        }
        response = WorkspaceClient().serving_endpoints.query(
            endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=_COLUMN_ENRICHMENT_PROMPT),
                ChatMessage(role=ChatMessageRole.USER, content=json.dumps(payload)),
            ],
            temperature=0.0,
            max_tokens=8000,
        )
        return _json_from_text(_response_text(response))
    except Exception:
        return None


def _merge_model_column_strategies(
    intent: dict[str, Any], enrichment: dict[str, Any]
) -> dict[str, Any]:
    """Apply only model-authored column strategies to the model's original schema."""
    merged = json.loads(json.dumps(intent))
    entries = enrichment.get("columns")
    if not isinstance(entries, list):
        return merged
    tables = {
        table.get("name"): table
        for table in merged.get("table_specs", [])
        if isinstance(table, dict)
    }
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        table = tables.get(entry.get("table"))
        if not isinstance(table, dict):
            continue
        column_name = entry.get("column") or entry.get("name")
        target = next(
            (
                column
                for column in table.get("columns", [])
                if isinstance(column, dict) and column.get("name") == column_name
            ),
            None,
        )
        if not isinstance(target, dict):
            if isinstance(column_name, str) and entry.get("type"):
                target = {
                    key: value for key, value in entry.items() if key not in {"table", "column"}
                }
                target["name"] = column_name
                table.setdefault("columns", []).append(target)
            else:
                continue
        for key in ("faker", "values", "semantic", "min_value", "max_value"):
            if key in entry:
                target[key] = entry[key]
        if "weights" in entry:
            values = entry.get("values", target.get("values"))
            weights = entry["weights"]
            if (
                isinstance(values, list)
                and isinstance(weights, list)
                and len(values) == len(weights)
            ):
                target["weights"] = weights
    return merged


def _enrich_issue_parameters_with_model(
    prompt: str, intent: dict[str, Any], validation_error: str
) -> dict[str, Any] | None:
    """Ask the planner to complete execution parameters without changing its schema."""
    endpoint = os.getenv("SDF_MODEL_ENDPOINT")
    if not endpoint:
        return None
    try:  # pragma: no cover - exercised in Databricks App runtime
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        payload = {
            "user_prompt": prompt,
            "validation_error": validation_error,
            "schema_intent": intent,
        }
        response = WorkspaceClient().serving_endpoints.query(
            endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=_ISSUE_ENRICHMENT_PROMPT),
                ChatMessage(role=ChatMessageRole.USER, content=json.dumps(payload)),
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        return _json_from_text(_response_text(response))
    except Exception:
        return None


def _merge_model_issue_parameters(
    intent: dict[str, Any], enrichment: dict[str, Any]
) -> dict[str, Any]:
    """Merge only planner-supplied issue parameters into the original issue plan."""
    merged = json.loads(json.dumps(intent))
    entries = enrichment.get("issues")
    issues = merged.get("issues")
    if not isinstance(entries, list) or not isinstance(issues, list):
        return merged

    for entry in entries:
        if not isinstance(entry, dict) or (
            not isinstance(entry.get("parameters"), dict)
            and not isinstance(entry.get("correlation"), dict)
        ):
            continue
        target = next(
            (
                issue
                for issue in issues
                if isinstance(issue, dict)
                and (
                    entry.get("issue_id")
                    and entry.get("issue_id") == issue.get("issue_id")
                    or (
                        entry.get("type") == issue.get("type")
                        and entry.get("table") == issue.get("table")
                        and entry.get("column") == issue.get("column")
                    )
                )
            ),
            None,
        )
        if isinstance(target, dict):
            if isinstance(entry.get("parameters"), dict):
                parameters = target.setdefault("parameters", {})
                if isinstance(parameters, dict):
                    parameters.update(entry["parameters"])
                    if (
                        target.get("type") == IssueType.FILE_REPLAY.value
                        and parameters.get("file_count") is not None
                    ):
                        target["rate"] = None
                        target["exact_count"] = None
            if isinstance(entry.get("correlation"), dict):
                target["correlation"] = entry["correlation"]
        elif (
            entry.get("type") in {issue_type.value for issue_type in IssueType}
            and isinstance(entry.get("table"), str)
            and (
                entry.get("rate") is not None
                or entry.get("exact_count") is not None
                or isinstance(entry.get("parameters"), dict)
                and entry["parameters"].get("file_count") is not None
            )
        ):
            issues.append(
                {
                    "issue_id": entry.get("issue_id")
                    or f"iss_agent_completion_{len(issues) + 1:02d}",
                    "type": entry["type"],
                    "table": entry["table"],
                    "column": entry.get("column"),
                    "rate": entry.get("rate"),
                    "exact_count": entry.get("exact_count"),
                    "parameters": entry.get("parameters") or {},
                    "correlation": entry.get("correlation"),
                }
            )
    return merged


def _enrich_contextual_issue_targets_with_model(
    prompt: str, intent: dict[str, Any], validation_error: str
) -> dict[str, Any] | None:
    """Ask the agent for a small, domain-aware correction instead of a full redesign."""
    endpoint = os.getenv("SDF_MODEL_ENDPOINT")
    if not endpoint:
        return None
    try:  # pragma: no cover - exercised in Databricks App runtime
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        payload = {
            "user_prompt": prompt,
            "validation_error": validation_error,
            "schema_intent": intent,
        }
        response = WorkspaceClient().serving_endpoints.query(
            endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=_CONTEXTUAL_ISSUE_PROMPT),
                ChatMessage(role=ChatMessageRole.USER, content=json.dumps(payload)),
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        return _json_from_text(_response_text(response))
    except Exception:
        return None


def _merge_model_contextual_issue_targets(
    intent: dict[str, Any], enrichment: dict[str, Any]
) -> dict[str, Any]:
    """Apply the agent's explicit issue replacements and supporting column additions."""
    merged = _merge_model_column_strategies(intent, enrichment)
    entries = enrichment.get("issues")
    issues = merged.get("issues")
    if not isinstance(entries, list) or not isinstance(issues, list):
        return merged

    remove_ids = enrichment.get("remove_issue_ids")
    if isinstance(remove_ids, list):
        removable = {value for value in remove_ids if isinstance(value, str)}
        issues[:] = [
            item
            for item in issues
            if not isinstance(item, dict) or item.get("issue_id") not in removable
        ]

    valid_types = {issue_type.value for issue_type in IssueType}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") not in valid_types:
            continue
        replacement_id = entry.get("replaces_issue_id")
        replacement_index = next(
            (
                index
                for index, existing in enumerate(issues)
                if isinstance(existing, dict)
                and replacement_id
                and existing.get("issue_id") == replacement_id
            ),
            None,
        )
        if replacement_index is None:
            same_defect = [
                index
                for index, existing in enumerate(issues)
                if isinstance(existing, dict)
                and existing.get("type") == entry.get("type")
                and existing.get("column") == entry.get("column")
            ]
            if len(same_defect) == 1:
                replacement_index = same_defect[0]
            elif entry.get("type") == IssueType.SCHEMA_DRIFT.value:
                existing_drift = [
                    index
                    for index, existing in enumerate(issues)
                    if isinstance(existing, dict)
                    and existing.get("type") == IssueType.SCHEMA_DRIFT.value
                ]
                if len(existing_drift) == 1:
                    replacement_index = existing_drift[0]
        replacement = {
            key: value
            for key, value in entry.items()
            if key not in {"replaces_issue_id", "issue_id"}
        }
        replacement.setdefault("parameters", {})
        if replacement_index is not None:
            previous = issues[replacement_index]
            assert isinstance(previous, dict)
            replacement["issue_id"] = entry.get("issue_id") or previous.get("issue_id")
            issues[replacement_index] = replacement
        else:
            replacement["issue_id"] = entry.get("issue_id") or (
                f"iss_agent_context_{len(issues) + 1:02d}"
            )
            issues.append(replacement)
    return merged


def _enrich_operational_contracts_with_model(
    prompt: str, intent: dict[str, Any], validation_error: str
) -> dict[str, Any] | None:
    """Complete a cross-cutting issue contract that requires both a new field and parameters."""
    endpoint = os.getenv("SDF_MODEL_ENDPOINT")
    if not endpoint:
        return None
    try:  # pragma: no cover - exercised in Databricks App runtime
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        response = WorkspaceClient().serving_endpoints.query(
            endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=_OPERATIONAL_ENRICHMENT_PROMPT),
                ChatMessage(
                    role=ChatMessageRole.USER,
                    content=json.dumps(
                        {
                            "user_prompt": prompt,
                            "validation_error": validation_error,
                            "schema_intent": intent,
                        }
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        return _json_from_text(_response_text(response))
    except Exception:
        return None


def _merge_model_operational_contracts(
    intent: dict[str, Any], enrichment: dict[str, Any]
) -> dict[str, Any]:
    """Merge a model-authored cross-cutting patch using the existing narrow mergers."""
    merged = _merge_model_column_strategies(intent, enrichment)
    return _merge_model_issue_parameters(merged, enrichment)


def _enrich_relationship_contracts_with_model(
    prompt: str, intent: dict[str, Any], validation_error: str
) -> dict[str, Any] | None:
    """Ask the planner to complete parent-filter contracts without replacing its schema."""
    endpoint = os.getenv("SDF_MODEL_ENDPOINT")
    if not endpoint:
        return None
    try:  # pragma: no cover - exercised in Databricks App runtime
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        payload = {
            "user_prompt": prompt,
            "validation_error": validation_error,
            "schema_intent": intent,
        }
        response = WorkspaceClient().serving_endpoints.query(
            endpoint,
            messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM,
                    content=_RELATIONSHIP_ENRICHMENT_PROMPT,
                ),
                ChatMessage(role=ChatMessageRole.USER, content=json.dumps(payload)),
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        return _json_from_text(_response_text(response))
    except Exception:
        return None


def _merge_model_relationship_contracts(
    intent: dict[str, Any], enrichment: dict[str, Any]
) -> dict[str, Any]:
    """Merge model-authored parent filters and only the columns needed to support them."""
    merged = json.loads(json.dumps(intent))
    relationships = merged.get("relationships")
    updates = enrichment.get("relationships")
    if isinstance(relationships, list) and isinstance(updates, list):
        for update in updates:
            if not isinstance(update, dict):
                continue
            target = next(
                (
                    relationship
                    for relationship in relationships
                    if isinstance(relationship, dict)
                    and (
                        update.get("name") == relationship.get("name")
                        or (
                            update.get("parent_table") == relationship.get("parent_table")
                            and update.get("child_table") == relationship.get("child_table")
                            and update.get("parent_column")
                            == relationship.get("parent_column")
                            and update.get("child_column")
                            == relationship.get("child_column")
                        )
                    )
                ),
                None,
            )
            if isinstance(target, dict):
                if isinstance(update.get("parent_filter"), dict):
                    target["parent_filter"] = update["parent_filter"]
                if isinstance(update.get("constraints"), dict):
                    target["constraints"] = update["constraints"]

    tables = {
        table.get("name"): table
        for table in merged.get("table_specs", [])
        if isinstance(table, dict)
    }
    columns = enrichment.get("columns")
    if not isinstance(columns, list):
        return merged
    for column in columns:
        if not isinstance(column, dict):
            continue
        table = tables.get(column.get("table"))
        name = column.get("name")
        if not isinstance(table, dict) or not isinstance(name, str):
            continue
        existing = next(
            (
                item
                for item in table.get("columns", [])
                if isinstance(item, dict) and item.get("name") == name
            ),
            None,
        )
        model_column = {key: value for key, value in column.items() if key != "table"}
        if isinstance(existing, dict):
            existing.update(model_column)
        elif model_column.get("type"):
            table.setdefault("columns", []).append(model_column)
    return merged




def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, list):
            text_items = [
                str(item["text"])
                for item in content
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
            ]
            if text_items:
                return text_items[-1]
        if content:
            return str(content)
        text = getattr(choices[0], "text", None)
        if text:
            return str(text)
    outputs = getattr(response, "outputs", None)
    if outputs:
        first = outputs[0]
        if isinstance(first, list):
            text_items = [
                str(item["text"])
                for item in first
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
            ]
            if text_items:
                return text_items[-1]
        if isinstance(first, dict) and first.get("type") == "text" and first.get("text"):
            return str(first["text"])
        return str(outputs[0])
    predictions = getattr(response, "predictions", None)
    if predictions:
        return str(predictions[0])
    return ""


def _json_from_text(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return None
    return parsed if isinstance(parsed, dict) else None




_COLUMN_ENRICHMENT_PROMPT = """
You complete a Scenario Data Factory schema designed by another model.
Return compact JSON only. Do not return tables, relationships, issues, commentary, or code.

Input contains a user prompt, an existing schema_intent, and validation_error listing
columns that lack executable data generation strategies. Return exactly:
{"columns":[{"table":"...","column":"...","faker":"...","values":[],
"weights":[],"semantic":{}}]}

Return one entry for every missing column in validation_error. Do not change columns
that are not listed. When validation_error requires a column absent from the schema
(for example a late-arrival ingestion timestamp), add exactly one entry with matching
column and name fields, plus type and a complete generation strategy. Supply only
relevant keys for each entry.

Rules:
- person name fields: faker "name"; person first/last names: matching Faker provider.
- account_number/bank_account: faker "iban"; company/merchant names: faker "company".
- categories, statuses, currencies, and labels: meaningful values, with weights when needed.
- For an insurance prompt where province volume governs policies or claims, put the
  requested Ontario/Quebec/Alberta/British Columbia weights on policies.province;
  customer geography alone does not satisfy policy or claim volume.
- independent date/timestamp: semantic {"kind":"timeline"}; dependent date:
  {"kind":"date_offset","base_column":"existing_column","min_days":0,"max_days":7}.
- decimal money/score/measure: semantic log_normal, uniform_range, or normal with
  all needed numeric parameters. Example: {"kind":"log_normal","median":85,
  "sigma":1.0,"max":5000}.
- boolean: values [true, false] with meaningful weights.
- independent integer/long business measures: min_value and max_value, or a semantic rule.
- values must be real domain values, never name_1/city_1/field_42 placeholders.
- When a null-value correlation has `where.source_system`, the affected table must contain a
  string `source_system` column with that source value and at least one contrasting real source
  value. For example, claims correlated with `legacy_batch` need values
  ["legacy_batch", "online_portal"] on claims.source_system.
Use the user prompt and existing schema names to choose realistic strategies.
"""

_CONTEXTUAL_ISSUE_PROMPT = """
You are a Scenario Data Factory contextual-correction agent. Return compact JSON only.
Do not return a complete schema, prose, markdown, or a deterministic template.

Input contains user_prompt, validation_error, and a model-authored schema_intent. Correct only
the columns and issue rules named by validation_error while preserving the model's tables,
relationships, row counts, requested rates, and all unrelated fields.

Return exactly:
{"columns":[{"table":"...","name":"...","type":"timestamp","semantic":{}}],
 "remove_issue_ids":["optional existing issue id"],
 "issues":[{"replaces_issue_id":"optional existing issue id","type":"...",
 "table":"...","column":"...","rate":0.01,"exact_count":null,"parameters":{}}]}

For AI model operations, apply the semantic owner of each request literally:
- model_inferences owns model_id, user_id, model_version, response_text,
  response_latency_ms, replay, and schema-drift defects;
- prompt_requests owns prompt_category and prompt_text;
- feedback_scores or feedback owns late feedback and feedback-before-inference;
- evaluation_results or evaluations owns evaluation_label;
- tenant_metadata or tenants owns region.
Replace an incorrectly targeted existing issue by setting replaces_issue_id. Preserve its exact
rate/count unless the prompt specifies a different one. Add a new issue only when one distinct
defect is absent. A table-level issue has column null.
When the prompt says duplicate inference records are caused by replayed ingestion files, remove
any duplicate_record rule for model_inferences and retain one file_replay rule as the only
duplicate source. Put that removed rule's id in remove_issue_ids.

When feedback-before-inference is requested, add feedback.inference_created_at as a timeline
timestamp if absent, generate feedback.created_at as a date_offset from it with non-negative
days, and make the intentional date_rule_violation use after_column=inference_created_at and
days_after=-1. For late feedback, use created_at as event_time_column and an existing or added
feedback ingestion timestamp as arrival_column. For a null or malformed response_text request,
return both null_value and invalid_format issues on model_inferences.response_text. For schema
drift, retain or provide executable batch and mutation parameters on model_inferences.

Every added column needs a complete executable generation strategy. Do not add placeholder
values or use an unrelated table merely because it has a similarly named field.
"""

_ISSUE_ENRICHMENT_PROMPT = """
You complete executable parameters for issue rules in a Scenario Data Factory schema.
Return compact JSON only. Do not return tables, columns, relationships, commentary, or code.

Input contains the user prompt, schema_intent, and a validation_error. Return exactly:
{"issues":[{"issue_id":"optional","type":"...","table":"...","column":"...",
"rate":null,"exact_count":null,"parameters":{},"correlation":{}}]}

Return an entry only for issue rules whose parameters are missing or invalid according
to validation_error. If validation_error says a requested issue type is missing, return one
complete new issue with its requested table, column, and rate/count. Do not otherwise change
rates, counts, tables, columns, issue types, or the schema. Select only existing columns from
the supplied schema_intent.

Rules:
- Every date_rule_violation requires parameters.after_column naming an existing column
  on the same table. For a request such as "ship_date before order_date", use
  column ship_date, after_column order_date, and days_after -1. For an ordinary
  "must be on or after" rule, use the earlier/source date as after_column and
  days_after -1 only when the requested defect must make the target date earlier.
- late_arrival needs event_time_column and arrival_column when the schema has both;
  use the actual business event timestamp and ingestion/arrival timestamp.
- For a request that a specified share of a null-value issue is correlated, retain the
  requested total null rate and return correlation {"share":...,"where":{...}} on
  that same null_value issue. Do not create a second generic null issue.
- schema_drift needs a concrete activation_batch and add_columns or another concrete
  mutation using existing table context.
- file_replay needs file_count, source_batch, and target_batch. Batches must be
  valid for the requested timeline.
  For replayed inference files, return {"type":"file_replay","table":"model_inferences",
  "column":null,"rate":null,"exact_count":null,"parameters":{"file_count":1,
  "source_batch":2,"target_batch":4}}.
Use the user's literal business rule to choose the parameters. Never invent a
generic field name or replace the issue with prose.
"""

_OPERATIONAL_ENRICHMENT_PROMPT = """
You complete one cross-cutting operational contract in a Scenario Data Factory schema.
Return compact JSON only. Do not return tables, relationships, commentary, or code.

Input contains the user prompt, schema_intent, and validation_error. Return exactly:
{"columns":[{"table":"...","column":"...","type":"timestamp","semantic":{}}],
"issues":[{"issue_id":"optional","type":"...","table":"...","column":"...",
"parameters":{},"correlation":{}}]}

Return entries only when validation_error requires them. A late-arrival defect is an
event-versus-arrival contract, not a mutation of the event time. When a requested
late-arrival issue lacks an arrival field, add a timestamp column with matching table
and column fields plus semantic {"kind":"timeline"}; then patch the same issue.

For a request such as "late_arrival on payments: 5%, delayed 1-7 days", return these
exact executable fragments, adapting only names present in the schema:
{"table":"payments","column":"ingestion_ts","type":"timestamp",
"semantic":{"kind":"timeline"}}
{"type":"late_arrival","table":"payments","column":"payment_date",
"parameters":{"event_time_column":"payment_date","arrival_column":"ingestion_ts",
"delay_days_min":1,"delay_days_max":7}}
Do not put the rate inside parameters and do not return prose.

For a date_rule_violation whose comparison column is absent from the affected table,
add a same-table comparison timestamp/date and patch the issue. For an appointment
before encounter defect on encounters, add encounters.appointment_date with semantic
{"kind":"timeline"}, update encounters.encounter_date to semantic
{"kind":"date_offset","base_column":"appointment_date","min_days":0,"max_days":7},
and set the issue parameters to {"after_column":"appointment_date","days_after":-1}.
The comparison column and target must be on the same table; do not point at a parent
table's column.
"""

_RELATIONSHIP_ENRICHMENT_PROMPT = """
You complete relationship execution contracts for a Scenario Data Factory schema.
Return compact JSON only. Do not return tables, issues, commentary, or code.

Input contains the user prompt, schema_intent, and a validation_error. Return exactly:
{"relationships":[{"name":"...","parent_filter":{"column":"...","values":["..."]},
"constraints":{}}],
"columns":[{"table":"...","name":"...","type":"string","values":["..."],
"weights":[...]}]}

Return only the relationships and parent-table columns needed to resolve a relationship
validation error. Do not change table names, keys, row counts, issue rules, or data
generation strategies unrelated to the error.

Rules:
- A relationship parent_filter must reference an existing column on its parent table.
- Use relationship constraints to make temporal and aggregate business rules executable:
  child_date_ranges uses child_column, parent_start_column, and parent_end_column;
  aggregate_caps uses child_amount_column, parent_amount_column, and maximum_fraction.
- When the prompt requires a child record to exist only for a subset of parent records,
  such as returns only for delivered orders, retain that semantic restriction. Either
  choose an existing meaningful parent status column or add the required parent column
  with a complete realistic categorical strategy. Do not remove the parent_filter.
- Any added string column must include real values (and weights when appropriate), not
  generic placeholders. For delivered-order semantics, include "delivered" among the
  possible values.
- Relationship updates must match an existing relationship name exactly and columns
  must match an existing parent table exactly.
- For insurance validation errors, return the full executable patches rather than
  prose. Use these exact JSON shapes, not shorthand arrays or maps:
  {"name":"policies_claims","parent_filter":{"column":"status","values":["active"]},
  "constraints":{"child_date_ranges":[{"child_column":"loss_date",
  "parent_start_column":"effective_date","parent_end_column":"expiry_date"}]}}
  {"name":"claims_payments","parent_filter":{"column":"claim_status",
  "values":["approved","settled"]},"constraints":{"aggregate_caps":[{
  "child_amount_column":"payment_amount","parent_amount_column":"claim_amount",
  "maximum_fraction":0.95}]}}
"""

_BASIC_SCHEMA_SYSTEM_PROMPT = """
You are a fast Scenario Data Factory schema-design agent. Return compact JSON only.

Design a small, executable synthetic relational dataset from the user's short request.
This is not a full enterprise architecture exercise: use 3 to 5 connected tables unless
the user explicitly asks for more. Infer a central fact/event table; when the user gives
one overall record count, assign it to that fact/event table exactly and choose plausible
smaller dimensions. Do not create data-quality issues unless the user asks for them.

For a basic Canadian banking request, create exactly these four practical tables unless
the user asks for extra entities: customers, branches, accounts, transactions. Put the
requested overall count on transactions. Connect customers to accounts and branches, and
accounts to transactions. Include real Canadian cities/provinces, actual banking product
categories, transaction types, channels, and readable categorical values. Use customer
names through faker "name", cities through faker "city" or a province lookup, and never
use city_1/name_1/place_1 placeholders. Use a timeline semantic for independent dates or
timestamps, a date_offset for dependent dates, a numeric semantic for decimal amounts,
and values/weights for categorical fields. Use locale "en_CA" for Canadian data.

Allowed types: string, integer, long, decimal, date, timestamp, boolean.
Supported Faker providers: address, ascii_email, city, company, country,
credit_card_number, date_time, domain_name, email, first_name, iban, job, last_name,
name, paragraph, phone_number, postcode, sentence, state, state_abbr, street_address,
text, url, user_name, uuid4, word.

Every table needs exactly one primary key. Every relationship needs existing parent and
child key columns. Every non-key business field needs an executable strategy: faker,
meaningful values, a lookup, a timeline/date-offset semantic, a numeric range, or a
numeric distribution. Keep output mode as Delta plus raw and metadata.synthetic_data=true.

Return this exact JSON shape:
{
  "domain":"custom_schema",
  "name":"short descriptive name",
  "dataset_code":"short_table_prefix",
  "seed":42,
  "locale":"en_CA",
  "timeline":{"start_date":"2025-01-01","batches":12,"frequency":"monthly"},
  "table_counts":{},
  "table_specs":[{
    "name":"customers","row_count":1000,
    "columns":[{"name":"customer_id","type":"long","primary_key":true,"nullable":false}]
  }],
  "relationships":[],
  "issues":[],
  "metadata":{"synthetic_data":true}
}
"""


_CUSTOM_SCHEMA_SYSTEM_PROMPT = """
You are a bounded Scenario Data Factory custom-schema planner.
Return compact JSON only. Do not include markdown or explanation.

Create a complete synthetic relational schema for the user's domain. The user gives
business nouns, rough scale, and bad-data requirements. Interpret those nouns into
all required tables, schemas, columns, row counts, relationships, and issue rules.

Do not return a generic two-table schema when the prompt names multiple domain
entities. Every named business entity must be represented as a table or as a
relationship-bearing column on a table.

Examples of entity interpretation:
- "orphan cell tower IDs" requires a cell_towers parent table and tower_id FK on events.
- "missing network engineers" requires a network_engineers table and nullable engineer FK
  on incidents, closures, tickets, or work orders.
- "late incident closures" requires an incident_closures/closures table with closure_ts.
- "invalid customer regions" requires customer_regions/regions with region_code/province.
- "orphan patient IDs" requires patients plus patient_id FK on claims.
- "missing provider IDs" requires providers plus provider_id FK on claims.
- "missing merchant category" requires merchants.category.
- "retail promotions" requires promotions plus products, stores, customers, coupons,
  and coupon_redemptions when coupon redemption is mentioned.
- "duplicate promotions" applies duplicate_record to promotions, not coupons.
- "orphan product IDs" applies referential_orphan to the primary requested table's
  product_id when that table has product_id.
- "missing store IDs" applies null_value to the primary requested table's store_id.
- "late coupon redemptions" applies late_arrival to coupon_redemptions.redemption_ts.
- "invalid customer postal codes" applies invalid_value to customers.postal_code.

Include:
- domain: "custom_schema"
- name
- dataset_code: a concise lower_snake_case table prefix, 3-16 characters. Use a memorable
  domain abbreviation such as "cbank" for Canadian banking or "retail" for retail sales;
  never include row counts, run IDs, hashes, or quality markers.
- locale when provided by the user
- timeline when provided by the user, e.g.
  {"start_date":"2025-01-01","batches":12,"frequency":"monthly"}
- scale: "demo"
- seed: 42
- table_counts: {}
- table_specs: a list of tables with row_count and columns
- relationships: parent/child relationships
- issues: supported issue rules
- metadata: preserve business_rules, statistical_anchors, distribution_rules,
  seasonality, output_settings, and assumptions as structured JSON

Allowed column types:
string, integer, long, decimal, date, timestamp, boolean.

Every table must have exactly one primary key column.
Table names must be unique after lower_snake_case normalization.
Column names must be unique within a table after lower_snake_case normalization.
Foreign key columns must exist on child tables.
Use relationship parent_filter {"column":"record_status","values":["delivered"]}
when a child table must reference only a defined subset of parent records.
Use relationship constraints for executable parent-child business rules. Supported forms are
{"child_date_ranges":[{"child_column":"...","parent_start_column":"...",
"parent_end_column":"..."}]} and {"aggregate_caps":[{"child_amount_column":"...",
"parent_amount_column":"...","maximum_fraction":0.95}]}. Metadata records context only;
every explicit statistic and business rule must also be encoded in executable columns,
relationships, or issue parameters.
For the insurance rules "active policy on the loss_date", "payments only for
approved or settled claims", and "late_arrival on payments", use this exact
shape (the `parent_filter` belongs beside `constraints`, never inside it):
{"name":"policies_claims","parent_table":"policies","parent_column":"policy_id",
 "child_table":"claims","child_column":"policy_id",
 "parent_filter":{"column":"status","values":["active"]},
 "constraints":{"child_date_ranges":[{"child_column":"loss_date",
 "parent_start_column":"effective_date","parent_end_column":"expiry_date"}]}}
{"name":"claims_payments","parent_table":"claims","parent_column":"claim_id",
 "child_table":"payments","child_column":"claim_id",
 "parent_filter":{"column":"claim_status","values":["approved","settled"]},
 "constraints":{"aggregate_caps":[{"child_amount_column":"payment_amount",
 "parent_amount_column":"claim_amount","maximum_fraction":0.95}]}}
{"type":"late_arrival","table":"payments","column":"payment_date","rate":0.05,
 "parameters":{"event_time_column":"payment_date","arrival_column":"ingestion_ts",
 "delay_days_min":1,"delay_days_max":7}}
Issues must reference existing tables and columns, except table-level issues.
If the user supplies a count for a central fact/event table, preserve that count
exactly and infer plausible counts for parent dimensions and downstream event tables.
Use 4 to 8 tables for multi-entity domains unless the prompt explicitly asks for one table.
If the user explicitly lists tables and columns, preserve those names. Add only columns
needed for relationships, time behavior, or issue rules.
If the user explicitly lists issue rules with table, column, and rate, preserve them exactly.
Do not silently remap issue rules to another table.

For healthcare claims, prefer tables like patients, providers, claims, adjudications,
and payments when appropriate. Use patient_id/provider_id foreign keys on claims.
For insurance claims with customers, policies, claims, and payments, encode literal
anchors on the actual columns: policy_type values auto/home/tenant with the requested
weights; claim_status weights whose closed plus settled share equals the request; and
province weights where Ontario and Quebec exceed Alberta and British Columbia. For a
material tail above a threshold, use log_normal semantic fields tail_share, tail_min,
and tail_max in addition to median, sigma, and max. A valid policy-to-claim link must
filter active policies and use child_date_ranges to make loss_date fall within
effective_date and expiry_date. A payments link restricted to approved/settled claims
must use parent_filter and aggregate_caps so total payment amount cannot exceed the
parent claim amount. Late arrival means a separate event timestamp and ingestion
timestamp: put event_time_column and arrival_column in issue parameters. For a request
that half of a null-value issue is correlated, keep the requested total null rate and
put correlation {"share":0.5,"where":{"source_system":"legacy_batch","after_batch":6}}
on that null_value issue. Never place these requirements only in metadata.
For telecom network events, prefer tables like customer_regions, cell_towers,
network_engineers, network_events, incidents, and incident_closures when appropriate.
For AI model operations and evaluation datasets, prefer tables like tenant_metadata,
user_directory, user_events, prompt_requests, model_inferences, feedback_scores,
evaluation_results, incident_logs, and model_registry. Preserve a prompt count like
"100,000 records representing" as model_inferences=100000. Map orphan model IDs to
model_inferences.model_id, orphan user IDs to model_inferences.user_id, missing prompt
categories to prompt_requests.prompt_category, missing evaluation labels to
evaluation_results.evaluation_label, missing latency to model_inferences.response_latency_ms,
late feedback to feedback_scores.created_at with ingestion_ts as arrival column, replayed
inference files to file_replay on model_inferences, invalid model versions to
model_inferences.model_version, invalid tenant regions to tenant_metadata.region, and
feedback-before-inference to feedback_scores.created_at using inference_created_at.
For feedback-before-inference, feedback_scores must contain both timestamp columns.
Generate clean feedback_scores.created_at as a date_offset after inference_created_at, and encode
the intentional defect as {"type":"date_rule_violation","table":"feedback_scores",
"column":"created_at","parameters":{"after_column":"inference_created_at","days_after":-1}}.
The model_inferences-to-feedback_scores relationship is an ordinary inference_id foreign-key
relationship with no child_date_ranges constraint: both feedback timestamp fields are local to
feedback_scores. Issue types are not physical data columns. Never create columns named
file_replay, schema_drift, duplicate_record, late_arrival, or referential_orphan; encode those
only in the issues list with executable parameters.
For decimal scores, use semantic {"kind":"uniform_range","min":0,"max":1} (or 0 to 5 for
rating scores), not `uniform`. Do not infer a products table from the word "production".
For retail sales guardrail prompts with customers, orders, and returns, preserve those
tables. Treat a prompt like "Generate 500,000 records for a retail sales scenario"
as orders=500000 unless the user explicitly assigns the count elsewhere. Preserve
North-East US state rules in metadata. Use customers.state values from the requested
states. Map duplicate_record on orders to orders, referential_orphan on customer_id
to orders.customer_id, date_rule_violation "ship before order" to orders.ship_date
with after_column order_date, and null_value on returns.reason to returns.reason.
Use population-derived weights for customers.state, a state-keyed city lookup with
real cities, weighted online/in_store values for channel, a log_normal amount rule,
and a date_offset rule for clean ship_date values.
For Canadian banking prompts that mention SAP, loyalty, cards, campaigns, or advertising,
design a connected banking operating model, for example customer master/business partners,
branches, banking products, credit cards, account or card transactions, loyalty members and
point-ledger activity, campaigns, and advertising events. Use SAP-style business identifiers
where helpful (for example customer_number/KUNNR, company_code/BUKRS, document_number/BELNR)
alongside readable analytics fields. Use actual Canadian cities, branches, products, campaign
names, transaction types, and channel values through categorical values, semantic lookups, or
supported Faker providers. Do not add data-quality issue rules unless the user explicitly asks
for them. Supported Faker providers are: address, bank_country, city, company, country,
credit_card_number, email, first_name, iban, job, last_name, name, paragraph, postcode,
sentence, state, state_abbr, street_address, text, url, user_name, uuid4, and word.

Final silent checklist for AI model operations prompts: model_inferences must carry model_id,
user_id, model_version, response_text, and response_latency_ms even when prompt_requests also
has identity fields. Map orphan model/user, invalid model version, latency, response-text,
replay, and schema-drift defects to model_inferences. Map missing prompt category to
prompt_requests; missing evaluation label to evaluation_results/evaluations; late feedback and
feedback-before-inference to feedback_scores/feedback; and invalid tenant region to
tenant_metadata/tenants. For "null or malformed response_text", create both null_value and
invalid_format on model_inferences.response_text. Verify this mapping before emitting JSON.

Supported issue types:
null_value, blank_value, duplicate_record, invalid_format, invalid_value,
referential_orphan, date_rule_violation, late_arrival, out_of_order,
file_replay, schema_drift, correlated_missingness.

Return shape:
{
  "domain": "custom_schema",
  "name": "healthcare claims scenario",
  "scale": "demo",
  "seed": 42,
  "table_counts": {},
  "table_specs": [
    {
      "name": "patients",
      "row_count": 50000,
      "columns": [
        {"name": "patient_id", "type": "long", "primary_key": true, "nullable": false},
        {"name": "province", "type": "string", "values": ["ON", "QC", "BC", "AB"]},
        {"name": "city", "type": "string", "semantic": {
          "kind": "lookup", "key_column": "province", "values_by_key": {
            "ON": ["Toronto", "Ottawa"], "QC": ["Montreal", "Quebec City"],
            "BC": ["Vancouver", "Victoria"], "AB": ["Calgary", "Edmonton"]
          }
        }}
      ]
    }
  ],
  "relationships": [
    {
      "name": "patients_claims",
      "parent_table": "patients",
      "parent_column": "patient_id",
      "child_table": "claims",
      "child_column": "patient_id",
      "constraints": {}
    }
  ],
  "issues": [
    {
      "type": "referential_orphan",
      "table": "claims",
      "column": "patient_id",
      "rate": 0.02,
      "exact_count": null,
      "parameters": {}
    }
  ]
}
"""

_CUSTOM_SCHEMA_REPAIR_PROMPT = """
You repair invalid Scenario Data Factory custom-schema JSON.
Return compact JSON only. Do not include markdown or explanation.

Input contains:
- user_prompt
- validation_error
- invalid_intent

Return a corrected custom_schema intent with complete table_specs, relationships,
and issues. Preserve requested table counts, rates, and exact counts. Fix duplicate
table names, duplicate column names, missing primary keys, missing FK columns, bad
issue references, and semantic coverage gaps.
Do not replace requested executable issue rules with prose-only diagnostic issue
objects. Every returned issue must have a supported type, table, and rate, exact_count,
or file_count.
Also preserve requested locale, timeline, output mode, business rules, statistical
anchors, distributions, seasonality, and table/column names in metadata. Do not replace
explicit user-listed tables with a generic source/event schema.

Every named business entity in the prompt must be represented as a table or a
relationship-bearing column. Multi-entity prompts should use 4 to 8 related tables.
Issue types belong only in the `issues` list, never as physical schema columns. In
particular, do not create columns named file_replay, schema_drift, duplicate_record,
late_arrival, or referential_orphan. A relationship child_date_ranges constraint may
reference only columns that actually exist on its parent and child tables; remove it
when the user's rule is instead represented by local event timestamps.

Every non-key string column must use a real semantic value strategy: a supported
Faker provider, meaningful categorical values, or a bounded semantic lookup tied
to another existing column. Do not use ID-derived placeholders such as city_1,
name_42, or source_7. Preserve geographic consistency with semantic lookups when
the user requires it.

Preserve every statistical or temporal guardrail as an executable column rule. Use
values plus weights for population/percentage distributions, semantic
{"kind":"log_normal",...} for a monetary long-tail request, and semantic
{"kind":"date_offset",...} when one date must follow another. For a rule such as
"returns only exist for delivered orders", add a meaningful delivered-status column
to orders and put parent_filter {"column":"record_status","values":["delivered"]}
on the orders-to-returns relationship. Do not leave these rules as metadata only.
Every date_rule_violation must name an existing same-table comparison column in
parameters.after_column. A requested defect of ship_date before order_date must be
encoded as column ship_date, parameters {"after_column":"order_date",
"days_after":-1}; do not omit either parameter.

For a retail prompt mentioning population weighting, a 65/35 channel split,
log-normal amounts, ordered shipping dates, and delivered-only returns, a compliant
fragment has this shape (adapt the actual values to the user's request):
{
  "name":"state", "type":"string", "values":["NY","PA"],
  "weights":[19500000,13000000]
}
{
  "name":"city", "type":"string", "semantic":{"kind":"lookup",
  "key_column":"state", "values_by_key":{"NY":["New York City","Buffalo"],
  "PA":["Philadelphia","Pittsburgh"]}}
}
{
  "name":"channel", "type":"string", "values":["online","in_store"],
  "weights":[65,35]
}
{
  "name":"amount", "type":"decimal", "precision":12, "scale":2,
  "semantic":{"kind":"log_normal","median":85,"sigma":1.0,"max":5000}
}
{
  "name":"ship_date", "type":"date", "semantic":{"kind":"date_offset",
  "base_column":"order_date","min_days":0,"max_days":7}
}
The independent order_date must use semantic {"kind":"timeline"}. A monetary
returns.refund_amount must also use an explicit numeric distribution, such as a
log_normal rule with a realistic lower median and the same maximum bound.
The orders table also needs record_status values including "delivered", and the
orders-to-returns relationship needs parent_filter
{"column":"record_status","values":["delivered"]}. This is an example of
the contract, not a reason to replace the user's tables or names.
For a returns.reason field, include meaningful values such as
["defective", "wrong_item", "changed_mind", "damaged_in_transit"] rather than
leaving it without a generation strategy.
Every non-key, non-foreign-key business field must be explicit: use timeline for
independent dates, date_offset for dependent dates, a stated distribution for every
decimal, numeric bounds/semantics for independent numeric measures, and real values,
lookups, or Faker for strings. No field may rely on a generic compiler default.
Treat validation errors about distributions, correlations, or relationship constraints
as executable requirements. Metadata never satisfies them. For insurance claims,
preserve policy_type weights, the combined closed/settled claim-status share, weighted
province volumes, an explicit tail_share/tail_min/tail_max for a stated high-value tail,
and a null_value correlation object when requested. Use relationship constraints
child_date_ranges and aggregate_caps to enforce policy-date and payment-total rules.
Late-arrival repairs must add a separate ingestion timestamp and name both the event
and arrival columns in the issue parameters.
For insurance claims, repair the three core relationship contracts exactly as follows:
`policies_claims.parent_filter` is `{"column":"status","values":["active"]}` and its
`constraints.child_date_ranges` maps claims.loss_date to policies.effective_date and
policies.expiry_date; `claims_payments.parent_filter` is
`{"column":"claim_status","values":["approved","settled"]}` and its
`constraints.aggregate_caps` maps payments.payment_amount to claims.claim_amount; and a
late_arrival payment issue has `event_time_column`, `arrival_column`, `delay_days_min`, and
`delay_days_max`. These keys must be executable JSON fields, not metadata or prose.
For a financial-crime or banking schema inferred from a short prompt, use faker
"name" for suspects.full_name (or equivalent person names), faker "iban" for
accounts.account_number, meaningful categorical values for risk/status/type fields,
and explicit date/numeric rules for every business measure.
For AI feedback-before-inference requests, feedback_scores must have local
inference_created_at and created_at timestamps, with created_at generated as a
date_offset from inference_created_at. The model_inferences-to-feedback_scores
relationship must be a plain inference_id relationship without child_date_ranges.
For AI model operations requests, keep defects on the operational entity named by
the request: orphan model_id and user_id defects belong on model_inferences, not
prompt_requests; missing prompt categories belong on prompt_requests; missing
evaluation labels belong on evaluation_results; missing, negative, malformed, and
invalid inference fields belong on model_inferences; late feedback belongs on
feedback_scores/feedback with created_at and a separate ingestion timestamp; replay
and schema-drift defects belong on model_inferences. Invalid tenant regions belong on
tenant_metadata/tenants. Do not satisfy these requirements with the same issue type on
an unrelated table. A request for null or malformed response_text needs both a
null_value and an invalid_format rule on model_inferences.response_text.

Allowed column types:
string, integer, long, decimal, date, timestamp, boolean.

Supported issue types:
null_value, blank_value, duplicate_record, invalid_format, invalid_value,
referential_orphan, date_rule_violation, late_arrival, out_of_order,
file_replay, schema_drift, correlated_missingness.
"""


def _ensure_databricks_outputs(spec: ScenarioSpec) -> ScenarioSpec:
    data = spec.model_dump(mode="json")
    catalog, schema = _workspace_output_location()
    data["outputs"].update(
        {
            "mode": OutputMode.BOTH.value,
            "include_clean": True,
            "catalog": data["outputs"].get("catalog") or catalog,
            "schema_name": data["outputs"].get("schema_name") or schema,
        }
    )
    return ScenarioSpec.model_validate(data)




def _extract_name(prompt: str, domain: str) -> str:
    quoted = re.search(r'"([^"]{4,120})"', prompt)
    if quoted:
        return quoted.group(1)
    if domain == "retail_orders":
        return "Retail Bad Data Scenario"
    return "Canadian Insurance Claims Reliability Scenario"














def _keyword_window(text: str, keyword: str) -> str:
    index = text.find(keyword)
    if index < 0:
        return text
    return text[max(0, index - 80) : index + 180]


def _keyword_clause(text: str, keyword: str) -> str:
    window = _keyword_window(text, keyword)
    index = window.find(keyword)
    if index < 0:
        return window
    before = re.split(r"[,.;]|\band\b", window[:index])[-1]
    after = re.split(r"[,.;]|\band\b", window[index + len(keyword) :])[0]
    return f"{before} {keyword} {after}"


def _write_databricks_scenario_spec(spec: ScenarioSpec) -> str:
    root = _control_volume_root()
    path = f"{root.rstrip('/')}/scenarios/{spec.scenario_id}.yaml"
    if path.startswith("/Volumes/"):
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient()
        directory = path.rsplit("/", 1)[0]
        client.files.create_directory(directory)
        client.files.upload(path, BytesIO(spec.to_yaml().encode("utf-8")), overwrite=True)
        return path
    local_path = Path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(spec.to_yaml(), encoding="utf-8")
    return local_path.as_posix()


def _control_volume_root() -> str:
    return os.getenv("SDF_CONTROL_VOLUME") or "/Volumes/sdf/scenario_data_factory/sdf_control"


def _raw_runs_root() -> str:
    configured_root = os.getenv("SDF_RAW_RUNS_ROOT")
    if configured_root:
        return configured_root
    raw_volume = os.getenv("SDF_RAW_VOLUME")
    if raw_volume:
        return f"{raw_volume.rstrip('/')}/runs"
    return "/Volumes/sdf/scenario_data_factory/sdf_raw/runs"


def _workspace_output_location() -> tuple[str, str]:
    """Resolve output storage from the bound control volume in any bundle target."""
    root = os.getenv("SDF_CONTROL_VOLUME", "")
    parts = root.strip("/").split("/")
    if len(parts) >= 4 and parts[0] == "Volumes":
        return parts[1], parts[2]
    return (
        os.getenv("SDF_CATALOG", "sdf"),
        os.getenv("SDF_SCHEMA", "scenario_data_factory"),
    )
