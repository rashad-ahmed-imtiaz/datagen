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
    RelationshipSpec,
    ScenarioSpec,
    TableSpec,
    TimelineSpec,
)
from scenario_data_factory.persistence.run_repository import RunRepository
from scenario_data_factory.persistence.scenario_repository import ScenarioRepository


class ScenarioService:
    def __init__(
        self,
        scenarios: ScenarioRepository | None = None,
        runs: RunRepository | None = None,
    ) -> None:
        self.scenarios = scenarios or ScenarioRepository()
        self.runs = runs or RunRepository()

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
            databricks_run_id = getattr(waiter, "run_id", None)
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
        run = self.runs.get(run_id)
        return {"run_id": run_id, "status": run["status"]}

    def get_run_summary(self, run_id: str) -> dict[str, object]:
        return self.runs.get(run_id)

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
        "tables": {table.name: table.row_count for table in spec.tables},
        "columns": {
            table.name: [
                {
                    "name": column.name,
                    "type": column.type,
                    "primary_key": column.primary_key,
                    "nullable": column.nullable,
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


def _spec_from_prompt(prompt: str) -> tuple[ScenarioSpec, list[str]]:
    if _blueprint_domain_from_prompt(prompt.lower()) is None:
        spec, assumptions = _custom_spec_from_agent_or_fallback(prompt)
        return spec, assumptions

    intent, intent_note = _agent_intent_from_model(prompt)
    if intent:
        try:
            spec, assumptions = _spec_from_intent(intent, prompt)
            return spec, [intent_note, *assumptions]
        except Exception as exc:
            fallback_note = (
                "Agent intent extraction was invalid; used deterministic fallback: "
                f"{exc}"
            )
    else:
        fallback_note = intent_note
    spec, assumptions = _heuristic_spec_from_prompt(prompt)
    return spec, [fallback_note, *assumptions]


def _custom_spec_from_agent_or_fallback(prompt: str) -> tuple[ScenarioSpec, list[str]]:
    normalized = prompt.lower()
    if _looks_like_ai_model_ops(normalized):
        text = prompt.strip()
        seed = _extract_seed(normalized)
        table_counts = _extract_table_count_mentions(normalized)
        spec = _ai_model_ops_custom_spec(text, table_counts, seed)
        spec = _ensure_databricks_outputs(spec)
        spec = _apply_custom_prompt_issues(spec, normalized)
        return spec, [
            "Recognized AI model operations domain; used deterministic domain planner.",
            (
                "Preserved named AI platform tables, relationships, row counts, and "
                "issue targets before validation."
            ),
            *_custom_issue_assumptions(normalized, spec),
            "Generation still requires hash confirmation before the Databricks job is submitted.",
        ]

    intent = _custom_schema_intent_from_model(prompt)
    if intent and isinstance(intent.get("table_specs"), list):
        try:
            spec, assumptions = _custom_spec_from_intent(intent, prompt)
            return spec, ["Schema-design agent created the custom ScenarioSpec.", *assumptions]
        except Exception as exc:
            repaired = _repair_custom_schema_intent_with_model(prompt, intent, str(exc))
            if repaired and isinstance(repaired.get("table_specs"), list):
                try:
                    spec, assumptions = _custom_spec_from_intent(repaired, prompt)
                    return spec, [
                        "Schema-design agent returned an invalid draft; repair agent fixed it.",
                        *assumptions,
                    ]
                except Exception:
                    pass
            fallback_note = f"Schema-design agent output was invalid: {exc}"
    else:
        fallback_note = "Schema-design agent did not return usable table_specs."

    spec, assumptions = _heuristic_custom_spec_from_prompt(prompt)
    return spec, [fallback_note, *assumptions]


def _heuristic_spec_from_prompt(prompt: str) -> tuple[ScenarioSpec, list[str]]:
    text = prompt.strip()
    normalized = text.lower()
    domain = _blueprint_domain_from_prompt(normalized)
    if domain is None:
        return _heuristic_custom_spec_from_prompt(prompt)
    scale = "small" if any(word in normalized for word in ("small", "sample", "test")) else "demo"
    seed = _extract_seed(normalized)
    name = _extract_name(text, domain)
    spec = get_blueprint(domain).build(name=name, seed=seed, scale=scale)
    spec = _ensure_databricks_outputs(spec)
    spec, explicit_counts = _apply_table_counts(spec, normalized)
    spec = _refresh_default_issue_counts(spec)
    spec = _apply_prompt_issues(spec, normalized)
    assumptions = [
        f"Interpreted domain as {domain}.",
        f"Using {scale} scale unless explicit table counts were supplied.",
        (
            "Model table planner was not available; only explicitly supplied table counts "
            "were changed."
            if explicit_counts
            else "Using blueprint table counts."
        ),
        "Generation still requires hash confirmation before the Databricks job is submitted.",
    ]
    return spec, assumptions


def _spec_from_intent(intent: dict[str, Any], prompt: str) -> tuple[ScenarioSpec, list[str]]:
    if isinstance(intent.get("table_specs"), list):
        return _custom_spec_from_intent(intent, prompt)

    domain = str(intent.get("domain") or "").strip()
    if domain == "custom_schema" or domain not in {"insurance_claims", "retail_orders"}:
        custom_intent = _custom_schema_intent_from_model(prompt)
        if custom_intent and isinstance(custom_intent.get("table_specs"), list):
            return _custom_spec_from_intent(custom_intent, prompt)
        raise ValueError(
            "non-blueprint domains require model-provided custom table_specs"
        )
    scale = str(intent.get("scale") or "demo")
    if scale not in {"small", "demo"}:
        scale = "demo"
    name = str(intent.get("name") or _extract_name(prompt, domain))
    seed = int(intent.get("seed") or 42)
    spec = get_blueprint(domain).build(name=name, seed=seed, scale=scale)
    spec = _ensure_databricks_outputs(spec)
    data = spec.model_dump(mode="json")
    table_counts = intent.get("table_counts") or {}
    explicit_counts: set[str] = set()
    if isinstance(table_counts, dict):
        by_name = {table["name"]: table for table in data["tables"]}
        for table, count in table_counts.items():
            if table in by_name and count is not None:
                by_name[str(table)]["row_count"] = max(1, int(count))
                explicit_counts.add(str(table))
    spec = ScenarioSpec.model_validate(data)
    if {table.name for table in spec.tables}.issubset(explicit_counts):
        table_plan_note = "Agent provided a complete table-count plan."
    else:
        spec, table_plan_note = _complete_table_counts_with_model(spec, explicit_counts, prompt)
    data = spec.model_dump(mode="json")
    parsed_issues = _issues_from_intent(intent.get("issues"), spec)
    if parsed_issues:
        data["issues"] = [issue.model_dump(mode="json") for issue in parsed_issues]
    else:
        counted_spec = _refresh_default_issue_counts(ScenarioSpec.model_validate(data))
        data["issues"] = _apply_prompt_issues(counted_spec, prompt.lower()).model_dump(
            mode="json"
        )["issues"]
    return ScenarioSpec.model_validate(data), [
        f"Interpreted domain as {domain}.",
        table_plan_note,
        "Model intent was constrained to supported blueprints, tables, columns, and issue types.",
        "Generation still requires hash confirmation before the Databricks job is submitted.",
    ]


def _custom_spec_from_intent(intent: dict[str, Any], prompt: str) -> tuple[ScenarioSpec, list[str]]:
    name = str(intent.get("name") or _extract_name(prompt, "custom_schema"))
    seed = int(intent.get("seed") or 42)
    tables = _unique_table_specs_from_intent(intent["table_specs"])
    if not tables:
        raise ValueError("custom schema must contain at least one valid table")
    table_map = {table.name: table for table in tables}
    relationships: list[RelationshipSpec] = []
    for relationship in intent.get("relationships", []):
        if not isinstance(relationship, dict):
            continue
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
        except Exception:
            continue
        if parsed.parent_table not in table_map or parsed.child_table not in table_map:
            continue
        if parsed.parent_column not in table_map[parsed.parent_table].column_names():
            continue
        if parsed.child_column not in table_map[parsed.child_table].column_names():
            continue
        relationships.append(parsed)
    spec = ScenarioSpec(
        name=name,
        domain="custom_schema",
        seed=seed,
        locale=str(intent.get("locale") or "en_CA"),
        timeline=TimelineSpec(start_date=date(2026, 1, 1), batches=30),
        tables=tables,
        relationships=relationships,
    )
    spec = _ensure_databricks_outputs(spec)
    data = spec.model_dump(mode="json")
    data["issues"] = []
    base_spec = _ensure_requested_relationship_columns(
        ScenarioSpec.model_validate(data), prompt.lower()
    )
    base_spec = _ensure_requested_operational_columns(base_spec, prompt.lower())
    data = base_spec.model_dump(mode="json")
    prompt_issue_spec = (
        _apply_custom_prompt_issues(base_spec, prompt.lower())
        if _mentions_any_issue(prompt.lower())
        else base_spec
    )
    if prompt_issue_spec.issues:
        data["issues"] = [issue.model_dump(mode="json") for issue in prompt_issue_spec.issues]
    else:
        data["issues"] = [
            issue.model_dump(mode="json")
            for issue in _issues_from_intent(intent.get("issues"), base_spec)
        ]
    spec = ScenarioSpec.model_validate(data)
    semantic_gaps = _custom_semantic_gaps(spec, prompt)
    if semantic_gaps:
        raise ValueError(f"custom schema missed requested concepts: {', '.join(semantic_gaps)}")
    spec = _ensure_timeline_supports_issue_batches(spec)
    return spec, [
        "Agent created a custom schema because the request was not limited to a known blueprint.",
        (
            "Custom tables, relationships, row counts, and issue rules were "
            "validated deterministically."
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
        row_count=max(1, int(raw_table.get("row_count") or 1000)),
        columns=columns,
        source_systems=list(raw_table.get("source_systems") or []),
    )


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


def _ensure_requested_relationship_columns(spec: ScenarioSpec, text: str) -> ScenarioSpec:
    primary = _primary_requested_table(spec, text) or _central_table(spec)
    data = spec.model_dump(mode="json")
    tables = {table["name"]: table for table in data["tables"]}
    if primary.name not in tables:
        return spec
    relationships = data["relationships"]

    def ensure_fk(parent_table: str, parent_column: str, child_column: str) -> None:
        if parent_table not in tables:
            return
        child_columns = {column["name"] for column in tables[primary.name]["columns"]}
        if child_column not in child_columns:
            tables[primary.name]["columns"].append(
                {
                    "name": child_column,
                    "type": ColumnType.LONG.value,
                    "nullable": True,
                    "primary_key": False,
                }
            )
        rel_name = f"{parent_table}_{primary.name}"
        if not any(
            rel["parent_table"] == parent_table
            and rel["child_table"] == primary.name
            and rel["child_column"] == child_column
            for rel in relationships
        ):
            relationships.append(
                {
                    "name": rel_name,
                    "parent_table": parent_table,
                    "parent_column": parent_column,
                    "child_table": primary.name,
                    "child_column": child_column,
                    "required": True,
                }
            )

    if "product" in _keyword_clause(text, "orphan") or "product" in _keyword_clause(
        text, "missing"
    ):
        ensure_fk("products", "product_id", "product_id")
    if "store" in _keyword_clause(text, "orphan") or "store" in _keyword_clause(
        text, "missing"
    ):
        ensure_fk("stores", "store_id", "store_id")
    return ScenarioSpec.model_validate(data)


def _ensure_requested_operational_columns(spec: ScenarioSpec, text: str) -> ScenarioSpec:
    if "late" not in text:
        return spec
    data = spec.model_dump(mode="json")
    tables = {table["name"]: table for table in data["tables"]}
    late_table = _table_for_keyword(spec, text, "late")
    if late_table is None and "closure" in _keyword_window(text, "late"):
        late_table = spec.table("incident_closures") if "incident_closures" in tables else None
    if late_table is None:
        return spec
    columns = tables[late_table.name]["columns"]
    if not any(column["name"] == "ingestion_ts" for column in columns):
        columns.append(
            {
                "name": "ingestion_ts",
                "type": ColumnType.TIMESTAMP.value,
                "nullable": True,
                "primary_key": False,
            }
        )
    return ScenarioSpec.model_validate(data)


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
        "store": ("store",),
        "product": ("product",),
        "customer": ("customer",),
        "postal code": ("postal code", "postal"),
    }
    gaps: list[str] = []
    mentioned_concepts = 0
    for concept, markers in concept_rules.items():
        if any(marker in text for marker in markers):
            mentioned_concepts += 1
            has_concept = any(
                marker.replace(" ", "_") in searchable or marker in searchable
                for marker in markers
            )
            if not has_concept:
                gaps.append(concept)
    if mentioned_concepts >= 3 and len(spec.tables) < 4:
        gaps.append("multi-entity relational design")
    if "event" in text and not any("event" in table.name for table in spec.tables):
        gaps.append("event fact table")
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


def _blueprint_domain_from_prompt(text: str) -> str | None:
    if "order" in text and any(word in text for word in ("retail", "ecommerce", "e-commerce")):
        return "retail_orders"
    if "insurance" in text:
        return "insurance_claims"
    return None


def _heuristic_custom_spec_from_prompt(prompt: str) -> tuple[ScenarioSpec, list[str]]:
    text = prompt.strip()
    normalized = text.lower()
    seed = _extract_seed(normalized)
    table_counts = _extract_table_count_mentions(normalized)
    if _looks_like_ai_model_ops(normalized):
        spec = _ai_model_ops_custom_spec(text, table_counts, seed)
    elif any(word in normalized for word in ("healthcare", "health care", "medical", "patient")):
        spec = _healthcare_custom_spec(text, table_counts, seed)
    elif any(word in normalized for word in ("promotion", "coupon", "redemption")):
        spec = _retail_promotions_custom_spec(text, table_counts, seed)
    elif any(word in normalized for word in ("telecom", "network", "cell tower", "tower")):
        spec = _telecom_custom_spec(text, table_counts, seed)
    elif any(word in normalized for word in ("bank", "banking", "transaction", "merchant")):
        spec = _banking_custom_spec(text, table_counts, seed)
    else:
        spec = _generic_custom_spec(text, table_counts, seed)
    spec = _ensure_databricks_outputs(spec)
    spec = _apply_custom_prompt_issues(spec, normalized)
    return spec, [
        "Model planner was unavailable or invalid; used deterministic custom-schema fallback.",
        "The fallback stayed in custom_schema instead of using an unrelated blueprint.",
        *_custom_issue_assumptions(normalized, spec),
        "Generation still requires hash confirmation before the Databricks job is submitted.",
    ]


def _looks_like_ai_model_ops(text: str) -> bool:
    has_ai_domain = any(
        marker in text
        for marker in (
            "ai model",
            "model operations",
            "model ops",
            "model training",
            "model inference",
            "inference",
            "prompt",
            "response",
            "evaluation",
        )
    )
    has_platform_shape = any(
        marker in text
        for marker in (
            "model_registry",
            "prompt_requests",
            "model_inferences",
            "feedback_scores",
            "evaluation_results",
            "tenant_metadata",
            "production ai platform",
        )
    )
    return has_ai_domain and has_platform_shape


def _extract_table_count_mentions(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    number_pattern = r"\d[\d,]*(?:\.\d+)?\s*[km]?"
    count_before_table = (
        rf"\b({number_pattern})\s+([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)?)\b"
    )
    for match in re.finditer(count_before_table, text):
        table = _table_name_from_phrase(match.group(2))
        if table in {
            "bad_data",
            "dirty_data",
            "synthetic_data",
            "healthcare",
            "insurance",
            "retail",
        }:
            continue
        counts[table] = _parse_number(match.group(1))
    for match in re.finditer(
        rf"\b([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)?)\s*(?:=|:|to|of)\s*({number_pattern})",
        text,
    ):
        table = _table_name_from_phrase(match.group(1))
        counts[table] = _parse_number(match.group(2))
    return counts


def _table_name_from_phrase(phrase: str) -> str:
    phrase = phrase.replace("-", " ").strip()
    words = [word for word in phrase.split() if word not in {"with", "and", "of"}]
    if len(words) >= 2 and words[-2:] == ["order", "lines"]:
        return "order_lines"
    return _identifier(words[-1] if words else phrase, "records")


def _healthcare_custom_spec(
    prompt: str, table_counts: dict[str, int], seed: int
) -> ScenarioSpec:
    claims = table_counts.get("claims", 100_000)
    tables = [
        _table(
            "patients",
            table_counts.get("patients", max(1_000, claims // 2)),
            [
                _pk("patient_id"),
                _col("province", values=["ON", "QC", "BC", "AB"]),
                _col("birth_date", ColumnType.DATE),
                _col("member_status", values=["active", "inactive", "pending"]),
            ],
        ),
        _table(
            "providers",
            table_counts.get("providers", max(100, claims // 20)),
            [
                _pk("provider_id"),
                _col("provider_type", values=["hospital", "clinic", "physician", "pharmacy"]),
                _col("province", values=["ON", "QC", "BC", "AB"]),
            ],
        ),
        _table(
            "claims",
            claims,
            [
                _pk("claim_id"),
                _col("patient_id", ColumnType.LONG, nullable=False),
                _col("provider_id", ColumnType.LONG, nullable=False),
                _col("service_date", ColumnType.DATE),
                _col("claim_amount", ColumnType.DECIMAL),
                _col("claim_status", values=["submitted", "approved", "denied", "pending"]),
                _col("ingestion_ts", ColumnType.TIMESTAMP),
            ],
        ),
        _table(
            "adjudications",
            table_counts.get("adjudications", claims),
            [
                _pk("adjudication_id"),
                _col("claim_id", ColumnType.LONG, nullable=False),
                _col("adjudication_ts", ColumnType.TIMESTAMP),
                _col("decision", values=["paid", "denied", "review"]),
            ],
        ),
        _table(
            "payments",
            table_counts.get("payments", max(1, round(claims * 0.6))),
            [
                _pk("payment_id"),
                _col("claim_id", ColumnType.LONG, nullable=False),
                _col("payment_amount", ColumnType.DECIMAL),
                _col("payment_ts", ColumnType.TIMESTAMP),
                _col("ingestion_ts", ColumnType.TIMESTAMP),
            ],
        ),
    ]
    relationships = [
        _relationship("patients_claims", "patients", "patient_id", "claims", "patient_id"),
        _relationship("providers_claims", "providers", "provider_id", "claims", "provider_id"),
        _relationship("claims_adjudications", "claims", "claim_id", "adjudications", "claim_id"),
        _relationship("claims_payments", "claims", "claim_id", "payments", "claim_id"),
    ]
    return ScenarioSpec(
        name=_extract_custom_name(prompt, "Healthcare Claims Scenario"),
        domain="custom_schema",
        seed=seed,
        locale="en_CA",
        timeline=TimelineSpec(start_date=date(2026, 1, 1), batches=30),
        tables=tables,
        relationships=relationships,
    )


def _banking_custom_spec(prompt: str, table_counts: dict[str, int], seed: int) -> ScenarioSpec:
    transactions = table_counts.get("transactions", table_counts.get("payments", 100_000))
    tables = [
        _table(
            "accounts",
            table_counts.get("accounts", max(1_000, transactions // 4)),
            [
                _pk("account_id"),
                _col("account_type", values=["checking", "savings", "credit", "loan"]),
                _col("province", values=["ON", "QC", "BC", "AB"]),
            ],
        ),
        _table(
            "merchants",
            table_counts.get("merchants", max(100, transactions // 50)),
            [
                _pk("merchant_id"),
                _col("merchant_category", values=["grocery", "fuel", "travel", "retail"]),
            ],
        ),
        _table(
            "transactions",
            transactions,
            [
                _pk("transaction_id"),
                _col("account_id", ColumnType.LONG, nullable=False),
                _col("merchant_id", ColumnType.LONG, nullable=False),
                _col("transaction_amount", ColumnType.DECIMAL),
                _col("transaction_ts", ColumnType.TIMESTAMP),
                _col("ingestion_ts", ColumnType.TIMESTAMP),
            ],
        ),
        _table(
            "settlements",
            table_counts.get("settlements", max(1, round(transactions * 0.8))),
            [
                _pk("settlement_id"),
                _col("transaction_id", ColumnType.LONG, nullable=False),
                _col("settlement_ts", ColumnType.TIMESTAMP),
                _col("settlement_status", values=["settled", "reversed", "pending"]),
            ],
        ),
    ]
    relationships = [
        _relationship(
            "accounts_transactions", "accounts", "account_id", "transactions", "account_id"
        ),
        _relationship(
            "merchants_transactions", "merchants", "merchant_id", "transactions", "merchant_id"
        ),
        _relationship(
            "transactions_settlements",
            "transactions",
            "transaction_id",
            "settlements",
            "transaction_id",
        ),
    ]
    return ScenarioSpec(
        name=_extract_custom_name(prompt, "Banking Transactions Scenario"),
        domain="custom_schema",
        seed=seed,
        locale="en_CA",
        timeline=TimelineSpec(start_date=date(2026, 1, 1), batches=30),
        tables=tables,
        relationships=relationships,
    )


def _retail_promotions_custom_spec(
    prompt: str, table_counts: dict[str, int], seed: int
) -> ScenarioSpec:
    promotions = table_counts.get("promotions", 100_000)
    coupons = table_counts.get("coupons", max(promotions, round(promotions * 2)))
    redemptions = table_counts.get("coupon_redemptions", round(promotions * 1.5))
    tables = [
        _table(
            "products",
            table_counts.get("products", max(1_000, promotions // 5)),
            [
                _pk("product_id"),
                _col("product_name"),
                _col("category", values=["grocery", "apparel", "electronics", "home"]),
            ],
        ),
        _table(
            "stores",
            table_counts.get("stores", 500),
            [
                _pk("store_id"),
                _col("store_name"),
                _col("region", values=["east", "central", "west", "north"]),
            ],
        ),
        _table(
            "customers",
            table_counts.get("customers", max(redemptions, promotions * 2)),
            [
                _pk("customer_id"),
                _col("first_name"),
                _col("last_name"),
                _col("postal_code"),
            ],
        ),
        _table(
            "promotions",
            promotions,
            [
                _pk("promotion_id"),
                _col("promotion_name"),
                _col("product_id", ColumnType.LONG, nullable=False),
                _col("store_id", ColumnType.LONG, nullable=False),
                _col("start_date", ColumnType.DATE),
                _col("end_date", ColumnType.DATE),
            ],
        ),
        _table(
            "coupons",
            coupons,
            [
                _pk("coupon_id"),
                _col("promotion_id", ColumnType.LONG, nullable=False),
                _col("product_id", ColumnType.LONG, nullable=False),
                _col("store_id", ColumnType.LONG, nullable=False),
                _col("issue_date", ColumnType.DATE),
            ],
        ),
        _table(
            "coupon_redemptions",
            redemptions,
            [
                _pk("redemption_id"),
                _col("coupon_id", ColumnType.LONG, nullable=False),
                _col("customer_id", ColumnType.LONG, nullable=False),
                _col("redemption_ts", ColumnType.TIMESTAMP),
                _col("ingestion_ts", ColumnType.TIMESTAMP),
            ],
        ),
    ]
    relationships = [
        _relationship("products_promotions", "products", "product_id", "promotions", "product_id"),
        _relationship("stores_promotions", "stores", "store_id", "promotions", "store_id"),
        _relationship(
            "promotions_coupons", "promotions", "promotion_id", "coupons", "promotion_id"
        ),
        _relationship("products_coupons", "products", "product_id", "coupons", "product_id"),
        _relationship("stores_coupons", "stores", "store_id", "coupons", "store_id"),
        _relationship(
            "coupons_redemptions",
            "coupons",
            "coupon_id",
            "coupon_redemptions",
            "coupon_id",
        ),
        _relationship(
            "customers_redemptions",
            "customers",
            "customer_id",
            "coupon_redemptions",
            "customer_id",
        ),
    ]
    return ScenarioSpec(
        name=_extract_custom_name(prompt, "Retail Promotions Scenario"),
        domain="custom_schema",
        seed=seed,
        locale="en_CA",
        timeline=TimelineSpec(start_date=date(2026, 1, 1), batches=30),
        tables=tables,
        relationships=relationships,
    )


def _telecom_custom_spec(prompt: str, table_counts: dict[str, int], seed: int) -> ScenarioSpec:
    events = table_counts.get("events", table_counts.get("network_events", 100_000))
    incidents = table_counts.get("incidents", max(1, round(events * 0.08)))
    tables = [
        _table(
            "customer_regions",
            table_counts.get("customer_regions", 120),
            [
                _pk("region_id"),
                _col("region_code", values=["ON-GTA", "QC-MTL", "BC-LMV", "AB-CGY"]),
                _col("province", values=["ON", "QC", "BC", "AB"]),
            ],
        ),
        _table(
            "cell_towers",
            table_counts.get("cell_towers", max(500, events // 50)),
            [
                _pk("tower_id"),
                _col("region_id", ColumnType.LONG, nullable=False),
                _col("tower_type", values=["macro", "micro", "small_cell", "distributed"]),
                _col("commissioned_date", ColumnType.DATE),
            ],
        ),
        _table(
            "network_engineers",
            table_counts.get("network_engineers", max(100, events // 1_000)),
            [
                _pk("engineer_id"),
                _col("region_id", ColumnType.LONG, nullable=False),
                _col("skill_area", values=["radio", "transport", "core", "field_ops"]),
                _col("active_flag", ColumnType.BOOLEAN),
            ],
        ),
        _table(
            "network_events",
            events,
            [
                _pk("event_id"),
                _col("tower_id", ColumnType.LONG, nullable=False),
                _col("region_id", ColumnType.LONG, nullable=False),
                _col("event_ts", ColumnType.TIMESTAMP),
                _col("ingestion_ts", ColumnType.TIMESTAMP),
                _col("event_type", values=["alarm", "handoff_failure", "packet_loss", "outage"]),
                _col("severity", values=["low", "medium", "high", "critical"]),
            ],
        ),
        _table(
            "incidents",
            incidents,
            [
                _pk("incident_id"),
                _col("event_id", ColumnType.LONG, nullable=False),
                _col("assigned_engineer_id", ColumnType.LONG, nullable=True),
                _col("opened_ts", ColumnType.TIMESTAMP),
                _col("incident_status", values=["open", "investigating", "resolved"]),
            ],
        ),
        _table(
            "incident_closures",
            table_counts.get("incident_closures", incidents),
            [
                _pk("closure_id"),
                _col("incident_id", ColumnType.LONG, nullable=False),
                _col("engineer_id", ColumnType.LONG, nullable=True),
                _col("closure_ts", ColumnType.TIMESTAMP),
                _col("ingestion_ts", ColumnType.TIMESTAMP),
                _col("closure_code", values=["fixed", "duplicate", "monitoring", "no_fault_found"]),
            ],
        ),
    ]
    relationships = [
        _relationship(
            "regions_cell_towers", "customer_regions", "region_id", "cell_towers", "region_id"
        ),
        _relationship(
            "regions_engineers",
            "customer_regions",
            "region_id",
            "network_engineers",
            "region_id",
        ),
        _relationship("towers_events", "cell_towers", "tower_id", "network_events", "tower_id"),
        _relationship(
            "regions_events", "customer_regions", "region_id", "network_events", "region_id"
        ),
        _relationship("events_incidents", "network_events", "event_id", "incidents", "event_id"),
        _relationship(
            "engineers_incidents",
            "network_engineers",
            "engineer_id",
            "incidents",
            "assigned_engineer_id",
        ),
        _relationship(
            "incidents_closures",
            "incidents",
            "incident_id",
            "incident_closures",
            "incident_id",
        ),
        _relationship(
            "engineers_closures",
            "network_engineers",
            "engineer_id",
            "incident_closures",
            "engineer_id",
        ),
    ]
    return ScenarioSpec(
        name=_extract_custom_name(prompt, "Telecom Network Events Scenario"),
        domain="custom_schema",
        seed=seed,
        locale="en_CA",
        timeline=TimelineSpec(start_date=date(2026, 1, 1), batches=30),
        tables=tables,
        relationships=relationships,
    )


def _ai_model_ops_custom_spec(
    prompt: str, table_counts: dict[str, int], seed: int
) -> ScenarioSpec:
    inferences = _central_ai_ops_count(prompt, table_counts)
    prompt_requests = table_counts.get("prompt_requests", inferences)
    feedback_scores = table_counts.get("feedback_scores", max(1, round(inferences * 0.60)))
    evaluation_results = table_counts.get(
        "evaluation_results", max(1, round(inferences * 0.50))
    )
    user_events = table_counts.get("user_events", max(inferences, round(inferences * 1.20)))
    incident_logs = table_counts.get("incident_logs", max(500, round(inferences * 0.01)))
    tenants = table_counts.get("tenant_metadata", 250)
    users = table_counts.get(
        "user_directory", table_counts.get("users", max(1_000, inferences // 4))
    )
    models = table_counts.get("model_registry", 240)
    tables = [
        _table(
            "tenant_metadata",
            tenants,
            [
                _pk("tenant_id"),
                _col("tenant_name", nullable=False),
                _col("region", values=["CA-ON", "CA-QC", "US-NY", "US-CA", "EU-DE"]),
                _col("industry", values=["financial_services", "retail", "healthcare", "telecom"]),
                _col("usage_tier", values=["enterprise", "growth", "sandbox"]),
                _col("created_at", ColumnType.TIMESTAMP),
                _col("updated_at", ColumnType.TIMESTAMP),
                _col("source_system", values=["tenant_admin", "crm_sync", "billing_platform"]),
                _col("record_status", values=["active", "suspended", "deleted"]),
            ],
        ),
        _table(
            "user_directory",
            users,
            [
                _pk("user_id"),
                _col("tenant_id", ColumnType.LONG, nullable=False),
                _col("user_role", values=["admin", "developer", "analyst", "reviewer"]),
                _col("region", values=["CA-ON", "CA-QC", "US-NY", "US-CA", "EU-DE"]),
                _col("created_at", ColumnType.TIMESTAMP),
                _col("updated_at", ColumnType.TIMESTAMP),
                _col("source_system", values=["identity_provider", "tenant_admin"]),
                _col("record_status", values=["active", "inactive", "deleted"]),
            ],
        ),
        _table(
            "model_registry",
            models,
            [
                _pk("model_id"),
                _col("tenant_id", ColumnType.LONG, nullable=False),
                _col("model_family", values=["llm", "embedding", "classifier", "reranker"]),
                _col("model_version", values=["v1.0.0", "v1.1.0", "v2.0.0", "v2.1.3"]),
                _col("deployment_stage", values=["training", "staging", "production", "retired"]),
                _col("created_at", ColumnType.TIMESTAMP),
                _col("updated_at", ColumnType.TIMESTAMP),
                _col("source_system", values=["mlflow_registry", "ci_cd", "model_ops"]),
                _col("record_status", values=["active", "archived", "deprecated"]),
            ],
        ),
        _table(
            "user_events",
            user_events,
            [
                _pk("event_id"),
                _col("user_id", ColumnType.LONG, nullable=False),
                _col("tenant_id", ColumnType.LONG, nullable=False),
                _col(
                    "event_type",
                    values=["login", "prompt_submit", "feedback_submit", "eval_view"],
                ),
                _col("region", values=["CA-ON", "CA-QC", "US-NY", "US-CA", "EU-DE"]),
                _col("created_at", ColumnType.TIMESTAMP),
                _col("updated_at", ColumnType.TIMESTAMP),
                _col("ingestion_batch_id", ColumnType.INTEGER),
                _col("source_system", values=["web_app", "api_gateway", "batch_import"]),
                _col("record_status", values=["completed", "failed", "retrying"]),
            ],
        ),
        _table(
            "prompt_requests",
            prompt_requests,
            [
                _pk("prompt_request_id"),
                _col("event_id", ColumnType.LONG, nullable=False),
                _col("user_id", ColumnType.LONG, nullable=False),
                _col("tenant_id", ColumnType.LONG, nullable=False),
                _col("model_id", ColumnType.LONG, nullable=False),
                _col("model_version", values=["v1.0.0", "v1.1.0", "v2.0.0", "v2.1.3"]),
                _col("prompt_text", nullable=False),
                _col("prompt_category", values=["support", "summarization", "coding", "analytics"]),
                _col("region", values=["CA-ON", "CA-QC", "US-NY", "US-CA", "EU-DE"]),
                _col("created_at", ColumnType.TIMESTAMP),
                _col("updated_at", ColumnType.TIMESTAMP),
                _col("ingestion_batch_id", ColumnType.INTEGER),
                _col("source_system", values=["chat_ui", "api_gateway", "workflow_job"]),
                _col("record_status", values=["completed", "failed", "cancelled"]),
            ],
        ),
        _table(
            "model_inferences",
            inferences,
            [
                _pk("inference_id"),
                _col("event_id", ColumnType.LONG, nullable=False),
                _col("prompt_request_id", ColumnType.LONG, nullable=False),
                _col("user_id", ColumnType.LONG, nullable=False),
                _col("tenant_id", ColumnType.LONG, nullable=False),
                _col("model_id", ColumnType.LONG, nullable=False),
                _col("model_version", values=["v1.0.0", "v1.1.0", "v2.0.0", "v2.1.3"]),
                _col("prompt_text", nullable=False),
                _col("prompt_category", values=["support", "summarization", "coding", "analytics"]),
                _col("response_text"),
                _col("response_latency_ms", ColumnType.DECIMAL),
                _col("confidence_score", ColumnType.DECIMAL),
                _col("region", values=["CA-ON", "CA-QC", "US-NY", "US-CA", "EU-DE"]),
                _col("created_at", ColumnType.TIMESTAMP),
                _col("updated_at", ColumnType.TIMESTAMP),
                _col("ingestion_batch_id", ColumnType.INTEGER),
                _col(
                    "source_system",
                    values=["serving_endpoint", "batch_inference", "agent_runtime"],
                ),
                _col("record_status", values=["completed", "failed", "timeout"]),
            ],
        ),
        _table(
            "feedback_scores",
            feedback_scores,
            [
                _pk("feedback_id"),
                _col("event_id", ColumnType.LONG, nullable=False),
                _col("inference_id", ColumnType.LONG, nullable=False),
                _col("user_id", ColumnType.LONG, nullable=False),
                _col("tenant_id", ColumnType.LONG, nullable=False),
                _col("feedback_label", values=["thumbs_up", "thumbs_down", "accepted", "rejected"]),
                _col("confidence_score", ColumnType.DECIMAL),
                _col("inference_created_at", ColumnType.TIMESTAMP),
                _col("created_at", ColumnType.TIMESTAMP),
                _col("ingestion_ts", ColumnType.TIMESTAMP),
                _col("updated_at", ColumnType.TIMESTAMP),
                _col("ingestion_batch_id", ColumnType.INTEGER),
                _col("source_system", values=["feedback_widget", "review_queue", "api_gateway"]),
                _col("record_status", values=["completed", "pending_review", "retracted"]),
            ],
        ),
        _table(
            "evaluation_results",
            evaluation_results,
            [
                _pk("evaluation_id"),
                _col("inference_id", ColumnType.LONG, nullable=False),
                _col("model_id", ColumnType.LONG, nullable=False),
                _col("tenant_id", ColumnType.LONG, nullable=False),
                _col("model_version", values=["v1.0.0", "v1.1.0", "v2.0.0", "v2.1.3"]),
                _col("evaluation_label", values=["pass", "fail", "needs_review", "unsafe"]),
                _col("feedback_label", values=["helpful", "not_helpful", "irrelevant"]),
                _col("evaluation_score", ColumnType.DECIMAL),
                _col("region", values=["CA-ON", "CA-QC", "US-NY", "US-CA", "EU-DE"]),
                _col("created_at", ColumnType.TIMESTAMP),
                _col("updated_at", ColumnType.TIMESTAMP),
                _col("ingestion_batch_id", ColumnType.INTEGER),
                _col("source_system", values=["eval_harness", "human_review", "monitoring_job"]),
                _col("record_status", values=["completed", "failed", "queued"]),
            ],
        ),
        _table(
            "incident_logs",
            incident_logs,
            [
                _pk("incident_id"),
                _col("event_id", ColumnType.LONG, nullable=False),
                _col("inference_id", ColumnType.LONG, nullable=False),
                _col("tenant_id", ColumnType.LONG, nullable=False),
                _col("model_id", ColumnType.LONG, nullable=False),
                _col("incident_type", values=["latency_spike", "safety_filter", "model_error"]),
                _col("severity", values=["low", "medium", "high", "critical"]),
                _col("region", values=["CA-ON", "CA-QC", "US-NY", "US-CA", "EU-DE"]),
                _col("created_at", ColumnType.TIMESTAMP),
                _col("updated_at", ColumnType.TIMESTAMP),
                _col("ingestion_batch_id", ColumnType.INTEGER),
                _col("source_system", values=["monitoring_job", "pager", "serving_endpoint"]),
                _col("record_status", values=["open", "mitigated", "resolved"]),
            ],
        ),
    ]
    relationships = [
        _relationship(
            "tenants_users", "tenant_metadata", "tenant_id", "user_directory", "tenant_id"
        ),
        _relationship(
            "tenants_models", "tenant_metadata", "tenant_id", "model_registry", "tenant_id"
        ),
        _relationship("users_events", "user_directory", "user_id", "user_events", "user_id"),
        _relationship("tenants_events", "tenant_metadata", "tenant_id", "user_events", "tenant_id"),
        _relationship("events_prompts", "user_events", "event_id", "prompt_requests", "event_id"),
        _relationship("users_prompts", "user_directory", "user_id", "prompt_requests", "user_id"),
        _relationship(
            "models_prompts", "model_registry", "model_id", "prompt_requests", "model_id"
        ),
        _relationship(
            "prompts_inferences",
            "prompt_requests",
            "prompt_request_id",
            "model_inferences",
            "prompt_request_id",
        ),
        _relationship(
            "users_inferences", "user_directory", "user_id", "model_inferences", "user_id"
        ),
        _relationship(
            "models_inferences",
            "model_registry",
            "model_id",
            "model_inferences",
            "model_id",
        ),
        _relationship(
            "inferences_feedback",
            "model_inferences",
            "inference_id",
            "feedback_scores",
            "inference_id",
        ),
        _relationship("users_feedback", "user_directory", "user_id", "feedback_scores", "user_id"),
        _relationship(
            "inferences_evaluations",
            "model_inferences",
            "inference_id",
            "evaluation_results",
            "inference_id",
        ),
        _relationship(
            "models_evaluations",
            "model_registry",
            "model_id",
            "evaluation_results",
            "model_id",
        ),
        _relationship(
            "inferences_incidents",
            "model_inferences",
            "inference_id",
            "incident_logs",
            "inference_id",
        ),
    ]
    return ScenarioSpec(
        name=_extract_custom_name(prompt, "AI Model Operations Evaluation Scenario"),
        domain="custom_schema",
        seed=seed,
        locale="en_CA",
        timeline=TimelineSpec(start_date=date(2026, 1, 1), batches=30),
        tables=tables,
        relationships=relationships,
    )


def _central_ai_ops_count(prompt: str, table_counts: dict[str, int]) -> int:
    for table_name in (
        "model_inferences",
        "inferences",
        "inference_records",
        "records",
        "representing",
    ):
        if table_name in table_counts:
            return table_counts[table_name]
    match = re.search(
        r"\b(\d[\d,]*(?:\.\d+)?\s*[km]?)\s+records?\s+representing\b",
        prompt.lower(),
    )
    if match:
        return _parse_number(match.group(1))
    return 100_000


def _generic_custom_spec(prompt: str, table_counts: dict[str, int], seed: int) -> ScenarioSpec:
    central_table = next(iter(table_counts), _identifier(_domain_words(prompt), "records"))
    central_count = table_counts.get(central_table, 10_000)
    entity_table = _identifier(f"{_singular(central_table)}_sources", "sources")
    entity_id = _default_pk_name(entity_table)
    central_id = _default_pk_name(central_table)
    tables = [
        _table(
            entity_table,
            max(100, central_count // 20),
            [
                _pk(entity_id),
                _col("source_name"),
                _col("province", values=["ON", "QC", "BC", "AB"]),
            ],
        ),
        _table(
            central_table,
            central_count,
            [
                _pk(central_id),
                _col(entity_id, ColumnType.LONG, nullable=False),
                _col("event_ts", ColumnType.TIMESTAMP),
                _col("ingestion_ts", ColumnType.TIMESTAMP),
                _col("amount", ColumnType.DECIMAL),
                _col("status", values=["new", "active", "closed", "exception"]),
            ],
        ),
    ]
    return ScenarioSpec(
        name=_extract_custom_name(prompt, "Custom Synthetic Data Scenario"),
        domain="custom_schema",
        seed=seed,
        locale="en_CA",
        timeline=TimelineSpec(start_date=date(2026, 1, 1), batches=30),
        tables=tables,
        relationships=[
            _relationship(
                f"{entity_table}_{central_table}",
                entity_table,
                entity_id,
                central_table,
                entity_id,
            )
        ],
    )


def _apply_custom_prompt_issues(spec: ScenarioSpec, text: str) -> ScenarioSpec:
    if _is_ai_model_ops_spec(spec):
        return _apply_ai_model_ops_prompt_issues(spec, text)

    tables = {table.name: table for table in spec.tables}
    central = _central_table(spec)
    primary = _primary_requested_table(spec, text) or central
    raw_issues: list[dict[str, Any]] = []

    def add(
        issue_type: IssueType,
        table: str,
        column: str | None = None,
        *,
        keyword: str,
        parameters: dict[str, Any] | None = None,
        default_rate: float = 0.01,
    ) -> None:
        raw_issues.append(
            {
                "type": issue_type.value,
                "table": table,
                "column": column,
                "parameters": parameters or {},
                **_count_or_rate(text, keyword, default_rate=default_rate),
            }
        )

    if "duplicate" in text:
        duplicate_table = _table_for_keyword(spec, text, "duplicate") or primary
        add(
            IssueType.DUPLICATE_RECORD,
            duplicate_table.name,
            keyword="duplicate",
            parameters={
                "duplicate_semantics": "exact_record_duplicate",
                "meaning": "same primary key and same business values are emitted again",
                "duplicate_source": "duplicate_record",
                "overlap_policy": (
                    "file_replay may introduce additional duplicates; those remain tracked "
                    "under file_replay, not duplicate_record"
                ),
            },
        )
    if any(word in text for word in ("orphan", "non-existent", "nonexistent", "bad foreign")):
        orphan_column = _requested_id_column(text, primary, "orphan")
        orphan_table = primary
        if not orphan_column:
            orphan_table, orphan_column = _relationship_target_for_requested_id(
                spec, text, "orphan", primary
            )
        orphan_column = orphan_column or _first_foreign_key(spec, orphan_table)
        if orphan_column:
            add(IssueType.REFERENTIAL_ORPHAN, orphan_table.name, orphan_column, keyword="orphan")
    if any(word in text for word in ("missing", "null", "blank")):
        missing_table = primary
        missing_column = _requested_id_column(text, primary, "missing")
        if not missing_column and "engineer" in _keyword_window(text, "missing"):
            if (
                "incidents" in tables
                and "assigned_engineer_id" in tables["incidents"].column_names()
            ):
                missing_table = tables["incidents"]
                missing_column = "assigned_engineer_id"
            elif (
                "incident_closures" in tables
                and "engineer_id" in tables["incident_closures"].column_names()
            ):
                missing_table = tables["incident_closures"]
                missing_column = "engineer_id"
        if not missing_column:
            missing_table, missing_column = _relationship_target_for_requested_id(
                spec, text, "missing", primary
            )
        missing_column = missing_column or _first_nullable_non_pk(missing_table)
        if missing_column:
            add(IssueType.NULL_VALUE, missing_table.name, missing_column, keyword="missing")
    if "invalid" in text or "bad value" in text:
        invalid_table, invalid_column = _invalid_value_target(spec, text)
        if invalid_column:
            add(
                IssueType.INVALID_VALUE,
                invalid_table,
                invalid_column,
                keyword="invalid",
                parameters=_invalid_value_parameters(invalid_column),
            )
            invalid_issue = raw_issues[-1]
            if _should_use_exact_invalid_count(spec.table(invalid_table), invalid_issue):
                invalid_issue["exact_count"] = 1
                invalid_issue["rate"] = None
    if "late" in text:
        explicit_late_table = _table_for_keyword(spec, text, "late")
        if explicit_late_table:
            late_table = explicit_late_table
        elif "closure" in _keyword_window(text, "late") and "incident_closures" in tables:
            late_table = tables["incident_closures"]
        else:
            late_table = tables.get("adjudications") or tables.get("settlements") or primary
        late_column = _timestamp_column_for_keyword(late_table, "redemption", text) or (
            _timestamp_column_for_keyword(late_table, "closure", text)
            or _first_timestamp_column(late_table)
            or "ingestion_ts"
        )
        add(
            IssueType.LATE_ARRIVAL,
            late_table.name,
            late_column,
            keyword="late",
            parameters=_late_arrival_parameters(late_table, late_column, text),
        )
    if "schema drift" in text or "new column" in text:
        schema_table = _table_for_keyword(spec, text, "schema drift") or primary
        add(
            IssueType.SCHEMA_DRIFT,
            schema_table.name,
            keyword="schema drift",
            parameters=_schema_drift_parameters(schema_table),
            default_rate=1.0,
        )
        raw_issues[-1]["exact_count"] = 1
        raw_issues[-1]["rate"] = None
    if "replay" in text:
        replay_table = _table_for_keyword(spec, text, "replay") or primary
        add(
            IssueType.FILE_REPLAY,
            replay_table.name,
            keyword="replay",
            parameters=_file_replay_parameters(),
        )
        raw_issues[-1]["rate"] = None

    data = spec.model_dump(mode="json")
    data["issues"] = [
        issue.model_dump(mode="json") for issue in _issues_from_intent(raw_issues, spec)
    ]
    return _ensure_timeline_supports_issue_batches(ScenarioSpec.model_validate(data))


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


def _apply_ai_model_ops_prompt_issues(spec: ScenarioSpec, text: str) -> ScenarioSpec:
    raw_issues: list[dict[str, Any]] = []

    def add(
        issue_type: IssueType,
        table: str,
        column: str | None,
        *,
        issue_id: str,
        quantity_keyword: str,
        default_rate: float = 0.01,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        raw_issues.append(
            {
                "issue_id": issue_id,
                "type": issue_type.value,
                "table": table,
                "column": column,
                "parameters": parameters or {},
                **_count_or_rate(text, quantity_keyword, default_rate=default_rate),
            }
        )

    add(
        IssueType.REFERENTIAL_ORPHAN,
        "model_inferences",
        "model_id",
        issue_id="iss_ai_orphan_model_ids",
        quantity_keyword="model id",
        default_rate=0.02,
        parameters={"parent_table": "model_registry", "parent_column": "model_id"},
    )
    add(
        IssueType.REFERENTIAL_ORPHAN,
        "model_inferences",
        "user_id",
        issue_id="iss_ai_orphan_user_ids",
        quantity_keyword="user id",
        default_rate=0.02,
        parameters={"parent_table": "user_directory", "parent_column": "user_id"},
    )
    add(
        IssueType.NULL_VALUE,
        "prompt_requests",
        "prompt_category",
        issue_id="iss_ai_missing_prompt_categories",
        quantity_keyword="prompt categor",
        default_rate=0.05,
    )
    add(
        IssueType.NULL_VALUE,
        "evaluation_results",
        "evaluation_label",
        issue_id="iss_ai_missing_evaluation_labels",
        quantity_keyword="evaluation label",
        default_rate=0.05,
    )
    add(
        IssueType.NULL_VALUE,
        "model_inferences",
        "response_latency_ms",
        issue_id="iss_ai_missing_latency",
        quantity_keyword="latency",
        default_rate=0.05,
    )
    add(
        IssueType.LATE_ARRIVAL,
        "feedback_scores",
        "created_at",
        issue_id="iss_ai_late_feedback",
        quantity_keyword="late-arriving feedback",
        default_rate=0.10,
        parameters={
            "semantics": "late_arriving_data",
            "event_time_column": "created_at",
            "arrival_column": "ingestion_ts",
            "delay_days_min": 1,
            "delay_days_max": 5,
            "meaning": (
                "feedback was created at event time but ingestion_ts arrives in a later batch"
            ),
        },
    )
    replay_quantity = _count_or_rate(text, "duplicated inference", default_rate=0.10)
    raw_issues.append(
        {
            "issue_id": "iss_ai_replayed_inference_files",
            "type": IssueType.FILE_REPLAY.value,
            "table": "model_inferences",
            "column": None,
            "parameters": _percent_file_replay_parameters(),
            **replay_quantity,
        }
    )
    raw_issues.append(
        {
            "issue_id": "iss_ai_schema_drift",
            "type": IssueType.SCHEMA_DRIFT.value,
            "table": "model_inferences",
            "column": None,
            "exact_count": 1,
            "parameters": _ai_model_ops_schema_drift_parameters(),
        }
    )
    add(
        IssueType.INVALID_VALUE,
        "model_inferences",
        "model_version",
        issue_id="iss_ai_invalid_model_versions",
        quantity_keyword="model version",
        default_rate=0.01,
        parameters={
            "valid_format": "semantic model version like v2.1.3",
            "invalid_values": ["latest", "vNext", "model-2026-beta", ""],
            "value": "vNext",
        },
    )
    add(
        IssueType.INVALID_VALUE,
        "tenant_metadata",
        "region",
        issue_id="iss_ai_invalid_tenant_regions",
        quantity_keyword="region",
        default_rate=0.01,
        parameters={
            "valid_values": ["CA-ON", "CA-QC", "US-NY", "US-CA", "EU-DE"],
            "invalid_values": ["UNKNOWN", "GLOBAL", "12345", "REGION_@@"],
            "value": "REGION_@@",
        },
    )
    add(
        IssueType.DATE_RULE_VIOLATION,
        "feedback_scores",
        "created_at",
        issue_id="iss_ai_feedback_before_inference",
        quantity_keyword="feedback occurs before inference",
        default_rate=0.01,
        parameters={
            "rule": "feedback_before_inference",
            "after_column": "inference_created_at",
            "days_after": -1,
        },
    )
    add(
        IssueType.NULL_VALUE,
        "model_inferences",
        "response_text",
        issue_id="iss_ai_null_response_text",
        quantity_keyword="response_text",
        default_rate=0.01,
    )
    add(
        IssueType.INVALID_FORMAT,
        "model_inferences",
        "response_text",
        issue_id="iss_ai_malformed_response_text",
        quantity_keyword="malformed response_text",
        default_rate=0.01,
        parameters={"value": "{malformed_json: true"},
    )
    raw_issues.append(
        {
            "issue_id": "iss_ai_inconsistent_status_values",
            "type": IssueType.INVALID_VALUE.value,
            "table": "model_inferences",
            "column": "record_status",
            "exact_count": 100,
            "parameters": {
                "valid_values": ["completed", "failed", "timeout"],
                "invalid_values": ["complete", "done", "success"],
                "value": "success",
            },
        }
    )
    raw_issues.append(
        {
            "issue_id": "iss_ai_negative_latency",
            "type": IssueType.INVALID_VALUE.value,
            "table": "model_inferences",
            "column": "response_latency_ms",
            "exact_count": 100,
            "parameters": {"value": -1, "meaning": "negative latency is impossible"},
        }
    )
    raw_issues.append(
        {
            "issue_id": "iss_ai_future_inference_timestamps",
            "type": IssueType.DATE_RULE_VIOLATION.value,
            "table": "model_inferences",
            "column": "created_at",
            "exact_count": 100,
            "parameters": {
                "rule": "future_created_at",
                "after_column": "updated_at",
                "days_after": 30,
            },
        }
    )
    raw_issues.append(
        {
            "issue_id": "iss_ai_empty_prompt_text",
            "type": IssueType.BLANK_VALUE.value,
            "table": "prompt_requests",
            "column": "prompt_text",
            "exact_count": 100,
            "parameters": {"meaning": "empty prompt text is not valid for inference requests"},
        }
    )

    data = spec.model_dump(mode="json")
    data["issues"] = [
        issue.model_dump(mode="json") for issue in _issues_from_intent(raw_issues, spec)
    ]
    return _ensure_timeline_supports_issue_batches(ScenarioSpec.model_validate(data))


def _table(name: str, row_count: int, columns: list[ColumnSpec]) -> TableSpec:
    return TableSpec(
        name=name,
        row_count=row_count,
        columns=columns,
        source_systems=["synthetic_feed", "partner_feed", "legacy_batch"],
    )


def _pk(name: str) -> ColumnSpec:
    return ColumnSpec(name=name, type=ColumnType.LONG, nullable=False, primary_key=True)


def _col(
    name: str,
    column_type: ColumnType = ColumnType.STRING,
    *,
    nullable: bool = True,
    values: list[Any] | None = None,
) -> ColumnSpec:
    return ColumnSpec(name=name, type=column_type, nullable=nullable, values=values)


def _relationship(
    name: str, parent_table: str, parent_column: str, child_table: str, child_column: str
) -> RelationshipSpec:
    return RelationshipSpec(
        name=name,
        parent_table=parent_table,
        parent_column=parent_column,
        child_table=child_table,
        child_column=child_column,
    )


def _central_table(spec: ScenarioSpec) -> TableSpec:
    child_tables = {relationship.child_table for relationship in spec.relationships}
    candidates = [table for table in spec.tables if table.name in child_tables]
    if candidates:
        return max(candidates, key=lambda table: table.row_count)
    return max(spec.tables, key=lambda table: table.row_count)


def _primary_requested_table(spec: ScenarioSpec, text: str) -> TableSpec | None:
    counts = _extract_table_count_mentions(text)
    candidates = [
        (table_name, count)
        for table_name, count in counts.items()
        if table_name in {table.name for table in spec.tables}
    ]
    if not candidates:
        return None
    table_name = max(candidates, key=lambda item: item[1])[0]
    return spec.table(table_name)


def _table_for_keyword(spec: ScenarioSpec, text: str, keyword: str) -> TableSpec | None:
    window = _keyword_clause(text, keyword)
    matches = []
    for table in spec.tables:
        names = {table.name, table.name.replace("_", " "), _singular(table.name)}
        names.add(_singular(table.name).replace("_", " "))
        if any(name in window for name in names):
            matches.append(table)
    if not matches:
        return None
    return max(matches, key=lambda table: len(table.name))


def _relationship_target_for_requested_id(
    spec: ScenarioSpec, text: str, keyword: str, default_table: TableSpec
) -> tuple[TableSpec, str | None]:
    window = _keyword_clause(text, keyword)
    for relationship in spec.relationships:
        parent_markers = {
            relationship.parent_table,
            relationship.parent_table.replace("_", " "),
            _singular(relationship.parent_table),
            _singular(relationship.parent_table).replace("_", " "),
        }
        column_markers = {
            relationship.child_column,
            relationship.child_column.replace("_", " "),
        }
        if any(marker in window for marker in parent_markers | column_markers):
            primary = _primary_requested_table(spec, text)
            if primary and relationship.child_column in primary.column_names():
                return primary, relationship.child_column
            child = spec.table(relationship.child_table)
            return child, relationship.child_column
    return default_table, None


def _first_foreign_key(spec: ScenarioSpec, table: TableSpec) -> str | None:
    for relationship in spec.relationships:
        if relationship.child_table == table.name:
            return relationship.child_column
    return None


def _requested_id_column(text: str, table: TableSpec, keyword: str) -> str | None:
    window = _keyword_window(text, keyword)
    keyword_position = window.find(keyword)
    if keyword_position >= 0:
        before = re.split(r"[,.;]|\band\b", window[:keyword_position])[-1]
        after = re.split(r"[,.;]|\band\b", window[keyword_position + len(keyword) :])[0]
        window = f"{before} {after}"
    for column in table.columns:
        token = column.name.replace("_", " ")
        if column.name.endswith("_id") and (token in window or f"{token}s" in window):
            return column.name
    return None


def _first_nullable_non_pk(table: TableSpec) -> str | None:
    for column in table.columns:
        if not column.primary_key and column.nullable:
            return column.name
    return None


def _first_timestamp_column(table: TableSpec) -> str | None:
    for column in table.columns:
        if ColumnType(column.type) in {ColumnType.TIMESTAMP, ColumnType.DATE}:
            return column.name
    return None


def _invalid_value_parameters(column_name: str) -> dict[str, Any]:
    if column_name == "region_code":
        return {
            "valid_values": ["CA-ON", "CA-QC", "US-NY"],
            "invalid_values": ["UNKNOWN", "12345", "", "REGION_@@"],
            "value": "REGION_@@",
        }
    if column_name == "postal_code":
        return {
            "valid_format": "Canadian postal code, for example M5V 2T6",
            "invalid_values": ["UNKNOWN", "12345", "", "POSTAL_@@"],
            "value": "POSTAL_@@",
        }
    return {"value": "__INVALID__"}


def _should_use_exact_invalid_count(table: TableSpec, raw_issue: dict[str, Any]) -> bool:
    if raw_issue.get("exact_count") is not None:
        return False
    if raw_issue.get("column") == "region_code":
        return True
    rate = float(raw_issue.get("rate") or 0)
    return 0 < table.row_count * rate < 1


def _late_arrival_parameters(table: TableSpec, event_column: str, text: str) -> dict[str, Any]:
    arrival_column = "ingestion_ts" if "ingestion_ts" in table.column_names() else event_column
    parameters: dict[str, Any] = {
        "semantics": "late_arriving_data",
        "event_time_column": event_column,
        "arrival_column": arrival_column,
        "delay_days_min": 1,
        "delay_days_max": 5,
    }
    if "closure" in _keyword_window(text, "late"):
        parameters["meaning"] = (
            "closure_ts is the business event time; ingestion_ts is delayed into a later batch"
        )
    return parameters


def _schema_drift_parameters(table: TableSpec) -> dict[str, Any]:
    if table.name == "model_inferences":
        return _ai_model_ops_schema_drift_parameters()
    if table.name in {"network_events", "events"}:
        column = {"name": "signal_strength", "type": "double"}
    else:
        column = {"name": "sdf_extra_attribute", "type": "string"}
    return {
        "activation_batch": 3,
        "batch": 3,
        "operation": "add_column",
        "column": column,
        "add_columns": [column],
    }


def _file_replay_parameters() -> dict[str, Any]:
    return {
        "replay_granularity": "file",
        "file_count": 1,
        "source_batch": 2,
        "target_batch": 4,
        "source_batch_label": "batch_002",
        "replay_batch": "batch_004",
        "duplicate_effect": "replayed file creates additional duplicate business events",
        "overlap_policy": (
            "replay-created duplicates are tracked as file_replay and do not count "
            "against duplicate_record target validation"
        ),
    }


def _percent_file_replay_parameters() -> dict[str, Any]:
    parameters = _file_replay_parameters()
    parameters.pop("file_count", None)
    parameters["replay_granularity"] = "batch_fraction"
    parameters["meaning"] = "records from source batch are replayed into a later ingestion batch"
    return parameters


def _ai_model_ops_schema_drift_parameters() -> dict[str, Any]:
    return {
        "activation_batch": 3,
        "batch": 3,
        "operation": "multi_mutation",
        "column": {"name": "safety_filter_score", "type": "double"},
        "add_columns": [
            {"name": "safety_filter_score", "type": "double"},
            {"name": "token_count", "type": "long"},
        ],
        "rename_columns": [
            {"from": "response_latency_ms", "to": "latency_ms", "batch": 4},
        ],
        "type_changes": [
            {"column": "confidence_score", "from": "decimal", "to": "string", "batch": 5},
        ],
        "meaning": (
            "schema drift covers extra fields, renamed latency column, and inconsistent "
            "confidence_score typing across later raw batches"
        ),
    }


def _timestamp_column_for_keyword(table: TableSpec, keyword: str, text: str) -> str | None:
    if keyword not in _keyword_window(text, "late"):
        return None
    for column in table.columns:
        is_temporal = ColumnType(column.type) in {ColumnType.TIMESTAMP, ColumnType.DATE}
        if keyword in column.name and is_temporal:
            return column.name
    return None


def _invalid_value_target(spec: ScenarioSpec, text: str) -> tuple[str, str | None]:
    invalid_clause = _keyword_clause(text, "invalid")
    if "postal" in invalid_clause:
        for table_name in ("customers", "customer_regions"):
            if table_name in {table.name for table in spec.tables}:
                table = spec.table(table_name)
                for column_name in ("postal_code", "zip_code", "postcode"):
                    if column_name in table.column_names():
                        return table.name, column_name
    if "region" in text and "customer_regions" in {table.name for table in spec.tables}:
        table = spec.table("customer_regions")
        if "region_code" in table.column_names():
            return table.name, "region_code"
        if "province" in table.column_names():
            return table.name, "province"
    if "province" in text:
        for cue, table_name in (
            ("patient", "patients"),
            ("account", "accounts"),
            ("customer", "customers"),
        ):
            if cue in text and table_name in {table.name for table in spec.tables}:
                table = spec.table(table_name)
                if "province" in table.column_names():
                    return table.name, "province"
    for table in spec.tables:
        for column in table.columns:
            if column.name in text or column.name.replace("_", " ") in text:
                return table.name, column.name
    for preferred_table in ("patients", "accounts", "customers"):
        if preferred_table in {table.name for table in spec.tables}:
            table = spec.table(preferred_table)
            if "province" in table.column_names():
                return table.name, "province"
    central = _central_table(spec)
    return central.name, _first_nullable_non_pk(central)


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


def _default_pk_name(table_name: str) -> str:
    return f"{_singular(table_name)}_id"


def _singular(value: str) -> str:
    if value.endswith("ies"):
        return f"{value[:-3]}y"
    if value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def _domain_words(prompt: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", prompt.lower())
    ignored = {
        "build",
        "create",
        "generate",
        "with",
        "data",
        "synthetic",
        "bad",
        "dirty",
        "duplicate",
        "missing",
        "invalid",
        "late",
        "schema",
        "drift",
        "replayed",
        "files",
    }
    useful = [word for word in words if word not in ignored and not re.fullmatch(r"\d+k?", word)]
    return useful[-1] if useful else "records"


def _extract_custom_name(prompt: str, default: str) -> str:
    quoted = re.search(r'"([^"]{4,120})"', prompt)
    return quoted.group(1) if quoted else default


def _refresh_default_issue_counts(spec: ScenarioSpec) -> ScenarioSpec:
    if spec.domain != "insurance_claims":
        return spec
    data = spec.model_dump(mode="json")
    counts = {table["name"]: int(table["row_count"]) for table in data["tables"]}
    claims = counts.get("claims", 0)
    payments = counts.get("payments", 0)
    customers = counts.get("customers", 0)
    policies = counts.get("policies", 0)
    refreshed = {
        "iss_invalid_customer_provinces": max(1, round(customers * 0.01)),
        "iss_missing_customer_last_names": max(1, round(customers * 0.015)),
        "iss_policy_effective_after_expiry": max(1, round(policies * 0.01)),
        "iss_duplicate_claims": round(claims * 0.03),
        "iss_policy_orphans": round(claims * 0.01),
        "iss_missing_adjusters": round(claims * 0.05),
        "iss_late_payments": round(payments * 0.04),
        "iss_invalid_loss_dates": round(claims * 0.02),
        "iss_file_replay_batch_10_12": round((claims / 30) * 0.25),
        "iss_legacy_adjuster_correlation": round(claims * 0.02),
    }
    for issue in data["issues"]:
        if issue["issue_id"] in refreshed:
            issue["exact_count"] = refreshed[issue["issue_id"]]
            issue["rate"] = None
    return ScenarioSpec.model_validate(data)


def _issues_from_intent(raw_issues: object, spec: ScenarioSpec) -> list[IssueSpec]:
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
            str(issue_type), table, column, raw_issue, spec
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
        )
        result.append(issue)
    return result


def _normalize_issue_intent(
    issue_type: str,
    table: object,
    column: object,
    raw_issue: dict[str, Any],
    spec: ScenarioSpec,
) -> tuple[str | None, str | None, dict[str, Any], object, object]:
    table_names = {table_spec.name for table_spec in spec.tables}
    parameters = dict(raw_issue.get("parameters") or {})
    count = raw_issue.get("exact_count")
    rate = raw_issue.get("rate")
    table_name = str(table) if table else None
    column_name = str(column) if column else None
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


def _agent_intent_from_model(prompt: str) -> tuple[dict[str, Any] | None, str]:
    endpoint = os.getenv("SDF_MODEL_ENDPOINT")
    if not endpoint:
        return None, "Model endpoint is not bound locally; used deterministic parser."
    try:  # pragma: no cover - exercised in Databricks App runtime
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        response = WorkspaceClient().serving_endpoints.query(
            endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=_INTENT_SYSTEM_PROMPT),
                ChatMessage(role=ChatMessageRole.USER, content=prompt),
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        intent = _json_from_text(_response_text(response))
        if intent is None:
            return None, "Model did not return usable scenario intent; used deterministic parser."
        return intent, "Model extracted scenario intent."
    except Exception as exc:
        return None, f"Model intent extraction failed; used deterministic parser: {exc}"


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
                ChatMessage(role=ChatMessageRole.SYSTEM, content=_CUSTOM_SCHEMA_SYSTEM_PROMPT),
                ChatMessage(role=ChatMessageRole.USER, content=prompt),
            ],
            temperature=0.0,
            max_tokens=4000,
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
            max_tokens=5000,
        )
        return _json_from_text(_response_text(response))
    except Exception:
        return None


def _complete_table_counts_with_model(
    spec: ScenarioSpec, explicit_counts: set[str], prompt: str
) -> tuple[ScenarioSpec, str]:
    if not explicit_counts:
        return spec, "Agent accepted the blueprint table counts."
    endpoint = os.getenv("SDF_MODEL_ENDPOINT")
    if not endpoint:
        return spec, (
            "Model table planner was not available; only explicitly supplied table counts "
            "were changed."
        )
    try:  # pragma: no cover - exercised in Databricks App runtime
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        payload = {
            "user_prompt": prompt,
            "domain": spec.domain,
            "tables": {table.name: table.row_count for table in spec.tables},
            "explicit_table_counts": sorted(explicit_counts),
            "relationships": [
                {
                    "parent_table": relationship.parent_table,
                    "child_table": relationship.child_table,
                }
                for relationship in spec.relationships
            ],
        }
        response = WorkspaceClient().serving_endpoints.query(
            endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=_TABLE_PLAN_SYSTEM_PROMPT),
                ChatMessage(role=ChatMessageRole.USER, content=json.dumps(payload)),
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        planned = _json_from_text(_response_text(response)) or {}
        table_counts = planned.get("table_counts")
        if not isinstance(table_counts, dict):
            return spec, "Agent table planner returned no usable table-count plan."
        data = spec.model_dump(mode="json")
        by_name = {table["name"]: table for table in data["tables"]}
        for table, count in table_counts.items():
            if table in by_name and count is not None:
                by_name[str(table)]["row_count"] = max(1, int(count))
        return ScenarioSpec.model_validate(data), (
            "Agent selected related table counts from the requested domain and relationships."
        )
    except Exception as exc:
        return spec, f"Agent table planner failed; kept explicit counts only: {exc}"


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


_INTENT_SYSTEM_PROMPT = """
Extract a Scenario Data Factory intent as JSON only.
Do not emit code, SQL, markdown, commentary, or unsupported issue types.
Return compact JSON. Do not include reasoning text in the JSON payload.

Allowed blueprint domains: insurance_claims, retail_orders.
Allowed scales: small, demo.
If the user asks for a domain or table shape outside those blueprints, use domain
custom_schema and provide table_specs and relationships.

Blueprint table boundaries:
- insurance_claims tables are exactly customers, policies, claims, payments.
- insurance_claims has adjuster_id as a claims column; do not invent an adjusters table.
- retail_orders tables are exactly customers, products, orders, order_lines.

For known blueprint domains, if the user specifies one table count, infer and return
complete table_counts for every table in the blueprint unless the user explicitly
specified those other table counts. Do not leave related table counts implicit.

Allowed issue types:
null_value, blank_value, duplicate_record, invalid_format, invalid_value,
referential_orphan, date_rule_violation, late_arrival, out_of_order,
file_replay, schema_drift, correlated_missingness.

Allowed column types:
string, integer, long, decimal, date, timestamp, boolean.

Return this shape:
{
  "domain": "insurance_claims",
  "name": "short scenario name",
  "scale": "demo",
  "seed": 42,
  "table_counts": {"customers": 10000, "policies": 15000},
  "table_specs": null,
  "relationships": [],
  "issues": [
    {
      "type": "duplicate_record",
      "table": "claims",
      "column": null,
      "rate": 0.03,
      "exact_count": null,
      "parameters": {}
    }
  ]
}
"""

_TABLE_PLAN_SYSTEM_PROMPT = """
You are a bounded table-count planner for synthetic relational data.
Return JSON only.

Input contains:
- user_prompt
- domain
- current table counts
- explicit_table_counts that must not be changed
- relationships

Choose realistic row counts for every listed table. Preserve explicitly supplied
table counts exactly. Infer related table counts from the business domain,
relationship cardinalities, and user intent. Do not use fixed global ratios unless
the prompt states them; make a domain-aware choice.

Return:
{
  "table_counts": {"table_name": 1000},
  "reasoning": "short non-sensitive note"
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
- scale: "demo"
- seed: 42
- table_counts: {}
- table_specs: a list of tables with row_count and columns
- relationships: parent/child relationships
- issues: supported issue rules

Allowed column types:
string, integer, long, decimal, date, timestamp, boolean.

Every table must have exactly one primary key column.
Table names must be unique after lower_snake_case normalization.
Column names must be unique within a table after lower_snake_case normalization.
Foreign key columns must exist on child tables.
Issues must reference existing tables and columns, except table-level issues.
If the user supplies a count for a central fact/event table, preserve that count
exactly and infer plausible counts for parent dimensions and downstream event tables.
Use 4 to 8 tables for multi-entity domains unless the prompt explicitly asks for one table.

For healthcare claims, prefer tables like patients, providers, claims, adjudications,
and payments when appropriate. Use patient_id/provider_id foreign keys on claims.
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
        {"name": "province", "type": "string", "values": ["ON", "QC", "BC", "AB"]}
      ]
    }
  ],
  "relationships": [
    {
      "name": "patients_claims",
      "parent_table": "patients",
      "parent_column": "patient_id",
      "child_table": "claims",
      "child_column": "patient_id"
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

Every named business entity in the prompt must be represented as a table or a
relationship-bearing column. Multi-entity prompts should use 4 to 8 related tables.

Allowed column types:
string, integer, long, decimal, date, timestamp, boolean.

Supported issue types:
null_value, blank_value, duplicate_record, invalid_format, invalid_value,
referential_orphan, date_rule_violation, late_arrival, out_of_order,
file_replay, schema_drift, correlated_missingness.
"""


def _ensure_databricks_outputs(spec: ScenarioSpec) -> ScenarioSpec:
    data = spec.model_dump(mode="json")
    data["outputs"].update(
        {
            "mode": OutputMode.BOTH.value,
            "include_clean": True,
            "catalog": data["outputs"].get("catalog") or "sdf",
            "schema_name": data["outputs"].get("schema_name") or "scenario_data_factory",
        }
    )
    return ScenarioSpec.model_validate(data)


def _extract_seed(text: str) -> int:
    match = re.search(r"\bseed\s*(?:=|is|:)?\s*(\d+)\b", text)
    return int(match.group(1)) if match else 42


def _extract_name(prompt: str, domain: str) -> str:
    quoted = re.search(r'"([^"]{4,120})"', prompt)
    if quoted:
        return quoted.group(1)
    if domain == "retail_orders":
        return "Retail Bad Data Scenario"
    return "Canadian Insurance Claims Reliability Scenario"


def _apply_table_counts(spec: ScenarioSpec, text: str) -> tuple[ScenarioSpec, set[str]]:
    data = spec.model_dump(mode="json")
    tables = {table["name"]: table for table in data["tables"]}
    explicit_counts: set[str] = set()
    table_aliases = {
        "order lines": "order_lines",
        "order_lines": "order_lines",
        "customers": "customers",
        "policies": "policies",
        "claims": "claims",
        "payments": "payments",
        "products": "products",
        "orders": "orders",
    }
    table_pattern = "|".join(
        re.escape(name) for name in sorted(table_aliases, key=len, reverse=True)
    )
    number_pattern = r"\d[\d,]*(?:\.\d+)?\s*[km]?"
    for match in re.finditer(rf"({number_pattern})\s+({table_pattern})\b", text):
        table = table_aliases[match.group(2)]
        if table in tables:
            tables[table]["row_count"] = _parse_number(match.group(1))
            explicit_counts.add(table)
    for match in re.finditer(rf"\b({table_pattern})\s*(?:=|:|to|of)\s*({number_pattern})", text):
        table = table_aliases[match.group(1)]
        if table in tables:
            tables[table]["row_count"] = _parse_number(match.group(2))
            explicit_counts.add(table)
    return ScenarioSpec.model_validate(data), explicit_counts


def _parse_number(value: str) -> int:
    compact = value.replace(",", "").replace(" ", "")
    multiplier = 1
    if compact.endswith("k"):
        multiplier = 1_000
        compact = compact[:-1]
    elif compact.endswith("m"):
        multiplier = 1_000_000
        compact = compact[:-1]
    return max(1, round(float(compact) * multiplier))


def _apply_prompt_issues(spec: ScenarioSpec, text: str) -> ScenarioSpec:
    data = spec.model_dump(mode="json")
    issues = [] if "only" in text and _mentions_any_issue(text) else list(data["issues"])

    def add(issue: IssueSpec) -> None:
        replacement = issue.model_dump(mode="json")
        for index, existing in enumerate(issues):
            if existing["issue_id"] == issue.issue_id:
                if issue.exact_count is not None or issue.rate is not None:
                    issues[index] = replacement
                return
        issues.append(replacement)

    if "duplicate" in text:
        table = "orders" if spec.domain == "retail_orders" else "claims"
        add(
            IssueSpec(
                issue_id=f"iss_duplicate_{table}",
                type=IssueType.DUPLICATE_RECORD,
                table=table,
                **_count_or_rate(text, "duplicate", default_rate=0.03),
            )
        )
    if any(word in text for word in ("orphan", "non-existent", "nonexistent", "bad foreign")):
        if spec.domain == "retail_orders":
            add(
                IssueSpec(
                    issue_id="iss_order_customer_orphans",
                    type=IssueType.REFERENTIAL_ORPHAN,
                    table="orders",
                    column="customer_id",
                    **_count_or_rate(text, "orphan"),
                )
            )
        else:
            add(
                IssueSpec(
                    issue_id="iss_policy_orphans",
                    type=IssueType.REFERENTIAL_ORPHAN,
                    table="claims",
                    column="policy_id",
                    **_count_or_rate(text, "orphan"),
                )
            )
    if "email" in text and any(word in text for word in ("missing", "null", "blank")):
        add(
            IssueSpec(
                issue_id="iss_missing_customer_emails",
                type=IssueType.NULL_VALUE,
                table="customers",
                column="email",
                **_count_or_rate(text, "email"),
            )
        )
    if "last name" in text or "last_name" in text:
        add(
            IssueSpec(
                issue_id="iss_missing_customer_last_names",
                type=IssueType.NULL_VALUE,
                table="customers",
                column="last_name",
                **_count_or_rate(text, "last"),
            )
        )
    if "adjuster" in text and any(word in text for word in ("missing", "null", "blank")):
        add(
            IssueSpec(
                issue_id="iss_missing_adjusters",
                type=IssueType.NULL_VALUE,
                table="claims",
                column="adjuster_id",
                **_count_or_rate(text, "adjuster"),
            )
        )
    if "province" in text and any(word in text for word in ("invalid", "bad", "wrong")):
        add(
            IssueSpec(
                issue_id="iss_invalid_customer_provinces",
                type=IssueType.INVALID_VALUE,
                table="customers",
                column="province",
                parameters={"value": "ZZ"},
                **_count_or_rate(text, "province"),
            )
        )
    if "late" in text:
        table = "payments" if spec.domain == "insurance_claims" else "orders"
        column = "ingestion_ts" if table == "payments" else "order_date"
        add(
            IssueSpec(
                issue_id=f"iss_late_{table}",
                type=IssueType.LATE_ARRIVAL,
                table=table,
                column=column,
                parameters={"delay_days_min": 1, "delay_days_max": 5},
                **_count_or_rate(text, "late"),
            )
        )
    if "schema drift" in text or "new column" in text or "fraud_score" in text:
        add(
            IssueSpec(
                issue_id="iss_schema_drift_fraud_score",
                type=IssueType.SCHEMA_DRIFT,
                table="claims",
                exact_count=1,
                parameters={
                    "activation_batch": 20,
                    "add_columns": [{"name": "fraud_score", "type": "decimal"}],
                },
            )
        )
    if "replay" in text:
        add(
            IssueSpec(
                issue_id="iss_file_replay_batch_10_12",
                type=IssueType.FILE_REPLAY,
                table="claims",
                parameters={"source_batch": 10, "target_batch": 12},
                **_count_or_rate(text, "replay", default_rate=0.01),
            )
        )
    has_date_issue = "date" in text and any(
        word in text for word in ("invalid", "after", "before", "impossible")
    )
    if has_date_issue:
        if spec.domain == "insurance_claims":
            add(
                IssueSpec(
                    issue_id="iss_invalid_loss_dates",
                    type=IssueType.DATE_RULE_VIOLATION,
                    table="claims",
                    column="loss_date",
                    parameters={"after_column": "settlement_date"},
                    **_count_or_rate(text, "date"),
                )
            )
        elif "orders" in {table.name for table in spec.tables}:
            add(
                IssueSpec(
                    issue_id="iss_invalid_order_dates",
                    type=IssueType.DATE_RULE_VIOLATION,
                    table="orders",
                    column="order_date",
                    parameters={"after_column": "order_date", "days_after": 30},
                    **_count_or_rate(text, "date"),
                )
            )
    data["issues"] = issues
    return ScenarioSpec.model_validate(data)


def _mentions_any_issue(text: str) -> bool:
    markers = (
        "duplicate",
        "missing",
        "null",
        "blank",
        "invalid",
        "orphan",
        "late",
        "replay",
        "schema drift",
    )
    return any(marker in text for marker in markers)


def _count_or_rate(text: str, keyword: str, default_rate: float = 0.01) -> dict[str, object]:
    window = _keyword_window(text, keyword)
    keyword_position = window.find(keyword)
    before_keyword = window[:keyword_position] if keyword_position >= 0 else window
    before_clause = re.split(r"[,.;]|\band\b", before_keyword)[-1]
    percent_matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*%", before_clause))
    if percent_matches:
        percent = percent_matches[-1]
        return {"rate": float(percent.group(1)) / 100}
    after_keyword = window[window.find(keyword) + len(keyword) :] if keyword in window else window
    after_clause = re.split(r"[,.;]|\band\b", after_keyword)[0]
    trailing_percent = re.search(r"\b(\d+(?:\.\d+)?)\s*%", after_clause)
    if trailing_percent:
        return {"rate": float(trailing_percent.group(1)) / 100}
    count = re.search(r"\b(\d[\d,]*(?:\.\d+)?\s*[km]?)(?!\s*%)\b", after_clause)
    if count:
        return {"exact_count": _parse_number(count.group(1))}
    return {"rate": default_rate}


def _keyword_window(text: str, keyword: str) -> str:
    index = text.find(keyword)
    if index < 0:
        return text
    return text[max(0, index - 60) : index + 80]


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
    return os.getenv("SDF_RAW_RUNS_ROOT") or "/Volumes/sdf/scenario_data_factory/sdf_raw/runs"
