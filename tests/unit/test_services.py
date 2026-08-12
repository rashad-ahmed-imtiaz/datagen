from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

import scenario_data_factory.app_services.scenario_service as scenario_service
from scenario_data_factory.app_services.scenario_service import (
    AgentPlanningError,
    ScenarioService,
    _ai_model_ops_execution_rule_gaps,
    _custom_semantic_gaps,
    _databricks_run_id,
    _faker_provider_available,
    _json_from_text,
    _merge_model_contextual_issue_targets,
    _normalize_agent_intent,
)
from scenario_data_factory.persistence.run_repository import RunRepository
from scenario_data_factory.persistence.scenario_repository import ScenarioRepository


def _service(tmp_path) -> ScenarioService:
    return ScenarioService(
        ScenarioRepository(tmp_path / "scenarios"),
        RunRepository(tmp_path / "runs"),
    )


def _agent_intent(*, include_city_strategy: bool = True) -> dict[str, object]:
    city = {"name": "city", "type": "string"}
    if include_city_strategy:
        city["faker"] = "city"
    return {
        "domain": "custom_schema",
        "name": "Customer Orders",
        "seed": 42,
        "locale": "en_US",
        "timeline": {"start_date": "2025-01-01", "batches": 12, "frequency": "monthly"},
        "table_specs": [
            {
                "name": "customers",
                "row_count": 100,
                "columns": [
                    {
                        "name": "customer_id",
                        "type": "long",
                        "primary_key": True,
                        "nullable": False,
                    },
                    city,
                ],
            },
            {
                "name": "orders",
                "row_count": 300,
                "columns": [
                    {
                        "name": "order_id",
                        "type": "long",
                        "primary_key": True,
                        "nullable": False,
                    },
                    {"name": "customer_id", "type": "long", "nullable": False},
                    {"name": "order_date", "type": "date", "semantic": {"kind": "timeline"}},
                    {
                        "name": "amount",
                        "type": "decimal",
                        "semantic": {"kind": "uniform_range", "min": 10, "max": 100},
                    },
                ],
            },
        ],
        "relationships": [
            {
                "name": "customers_orders",
                "parent_table": "customers",
                "parent_column": "customer_id",
                "child_table": "orders",
                "child_column": "customer_id",
            }
        ],
        "issues": [
            {
                "type": "null_value",
                "table": "customers",
                "column": "city",
                "rate": 0.03,
            }
        ],
    }


def test_generation_confirmation_hash_guard(tmp_path) -> None:
    service = _service(tmp_path)
    draft = service.create_scenario_draft(
        {"domain": "insurance_claims", "name": "demo", "seed": 42, "scale": "small"}
    )
    prepared = service.prepare_generation(str(draft["scenario_id"]))
    rejected = service.confirm_generation(str(prepared["run_id"]), "wrong")
    assert rejected["status"] == "rejected"
    confirmed = service.confirm_generation(str(prepared["run_id"]), str(draft["spec_hash"]))
    assert confirmed["status"] == "confirmed"


def test_generation_rejects_a_scenario_changed_after_prepare(tmp_path) -> None:
    service = _service(tmp_path)
    draft = service.create_scenario_draft(
        {"domain": "insurance_claims", "name": "demo", "seed": 42, "scale": "small"}
    )
    prepared = service.prepare_generation(str(draft["scenario_id"]))
    service.patch_scenario_draft(
        str(draft["scenario_id"]), int(draft["revision"]), {"name": "changed"}
    )
    result = service.confirm_and_submit_generation(
        str(prepared["run_id"]), str(draft["spec_hash"])
    )
    assert result["status"] == "rejected"
    assert "scenario changed" in str(result["reason"])


def test_generation_status_refreshes_from_databricks(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path)
    draft = service.create_scenario_draft(
        {"domain": "insurance_claims", "name": "demo", "seed": 42, "scale": "small"}
    )
    prepared = service.prepare_generation(str(draft["scenario_id"]))
    run_id = str(prepared["run_id"])
    service.runs.update_status(run_id, "submitted", databricks_run_id=123)
    monkeypatch.setattr(
        scenario_service,
        "_databricks_run_state",
        lambda _: ("TERMINATED", "SUCCESS"),
    )

    assert service.get_run_status(run_id) == {"run_id": run_id, "status": "succeeded"}
    assert service.get_run_summary(run_id)["databricks_result_state"] == "SUCCESS"


def test_generation_status_preserves_submission_when_databricks_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    service = _service(tmp_path)
    draft = service.create_scenario_draft(
        {"domain": "insurance_claims", "name": "demo", "seed": 42, "scale": "small"}
    )
    prepared = service.prepare_generation(str(draft["scenario_id"]))
    run_id = str(prepared["run_id"])
    service.runs.update_status(run_id, "submitted", databricks_run_id=123)
    monkeypatch.setattr(scenario_service, "_databricks_run_state", lambda _: None)

    assert service.get_run_status(run_id) == {"run_id": run_id, "status": "submitted"}


def test_generation_submission_persists_databricks_run_id(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path)
    draft = service.create_scenario_draft(
        {"domain": "insurance_claims", "name": "demo", "seed": 42, "scale": "small"}
    )
    prepared = service.prepare_generation(str(draft["scenario_id"]))
    jobs = SimpleNamespace(
        run_now=lambda *_args, **_kwargs: SimpleNamespace(response=SimpleNamespace(run_id=987))
    )
    sdk = ModuleType("databricks.sdk")
    sdk.WorkspaceClient = lambda: SimpleNamespace(jobs=jobs)
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk)
    monkeypatch.setenv("SDF_GENERATION_JOB_ID", "100")
    monkeypatch.setattr(scenario_service, "_write_databricks_scenario_spec", lambda _: "/Volumes/x")

    result = service.confirm_and_submit_generation(
        str(prepared["run_id"]), str(draft["spec_hash"])
    )

    assert result["status"] == "submitted"
    assert result["databricks_run_id"] == 987


def test_generation_submission_without_a_run_id_is_marked_failed(tmp_path, monkeypatch) -> None:
    service = _service(tmp_path)
    draft = service.create_scenario_draft(
        {"domain": "insurance_claims", "name": "demo", "seed": 42, "scale": "small"}
    )
    prepared = service.prepare_generation(str(draft["scenario_id"]))
    sdk = ModuleType("databricks.sdk")
    sdk.WorkspaceClient = lambda: SimpleNamespace(
        jobs=SimpleNamespace(run_now=lambda *_args, **_kwargs: SimpleNamespace(response={}))
    )
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk)
    monkeypatch.setenv("SDF_GENERATION_JOB_ID", "100")
    monkeypatch.setattr(scenario_service, "_write_databricks_scenario_spec", lambda _: "/Volumes/x")

    result = service.confirm_and_submit_generation(
        str(prepared["run_id"]), str(draft["spec_hash"])
    )

    assert result["status"] == "submit_failed"
    assert "without returning a run ID" in str(result["reason"])


def test_service_uses_control_volume_for_persistent_drafts(monkeypatch) -> None:
    monkeypatch.setenv("SDF_CONTROL_VOLUME", "/Volumes/sdf/scenario_data_factory/sdf_control")
    service = ScenarioService()
    assert service.scenarios.root.as_posix().endswith("sdf_control/drafts")
    assert service.runs.root.as_posix().endswith("sdf_control/runs")


def test_agentic_draft_creates_an_executable_custom_scenario(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service, "_custom_schema_intent_from_model", lambda _: _agent_intent()
    )
    result = _service(tmp_path).create_scenario_from_prompt("Create customer orders data.")
    assert result["domain"] == "custom_schema"
    assert result["tables"] == {"customers": 100, "orders": 300}
    assert result["issues"][0]["column"] == "city"
    assert any("Schema-design agent designed" in warning for warning in result["warnings"])


def test_agentic_draft_uses_model_completion_for_missing_strategy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "_custom_schema_intent_from_model",
        lambda _: _agent_intent(include_city_strategy=False),
    )
    monkeypatch.setattr(
        scenario_service,
        "_enrich_column_strategies_with_model",
        lambda *_: {"columns": [{"table": "customers", "column": "city", "faker": "city"}]},
    )
    result = _service(tmp_path).create_scenario_from_prompt("Create customer orders data.")
    assert result["tables"] == {"customers": 100, "orders": 300}


def test_agentic_draft_ignores_empty_optional_model_weights(tmp_path, monkeypatch) -> None:
    intent = _agent_intent()
    tables = intent["table_specs"]
    assert isinstance(tables, list)
    columns = tables[0]["columns"]
    assert isinstance(columns, list)
    columns[1]["weights"] = []
    monkeypatch.setattr(scenario_service, "_custom_schema_intent_from_model", lambda _: intent)

    result = _service(tmp_path).create_scenario_from_prompt("Create customer orders data.")

    assert result["tables"] == {"customers": 100, "orders": 300}


def test_agentic_draft_uses_focused_issue_completion(tmp_path, monkeypatch) -> None:
    intent = _agent_intent()
    intent["issues"] = [
        {
            "issue_id": "replay_orders",
            "type": "file_replay",
            "table": "orders",
            "exact_count": 1,
            "parameters": {"source_batch": 12, "target_batch": 10},
        }
    ]
    monkeypatch.setattr(scenario_service, "_custom_schema_intent_from_model", lambda _: intent)
    monkeypatch.setattr(scenario_service, "_enrich_column_strategies_with_model", lambda *_: None)
    monkeypatch.setattr(
        scenario_service,
        "_enrich_issue_parameters_with_model",
        lambda *_: {
            "issues": [
                {
                "issue_id": "replay_orders",
                "type": "file_replay",
                "table": "orders",
                "parameters": {"source_batch": 2, "target_batch": 4, "file_count": 1},
                }
            ]
        },
    )

    result = _service(tmp_path).create_scenario_from_prompt("Create customer orders data.")

    assert result["issues"][0]["display_value"] == "1 file"


def test_agentic_draft_rejects_no_model_plan_without_deterministic_fallback(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(scenario_service, "_custom_schema_intent_from_model", lambda _: None)
    with pytest.raises(AgentPlanningError, match="No data or tables were created"):
        _service(tmp_path).create_scenario_from_prompt("Create customer orders data.")


def test_agentic_draft_retries_after_an_unusable_repair_response(tmp_path, monkeypatch) -> None:
    repairs = iter([None, _agent_intent()])
    monkeypatch.setattr(
        scenario_service,
        "_custom_schema_intent_from_model",
        lambda _: _agent_intent(include_city_strategy=False),
    )
    monkeypatch.setattr(
        scenario_service, "_enrich_column_strategies_with_model", lambda *_: None
    )
    monkeypatch.setattr(
        scenario_service, "_repair_custom_schema_intent_with_model", lambda *_: next(repairs)
    )
    result = _service(tmp_path).create_scenario_from_prompt("Create customer orders data.")
    assert result["tables"] == {"customers": 100, "orders": 300}


def test_agentic_draft_rejects_missing_issue_column_before_submission(
    tmp_path, monkeypatch
) -> None:
    intent = _agent_intent()
    issues = intent["issues"]
    assert isinstance(issues, list)
    issues[0]["column"] = None
    monkeypatch.setattr(scenario_service, "_custom_schema_intent_from_model", lambda _: intent)
    monkeypatch.setattr(
        scenario_service, "_enrich_column_strategies_with_model", lambda *_: None
    )
    monkeypatch.setattr(
        scenario_service, "_repair_custom_schema_intent_with_model", lambda *_: None
    )
    with pytest.raises(AgentPlanningError, match="No data or tables were created"):
        _service(tmp_path).create_scenario_from_prompt("Create customer orders data.")


def test_truncated_model_json_returns_none() -> None:
    assert _json_from_text('{"table_specs": [') is None


def test_locale_aware_faker_provider_validation_accepts_file_paths() -> None:
    assert _faker_provider_available("file_path", "en_US")
    assert not _faker_provider_available("__class__", "en_US")


def test_canadian_city_fields_use_real_province_keyed_lookups() -> None:
    intent = _agent_intent()
    tables = intent["table_specs"]
    assert isinstance(tables, list)
    customers = tables[0]
    assert isinstance(customers, dict)
    columns = customers["columns"]
    assert isinstance(columns, list)
    columns.append({"name": "province", "type": "string", "values": ["ON", "QC", "BC"]})
    intent["locale"] = "en_CA"

    normalized = _normalize_agent_intent(intent)
    city = next(
        column
        for column in normalized["table_specs"][0]["columns"]
        if column["name"] == "city"
    )

    assert city["semantic"]["kind"] == "lookup"
    assert city["semantic"]["key_column"] == "province"
    assert city["semantic"]["values_by_key"]["ON"] == [
        "Toronto",
        "Ottawa",
        "Mississauga",
        "Hamilton",
        "London",
        "Kitchener",
    ]
    assert "faker" not in city


def test_canadian_city_lookup_accepts_full_province_names() -> None:
    intent = _agent_intent()
    tables = intent["table_specs"]
    assert isinstance(tables, list)
    customers = tables[0]
    assert isinstance(customers, dict)
    columns = customers["columns"]
    assert isinstance(columns, list)
    columns.append(
        {"name": "province", "type": "string", "values": ["Ontario", "Quebec"]}
    )
    intent["locale"] = "en_CA"

    normalized = _normalize_agent_intent(intent)
    city = next(
        column
        for column in normalized["table_specs"][0]["columns"]
        if column["name"] == "city"
    )

    assert city["semantic"]["key_column"] == "province"
    assert city["semantic"]["values_by_key"]["Ontario"][0] == "Toronto"
    assert city["semantic"]["values_by_key"]["Quebec"][0] == "Montreal"


def test_ai_model_ops_context_requires_issue_targets_on_operational_tables() -> None:
    def table(name: str, *columns: str) -> SimpleNamespace:
        return SimpleNamespace(name=name, column_names=lambda: set(columns))

    def issue(
        issue_type: str, table_name: str, column: str | None, parameters: dict | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            type=issue_type,
            table=table_name,
            column=column,
            parameters=parameters or {},
        )

    prompt = """
    Create AI model operations data with orphan model IDs, orphan user IDs, missing prompt
    categories, missing evaluation labels, missing inference latency values, late-arriving
    feedback, replayed inference files, schema drift, invalid model versions, invalid tenant
    regions, feedback occurs before inference, null or malformed response_text, negative
    latency, and empty prompt text.
    """
    tables = [
        table(
            "model_inferences",
            "model_id",
            "user_id",
            "model_version",
            "response_text",
            "response_latency_ms",
        ),
        table("prompt_requests", "prompt_category", "prompt_text"),
        table("feedback", "created_at", "inference_created_at", "ingestion_ts"),
        table("evaluations", "evaluation_label"),
        table("tenants", "region"),
    ]
    wrong = SimpleNamespace(
        tables=tables,
        issues=[
            issue("referential_orphan", "prompt_requests", "model_id"),
            issue("referential_orphan", "prompt_requests", "user_id"),
            issue("invalid_value", "model_registry", "model_version"),
            issue("schema_drift", "model_registry", None),
            issue("file_replay", "prompt_requests", None),
        ],
    )

    gaps = _ai_model_ops_execution_rule_gaps(wrong, prompt)

    assert "AI scenario must map orphan model IDs to model_inferences.model_id" in gaps
    assert "AI scenario must map orphan user IDs to model_inferences.user_id" in gaps
    assert "AI scenario must map invalid model versions to model_inferences.model_version" in gaps
    assert "AI scenario must replay model_inferences ingestion files" in gaps
    assert "AI scenario must apply schema drift to model_inferences batches" in gaps

    correct = SimpleNamespace(
        tables=tables,
        issues=[
            issue("referential_orphan", "model_inferences", "model_id"),
            issue("referential_orphan", "model_inferences", "user_id"),
            issue("null_value", "prompt_requests", "prompt_category"),
            issue("null_value", "evaluations", "evaluation_label"),
            issue("null_value", "model_inferences", "response_latency_ms"),
            issue(
                "late_arrival",
                "feedback",
                "created_at",
                {"event_time_column": "created_at", "arrival_column": "ingestion_ts"},
            ),
            issue("file_replay", "model_inferences", None),
            issue("schema_drift", "model_inferences", None),
            issue("invalid_value", "model_inferences", "model_version"),
            issue("invalid_value", "tenants", "region"),
            issue(
                "date_rule_violation",
                "feedback",
                "created_at",
                {"after_column": "inference_created_at"},
            ),
            issue("invalid_format", "model_inferences", "response_text"),
            issue("null_value", "model_inferences", "response_text"),
            issue("invalid_value", "model_inferences", "response_latency_ms"),
            issue("blank_value", "prompt_requests", "prompt_text"),
        ],
    )

    assert _ai_model_ops_execution_rule_gaps(correct, prompt) == []


def test_contextual_issue_merge_replaces_wrong_target_and_adds_missing_defect() -> None:
    intent = {
        "table_specs": [
            {
                "name": "model_inferences",
                "row_count": 10,
                "columns": [
                    {
                        "name": "inference_id",
                        "type": "long",
                        "primary_key": True,
                        "nullable": False,
                    }
                ],
            }
        ],
        "issues": [
            {
                "issue_id": "wrong_model_version",
                "type": "invalid_value",
                "table": "model_registry",
                "column": "model_version",
                "rate": 0.01,
                "parameters": {},
            },
            {
                "issue_id": "replay_duplicate_overlap",
                "type": "duplicate_record",
                "table": "model_inferences",
                "column": None,
                "rate": 0.1,
                "parameters": {},
            },
        ],
    }
    enrichment = {
        "remove_issue_ids": ["replay_duplicate_overlap"],
        "columns": [
            {
                "table": "model_inferences",
                "name": "model_version",
                "type": "string",
                "values": ["1.0.0", "1.1.0"],
            }
        ],
        "issues": [
            {
                "replaces_issue_id": "wrong_model_version",
                "type": "invalid_value",
                "table": "model_inferences",
                "column": "model_version",
                "rate": 0.01,
                "parameters": {"invalid_values": ["version_unknown"]},
            },
            {
                "type": "null_value",
                "table": "model_inferences",
                "column": "response_text",
                "rate": 0.01,
                "parameters": {},
            },
        ],
    }

    merged = _merge_model_contextual_issue_targets(intent, enrichment)

    assert merged["issues"][0]["issue_id"] == "wrong_model_version"
    assert merged["issues"][0]["table"] == "model_inferences"
    assert merged["issues"][1]["column"] == "response_text"
    assert len(merged["issues"]) == 2
    assert any(
        column["name"] == "model_version"
        for column in merged["table_specs"][0]["columns"]
    )


def test_normalization_clears_rate_when_file_replay_uses_file_count() -> None:
    intent = _agent_intent()
    intent["issues"] = [
        {
            "type": "file_replay",
            "table": "orders",
            "rate": 0.1,
            "parameters": {"file_count": 1, "source_batch": 2, "target_batch": 4},
        }
    ]

    normalized = _normalize_agent_intent(intent)

    assert normalized["issues"][0]["rate"] is None
    assert normalized["issues"][0]["exact_count"] is None


def test_ai_inferences_satisfy_event_fact_concept_without_generic_event_table() -> None:
    spec = SimpleNamespace(
        tables=[SimpleNamespace(name="model_inferences", columns=[])], relationships=[]
    )

    assert _custom_semantic_gaps(spec, "AI model inference events") == []


def test_databricks_submission_extracts_the_sdk_waiter_response_run_id() -> None:
    assert _databricks_run_id(SimpleNamespace(response=SimpleNamespace(run_id=123))) == 123
    assert _databricks_run_id(SimpleNamespace(run_id=456)) == 456
    assert _databricks_run_id(SimpleNamespace(response=SimpleNamespace(run_id=None))) is None
