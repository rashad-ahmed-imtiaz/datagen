from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

import scenario_data_factory.app_services.scenario_service as scenario_service
from scenario_data_factory.app_services.scenario_service import (
    AgentPlanningError,
    ScenarioService,
    _databricks_run_id,
    _faker_provider_available,
    _json_from_text,
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


def test_databricks_submission_extracts_the_sdk_waiter_response_run_id() -> None:
    assert _databricks_run_id(SimpleNamespace(response=SimpleNamespace(run_id=123))) == 123
    assert _databricks_run_id(SimpleNamespace(run_id=456)) == 456
    assert _databricks_run_id(SimpleNamespace(response=SimpleNamespace(run_id=None))) is None
