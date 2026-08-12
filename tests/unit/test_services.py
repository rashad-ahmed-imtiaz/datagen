from __future__ import annotations

import pytest

import scenario_data_factory.app_services.scenario_service as scenario_service
from scenario_data_factory.app_services.scenario_service import (
    AgentPlanningError,
    ScenarioService,
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
