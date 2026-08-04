from __future__ import annotations

from typing import Any

from scenario_data_factory.app_services.scenario_service import ScenarioService
from scenario_data_factory.blueprints.registry import get_blueprint, list_blueprints
from scenario_data_factory.issues.registry import list_issue_types

service = ScenarioService()


def _local_tools() -> dict[str, Any]:
    return {
        "list_domain_blueprints": lambda: [bp.__dict__ for bp in list_blueprints()],
        "get_domain_blueprint": lambda domain: get_blueprint(domain).metadata.__dict__,
        "list_supported_issue_types": list_issue_types,
        "get_issue_type": lambda issue_type: next(
            i for i in list_issue_types() if i["type"] == issue_type
        ),
        "create_scenario_draft": service.create_scenario_draft,
        "get_scenario_draft": service.get_scenario_draft,
        "patch_scenario_draft": service.patch_scenario_draft,
        "validate_scenario_draft": service.validate_scenario_draft,
        "estimate_scenario": service.estimate_scenario,
        "prepare_preview": service.prepare_preview,
        "get_preview_status": service.get_run_status,
        "get_preview_summary": service.get_run_summary,
        "prepare_generation": service.prepare_generation,
        "get_generation_status": service.get_run_status,
        "get_generation_summary": service.get_run_summary,
        "list_recent_scenarios": service.list_recent_scenarios,
        "clone_scenario": service.clone_scenario,
    }


def tool_registry() -> list[Any]:
    try:  # pragma: no cover - depends on OpenAI Agents SDK runtime
        from agents import function_tool
    except Exception:
        return []
    return [function_tool(fn) for fn in _local_tools().values()]
