from __future__ import annotations

from scenario_data_factory.blueprints.registry import get_blueprint
from scenario_data_factory.compiler.validation import validate_scenario
from scenario_data_factory.jobs.preview import preview_scenario


def run_smoke_test() -> dict[str, object]:
    spec = get_blueprint("insurance_claims").build(
        name="Canadian Insurance Claims Reliability Scenario",
        seed=42,
        scale="small",
    )
    warnings = validate_scenario(spec)
    preview = preview_scenario(spec)
    return {"status": "ok", "warnings": warnings, "preview": preview}
