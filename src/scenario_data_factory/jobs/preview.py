from __future__ import annotations

from scenario_data_factory.issues.planner import estimate_issues
from scenario_data_factory.models.scenario import ScenarioSpec


def preview_scenario(spec: ScenarioSpec) -> dict[str, object]:
    return {
        "scenario_id": spec.scenario_id,
        "spec_hash": spec.spec_hash(),
        "timeline": spec.timeline.model_dump(mode="json"),
        "tables": {table.name: table.row_count for table in spec.tables},
        "issue_counts": estimate_issues(spec),
    }
