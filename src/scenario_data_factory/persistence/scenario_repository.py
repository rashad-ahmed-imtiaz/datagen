from __future__ import annotations

from pathlib import Path
from typing import Any

from scenario_data_factory.exceptions import ScenarioRevisionConflict
from scenario_data_factory.models.scenario import ScenarioSpec


class ScenarioRepository:
    def __init__(self, root: str | Path = ".sdf/scenarios") -> None:
        self.root = Path(root)

    def save(self, spec: ScenarioSpec) -> ScenarioSpec:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{spec.scenario_id}.json"
        path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        return spec

    def get(self, scenario_id: str) -> ScenarioSpec:
        return ScenarioSpec.from_json(
            (self.root / f"{scenario_id}.json").read_text(encoding="utf-8")
        )

    def list_recent(self, limit: int = 20) -> list[ScenarioSpec]:
        files = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [ScenarioSpec.from_json(path.read_text(encoding="utf-8")) for path in files[:limit]]

    def patch(
        self, scenario_id: str, expected_revision: int, patch: dict[str, Any]
    ) -> ScenarioSpec:
        current = self.get(scenario_id)
        if current.revision != expected_revision:
            raise ScenarioRevisionConflict(
                "STALE_SCENARIO_REVISION",
                "Scenario draft changed since the caller last read it.",
                scenario_id=scenario_id,
                remediation=(
                    f"Reload the draft at revision {current.revision} and reapply the change."
                ),
            )
        data = current.model_dump(mode="json")
        _deep_update(data, patch)
        data["revision"] = current.revision + 1
        return self.save(ScenarioSpec.model_validate(data))


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
