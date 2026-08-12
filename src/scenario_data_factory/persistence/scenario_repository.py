from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from scenario_data_factory.exceptions import ScenarioDataFactoryError, ScenarioRevisionConflict
from scenario_data_factory.models.scenario import ScenarioSpec


class ScenarioRepository:
    def __init__(self, root: str | Path = ".sdf/scenarios") -> None:
        self.root = Path(root).resolve()
        self._lock = threading.RLock()

    def _path(self, scenario_id: str) -> Path:
        if not re.fullmatch(r"scn_[A-Za-z0-9]+", scenario_id):
            raise ScenarioDataFactoryError(
                "INVALID_SCENARIO_ID",
                "Scenario ID is invalid.",
                technical_detail=scenario_id,
            )
        return self.root / f"{scenario_id}.json"

    def save(self, spec: ScenarioSpec) -> ScenarioSpec:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            _atomic_write(self._path(spec.scenario_id), spec.model_dump_json(indent=2))
        return spec

    def get(self, scenario_id: str) -> ScenarioSpec:
        with self._lock:
            return ScenarioSpec.from_json(self._path(scenario_id).read_text(encoding="utf-8"))

    def list_recent(self, limit: int = 20) -> list[ScenarioSpec]:
        files = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [ScenarioSpec.from_json(path.read_text(encoding="utf-8")) for path in files[:limit]]

    def patch(
        self, scenario_id: str, expected_revision: int, patch: dict[str, Any]
    ) -> ScenarioSpec:
        with self._lock:
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


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
