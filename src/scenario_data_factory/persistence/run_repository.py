from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from scenario_data_factory.exceptions import ScenarioDataFactoryError


class RunRepository:
    def __init__(self, root: str | Path = ".sdf/runs") -> None:
        self.root = Path(root).resolve()
        self._lock = threading.RLock()

    def _path(self, run_id: str) -> Path:
        if not re.fullmatch(r"run_[A-Za-z0-9]+", run_id):
            raise ScenarioDataFactoryError(
                "INVALID_RUN_ID", "Run ID is invalid.", technical_detail=run_id
            )
        return self.root / f"{run_id}.json"

    def create(self, scenario_id: str, run_type: str, spec_hash: str) -> dict[str, object]:
        run = {
            "run_id": f"run_{uuid4().hex[:12]}",
            "scenario_id": scenario_id,
            "run_type": run_type,
            "spec_hash": spec_hash,
            "status": "prepared",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save(run)
        return run

    def save(self, run: dict[str, object]) -> None:
        run_id = str(run["run_id"])
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            _atomic_write(self._path(run_id), json.dumps(run, indent=2, sort_keys=True))

    def get(self, run_id: str) -> dict[str, object]:
        with self._lock:
            return json.loads(self._path(run_id).read_text(encoding="utf-8"))

    def update_status(self, run_id: str, status: str, **fields: object) -> dict[str, object]:
        with self._lock:
            run = self.get(run_id)
            run.update(fields)
            run["status"] = status
            run["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.save(run)
            return run


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
