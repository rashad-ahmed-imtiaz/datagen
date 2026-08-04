from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class RunRepository:
    def __init__(self, root: str | Path = ".sdf/runs") -> None:
        self.root = Path(root)

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
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{run['run_id']}.json").write_text(
            json.dumps(run, indent=2, sort_keys=True), encoding="utf-8"
        )

    def get(self, run_id: str) -> dict[str, object]:
        return json.loads((self.root / f"{run_id}.json").read_text(encoding="utf-8"))

    def update_status(self, run_id: str, status: str, **fields: object) -> dict[str, object]:
        run = self.get(run_id)
        run.update(fields)
        run["status"] = status
        run["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save(run)
        return run
