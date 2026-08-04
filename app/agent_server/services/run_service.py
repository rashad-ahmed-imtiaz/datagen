from __future__ import annotations

from scenario_data_factory.persistence.run_repository import RunRepository


class RunService:
    def __init__(self, runs: RunRepository | None = None) -> None:
        self.runs = runs or RunRepository()

    def status(self, run_id: str) -> dict[str, object]:
        run = self.runs.get(run_id)
        return {"run_id": run_id, "status": run["status"]}
