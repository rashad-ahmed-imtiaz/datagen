from __future__ import annotations

from dataclasses import dataclass

from scenario_data_factory.models.scenario import ScenarioSpec


@dataclass(frozen=True)
class BatchPlan:
    table: str
    batch_id: int
    expected_rows: int


def plan_batches(spec: ScenarioSpec) -> list[BatchPlan]:
    plans: list[BatchPlan] = []
    for table in spec.tables:
        base = table.row_count // spec.timeline.batches
        remainder = table.row_count % spec.timeline.batches
        for batch_id in range(1, spec.timeline.batches + 1):
            plans.append(
                BatchPlan(
                    table=table.name,
                    batch_id=batch_id,
                    expected_rows=base + (1 if batch_id <= remainder else 0),
                )
            )
    return plans
