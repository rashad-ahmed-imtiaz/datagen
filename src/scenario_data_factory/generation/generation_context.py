from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from scenario_data_factory.models.scenario import ScenarioSpec


@dataclass
class GenerationContext:
    spec: ScenarioSpec
    run_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: dict[str, object] = field(default_factory=dict)
