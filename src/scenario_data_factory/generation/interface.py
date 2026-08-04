from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from scenario_data_factory.models.scenario import ScenarioSpec


class BaselineGenerator(ABC):
    @abstractmethod
    def generate(
        self, spark: Any, spec: ScenarioSpec, *, partitions: int | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError
