from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from scenario_data_factory.models.scenario import ScenarioSpec


@dataclass(frozen=True)
class BlueprintMetadata:
    domain: str
    name: str
    description: str
    tables: list[str]


class DomainBlueprint(ABC):
    metadata: BlueprintMetadata

    @abstractmethod
    def build(self, *, name: str, seed: int, scale: str = "demo") -> ScenarioSpec:
        raise NotImplementedError
