from __future__ import annotations

from scenario_data_factory.blueprints.base import BlueprintMetadata, DomainBlueprint
from scenario_data_factory.models.scenario import ScenarioSpec


class CustomSchemaBlueprint(DomainBlueprint):
    metadata = BlueprintMetadata(
        domain="custom_schema",
        name="Custom schema",
        description="Pass-through blueprint for fully specified ScenarioSpecs.",
        tables=[],
    )

    def build(self, *, name: str, seed: int, scale: str = "demo") -> ScenarioSpec:
        raise NotImplementedError("custom_schema requires an explicit ScenarioSpec")
