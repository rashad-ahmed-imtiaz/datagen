from __future__ import annotations

from scenario_data_factory.blueprints.base import BlueprintMetadata, DomainBlueprint
from scenario_data_factory.blueprints.custom_schema import CustomSchemaBlueprint
from scenario_data_factory.blueprints.insurance_claims import InsuranceClaimsBlueprint
from scenario_data_factory.blueprints.retail_orders import RetailOrdersBlueprint
from scenario_data_factory.exceptions import UnsupportedBlueprintError

BLUEPRINTS: dict[str, DomainBlueprint] = {
    "insurance_claims": InsuranceClaimsBlueprint(),
    "retail_orders": RetailOrdersBlueprint(),
    "custom_schema": CustomSchemaBlueprint(),
}


def list_blueprints() -> list[BlueprintMetadata]:
    return [bp.metadata for bp in BLUEPRINTS.values()]


def get_blueprint(domain: str) -> DomainBlueprint:
    try:
        return BLUEPRINTS[domain]
    except KeyError as exc:
        raise UnsupportedBlueprintError(
            "UNSUPPORTED_BLUEPRINT",
            f"Unsupported domain blueprint: {domain}",
            remediation=f"Choose one of: {', '.join(sorted(BLUEPRINTS))}",
        ) from exc
