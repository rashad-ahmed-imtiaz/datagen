from __future__ import annotations

from scenario_data_factory.exceptions import ScenarioValidationError
from scenario_data_factory.models.scenario import RelationshipSpec, TableSpec


def dependency_order(tables: list[TableSpec], relationships: list[RelationshipSpec]) -> list[str]:
    names = {t.name for t in tables}
    incoming = {name: set() for name in names}
    outgoing = {name: set() for name in names}
    for rel in relationships:
        if rel.parent_table not in names or rel.child_table not in names:
            raise ScenarioValidationError(
                "INVALID_RELATIONSHIP",
                "A relationship references a missing table.",
                technical_detail=rel.name,
            )
        incoming[rel.child_table].add(rel.parent_table)
        outgoing[rel.parent_table].add(rel.child_table)

    ready = sorted(name for name, deps in incoming.items() if not deps)
    ordered: list[str] = []
    while ready:
        name = ready.pop(0)
        ordered.append(name)
        for child in sorted(outgoing[name]):
            incoming[child].remove(name)
            if not incoming[child]:
                ready.append(child)
        ready.sort()

    if len(ordered) != len(names):
        raise ScenarioValidationError(
            "RELATIONSHIP_CYCLE",
            "Table relationships contain a cycle.",
            technical_detail=str({k: sorted(v) for k, v in incoming.items() if v}),
        )
    return ordered
