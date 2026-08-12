from __future__ import annotations

from collections.abc import Iterator
from math import gcd

from scenario_data_factory.generation.seed import derive_seed
from scenario_data_factory.issues.base import IssueTarget
from scenario_data_factory.models.scenario import IssueSpec, IssueType, ScenarioSpec, TableSpec


def resolve_issue_count(issue: IssueSpec, table: TableSpec, *, batches: int | None = None) -> int:
    if IssueType(issue.type) == IssueType.FILE_REPLAY and issue.parameters.get("file_count"):
        source_batch = int(issue.parameters["source_batch"])
        if batches is None:
            raise ValueError("file_replay counts require the scenario batch count")
        return (table.row_count - source_batch) // batches + 1
    if issue.exact_count is not None:
        return min(issue.exact_count, table.row_count)
    assert issue.rate is not None
    return round(table.row_count * issue.rate)


def select_record_keys(spec: ScenarioSpec, issue: IssueSpec, table: TableSpec) -> list[int]:
    count = resolve_issue_count(issue, table, batches=spec.timeline.batches)
    return list(iter_record_keys(spec, issue, table))[:count]


def iter_record_keys(spec: ScenarioSpec, issue: IssueSpec, table: TableSpec) -> Iterator[int]:
    """Yield a deterministic permutation without sorting every table row in Python."""
    source_batch = None
    issue_type = IssueType(issue.type)
    if issue_type == IssueType.FILE_REPLAY:
        source_batch = int(issue.parameters["source_batch"])
    elif issue_type == IssueType.OUT_OF_ORDER:
        source_batch = int(issue.parameters.get("source_batch", 2))

    if source_batch is None:
        population = table.row_count

        def key_at(index: int) -> int:
            return index + 1

    else:
        population = (table.row_count - source_batch) // spec.timeline.batches + 1

        def key_at(index: int) -> int:
            return source_batch + index * spec.timeline.batches

    if population <= 0:
        return
    seed = derive_seed(spec.seed, spec.scenario_id, issue.issue_id, issue.type)
    start = seed % population
    step = max(1, (seed // max(population, 1)) % population)
    while gcd(step, population) != 1:
        step = (step + 1) % population or 1
    for index in range(population):
        yield key_at((start + index * step) % population)


def plan_targets(spec: ScenarioSpec, issue: IssueSpec, table: TableSpec) -> list[IssueTarget]:
    keys = select_record_keys(spec, issue, table)
    batches = spec.timeline.batches
    targets: list[IssueTarget] = []
    for key in keys:
        batch_id = ((key - 1) % batches) + 1
        if IssueType(issue.type) == IssueType.FILE_REPLAY:
            batch_id = int(issue.parameters.get("source_batch", batch_id))
        targets.append(
            IssueTarget(
                issue_id=issue.issue_id,
                issue_type=IssueType(issue.type).value,
                table=issue.table,
                column=issue.column,
                record_key=key,
                batch_id=batch_id,
                details=dict(issue.parameters),
            )
        )
    return targets
