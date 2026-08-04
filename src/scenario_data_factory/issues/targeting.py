from __future__ import annotations

from scenario_data_factory.generation.seed import derive_seed
from scenario_data_factory.issues.base import IssueTarget
from scenario_data_factory.models.scenario import IssueSpec, IssueType, ScenarioSpec, TableSpec


def resolve_issue_count(issue: IssueSpec, table: TableSpec) -> int:
    if IssueType(issue.type) == IssueType.FILE_REPLAY and issue.parameters.get("file_count"):
        return table.row_count
    if issue.exact_count is not None:
        return min(issue.exact_count, table.row_count)
    assert issue.rate is not None
    return round(table.row_count * issue.rate)


def select_record_keys(spec: ScenarioSpec, issue: IssueSpec, table: TableSpec) -> list[int]:
    count = resolve_issue_count(issue, table)
    salt = derive_seed(spec.seed, spec.scenario_id, issue.issue_id, issue.type)
    keys = range(1, table.row_count + 1)
    if IssueType(issue.type) == IssueType.FILE_REPLAY and "source_batch" in issue.parameters:
        source_batch = int(issue.parameters["source_batch"])
        keys = [key for key in keys if ((key - 1) % spec.timeline.batches) + 1 == source_batch]
    ranked = sorted(keys, key=lambda key: derive_seed(salt, key))
    return ranked[:count]


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
