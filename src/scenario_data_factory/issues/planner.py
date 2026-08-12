from __future__ import annotations

from collections import defaultdict

from scenario_data_factory.issues.base import IssueTarget
from scenario_data_factory.issues.registry import ISSUE_REGISTRY
from scenario_data_factory.issues.targeting import (
    iter_record_keys,
    plan_targets,
    resolve_issue_count,
)
from scenario_data_factory.models.scenario import IssueType, ScenarioSpec


def estimate_issues(spec: ScenarioSpec) -> dict[str, int]:
    return {
        issue.issue_id: len(plan_targets(spec, issue, spec.table(issue.table)))
        for issue in spec.issues
    }


def build_issue_plan(spec: ScenarioSpec) -> dict[str, list[IssueTarget]]:
    planned: dict[str, list[IssueTarget]] = defaultdict(list)
    occupied: set[tuple[str, int]] = set()
    # Reserve physical batches before row mutations. Otherwise a replay can copy a
    # separately injected duplicate and make the duplicate manifest understate its
    # actual effect. Keep declaration order within each priority for determinism.
    priority = {
        IssueType.FILE_REPLAY: 0,
        IssueType.OUT_OF_ORDER: 1,
        IssueType.DUPLICATE_RECORD: 2,
    }
    ordered_issues = sorted(
        enumerate(spec.issues), key=lambda item: (priority.get(IssueType(item[1].type), 3), item[0])
    )
    for _, issue in ordered_issues:
        plugin = ISSUE_REGISTRY[IssueType(issue.type)]
        plugin.validate(spec, issue)
        table = spec.table(issue.table)
        requested = resolve_issue_count(issue, table, batches=spec.timeline.batches)
        selected: list[int] = []
        for record_key in iter_record_keys(spec, issue, table):
            slot = (table.name, record_key)
            if slot in occupied:
                continue
            selected.append(record_key)
            occupied.add(slot)
            if len(selected) == requested:
                break
        if len(selected) != requested:
            raise ValueError(
                f"Issue {issue.issue_id} cannot reserve {requested} distinct targets on "
                f"{table.name}.{issue.column}."
            )
        batches = spec.timeline.batches
        for record_key in selected:
            batch_id = ((record_key - 1) % batches) + 1
            if IssueType(issue.type) == IssueType.FILE_REPLAY:
                batch_id = int(issue.parameters.get("source_batch", batch_id))
            planned[table.name].append(
                IssueTarget(
                    issue_id=issue.issue_id,
                    issue_type=IssueType(issue.type).value,
                    table=table.name,
                    column=issue.column,
                    record_key=record_key,
                    batch_id=batch_id,
                    details=dict(issue.parameters),
                )
            )
    return dict(planned)
