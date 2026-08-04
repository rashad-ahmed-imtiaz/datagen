from __future__ import annotations

from collections import defaultdict

from scenario_data_factory.issues.base import IssueTarget
from scenario_data_factory.issues.registry import ISSUE_REGISTRY
from scenario_data_factory.issues.targeting import plan_targets
from scenario_data_factory.models.scenario import IssueType, ScenarioSpec


def estimate_issues(spec: ScenarioSpec) -> dict[str, int]:
    return {
        issue.issue_id: len(plan_targets(spec, issue, spec.table(issue.table)))
        for issue in spec.issues
    }


def build_issue_plan(spec: ScenarioSpec) -> dict[str, list[IssueTarget]]:
    planned: dict[str, list[IssueTarget]] = defaultdict(list)
    occupied: set[tuple[str, str | None, int]] = set()
    for issue in spec.issues:
        plugin = ISSUE_REGISTRY[IssueType(issue.type)]
        plugin.validate(spec, issue)
        table = spec.table(issue.table)
        for target in plan_targets(spec, issue, table):
            slot = (target.table, target.column, target.record_key)
            if slot in occupied and target.column is not None:
                target = IssueTarget(
                    issue_id=target.issue_id,
                    issue_type=target.issue_type,
                    table=target.table,
                    column=target.column,
                    record_key=target.record_key,
                    batch_id=target.batch_id,
                    details={**target.details, "conflict": "shared_target"},
                )
            occupied.add(slot)
            planned[target.table].append(target)
    return dict(planned)
