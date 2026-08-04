from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scenario_data_factory.exceptions import IssueInjectionError, IssuePlanningError
from scenario_data_factory.models.scenario import IssueSpec, ScenarioSpec


@dataclass(frozen=True)
class IssueTarget:
    issue_id: str
    issue_type: str
    table: str
    column: str | None
    record_key: int
    batch_id: int
    details: dict[str, Any]


class IssuePlugin:
    issue_type: str
    requires_raw_output: bool = False

    def validate(self, spec: ScenarioSpec, issue: IssueSpec) -> None:
        if issue.table not in {t.name for t in spec.tables}:
            raise IssuePlanningError(
                "ISSUE_TABLE_MISSING",
                f"Issue references unknown table {issue.table}.",
                scenario_id=spec.scenario_id,
            )

    def apply_spark(
        self, df: Any, issue: IssueSpec, targets: list[IssueTarget], key_column: str
    ) -> Any:
        raise IssueInjectionError(
            "PLUGIN_NOT_IMPLEMENTED",
            f"{self.issue_type} does not implement Spark injection.",
            technical_detail=self.__class__.__name__,
        )


class MetadataOnlyIssuePlugin(IssuePlugin):
    def apply_spark(
        self, df: Any, issue: IssueSpec, targets: list[IssueTarget], key_column: str
    ) -> Any:
        return df
