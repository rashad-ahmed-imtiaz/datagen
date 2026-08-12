from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scenario_data_factory.exceptions import IssueInjectionError, IssuePlanningError
from scenario_data_factory.models.scenario import ColumnType, IssueSpec, IssueType, ScenarioSpec


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
        tables = {table.name: table for table in spec.tables}
        if issue.table not in tables:
            raise IssuePlanningError(
                "ISSUE_TABLE_MISSING",
                f"Issue references unknown table {issue.table}.",
                scenario_id=spec.scenario_id,
            )
        column_required_types = {
            IssueType.NULL_VALUE,
            IssueType.BLANK_VALUE,
            IssueType.INVALID_FORMAT,
            IssueType.INVALID_VALUE,
            IssueType.REFERENTIAL_ORPHAN,
            IssueType.DATE_RULE_VIOLATION,
            IssueType.LATE_ARRIVAL,
            IssueType.CORRELATED_MISSINGNESS,
        }
        table = tables[issue.table]
        columns = {column.name: column for column in table.columns}
        if IssueType(issue.type) in column_required_types and not issue.column:
            raise IssuePlanningError(
                "ISSUE_COLUMN_REQUIRED",
                f"{issue.type} on {issue.table} requires a target column.",
                scenario_id=spec.scenario_id,
            )
        if issue.column and issue.column not in columns:
            raise IssuePlanningError(
                "ISSUE_COLUMN_MISSING",
                f"{issue.type} references missing column {issue.table}.{issue.column}.",
                scenario_id=spec.scenario_id,
            )

        issue_type = IssueType(issue.type)
        target = columns.get(issue.column or "")
        if issue_type in {IssueType.BLANK_VALUE, IssueType.INVALID_FORMAT} and (
            target is None or target.type != ColumnType.STRING
        ):
            raise IssuePlanningError(
                "ISSUE_COLUMN_TYPE_INVALID",
                f"{issue.type} requires a string target column.",
                scenario_id=spec.scenario_id,
            )
        if issue_type == IssueType.REFERENTIAL_ORPHAN and (
            target is None or target.type not in {ColumnType.INTEGER, ColumnType.LONG}
        ):
            raise IssuePlanningError(
                "ISSUE_COLUMN_TYPE_INVALID",
                "referential_orphan requires an integer or long foreign-key column.",
                scenario_id=spec.scenario_id,
            )
        if issue_type == IssueType.DATE_RULE_VIOLATION:
            after_column = issue.parameters.get("after_column")
            after = columns.get(after_column) if isinstance(after_column, str) else None
            if target is None or target.type not in {ColumnType.DATE, ColumnType.TIMESTAMP}:
                raise IssuePlanningError(
                    "ISSUE_COLUMN_TYPE_INVALID",
                    "date_rule_violation requires a date or timestamp target column.",
                    scenario_id=spec.scenario_id,
                )
            if after is None or after.type not in {ColumnType.DATE, ColumnType.TIMESTAMP}:
                raise IssuePlanningError(
                    "ISSUE_PARAMETER_INVALID",
                    (
                        "date_rule_violation requires after_column to reference a date or "
                        "timestamp column."
                    ),
                    scenario_id=spec.scenario_id,
                )
        if issue_type == IssueType.LATE_ARRIVAL:
            arrival_column = issue.parameters.get("arrival_column") or issue.column
            arrival = columns.get(arrival_column) if isinstance(arrival_column, str) else None
            if arrival is None or arrival.type not in {ColumnType.DATE, ColumnType.TIMESTAMP}:
                raise IssuePlanningError(
                    "ISSUE_PARAMETER_INVALID",
                    "late_arrival requires arrival_column to reference a date or timestamp column.",
                    scenario_id=spec.scenario_id,
                )
        if issue_type in {IssueType.FILE_REPLAY, IssueType.SCHEMA_DRIFT}:
            batch_parameters = (
                ("source_batch", "target_batch")
                if issue_type == IssueType.FILE_REPLAY
                else ("activation_batch",)
            )
            for parameter in batch_parameters:
                value = issue.parameters.get(parameter)
                if value is not None and (
                    not isinstance(value, int) or not 1 <= value <= spec.timeline.batches
                ):
                    raise IssuePlanningError(
                        "ISSUE_BATCH_INVALID",
                        f"{issue.type} {parameter} must be within the configured timeline batches.",
                        scenario_id=spec.scenario_id,
                    )
        if issue_type == IssueType.FILE_REPLAY:
            source_batch = issue.parameters.get("source_batch")
            target_batch = issue.parameters.get("target_batch")
            if (
                isinstance(source_batch, int)
                and isinstance(target_batch, int)
                and target_batch <= source_batch
            ):
                raise IssuePlanningError(
                    "ISSUE_BATCH_INVALID",
                    "file_replay target_batch must be later than source_batch.",
                    scenario_id=spec.scenario_id,
                )
        if issue_type in {IssueType.NULL_VALUE, IssueType.CORRELATED_MISSINGNESS}:
            correlation = issue.correlation or issue.parameters.get("correlation")
            where = correlation.get("where") if isinstance(correlation, dict) else None
            if isinstance(where, dict):
                source_column = where.get("source_column", "source_system")
                if source_column not in columns:
                    raise IssuePlanningError(
                        "ISSUE_PARAMETER_INVALID",
                        (
                            f"{issue.type} correlation references missing column "
                            f"{issue.table}.{source_column}."
                        ),
                        scenario_id=spec.scenario_id,
                    )
                after_batch = where.get("after_batch")
                if after_batch is not None and (
                    not isinstance(after_batch, int)
                    or not 1 <= after_batch <= spec.timeline.batches
                ):
                    raise IssuePlanningError(
                        "ISSUE_BATCH_INVALID",
                        (
                            f"{issue.type} correlation after_batch must be within the "
                            "configured timeline batches."
                        ),
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
