from __future__ import annotations

from scenario_data_factory.issues.base import IssuePlugin
from scenario_data_factory.issues.blank_value import BlankValuePlugin
from scenario_data_factory.issues.correlated_missingness import CorrelatedMissingnessPlugin
from scenario_data_factory.issues.date_rule_violation import DateRuleViolationPlugin
from scenario_data_factory.issues.duplicate_record import DuplicateRecordPlugin
from scenario_data_factory.issues.file_replay import FileReplayPlugin
from scenario_data_factory.issues.invalid_format import InvalidFormatPlugin
from scenario_data_factory.issues.invalid_value import InvalidValuePlugin
from scenario_data_factory.issues.late_arrival import LateArrivalPlugin
from scenario_data_factory.issues.null_value import NullValuePlugin
from scenario_data_factory.issues.out_of_order import OutOfOrderPlugin
from scenario_data_factory.issues.referential_orphan import ReferentialOrphanPlugin
from scenario_data_factory.issues.schema_drift import SchemaDriftPlugin
from scenario_data_factory.models.scenario import IssueType

ISSUE_REGISTRY: dict[IssueType, IssuePlugin] = {
    IssueType.NULL_VALUE: NullValuePlugin(),
    IssueType.BLANK_VALUE: BlankValuePlugin(),
    IssueType.DUPLICATE_RECORD: DuplicateRecordPlugin(),
    IssueType.INVALID_FORMAT: InvalidFormatPlugin(),
    IssueType.INVALID_VALUE: InvalidValuePlugin(),
    IssueType.REFERENTIAL_ORPHAN: ReferentialOrphanPlugin(),
    IssueType.DATE_RULE_VIOLATION: DateRuleViolationPlugin(),
    IssueType.LATE_ARRIVAL: LateArrivalPlugin(),
    IssueType.OUT_OF_ORDER: OutOfOrderPlugin(),
    IssueType.FILE_REPLAY: FileReplayPlugin(),
    IssueType.SCHEMA_DRIFT: SchemaDriftPlugin(),
    IssueType.CORRELATED_MISSINGNESS: CorrelatedMissingnessPlugin(),
}


def list_issue_types() -> list[dict[str, object]]:
    return [
        {
            "type": issue_type.value,
            "requires_raw_output": plugin.requires_raw_output,
            "plugin": plugin.__class__.__name__,
        }
        for issue_type, plugin in sorted(ISSUE_REGISTRY.items(), key=lambda item: item[0].value)
    ]
