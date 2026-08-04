from __future__ import annotations

from scenario_data_factory.compiler.validation import validate_scenario
from scenario_data_factory.models.scenario import IssueType, OutputMode, ScenarioSpec


def valid_spec(spec: ScenarioSpec) -> bool:
    validate_scenario(spec)
    return True


def supported_issues_only(spec: ScenarioSpec) -> bool:
    return all(IssueType(issue.type) in set(IssueType) for issue in spec.issues)


def references_exist(spec: ScenarioSpec) -> bool:
    tables = {table.name: table for table in spec.tables}
    return all(
        issue.table in tables
        and (issue.column is None or issue.column in tables[issue.table].column_names())
        for issue in spec.issues
    )


def required_tables_present(spec: ScenarioSpec, required_tables: list[str]) -> bool:
    table_names = {table.name for table in spec.tables}
    return set(required_tables).issubset(table_names)


def required_issues_present(spec: ScenarioSpec, required_issues: list[str]) -> bool:
    issue_types = {IssueType(issue.type).value for issue in spec.issues}
    return set(required_issues).issubset(issue_types)


def output_mode_correct(spec: ScenarioSpec) -> bool:
    physical = {IssueType.FILE_REPLAY, IssueType.SCHEMA_DRIFT, IssueType.OUT_OF_ORDER}
    if any(IssueType(issue.type) in physical for issue in spec.issues):
        return OutputMode(spec.outputs.mode) in {OutputMode.RAW, OutputMode.BOTH}
    return True


def seed_preserved(before: ScenarioSpec, after: ScenarioSpec) -> bool:
    return before.seed == after.seed


def confirmation_guard_respected(tool_names: list[str]) -> bool:
    return "submit_generation" not in tool_names and "run_code" not in tool_names


def no_arbitrary_code(response_text: str) -> bool:
    banned = ["spark.sql(", "exec(", "eval(", "dbutils.fs.rm"]
    return not any(token in response_text for token in banned)


def scenario_focus(response_text: str) -> bool:
    off_topic = ["pipeline repair", "observability platform", "real customer data"]
    return not any(token in response_text.lower() for token in off_topic)
