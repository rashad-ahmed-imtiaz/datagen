from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = str
_UC_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class LifecycleState(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    PREVIEW_PREPARED = "preview_prepared"
    GENERATION_PREPARED = "generation_prepared"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ColumnType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    LONG = "long"
    DECIMAL = "decimal"
    DATE = "date"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"


class OutputMode(str, Enum):
    DELTA = "delta"
    RAW = "raw"
    BOTH = "both"


class IssueType(str, Enum):
    NULL_VALUE = "null_value"
    BLANK_VALUE = "blank_value"
    DUPLICATE_RECORD = "duplicate_record"
    INVALID_FORMAT = "invalid_format"
    INVALID_VALUE = "invalid_value"
    REFERENTIAL_ORPHAN = "referential_orphan"
    DATE_RULE_VIOLATION = "date_rule_violation"
    LATE_ARRIVAL = "late_arrival"
    OUT_OF_ORDER = "out_of_order"
    FILE_REPLAY = "file_replay"
    SCHEMA_DRIFT = "schema_drift"
    CORRELATED_MISSINGNESS = "correlated_missingness"


class ColumnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Identifier
    type: ColumnType
    nullable: bool = True
    primary_key: bool = False
    faker: str | None = None
    values: list[Any] | None = None
    weights: list[float] | None = None
    semantic: dict[str, Any] | None = None
    min_value: int | Decimal | None = None
    max_value: int | Decimal | None = None
    precision: int | None = None
    scale: int | None = None

    @field_validator("name")
    @classmethod
    def valid_identifier(cls, value: str) -> str:
        if len(value) > 128 or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(f"invalid identifier: {value}")
        return value

    @model_validator(mode="after")
    def round_decimal_bounds(self) -> ColumnSpec:
        if self.type == ColumnType.DECIMAL and self.scale is not None:
            quant = Decimal("1").scaleb(-self.scale)
            for attr in ("min_value", "max_value"):
                val = getattr(self, attr)
                if isinstance(val, Decimal):
                    setattr(self, attr, val.quantize(quant, rounding=ROUND_HALF_UP))
        return self

    @model_validator(mode="after")
    def validate_weighted_values(self) -> ColumnSpec:
        if self.weights is None:
            return self
        if not self.values or len(self.weights) != len(self.values):
            raise ValueError("weights must align one-to-one with values")
        if any(weight <= 0 for weight in self.weights):
            raise ValueError("weights must be positive")
        return self


class TableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Identifier
    row_count: int = Field(gt=0)
    columns: list[ColumnSpec]
    source_systems: list[str] = Field(default_factory=list)
    batch_column: str = "batch_id"

    @field_validator("name")
    @classmethod
    def valid_identifier(cls, value: str) -> str:
        if len(value) > 128 or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(f"invalid table identifier: {value}")
        return value

    @model_validator(mode="after")
    def require_columns_and_one_pk(self) -> TableSpec:
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate column in table {self.name}")
        if sum(column.primary_key for column in self.columns) != 1:
            raise ValueError(f"table {self.name} must define exactly one primary key")
        return self

    def column_names(self) -> set[str]:
        return {c.name for c in self.columns}


class RelationshipSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Identifier
    parent_table: Identifier
    parent_column: Identifier
    child_table: Identifier
    child_column: Identifier
    required: bool = True
    parent_filter: dict[str, Any] | None = None
    constraints: dict[str, Any] | None = None


class TimelineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: date
    batches: int = Field(gt=0)
    frequency: Literal["daily", "monthly"] = "daily"


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: OutputMode = OutputMode.BOTH
    catalog: str | None = None
    schema_name: str | None = None
    delta_namespace: str | None = None
    # These are appended after the business table name, not prepended to it.
    # The defaults therefore produce <scenario>_<table> and <scenario>_<table>_dq.
    clean_delta_prefix: str = ""
    dirty_delta_prefix: str = "dq"
    control_volume: str = "sdf_control"
    raw_volume: str = "sdf_raw"
    local_root: str = "out"
    include_clean: bool = True
    raw_format: Literal["json", "csv"] = "json"
    manifest_detail: Literal["full", "summary"] = "full"

    @field_validator("catalog", "schema_name", "delta_namespace")
    @classmethod
    def valid_delta_identifier(cls, value: str | None) -> str | None:
        if value is not None and not _UC_IDENTIFIER.fullmatch(value):
            raise ValueError("catalog, schema, and Delta prefixes must be valid identifiers")
        return value

    @field_validator("clean_delta_prefix", "dirty_delta_prefix")
    @classmethod
    def valid_delta_suffix(cls, value: str) -> str:
        if value and not _UC_IDENTIFIER.fullmatch(value):
            raise ValueError("Delta quality suffixes must be valid identifiers")
        return value


class IssueSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(default_factory=lambda: f"iss_{uuid4().hex[:12]}")
    type: IssueType
    table: Identifier
    column: Identifier | None = None
    rate: float | None = Field(default=None, ge=0, le=1)
    exact_count: int | None = Field(default=None, ge=0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    correlation: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_rate_or_count(self) -> IssueSpec:
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
        if self.type in column_required_types and not self.column:
            raise ValueError(f"{self.type.value} requires a target column")
        file_count = self.parameters.get("file_count")
        if file_count is not None:
            if self.type != IssueType.FILE_REPLAY:
                raise ValueError("file_count is supported only by file_replay")
            if self.rate is not None or self.exact_count is not None:
                raise ValueError(
                    "file_replay file_count cannot be combined with rate or exact_count"
                )
            return self
        if self.rate is not None and self.exact_count is not None:
            raise ValueError("issue must define either rate or exact_count, not both")
        if self.rate is None and self.exact_count is None:
            raise ValueError("issue must define rate or exact_count")
        return self


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    scenario_id: str = Field(default_factory=lambda: f"scn_{uuid4().hex[:12]}")
    name: str
    domain: str
    seed: int = 42
    locale: str = "en_CA"
    lifecycle_state: LifecycleState = LifecycleState.DRAFT
    revision: int = 1
    timeline: TimelineSpec
    tables: list[TableSpec]
    relationships: list[RelationshipSpec] = Field(default_factory=list)
    issues: list[IssueSpec] = Field(default_factory=list)
    outputs: OutputSpec = Field(default_factory=OutputSpec)
    metadata: dict[str, Any] = Field(default_factory=lambda: {"synthetic_data": True})

    @field_validator("scenario_id")
    @classmethod
    def valid_scenario_id(cls, value: str) -> str:
        if not re.fullmatch(r"scn_[A-Za-z0-9]+", value):
            raise ValueError("scenario_id must use the scn_<alphanumeric> format")
        return value

    @field_validator("name")
    @classmethod
    def valid_scenario_name(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 120:
            raise ValueError("scenario name must be between 1 and 120 characters")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> ScenarioSpec:
        tables = {t.name: t for t in self.tables}
        if len(tables) != len(self.tables):
            raise ValueError("duplicate table names")
        issue_ids = [issue.issue_id for issue in self.issues]
        if len(issue_ids) != len(set(issue_ids)):
            raise ValueError("duplicate issue IDs")
        relationship_targets: set[tuple[str, str]] = set()
        for rel in self.relationships:
            if rel.parent_table not in tables:
                raise ValueError(f"relationship {rel.name} references missing parent table")
            if rel.child_table not in tables:
                raise ValueError(f"relationship {rel.name} references missing child table")
            if rel.parent_column not in tables[rel.parent_table].column_names():
                raise ValueError(f"relationship {rel.name} references missing parent column")
            if rel.child_column not in tables[rel.child_table].column_names():
                raise ValueError(f"relationship {rel.name} references missing child column")
            parent_column = next(
                column
                for column in tables[rel.parent_table].columns
                if column.name == rel.parent_column
            )
            child_column = next(
                column
                for column in tables[rel.child_table].columns
                if column.name == rel.child_column
            )
            if parent_column.type != child_column.type:
                raise ValueError(f"relationship {rel.name} key column types must match")
            if not parent_column.primary_key:
                raise ValueError(f"relationship {rel.name} parent column must be a primary key")
            target = (rel.child_table, rel.child_column)
            if target in relationship_targets:
                raise ValueError("multiple relationships cannot populate the same child column")
            relationship_targets.add(target)
            if rel.parent_filter:
                filter_column = rel.parent_filter.get("column")
                filter_values = rel.parent_filter.get("values")
                if filter_column not in tables[rel.parent_table].column_names():
                    raise ValueError(
                        f"relationship {rel.name} filter references missing parent column"
                    )
                if not isinstance(filter_values, list) or not filter_values:
                    raise ValueError(f"relationship {rel.name} filter requires values")
                filter_spec = next(
                    column
                    for column in tables[rel.parent_table].columns
                    if column.name == filter_column
                )
                if filter_spec.values and not set(filter_values).intersection(filter_spec.values):
                    raise ValueError(f"relationship {rel.name} filter cannot match parent values")
            constraints = rel.constraints or {}
            for rule in constraints.get("child_date_ranges", []):
                if not isinstance(rule, dict):
                    raise ValueError(f"relationship {rel.name} has an invalid date-range rule")
                if rule.get("child_column") not in tables[rel.child_table].column_names():
                    raise ValueError(f"relationship {rel.name} date-range child column is missing")
                for key in ("parent_start_column", "parent_end_column"):
                    if rule.get(key) not in tables[rel.parent_table].column_names():
                        raise ValueError(
                            f"relationship {rel.name} date-range parent column is missing"
                        )
                child = next(
                    column
                    for column in tables[rel.child_table].columns
                    if column.name == rule["child_column"]
                )
                parent_types = {
                    next(
                        column
                        for column in tables[rel.parent_table].columns
                        if column.name == rule[key]
                    ).type
                    for key in ("parent_start_column", "parent_end_column")
                }
                temporal = {ColumnType.DATE, ColumnType.TIMESTAMP}
                if child.type not in temporal or not parent_types.issubset(temporal):
                    raise ValueError(f"relationship {rel.name} date-range columns must be temporal")
                if len(parent_types) != 1 or child.type not in parent_types:
                    raise ValueError(f"relationship {rel.name} date-range column types must match")
            for rule in constraints.get("aggregate_caps", []):
                if not isinstance(rule, dict):
                    raise ValueError(f"relationship {rel.name} has an invalid aggregate-cap rule")
                if rule.get("child_amount_column") not in tables[rel.child_table].column_names():
                    raise ValueError(
                        f"relationship {rel.name} aggregate-cap child column is missing"
                    )
                if rule.get("parent_amount_column") not in tables[rel.parent_table].column_names():
                        raise ValueError(
                            f"relationship {rel.name} aggregate-cap parent column is missing"
                        )
                child = next(
                    column
                    for column in tables[rel.child_table].columns
                    if column.name == rule["child_amount_column"]
                )
                parent = next(
                    column
                    for column in tables[rel.parent_table].columns
                    if column.name == rule["parent_amount_column"]
                )
                numeric = {ColumnType.INTEGER, ColumnType.LONG, ColumnType.DECIMAL}
                if child.type not in numeric or parent.type not in numeric:
                    raise ValueError(
                        f"relationship {rel.name} aggregate-cap columns must be numeric"
                    )
                maximum_fraction = rule.get("maximum_fraction", 1.0)
                if (
                    not isinstance(maximum_fraction, (int, float))
                    or isinstance(maximum_fraction, bool)
                    or not 0 < maximum_fraction <= 1
                ):
                    raise ValueError(
                        f"relationship {rel.name} aggregate cap must have fraction in (0, 1]"
                    )
        for issue in self.issues:
            if issue.table not in tables:
                raise ValueError(f"issue {issue.issue_id} references missing table {issue.table}")
            if issue.column and issue.column not in tables[issue.table].column_names():
                raise ValueError(f"issue {issue.issue_id} references missing column {issue.column}")
            if issue.exact_count is not None and issue.exact_count > tables[issue.table].row_count:
                raise ValueError(f"issue {issue.issue_id} exact_count exceeds table row count")
        if self.metadata.get("synthetic_data") is not True:
            raise ValueError("metadata.synthetic_data must be true")
        return self

    def table(self, name: str) -> TableSpec:
        return next(t for t in self.tables if t.name == name)

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def spec_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.canonical_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> ScenarioSpec:
        return cls.model_validate(yaml.safe_load(text))

    @classmethod
    def from_json(cls, text: str) -> ScenarioSpec:
        return cls.model_validate_json(text)

    def transition(self, target: LifecycleState) -> ScenarioSpec:
        allowed = {
            LifecycleState.DRAFT: {LifecycleState.VALIDATED, LifecycleState.FAILED},
            LifecycleState.VALIDATED: {
                LifecycleState.PREVIEW_PREPARED,
                LifecycleState.GENERATION_PREPARED,
                LifecycleState.DRAFT,
            },
            LifecycleState.PREVIEW_PREPARED: {LifecycleState.RUNNING, LifecycleState.VALIDATED},
            LifecycleState.GENERATION_PREPARED: {LifecycleState.RUNNING, LifecycleState.VALIDATED},
            LifecycleState.RUNNING: {LifecycleState.SUCCEEDED, LifecycleState.FAILED},
            LifecycleState.SUCCEEDED: set(),
            LifecycleState.FAILED: {LifecycleState.DRAFT},
        }
        current = LifecycleState(self.lifecycle_state)
        if target not in allowed[current]:
            raise ValueError(f"invalid lifecycle transition {current.value} -> {target.value}")
        return self.model_copy(update={"lifecycle_state": target, "revision": self.revision + 1})
