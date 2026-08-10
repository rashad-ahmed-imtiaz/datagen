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
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
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
            self.weights = None
            return self
        if any(weight <= 0 for weight in self.weights):
            self.weights = None
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
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(f"invalid table identifier: {value}")
        return value

    @model_validator(mode="after")
    def require_columns_and_one_pk(self) -> TableSpec:
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate column in table {self.name}")
        if not any(c.primary_key for c in self.columns):
            raise ValueError(f"table {self.name} must define a primary key")
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
    clean_delta_prefix: str = "clean"
    dirty_delta_prefix: str = "dirty"
    control_volume: str = "sdf_control"
    raw_volume: str = "sdf_raw"
    local_root: str = "out"
    include_clean: bool = True
    raw_format: Literal["json", "csv"] = "json"
    manifest_detail: Literal["full", "summary"] = "full"


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
        if self.type == IssueType.FILE_REPLAY and self.parameters.get("file_count") is not None:
            return self
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

    @model_validator(mode="after")
    def validate_references(self) -> ScenarioSpec:
        tables = {t.name: t for t in self.tables}
        if len(tables) != len(self.tables):
            raise ValueError("duplicate table names")
        for rel in self.relationships:
            if rel.parent_table not in tables:
                raise ValueError(f"relationship {rel.name} references missing parent table")
            if rel.child_table not in tables:
                raise ValueError(f"relationship {rel.name} references missing child table")
            if rel.parent_column not in tables[rel.parent_table].column_names():
                raise ValueError(f"relationship {rel.name} references missing parent column")
            if rel.child_column not in tables[rel.child_table].column_names():
                raise ValueError(f"relationship {rel.name} references missing child column")
            if rel.parent_filter:
                filter_column = rel.parent_filter.get("column")
                filter_values = rel.parent_filter.get("values")
                if filter_column not in tables[rel.parent_table].column_names():
                    raise ValueError(
                        f"relationship {rel.name} filter references missing parent column"
                    )
                if not isinstance(filter_values, list) or not filter_values:
                    raise ValueError(f"relationship {rel.name} filter requires values")
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
        for issue in self.issues:
            if issue.table not in tables:
                raise ValueError(f"issue {issue.issue_id} references missing table {issue.table}")
            if issue.column and issue.column not in tables[issue.table].column_names():
                raise ValueError(f"issue {issue.issue_id} references missing column {issue.column}")
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
