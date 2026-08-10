from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scenario_data_factory.models.scenario import ColumnSpec, ColumnType, TableSpec


@dataclass(frozen=True)
class DatagenColumn:
    name: str
    dtype: str
    options: dict[str, Any]


@dataclass(frozen=True)
class DatagenTableSpec:
    name: str
    rows: int
    columns: list[DatagenColumn]


def compile_table(table: TableSpec) -> DatagenTableSpec:
    return DatagenTableSpec(
        name=table.name,
        rows=table.row_count,
        columns=[compile_column(col) for col in table.columns],
    )


def compile_column(column: ColumnSpec) -> DatagenColumn:
    dtype = {
        ColumnType.STRING: "string",
        ColumnType.INTEGER: "integer",
        ColumnType.LONG: "long",
        ColumnType.DECIMAL: "decimal",
        ColumnType.DATE: "date",
        ColumnType.TIMESTAMP: "timestamp",
        ColumnType.BOOLEAN: "boolean",
    }[ColumnType(column.type)]
    options: dict[str, Any] = {}
    if column.primary_key:
        options["expr"] = "id + 1"
    elif column.faker:
        # Resolved by the engine, where the scenario locale is available.
        options["faker"] = column.faker
    elif column.values:
        options["values"] = column.values
        if column.weights:
            total = sum(column.weights)
            options["weights"] = [
                max(1, round(float(weight) / total * 10_000)) for weight in column.weights
            ]
    elif column.min_value is not None or column.max_value is not None:
        if column.min_value is not None:
            options["minValue"] = column.min_value
        if column.max_value is not None:
            options["maxValue"] = column.max_value
    elif column.type == ColumnType.DATE:
        options["expr"] = "date_add(DATE'2026-01-01', int(id % 365))"
    elif column.type == ColumnType.TIMESTAMP:
        options["expr"] = "timestampadd(HOUR, int(id % 720), TIMESTAMP'2026-01-01 00:00:00')"
    elif column.type == ColumnType.DECIMAL:
        options["expr"] = "cast(((id % 100000) / 100.0) as decimal(12,2))"
    elif column.type == ColumnType.BOOLEAN:
        options["expr"] = "id % 2 = 0"
    elif column.type == ColumnType.STRING:
        options["expr"] = f"concat('{column.name}_', cast(id as string))"
    return DatagenColumn(name=column.name, dtype=dtype, options=options)
