from __future__ import annotations

import re
from typing import Any


def write_delta_tables(
    dataframes: dict[str, Any],
    catalog: str,
    schema: str,
    prefix: str,
    *,
    namespace: str | None = None,
) -> list[str]:
    written: list[str] = []
    table_prefix = _safe_identifier(prefix)
    if namespace:
        table_prefix = f"{table_prefix}_{_safe_identifier(namespace)}"
    for table, df in dataframes.items():
        full_name = f"`{catalog}`.`{schema}`.`{table_prefix}_{_safe_identifier(table)}`"
        df.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(full_name)
        written.append(full_name)
    return written


def _safe_identifier(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    if not identifier:
        identifier = "table"
    if identifier[0].isdigit():
        identifier = f"_{identifier}"
    return identifier
