from __future__ import annotations

import re
from typing import Any


def write_delta_tables(
    dataframes: dict[str, Any],
    catalog: str,
    schema: str,
    quality_suffix: str,
    *,
    namespace: str | None = None,
) -> list[str]:
    written: list[str] = []
    for table, df in dataframes.items():
        name_parts = [
            *([_safe_identifier(namespace)] if namespace else []),
            _safe_identifier(table),
            *([_safe_identifier(quality_suffix)] if quality_suffix else []),
        ]
        full_name = f"`{catalog}`.`{schema}`.`{'_'.join(name_parts)}`"
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
