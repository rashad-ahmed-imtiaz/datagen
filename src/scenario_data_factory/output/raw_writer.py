from __future__ import annotations

from pathlib import Path
from typing import Any


def write_raw_batches(dataframes: dict[str, Any], root: Path, fmt: str = "json") -> list[str]:
    written: list[str] = []
    for table, df in dataframes.items():
        target = root / table
        writer = df.write.mode("overwrite").partitionBy("batch_id")
        if fmt == "csv":
            writer.option("header", True).csv(str(target))
        else:
            writer.json(str(target))
        written.append(str(target))
    return written
