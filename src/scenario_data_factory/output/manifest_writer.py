from __future__ import annotations

import json
from pathlib import Path

from scenario_data_factory.issues.base import IssueTarget


def manifest_rows(plan: dict[str, list[IssueTarget]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for table, targets in sorted(plan.items()):
        for target in targets:
            rows.append(
                {
                    "issue_id": target.issue_id,
                    "issue_type": target.issue_type,
                    "table": table,
                    "column": target.column,
                    "record_key": target.record_key,
                    "batch_id": target.batch_id,
                    "details": target.details,
                }
            )
    return rows


def write_manifest(path: Path, plan: dict[str, list[IssueTarget]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest_rows(plan), indent=2, sort_keys=True), encoding="utf-8")
    return path
