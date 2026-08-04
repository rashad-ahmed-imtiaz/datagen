from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scenario_data_factory.output.manifest_writer import manifest_rows


def build_issue_summary(plan: dict[str, list[object]]) -> dict[str, object]:
    rows = manifest_rows(plan)
    by_type = Counter(row["issue_type"] for row in rows)
    by_table = Counter(row["table"] for row in rows)
    by_issue_id = Counter(row["issue_id"] for row in rows)
    explicit_duplicates = by_type.get("duplicate_record", 0)
    replay_duplicates = by_type.get("file_replay", 0)
    return {
        "manifest_count": len(rows),
        "issue_counts": dict(sorted(by_type.items())),
        "issue_id_counts": dict(sorted(by_issue_id.items())),
        "table_counts": dict(sorted(by_table.items())),
        "duplicate_effects": {
            "duplicate_record": explicit_duplicates,
            "file_replay": replay_duplicates,
            "total_duplicate_like_records": explicit_duplicates + replay_duplicates,
        },
    }


def write_summary(path: Path, summary: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path
