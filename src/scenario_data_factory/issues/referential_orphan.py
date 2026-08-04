from __future__ import annotations

from typing import Any

from scenario_data_factory.issues.base import IssuePlugin, IssueTarget
from scenario_data_factory.models.scenario import IssueSpec


class ReferentialOrphanPlugin(IssuePlugin):
    issue_type = "referential_orphan"

    def apply_spark(
        self, df: Any, issue: IssueSpec, targets: list[IssueTarget], key_column: str
    ) -> Any:
        from pyspark.sql import functions as F

        offset = int(issue.parameters.get("orphan_offset", 9_000_000_000))
        keys = [t.record_key for t in targets if t.issue_id == issue.issue_id]
        return df.withColumn(
            issue.column,
            F.when(F.col(key_column).isin(keys), F.col(issue.column) + F.lit(offset)).otherwise(
                F.col(issue.column)
            ),
        )
