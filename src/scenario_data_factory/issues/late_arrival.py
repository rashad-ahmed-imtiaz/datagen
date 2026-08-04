from __future__ import annotations

from typing import Any

from scenario_data_factory.issues.base import IssuePlugin, IssueTarget
from scenario_data_factory.models.scenario import IssueSpec


class LateArrivalPlugin(IssuePlugin):
    issue_type = "late_arrival"

    def apply_spark(
        self, df: Any, issue: IssueSpec, targets: list[IssueTarget], key_column: str
    ) -> Any:
        from pyspark.sql import functions as F

        delay = int(issue.parameters.get("delay_days_max", issue.parameters.get("delay_days", 1)))
        target_column = issue.parameters.get("arrival_column") or issue.column
        keys = [t.record_key for t in targets if t.issue_id == issue.issue_id]
        return df.withColumn(
            target_column,
            F.when(
                F.col(key_column).isin(keys),
                F.col(target_column) + F.expr(f"INTERVAL {delay} DAYS"),
            ).otherwise(F.col(target_column)),
        )
