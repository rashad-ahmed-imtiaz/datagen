from __future__ import annotations

from typing import Any

from scenario_data_factory.issues.base import IssuePlugin, IssueTarget
from scenario_data_factory.models.scenario import IssueSpec


class DuplicateRecordPlugin(IssuePlugin):
    issue_type = "duplicate_record"

    def apply_spark(
        self, df: Any, issue: IssueSpec, targets: list[IssueTarget], key_column: str
    ) -> Any:
        from pyspark.sql import functions as F

        keys = [t.record_key for t in targets if t.issue_id == issue.issue_id]
        duplicates = (
            df.where(F.col(key_column).isin(keys))
            .withColumn("_sdf_duplicate_of", F.col(key_column))
            .withColumn("_sdf_issue_id", F.lit(issue.issue_id))
        )
        return df.unionByName(duplicates, allowMissingColumns=True)
