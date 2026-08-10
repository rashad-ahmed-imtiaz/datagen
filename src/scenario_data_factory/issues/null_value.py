from __future__ import annotations

from typing import Any

from scenario_data_factory.issues.base import IssuePlugin, IssueTarget
from scenario_data_factory.models.scenario import IssueSpec


class NullValuePlugin(IssuePlugin):
    issue_type = "null_value"

    def apply_spark(
        self, df: Any, issue: IssueSpec, targets: list[IssueTarget], key_column: str
    ) -> Any:
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        keys = [t.record_key for t in targets if t.issue_id == issue.issue_id]
        correlation = issue.correlation or issue.parameters.get("correlation") or {}
        where = correlation.get("where") if isinstance(correlation, dict) else None
        share = correlation.get("share") if isinstance(correlation, dict) else None
        if isinstance(where, dict) and isinstance(share, (int, float)) and 0 < share < 1:
            source_column = where.get("source_column", "source_system")
            source_value = where.get("source_value", where.get("source_system"))
            after_batch = where.get("after_batch")
            eligible = F.col(source_column) == F.lit(source_value)
            if isinstance(after_batch, int):
                eligible = eligible & (F.col("batch_id") > F.lit(after_batch))
            correlated_count = round(len(keys) * float(share))
            uncorrelated_count = len(keys) - correlated_count
            ranked = df.withColumn("_sdf_correlation_eligible", eligible)
            ranked = ranked.withColumn(
                "_sdf_correlation_rank",
                F.row_number().over(
                    Window.partitionBy("_sdf_correlation_eligible").orderBy(
                        F.xxhash64(F.col(key_column), F.lit(issue.issue_id))
                    )
                ),
            )
            missing = (
                (
                    F.col("_sdf_correlation_eligible")
                    & (F.col("_sdf_correlation_rank") <= correlated_count)
                )
                | (
                    ~F.col("_sdf_correlation_eligible")
                    & (F.col("_sdf_correlation_rank") <= uncorrelated_count)
                )
            )
            return (
                ranked.withColumn(
                    issue.column,
                    F.when(missing, F.lit(None)).otherwise(F.col(issue.column)),
                )
                .drop(
                    "_sdf_correlation_eligible",
                    "_sdf_correlation_rank",
                )
            )
        return df.withColumn(
            issue.column,
            F.when(F.col(key_column).isin(keys), F.lit(None)).otherwise(F.col(issue.column)),
        )
