from __future__ import annotations

from typing import Any

from scenario_data_factory.issues.base import IssuePlugin, IssueTarget
from scenario_data_factory.models.scenario import IssueSpec


class InvalidValuePlugin(IssuePlugin):
    issue_type = "invalid_value"

    def apply_spark(
        self, df: Any, issue: IssueSpec, targets: list[IssueTarget], key_column: str
    ) -> Any:
        from pyspark.sql import functions as F
        from pyspark.sql.types import BooleanType, DateType, NumericType, TimestampType

        data_type = df.schema[issue.column].dataType
        configured_value = issue.parameters.get("value")

        # An invalid value must still be representable by the physical Delta type.
        # For temporal fields, a future value is an executable impossible-value defect.
        if isinstance(data_type, TimestampType):
            replacement = F.expr("timestampadd(DAY, 3650, current_timestamp())")
        elif isinstance(data_type, DateType):
            replacement = F.date_add(F.current_date(), 3650)
        elif isinstance(data_type, NumericType):
            value = configured_value if isinstance(configured_value, (int, float)) else -1
            replacement = F.lit(value).cast(data_type.simpleString())
        elif isinstance(data_type, BooleanType):
            replacement = F.lit(False)
        else:
            values = issue.parameters.get("invalid_values")
            value = configured_value
            if value is None and isinstance(values, list) and values:
                value = values[0]
            replacement = F.lit(value if value is not None else "__INVALID__").cast(
                data_type.simpleString()
            )

        keys = [t.record_key for t in targets if t.issue_id == issue.issue_id]
        return df.withColumn(
            issue.column,
            F.when(F.col(key_column).isin(keys), replacement).otherwise(F.col(issue.column)),
        )
