from __future__ import annotations

from typing import Any

from scenario_data_factory.issues.base import IssueTarget
from scenario_data_factory.models.scenario import IssueType, ScenarioSpec


def apply_physical_raw_issues(
    dataframes: dict[str, Any], spec: ScenarioSpec, plan: dict[str, list[IssueTarget]]
) -> dict[str, Any]:
    result = dict(dataframes)
    for issue in spec.issues:
        issue_type = IssueType(issue.type)
        if issue_type == IssueType.FILE_REPLAY:
            result[issue.table] = _apply_file_replay(
                result[issue.table], issue.issue_id, issue.parameters, plan.get(issue.table, [])
            )
        elif issue_type == IssueType.SCHEMA_DRIFT:
            result[issue.table] = _apply_schema_drift(result[issue.table], issue.parameters)
        elif issue_type == IssueType.OUT_OF_ORDER:
            result[issue.table] = _apply_out_of_order(result[issue.table], issue.parameters)
    return result


def _apply_file_replay(
    df: Any, issue_id: str, parameters: dict[str, Any], targets: list[IssueTarget]
) -> Any:
    from pyspark.sql import functions as F

    target_batch = int(parameters["target_batch"])
    keys = [target.record_key for target in targets if target.issue_id == issue_id]
    replay = (
        df.where(F.col("_sdf_record_key").isin(keys))
        .withColumn("batch_id", F.lit(target_batch))
        .withColumn("_sdf_replayed_from_batch", F.lit(int(parameters["source_batch"])))
        .withColumn("_sdf_issue_id", F.lit(issue_id))
    )
    return df.unionByName(replay, allowMissingColumns=True)


def _apply_schema_drift(df: Any, parameters: dict[str, Any]) -> Any:
    from pyspark.sql import functions as F

    activation_batch = int(parameters["activation_batch"])
    drifted = df
    for column in parameters.get("add_columns", []):
        name = column["name"]
        dtype = column.get("type", "string")
        value = F.when(
            F.col("batch_id") >= activation_batch, _default_value(name, dtype)
        ).otherwise(F.lit(None))
        drifted = drifted.withColumn(name, value)
    return drifted


def _apply_out_of_order(df: Any, parameters: dict[str, Any]) -> Any:
    from pyspark.sql import functions as F

    source_batch = int(parameters.get("source_batch", 2))
    emitted_batch = int(parameters.get("emitted_batch", 1))
    return df.withColumn(
        "_sdf_emitted_batch_id",
        F.when(F.col("batch_id") == source_batch, F.lit(emitted_batch)).otherwise(
            F.col("batch_id")
        ),
    )


def _default_value(name: str, dtype: str) -> Any:
    from pyspark.sql import functions as F

    if dtype in {"double", "float"}:
        return ((F.col("_sdf_record_key") % F.lit(100)) / F.lit(100)).cast("double")
    if dtype == "decimal":
        return ((F.col("_sdf_record_key") % F.lit(100)) / F.lit(100)).cast("decimal(5,2)")
    if dtype in {"integer", "long"}:
        return (F.col("_sdf_record_key") % F.lit(100)).cast("long")
    return F.concat(F.lit(f"{name}_"), F.col("_sdf_record_key").cast("string"))
