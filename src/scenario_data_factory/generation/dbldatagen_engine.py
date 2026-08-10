from __future__ import annotations

import math
from typing import Any

from scenario_data_factory.compiler.datagen_compiler import compile_table
from scenario_data_factory.compiler.dependency_graph import dependency_order
from scenario_data_factory.exceptions import DatagenCompilationError
from scenario_data_factory.generation.interface import BaselineGenerator
from scenario_data_factory.models.scenario import ScenarioSpec


class DbldatagenEngine(BaselineGenerator):
    """Baseline generator adapter. All direct dbldatagen API usage belongs here."""

    def generate(
        self, spark: Any, spec: ScenarioSpec, *, partitions: int | None = None
    ) -> dict[str, Any]:
        try:
            import dbldatagen as dg
        except Exception as exc:  # pragma: no cover - depends on Spark runtime
            raise DatagenCompilationError(
                "DBLDATAGEN_IMPORT_FAILED",
                "dbldatagen and PySpark must be importable to generate baseline data.",
                technical_detail=str(exc),
                scenario_id=spec.scenario_id,
                remediation="Install the pinned project dependencies or run on Databricks.",
            ) from exc
        _patch_dbldatagen_for_serverless_spark_connect(dg)

        generated: dict[str, Any] = {}
        order = dependency_order(spec.tables, spec.relationships)
        tables = {t.name: t for t in spec.tables}
        for table_name in order:
            table = tables[table_name]
            compiled = compile_table(table)
            try:
                data_spec = dg.DataGenerator(
                    spark,
                    name=compiled.name,
                    rows=compiled.rows,
                    partitions=partitions or min(max(compiled.rows // 100_000, 1), 64),
                ).withIdOutput()
                for col in compiled.columns:
                    options = self._datagen_options(col.options, spec.locale)
                    data_spec = data_spec.withColumn(col.name, col.dtype, **options)
                df = data_spec.build().drop("id")
                df = self._apply_timeline(df, table, spec)
                df = self._apply_agent_semantics(df, table, spec)
                df = self._apply_relationships(df, table_name, spec, generated)
                df = self._add_operational_columns(df, table)
                generated[table_name] = df
            except Exception as exc:
                raise DatagenCompilationError(
                    "DBLDATAGEN_BUILD_FAILED",
                    f"dbldatagen failed while building table {table_name}.",
                    technical_detail=str(exc),
                    scenario_id=spec.scenario_id,
                ) from exc
        return generated

    @staticmethod
    def _datagen_options(options: dict[str, Any], locale: str) -> dict[str, Any]:
        resolved = dict(options)
        provider = resolved.pop("faker", None)
        if provider:
            from dbldatagen.text_generator_plugins import FakerTextFactory

            resolved["text"] = FakerTextFactory(locale=locale.replace("-", "_"))(provider)
        return resolved

    @staticmethod
    def _apply_timeline(df: Any, table: Any, spec: ScenarioSpec) -> Any:
        """Anchor generated temporal values to the agent's requested timeline."""
        from pyspark.sql import functions as F

        record_key = next(column.name for column in table.columns if column.primary_key)
        if spec.timeline.frequency == "monthly":
            timeline_days = max(1, round(365 * spec.timeline.batches / 12))
        else:
            timeline_days = spec.timeline.batches
        start = F.lit(spec.timeline.start_date.isoformat()).cast("date")
        offset = F.pmod(F.col(record_key) - F.lit(1), F.lit(timeline_days)).cast("int")
        for column in table.columns:
            if column.type == "date":
                df = df.withColumn(column.name, F.date_add(start, offset))
            elif column.type == "timestamp":
                df = df.withColumn(
                    column.name,
                    F.to_timestamp(F.date_add(start, offset)),
                )
        return df

    @staticmethod
    def _apply_agent_semantics(df: Any, table: Any, spec: ScenarioSpec) -> Any:
        """Apply the bounded value rules returned by the schema-design agent."""
        from pyspark.sql import functions as F

        record_key = next(column.name for column in table.columns if column.primary_key)
        for column in table.columns:
            semantic = column.semantic or {}
            kind = semantic.get("kind")
            if kind == "date_offset":
                base_column = semantic.get("base_column")
                min_days = semantic.get("min_days", 0)
                max_days = semantic.get("max_days", min_days)
                if (
                    base_column not in table.column_names()
                    or not isinstance(min_days, int)
                    or not isinstance(max_days, int)
                    or max_days < min_days
                ):
                    continue
                span = max_days - min_days + 1
                offset = F.pmod(F.col(record_key), F.lit(span)).cast("int") + F.lit(min_days)
                if column.type == "date":
                    df = df.withColumn(column.name, F.date_add(F.col(base_column), offset))
                elif column.type == "timestamp":
                    df = df.withColumn(
                        column.name,
                        F.expr(
                            f"timestampadd(DAY, {min_days} + pmod({record_key}, {span}), "
                            f"{base_column})"
                        ),
                    )
                continue
            if kind == "log_normal":
                median = semantic.get("median")
                maximum = semantic.get("max")
                sigma = semantic.get("sigma", 1.0)
                if not all(isinstance(value, (int, float)) for value in (median, maximum, sigma)):
                    continue
                if median <= 0 or maximum <= 0 or sigma <= 0:
                    continue
                log_normal = F.exp(
                    F.lit(math.log(float(median))) + F.randn(spec.seed) * F.lit(float(sigma))
                )
                value = F.least(log_normal, F.lit(float(maximum)))
                scale = column.scale if column.scale is not None else 2
                precision = column.precision if column.precision is not None else 12
                df = df.withColumn(
                    column.name,
                    F.round(value, scale).cast(f"decimal({precision},{scale})"),
                )
                continue
            if kind in {"uniform_range", "normal"}:
                minimum = semantic.get("min")
                maximum = semantic.get("max")
                if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
                    continue
                if maximum < minimum:
                    continue
                if kind == "uniform_range":
                    value = F.lit(float(minimum)) + F.rand(spec.seed) * F.lit(
                        float(maximum) - float(minimum)
                    )
                else:
                    mean = semantic.get("mean", (float(minimum) + float(maximum)) / 2)
                    stddev = semantic.get("stddev", (float(maximum) - float(minimum)) / 6)
                    if not isinstance(mean, (int, float)) or not isinstance(stddev, (int, float)):
                        continue
                    value = F.least(
                        F.greatest(
                            F.lit(float(minimum)),
                            F.lit(float(mean)) + F.randn(spec.seed) * F.lit(float(stddev)),
                        ),
                        F.lit(float(maximum)),
                    )
                scale = column.scale if column.scale is not None else 2
                precision = column.precision if column.precision is not None else 12
                df = df.withColumn(
                    column.name,
                    F.round(value, scale).cast(f"decimal({precision},{scale})"),
                )
                continue
            key_column = semantic.get("key_column")
            values_by_key = semantic.get("values_by_key")
            if (
                kind != "lookup"
                or not isinstance(key_column, str)
                or not isinstance(values_by_key, dict)
            ):
                continue
            expression = F.col(column.name)
            for key, values in values_by_key.items():
                if not isinstance(values, list) or not values:
                    continue
                choices = F.array(*[F.lit(str(value)) for value in values])
                position = (
                    F.pmod(F.col(record_key), F.lit(len(values))) + F.lit(1)
                ).cast("int")
                expression = F.when(
                    F.col(key_column) == F.lit(str(key)), F.element_at(choices, position)
                ).otherwise(expression)
            df = df.withColumn(column.name, expression)
        return df

    def _apply_relationships(
        self, df: Any, table_name: str, spec: ScenarioSpec, generated: dict[str, Any]
    ) -> Any:
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        for rel in spec.relationships:
            if rel.child_table != table_name:
                continue
            child_pk = next(c.name for c in spec.table(table_name).columns if c.primary_key)
            parent_count = spec.table(rel.parent_table).row_count
            parent_filter = rel.parent_filter or {}
            if parent_filter:
                parent_column = parent_filter["column"]
                filter_values = parent_filter["values"]
                parent_keys = (
                    generated[rel.parent_table]
                    .where(F.col(parent_column).isin(filter_values))
                    .select(F.col(rel.parent_column).alias("_sdf_parent_key"))
                    .withColumn(
                        "_sdf_relationship_slot",
                        F.row_number().over(Window.orderBy("_sdf_parent_key")) - F.lit(1),
                    )
                )
                parent_count = parent_keys.count()
                if parent_count == 0:
                    raise ValueError(f"relationship {rel.name} filter matched no parent rows")
                df = (
                    df.drop(rel.child_column)
                    .withColumn(
                        "_sdf_relationship_slot",
                        F.pmod(F.col(child_pk) - F.lit(1), F.lit(parent_count)),
                    )
                    .join(parent_keys, "_sdf_relationship_slot", "left")
                    .drop("_sdf_relationship_slot")
                    .withColumnRenamed("_sdf_parent_key", rel.child_column)
                )
                continue
            df = df.withColumn(
                rel.child_column, ((F.col(child_pk) - F.lit(1)) % F.lit(parent_count)) + F.lit(1)
            )
        return df

    def _add_operational_columns(self, df: Any, table: Any) -> Any:
        from pyspark.sql import functions as F

        key = next(c.name for c in table.columns if c.primary_key)
        return (
            df.withColumn("_sdf_record_key", F.col(key).cast("long"))
            .withColumn("batch_id", ((F.col(key) - F.lit(1)) % F.lit(30)) + F.lit(1))
            .withColumn("_sdf_is_synthetic", F.lit(True))
        )


def dbldatagen_version() -> str:
    import importlib.metadata

    return importlib.metadata.version("dbldatagen")


def _patch_dbldatagen_for_serverless_spark_connect(dg: Any) -> None:
    data_generator = dg.DataGenerator
    if getattr(data_generator, "_sdf_serverless_patch", False):
        return
    original_setup = data_generator._setupPandas

    def safe_setup(self: Any, pandasBatchSize: int | None) -> None:
        try:
            original_setup(self, pandasBatchSize)
        except Exception as exc:
            message = str(exc)
            if "spark.sql.execution.arrow.enabled" not in message:
                raise
            self.logger.warning(
                "Skipping dbldatagen Arrow config setup because Spark Connect "
                "does not allow changing spark.sql.execution.arrow.enabled."
            )
            self._batchSize = pandasBatchSize

    data_generator._setupPandas = safe_setup
    data_generator._sdf_serverless_patch = True
