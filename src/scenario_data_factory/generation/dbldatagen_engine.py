from __future__ import annotations

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
                    data_spec = data_spec.withColumn(col.name, col.dtype, **col.options)
                df = data_spec.build().drop("id")
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

    def _apply_relationships(
        self, df: Any, table_name: str, spec: ScenarioSpec, generated: dict[str, Any]
    ) -> Any:
        from pyspark.sql import functions as F

        for rel in spec.relationships:
            if rel.child_table != table_name:
                continue
            parent_count = spec.table(rel.parent_table).row_count
            child_pk = next(c.name for c in spec.table(table_name).columns if c.primary_key)
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
