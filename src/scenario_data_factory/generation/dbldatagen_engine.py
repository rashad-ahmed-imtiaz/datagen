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
                df = self._apply_dependent_date_offsets(df, table)
                df = self._add_operational_columns(df, table, spec)
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

            faker_locale = _supported_faker_locale(locale)
            resolved["text"] = FakerTextFactory(locale=faker_locale)(
                _supported_faker_provider(str(provider), faker_locale)
            )
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
            if column.values:
                spark_type = column.type.value
                choices = [F.lit(value).cast(spark_type) for value in column.values]
                if column.weights and len(column.weights) == len(choices):
                    total = sum(float(weight) for weight in column.weights)
                    if total > 0:
                        weights = [
                            max(1, round(float(weight) / total * 10_000))
                            for weight in column.weights
                        ]
                        weights[-1] += 10_000 - sum(weights)
                        bucket = F.pmod(F.xxhash64(F.col(record_key)), F.lit(10_000))
                        expression = choices[-1]
                        bounds: list[int] = []
                        running_total = 0
                        for weight in weights[:-1]:
                            running_total += weight
                            bounds.append(running_total)
                        for choice, upper_bound in reversed(
                            list(zip(choices[:-1], bounds, strict=True))
                        ):
                            expression = F.when(bucket < F.lit(upper_bound), choice).otherwise(
                                expression
                            )
                        df = df.withColumn(column.name, expression)
                        continue
                position = (
                    F.pmod(F.col(record_key) - F.lit(1), F.lit(len(choices))) + F.lit(1)
                ).cast("int")
                df = df.withColumn(column.name, F.element_at(F.array(*choices), position))
                continue
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
                tail_share = semantic.get("tail_share")
                tail_min = semantic.get("tail_min")
                tail_max = semantic.get("tail_max", maximum)
                if (
                    isinstance(tail_share, (int, float))
                    and isinstance(tail_min, (int, float))
                    and isinstance(tail_max, (int, float))
                    and 0 < tail_share < 1
                    and 0 < tail_min <= tail_max
                ):
                    tail_value = F.lit(float(tail_min)) + F.rand(spec.seed + 1) * F.lit(
                        float(tail_max) - float(tail_min)
                    )
                    is_tail = F.rand(spec.seed + 2) < F.lit(float(tail_share))
                    value = F.when(is_tail, tail_value).otherwise(value)
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

    @staticmethod
    def _apply_dependent_date_offsets(df: Any, table: Any) -> Any:
        """Reapply child date offsets after a relationship rewrites their base dates."""
        from pyspark.sql import functions as F

        record_key = next(column.name for column in table.columns if column.primary_key)
        for column in table.columns:
            semantic = column.semantic or {}
            if semantic.get("kind") != "date_offset":
                continue
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
            constraints = rel.constraints or {}
            parent_columns = self._constraint_parent_columns(constraints)
            if parent_filter:
                parent_column = parent_filter["column"]
                filter_values = parent_filter["values"]
                parent_frame = generated[rel.parent_table].where(
                    F.col(parent_column).isin(filter_values)
                )
                parent_count = parent_frame.count()
                if parent_count == 0:
                    raise ValueError(f"relationship {rel.name} filter matched no parent rows")
                bucket_count = min(64, max(1, parent_count // 10_000))
                parent_keys = (
                    parent_frame
                    .select(
                        F.col(rel.parent_column).alias("_sdf_parent_key"),
                        *[
                            F.col(column).alias(self._parent_alias(column))
                            for column in parent_columns
                        ],
                    )
                    .withColumn(
                        "_sdf_parent_bucket",
                        F.pmod(F.xxhash64(F.col("_sdf_parent_key")), F.lit(bucket_count)),
                    )
                )
                bucket_sizes = {
                    int(row["_sdf_parent_bucket"]): int(row["count"])
                    for row in parent_keys.groupBy("_sdf_parent_bucket").count().collect()
                }
                offsets: dict[int, int] = {}
                next_offset = 0
                for bucket in sorted(bucket_sizes):
                    offsets[bucket] = next_offset
                    next_offset += bucket_sizes[bucket]
                offset_map = F.create_map(
                    *[
                        item
                        for bucket, offset in offsets.items()
                        for item in (F.lit(bucket), F.lit(offset))
                    ]
                )
                parent_keys = (
                    parent_keys
                    .withColumn(
                        "_sdf_parent_rank",
                        F.row_number().over(
                            Window.partitionBy("_sdf_parent_bucket").orderBy("_sdf_parent_key")
                        )
                        - F.lit(1),
                    )
                    .withColumn(
                        "_sdf_relationship_slot",
                        F.element_at(offset_map, F.col("_sdf_parent_bucket"))
                        + F.col("_sdf_parent_rank"),
                    )
                    .select(
                        "_sdf_relationship_slot",
                        "_sdf_parent_key",
                        *[self._parent_alias(column) for column in parent_columns],
                    )
                )
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
            else:
                df = df.withColumn(
                    rel.child_column,
                    ((F.col(child_pk) - F.lit(1)) % F.lit(parent_count)) + F.lit(1),
                )
                if parent_columns:
                    parent_values = generated[rel.parent_table].select(
                        F.col(rel.parent_column).alias("_sdf_parent_key"),
                        *[
                            F.col(column).alias(self._parent_alias(column))
                            for column in parent_columns
                        ],
                    )
                    df = (
                        df.join(
                            parent_values,
                            F.col(rel.child_column) == F.col("_sdf_parent_key"),
                            "left",
                        )
                        .drop("_sdf_parent_key")
                    )
            df = self._apply_relationship_constraints(df, rel, constraints, child_pk)
        return df

    @staticmethod
    def _constraint_parent_columns(constraints: dict[str, Any]) -> list[str]:
        columns: list[str] = []
        for rule in constraints.get("child_date_ranges", []):
            if isinstance(rule, dict):
                columns.extend(
                    [str(rule.get("parent_start_column")), str(rule.get("parent_end_column"))]
                )
        for rule in constraints.get("aggregate_caps", []):
            if isinstance(rule, dict):
                columns.append(str(rule.get("parent_amount_column")))
        return list(dict.fromkeys(column for column in columns if column and column != "None"))

    @staticmethod
    def _parent_alias(column: str) -> str:
        return f"_sdf_parent_{column}"

    def _apply_relationship_constraints(
        self, df: Any, relationship: Any, constraints: dict[str, Any], child_pk: str
    ) -> Any:
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        aliases: set[str] = set()
        for rule in constraints.get("child_date_ranges", []):
            if not isinstance(rule, dict):
                continue
            child_column = rule["child_column"]
            start_alias = self._parent_alias(rule["parent_start_column"])
            end_alias = self._parent_alias(rule["parent_end_column"])
            aliases.update((start_alias, end_alias))
            span = F.greatest(F.datediff(F.col(end_alias), F.col(start_alias)), F.lit(0))
            offset = F.pmod(F.col(child_pk), span + F.lit(1)).cast("int")
            df = df.withColumn(child_column, F.date_add(F.col(start_alias), offset))
        for rule in constraints.get("aggregate_caps", []):
            if not isinstance(rule, dict):
                continue
            child_column = rule["child_amount_column"]
            parent_alias = self._parent_alias(rule["parent_amount_column"])
            aliases.add(parent_alias)
            maximum_fraction = float(rule.get("maximum_fraction", 1.0))
            payments_per_parent = F.count(F.lit(1)).over(
                Window.partitionBy(F.col(relationship.child_column))
            )
            cap = F.col(parent_alias) * F.lit(maximum_fraction) / payments_per_parent
            df = df.withColumn(child_column, F.least(F.col(child_column), cap))
        return df.drop(*aliases)

    def _add_operational_columns(self, df: Any, table: Any, spec: ScenarioSpec) -> Any:
        from pyspark.sql import functions as F

        key = next(c.name for c in table.columns if c.primary_key)
        result = (
            df.withColumn("_sdf_record_key", F.col(key).cast("long"))
            .withColumn(
                "batch_id",
                F.pmod(F.col(key) - F.lit(1), F.lit(spec.timeline.batches)) + F.lit(1),
            )
            .withColumn("_sdf_is_synthetic", F.lit(True))
        )
        if table.source_systems and "source_system" not in result.columns:
            sources = F.array(*[F.lit(source) for source in table.source_systems])
            result = result.withColumn(
                "source_system",
                F.element_at(
                    sources,
                    F.pmod(F.xxhash64(F.col(key)), F.lit(len(table.source_systems))) + F.lit(1),
                ),
            )
        return result


def dbldatagen_version() -> str:
    import importlib.metadata

    return importlib.metadata.version("dbldatagen")


def _supported_faker_locale(locale: str) -> str:
    normalized = locale.replace("-", "_")
    aliases = {"ca": "en_CA", "canada": "en_CA", "ca_es": "en_CA"}
    normalized = aliases.get(normalized.lower(), normalized)
    try:
        from faker.config import AVAILABLE_LOCALES

        return normalized if normalized in AVAILABLE_LOCALES else "en_US"
    except Exception:
        return normalized


def _supported_faker_provider(provider: str, locale: str) -> str:
    """Use a safe text provider instead of failing a distributed Spark write."""
    try:
        from faker import Faker

        getattr(Faker(locale), provider)
        return provider
    except Exception:
        return "word"


def _patch_dbldatagen_for_serverless_spark_connect(dg: Any) -> None:
    data_generator = dg.DataGenerator
    if getattr(data_generator, "_sdf_serverless_patch", False):
        return
    original_setup = data_generator._setupPandas
    warned = False

    def safe_setup(self: Any, pandasBatchSize: int | None) -> None:
        nonlocal warned
        try:
            original_setup(self, pandasBatchSize)
        except Exception as exc:
            message = str(exc)
            if "spark.sql.execution.arrow.enabled" not in message:
                raise
            if not warned:
                self.logger.warning(
                    "Skipping dbldatagen Arrow config setup because Spark Connect "
                    "does not allow changing spark.sql.execution.arrow.enabled."
                )
                warned = True
            self._batchSize = pandasBatchSize

    data_generator._setupPandas = safe_setup
    data_generator._sdf_serverless_patch = True
