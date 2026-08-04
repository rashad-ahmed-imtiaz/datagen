from __future__ import annotations

import os

import pytest

from scenario_data_factory.blueprints.registry import get_blueprint
from scenario_data_factory.jobs.generate import run_generation

pytestmark = pytest.mark.skipif(
    not os.getenv("JAVA_HOME"),
    reason="local Spark integration requires Java; Databricks jobs provide Spark runtime",
)


def test_small_insurance_generation_with_real_dbldatagen(tmp_path) -> None:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[2]").appName("sdf-spark-test").getOrCreate()
    try:
        spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
        summary = run_generation(spark, spec, local_root=tmp_path, partitions=1)
        assert summary["rows_by_table"]["claims"] == 300
        assert summary["issues"]["issue_counts"]["duplicate_record"] == 9
    finally:
        spark.stop()
