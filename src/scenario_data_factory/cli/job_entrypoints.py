from __future__ import annotations

import json
import sys
from pathlib import Path

from scenario_data_factory.jobs.generate import run_generation
from scenario_data_factory.jobs.preview import preview_scenario
from scenario_data_factory.jobs.smoke_test import run_smoke_test
from scenario_data_factory.models.scenario import ScenarioSpec


def smoke_test() -> None:
    print(json.dumps(run_smoke_test(), indent=2, sort_keys=True))


def preview() -> None:
    path = Path(sys.argv[1])
    spec = ScenarioSpec.from_yaml(path.read_text(encoding="utf-8"))
    print(json.dumps(preview_scenario(spec), indent=2, sort_keys=True))


def generate() -> None:
    from pyspark.sql import SparkSession

    path = Path(sys.argv[1])
    output_root = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    spec = ScenarioSpec.from_yaml(path.read_text(encoding="utf-8"))
    spark = SparkSession.builder.appName("scenario-data-factory").getOrCreate()
    try:
        summary = run_generation(spark, spec, local_root=output_root)
    finally:
        spark.stop()
    print(json.dumps(summary, indent=2, sort_keys=True))
