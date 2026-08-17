from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer
from rich.console import Console

from scenario_data_factory.blueprints.registry import get_blueprint, list_blueprints
from scenario_data_factory.compiler.validation import validate_scenario
from scenario_data_factory.generation.dbldatagen_engine import dbldatagen_version
from scenario_data_factory.issues.registry import list_issue_types
from scenario_data_factory.jobs.generate import run_generation
from scenario_data_factory.jobs.preview import preview_scenario
from scenario_data_factory.jobs.smoke_test import run_smoke_test
from scenario_data_factory.models.scenario import ScenarioSpec

app = typer.Typer(help="Scenario Data Factory")
console = Console()


@app.command()
def doctor(profile: str | None = None, target: str = "dev") -> None:
    console.print("[bold]Scenario Data Factory doctor[/bold]")
    console.print(f"PASS Python project is importable for target {target}")
    console.print(f"PASS dbldatagen pinned import check: {dbldatagen_version()}")
    if profile:
        console.print(f"WARN Databricks workspace checks require live authentication: {profile}")
    else:
        console.print("WARN Databricks profile not supplied; workspace checks skipped")


@app.command()
def blueprints() -> None:
    for bp in list_blueprints():
        console.print(f"{bp.domain}: {bp.description}")


@app.command("issues")
def issues_cmd() -> None:
    for issue in list_issue_types():
        console.print(f"{issue['type']} raw_required={issue['requires_raw_output']}")


@app.command()
def new(
    domain: str = "insurance_claims",
    name: str = "Canadian Insurance Claims Reliability Scenario",
    seed: int = 42,
    scale: str = "demo",
    output: Path = Path("examples/insurance_claims.yaml"),
) -> None:
    spec = get_blueprint(domain).build(name=name, seed=seed, scale=scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        output.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    else:
        output.write_text(spec.to_yaml(), encoding="utf-8")
    console.print(f"Wrote {output} ({spec.spec_hash()})")


@app.command()
def validate(path: Path) -> None:
    spec = ScenarioSpec.from_yaml(path.read_text(encoding="utf-8"))
    warnings = validate_scenario(spec)
    console.print(f"PASS {spec.scenario_id} revision={spec.revision} hash={spec.spec_hash()}")
    for warning in warnings:
        console.print(f"WARN {warning}")


@app.command()
def estimate(path: Path) -> None:
    spec = ScenarioSpec.from_yaml(path.read_text(encoding="utf-8"))
    console.print_json(data=preview_scenario(spec))


@app.command()
def generate(
    path: Path,
    output_root: Path | None = None,
    partitions: int | None = None,
    app_name: str = "scenario-data-factory",
) -> None:
    spec = ScenarioSpec.from_yaml(path.read_text(encoding="utf-8"))
    try:
        from pyspark.sql import SparkSession
    except Exception as exc:
        raise typer.BadParameter(
            "PySpark is required for generation. Install with `uv sync --extra spark` "
            "or run through the Databricks Lakeflow job."
        ) from exc
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    try:
        summary = run_generation(
            spark,
            spec,
            local_root=output_root or spec.outputs.local_root,
            partitions=partitions,
        )
    finally:
        spark.stop()
    console.print_json(data=summary)


@app.command("smoke-test")
def smoke_test_cmd() -> None:
    console.print_json(data=run_smoke_test())


def _run(command: list[str]) -> None:
    console.print(f"[bold]$[/bold] {' '.join(command)}")
    subprocess.run(command, check=True)


def _capture_json(command: list[str]) -> dict[str, object]:
    console.print(f"[bold]$[/bold] {' '.join(command)}")
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


@app.command()
def install(
    profile: str,
    target: str = "dev",
    model_endpoint: str = "databricks-gpt-oss-120b",
    catalog: str = "sdf",
    schema: str = "scenario_data_factory",
    app_name: str = "scenario-data-factory",
    engine_only: bool = False,
) -> None:
    bundle_vars = [
        "--var",
        f"model_endpoint={model_endpoint}",
        "--var",
        f"catalog={catalog}",
        "--var",
        f"schema={schema}",
        "--var",
        f"app_name={app_name}",
    ]
    _run(["uv", "sync", "--extra", "dev", "--extra", "spark", "--extra", "app"])
    _run(["uv", "build"])
    _run(
        ["databricks", "bundle", "validate", "--profile", profile, "--target", target, *bundle_vars]
    )
    _run(["databricks", "bundle", "deploy", "--profile", profile, "--target", target, *bundle_vars])
    summary = _capture_json(
        [
            "databricks",
            "bundle",
            "summary",
            "--profile",
            profile,
            "--target",
            target,
            *bundle_vars,
            "-o",
            "json",
        ]
    )
    control_volume = summary["resources"]["volumes"]["control_volume"]["id"]
    _run(
        [
            "databricks",
            "fs",
            "mkdirs",
            f"dbfs:/Volumes/{control_volume}/scenarios",
            "--profile",
            profile,
        ]
    )
    _run(
        [
            "databricks",
            "fs",
            "cp",
            "examples/insurance_claims.yaml",
            f"dbfs:/Volumes/{control_volume}/scenarios/current.yaml",
            "--profile",
            profile,
            "--overwrite",
        ]
    )
    _run(
        [
            "databricks",
            "bundle",
            "run",
            "smoke_test",
            "--profile",
            profile,
            "--target",
            target,
            *bundle_vars,
        ]
    )
    _run(
        [
            "databricks",
            "bundle",
            "run",
            "preview",
            "--profile",
            profile,
            "--target",
            target,
            *bundle_vars,
        ]
    )
    if not engine_only:
        app_source_path = summary["resources"]["apps"]["scenario_data_factory"]["source_code_path"]
        _run(["databricks", "apps", "start", app_name, "--profile", profile])
        _run(
            [
                "databricks",
                "apps",
                "deploy",
                app_name,
                "--profile",
                profile,
                "--source-code-path",
                str(app_source_path),
                "--timeout",
                "20m",
            ]
        )
    console.print("PASS Scenario Data Factory installation completed.")
