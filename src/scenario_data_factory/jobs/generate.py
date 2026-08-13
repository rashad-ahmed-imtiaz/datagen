from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from scenario_data_factory.compiler.validation import validate_scenario
from scenario_data_factory.generation.dbldatagen_engine import DbldatagenEngine
from scenario_data_factory.issues.planner import build_issue_plan
from scenario_data_factory.issues.registry import ISSUE_REGISTRY
from scenario_data_factory.models.scenario import IssueType, OutputMode, ScenarioSpec
from scenario_data_factory.output.delta_writer import write_delta_tables
from scenario_data_factory.output.manifest_writer import write_manifest
from scenario_data_factory.output.path_resolver import safe_child
from scenario_data_factory.output.physical_issues import apply_physical_raw_issues
from scenario_data_factory.output.raw_writer import write_raw_batches
from scenario_data_factory.output.summary_writer import build_issue_summary, write_summary


def run_generation(
    spark,
    spec: ScenarioSpec,
    *,
    run_id: str | None = None,
    local_root: str | Path | None = None,
    partitions: int | None = None,
) -> dict[str, object]:
    validate_scenario(spec)
    run_id = run_id or f"run_{uuid4().hex[:12]}"
    root = Path(local_root or spec.outputs.local_root)
    run_root = safe_child(root, spec.scenario_id, run_id)
    clean = DbldatagenEngine().generate(spark, spec, partitions=partitions)
    dirty = dict(clean)
    plan = build_issue_plan(spec)
    has_issues = bool(spec.issues)

    for issue in spec.issues:
        table = issue.table
        key = next(c.name for c in spec.table(table).columns if c.primary_key)
        dirty[table] = ISSUE_REGISTRY[IssueType(issue.type)].apply_spark(
            dirty[table], issue, plan.get(table, []), key
        )

    written: dict[str, object] = {}
    if (
        spec.outputs.mode in {OutputMode.DELTA, OutputMode.BOTH, "delta", "both"}
        and spec.outputs.catalog
        and spec.outputs.schema_name
    ):
        if has_issues:
            written["dirty_delta_tables"] = write_delta_tables(
                dirty,
                spec.outputs.catalog,
                spec.outputs.schema_name,
                spec.outputs.dirty_delta_prefix,
                namespace=spec.outputs.delta_namespace or spec.name,
            )
        if spec.outputs.include_clean or not has_issues:
            written["clean_delta_tables"] = write_delta_tables(
                clean,
                spec.outputs.catalog,
                spec.outputs.schema_name,
                spec.outputs.clean_delta_prefix,
                namespace=spec.outputs.delta_namespace or spec.name,
            )
    if spec.outputs.mode in {OutputMode.RAW, OutputMode.BOTH, "raw", "both"}:
        if has_issues:
            raw_dirty = apply_physical_raw_issues(dirty, spec, plan)
            written["raw_dirty_paths"] = write_raw_batches(
                raw_dirty, run_root / "raw" / "dirty", spec.outputs.raw_format
            )
        if spec.outputs.include_clean or not has_issues:
            written["raw_clean_paths"] = write_raw_batches(
                clean, run_root / "raw" / "clean", spec.outputs.raw_format
            )
    manifest_path = write_manifest(run_root / "issue_manifest.json", plan)
    summary = {
        "run_id": run_id,
        "scenario_id": spec.scenario_id,
        "spec_hash": spec.spec_hash(),
        "rows_by_table": {table.name: table.row_count for table in spec.tables},
        "issues": build_issue_summary(plan),
        "outputs": written,
        "manifest_path": str(manifest_path),
    }
    summary_path = write_summary(run_root / "summary.json", summary)
    summary["summary_path"] = str(summary_path)
    return summary
