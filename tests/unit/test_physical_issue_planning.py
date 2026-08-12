from __future__ import annotations

from pathlib import Path

from scenario_data_factory.blueprints.registry import get_blueprint
from scenario_data_factory.issues.planner import build_issue_plan, estimate_issues
from scenario_data_factory.models.scenario import ScenarioSpec


def test_file_replay_count_is_fraction_of_source_batch() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="demo")
    counts = estimate_issues(spec)
    assert counts["iss_file_replay_batch_10_12"] == 250


def test_file_replay_targets_only_source_batch() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    plan = build_issue_plan(spec)
    replay_targets = [
        target for target in plan["claims"] if target.issue_id == "iss_file_replay_batch_10_12"
    ]
    assert replay_targets
    assert {target.batch_id for target in replay_targets} == {10}
    assert all(
        ((target.record_key - 1) % spec.timeline.batches) + 1 == 10 for target in replay_targets
    )


def test_out_of_order_targets_only_source_batch() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    data = spec.model_dump(mode="json")
    data["issues"] = [
        {
            "issue_id": "out_of_order",
            "type": "out_of_order",
            "table": "claims",
            "exact_count": 3,
            "parameters": {"source_batch": 2, "emitted_batch": 1},
        }
    ]
    plan = build_issue_plan(ScenarioSpec.model_validate(data))
    targets = plan["claims"]
    assert {target.batch_id for target in targets} == {2}
    assert all(((target.record_key - 1) % spec.timeline.batches) + 1 == 2 for target in targets)


def test_file_count_replay_estimates_source_batch_records() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="small")
    data = spec.model_dump(mode="json")
    for issue in data["issues"]:
        if issue["type"] == "file_replay":
            issue["rate"] = None
            issue["exact_count"] = None
            issue["parameters"] = {
                **issue["parameters"],
                "file_count": 1,
                "source_batch": 10,
                "target_batch": 12,
            }
    replay_spec = ScenarioSpec.model_validate(data)

    counts = estimate_issues(replay_spec)

    assert counts["iss_file_replay_batch_10_12"] == 10
    plan = build_issue_plan(replay_spec)
    replay_targets = [
        target for target in plan["claims"] if target.issue_id == "iss_file_replay_batch_10_12"
    ]
    assert len(replay_targets) == 10
    assert {target.batch_id for target in replay_targets} == {10}


def test_replay_targets_do_not_overlap_explicit_duplicate_targets() -> None:
    spec = ScenarioSpec.from_yaml(
        (Path(__file__).parents[1] / "fixtures" / "all_issue_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    plan = build_issue_plan(spec)
    duplicate_keys = {
        target.record_key for target in plan["events"] if target.issue_id == "duplicate_events"
    }
    replay_keys = {
        target.record_key for target in plan["events"] if target.issue_id == "replay_batch_one"
    }
    assert duplicate_keys.isdisjoint(replay_keys)


def test_demo_dirty_outputs_have_issues_for_every_table() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="demo")
    plan = build_issue_plan(spec)

    assert {table.name for table in spec.tables} == set(plan)
    assert len(plan["customers"]) == 250
    assert len(plan["policies"]) == 150
