from __future__ import annotations

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


def test_demo_dirty_outputs_have_issues_for_every_table() -> None:
    spec = get_blueprint("insurance_claims").build(name="demo", seed=42, scale="demo")
    plan = build_issue_plan(spec)

    assert {table.name for table in spec.tables} == set(plan)
    assert len(plan["customers"]) == 250
    assert len(plan["policies"]) == 150
