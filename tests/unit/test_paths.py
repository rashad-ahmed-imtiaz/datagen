from __future__ import annotations

from scenario_data_factory.output.path_resolver import safe_child, sanitize_identifier


def test_sanitize_identifier() -> None:
    assert sanitize_identifier("Canadian Insurance Claims!") == "canadian_insurance_claims"


def test_safe_child_stays_under_root(tmp_path) -> None:
    child = safe_child(tmp_path, "../escape", "run")
    assert tmp_path.resolve() in child.parents
