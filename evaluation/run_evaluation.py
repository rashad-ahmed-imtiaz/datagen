from __future__ import annotations

import yaml

from evaluation.scorers import (
    output_mode_correct,
    references_exist,
    supported_issues_only,
    valid_spec,
)
from scenario_data_factory.blueprints.registry import get_blueprint


def main() -> None:
    cases = yaml.safe_load(open("evaluation/agent_cases.yaml", encoding="utf-8"))["cases"]
    spec = get_blueprint("insurance_claims").build(
        name="evaluation baseline", seed=42, scale="small"
    )
    results = {
        "case_count": len(cases),
        "valid_spec": valid_spec(spec),
        "supported_issues_only": supported_issues_only(spec),
        "references_exist": references_exist(spec),
        "output_mode_correct": output_mode_correct(spec),
    }
    print(results)


if __name__ == "__main__":
    main()
