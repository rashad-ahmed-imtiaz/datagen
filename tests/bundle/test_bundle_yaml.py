from __future__ import annotations

from pathlib import Path

import yaml


def test_bundle_yaml_files_parse() -> None:
    for path in [Path("databricks.yml"), *Path("resources").glob("*.yml")]:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None
