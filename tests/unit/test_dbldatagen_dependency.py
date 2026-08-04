from __future__ import annotations

import importlib.metadata


def test_dbldatagen_is_pinned_to_expected_version() -> None:
    assert importlib.metadata.version("dbldatagen") == "0.4.0"
