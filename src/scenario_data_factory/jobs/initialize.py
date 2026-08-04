from __future__ import annotations


def initialize() -> dict[str, object]:
    return {
        "status": "ok",
        "objects": [
            "scenario metadata tables",
            "run metadata tables",
            "control volume",
            "raw output volume",
        ],
    }
