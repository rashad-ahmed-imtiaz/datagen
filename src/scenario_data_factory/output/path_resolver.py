from __future__ import annotations

import re
from pathlib import Path

from scenario_data_factory.exceptions import OutputWriteError


def sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    if not sanitized:
        raise OutputWriteError("INVALID_PATH_NAME", "Name cannot be used in an output path.")
    return sanitized[:120]


def safe_child(root: str | Path, *parts: str) -> Path:
    root_path = Path(root).resolve()
    candidate = root_path.joinpath(*(sanitize_identifier(part) for part in parts)).resolve()
    if root_path != candidate and root_path not in candidate.parents:
        raise OutputWriteError(
            "PATH_TRAVERSAL",
            "Output path escapes the configured root.",
            technical_detail=str(candidate),
        )
    return candidate
