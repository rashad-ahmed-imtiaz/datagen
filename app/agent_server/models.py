from __future__ import annotations

from pydantic import BaseModel


class ToolResult(BaseModel):
    ok: bool
    data: dict[str, object]
    warnings: list[str] = []
