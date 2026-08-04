from __future__ import annotations

from pathlib import Path


class VolumeRepository:
    def __init__(self, local_root: str | Path = "out") -> None:
        self.local_root = Path(local_root)

    def ensure(self) -> Path:
        self.local_root.mkdir(parents=True, exist_ok=True)
        return self.local_root
