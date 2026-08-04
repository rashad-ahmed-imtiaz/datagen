from __future__ import annotations

import hashlib
import importlib.metadata
import json


def main() -> None:
    version = importlib.metadata.version("dbldatagen")
    payload = {"package": "dbldatagen", "version": version, "source": "PyPI"}
    payload["manifest_checksum"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
