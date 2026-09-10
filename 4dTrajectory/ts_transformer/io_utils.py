"""Small file/hash helpers shared across the package's writers and runners.

They were copied byte-for-byte into five modules (2026-09-07 package audit); one
definition each. Deliberately torch-free: `experiments/pipeline.py` is import-light and
`experiment_index` runs before any model is built.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any], *, allow_nan: bool = True) -> None:
    """Write via a sibling temp file and rename, so a reader never sees a partial file.

    ``allow_nan=False`` refuses a payload with NaN/inf instead of writing invalid JSON —
    cross-validation results use it because a NaN fold must fail loudly, not serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=allow_nan), encoding="utf-8")
    temporary.replace(path)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
