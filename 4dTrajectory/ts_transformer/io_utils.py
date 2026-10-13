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
import sys
from typing import Any

#: The line a campaign writes to a gated step's stdin when the step's turn comes (`wait_for_turn`).
GO = "go"


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


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """`write_json_atomic` for bytes as they are (a file copied byte for byte)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_turn(gated: bool) -> None:
    """A gated step (``--gate``) waits here for its turn: a line ``go`` on stdin from the campaign that started it early,
    so that its CPU work before this call overlaps the step before it. A step stops by name when stdin ends or gives
    another line (its campaign stopped before its turn). An ungated step goes on at once."""
    if not gated:
        return
    line = sys.stdin.readline()
    if line.strip() != GO:
        raise SystemExit(f"stopped before this step's turn: stdin gave {line!r}, not {GO!r}")
