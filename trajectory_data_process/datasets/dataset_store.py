"""Filesystem layout and JSONL helpers for the partitioned dataset store.

This module only handles where records go and how they are written; it does not
interpret trajectory content.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal


PartitionGranularity = Literal["hour", "day"]


def ensure_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def partition_path(
    output_root: Path,
    dataset_name: str,
    *,
    airport: str,
    timestamp: datetime,
    granularity: PartitionGranularity = "hour",
    version: str = "v3",
) -> Path:
    """Build a stable airport/time partition path."""
    ts = ensure_utc(timestamp)
    parts = [
        output_root,
        dataset_name,
        version,
        f"airport={airport.upper()}",
        f"year={ts.year:04d}",
        f"month={ts.month:02d}",
        f"day={ts.day:02d}",
    ]
    if granularity == "hour":
        parts.append(f"hour={ts.hour:02d}")
    return Path(*parts)


def write_jsonl_records(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Append records to a JSONL file and return the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from a file, or an empty list if it does not exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_json_dumps(value: Any) -> str:
    """Serialize deterministically for content-addressed identifiers."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    """Hash text as it will be written with UTF-8 encoding."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
