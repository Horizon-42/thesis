"""Dataset storage helpers for OpenSky training data.

This module only handles filesystem layout and JSONL writing. It deliberately
does not parse or normalize OpenSky payloads, so callers can save source
response bodies before any data transformation happens.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal


PartitionGranularity = Literal["hour", "day"]


def ensure_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def partition_path(
    output_root: Path,
    dataset_name: str,
    *,
    airport: str,
    timestamp: datetime,
    granularity: PartitionGranularity = "hour",
    version: str = "v2",
) -> Path:
    """Build a stable airport/time partition path."""
    if granularity not in {"hour", "day"}:
        raise ValueError(f"Unsupported partition granularity: {granularity}")

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


def stable_json_dumps(value: Any) -> str:
    """Serialize metadata deterministically for index and manifest records."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    """Hash text exactly as it will be written with UTF-8 encoding."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    """Convert an endpoint or tag into a filesystem-safe compact name."""
    text = value.strip().strip("/") or "root"
    text = text.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "item"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        f.write("\n")


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


def write_source_response(
    output_root: Path,
    *,
    airport: str,
    fetched_at: datetime,
    endpoint: str,
    params: dict[str, Any],
    body_text: str,
    http_status: int,
    granularity: PartitionGranularity = "hour",
) -> dict[str, Any]:
    """Persist an original OpenSky response body and append its index record.

    The body is written before the index record. Callers should parse JSON only
    from the returned body path or from the same text after this function
    succeeds.
    """
    fetched_at_utc = ensure_utc(fetched_at)
    body_sha256 = sha256_text(body_text)
    source_id = f"sha256:{body_sha256}"
    partition = partition_path(
        output_root,
        "source_responses",
        airport=airport,
        timestamp=fetched_at_utc,
        granularity=granularity,
    )
    partition.mkdir(parents=True, exist_ok=True)

    timestamp_tag = fetched_at_utc.strftime("%Y%m%dT%H%M%SZ")
    endpoint_tag = _safe_name(endpoint)
    params_tag = hashlib.sha1(stable_json_dumps(params).encode("utf-8")).hexdigest()[:10]
    body_name = f"{timestamp_tag}_{endpoint_tag}_{params_tag}_{body_sha256[:12]}.body.txt"
    body_path = partition / body_name
    body_path.write_text(body_text, encoding="utf-8")

    record = {
        "schema_version": "opensky-source-response-v2",
        "source_id": source_id,
        "fetched_at_utc": fetched_at_utc.isoformat().replace("+00:00", "Z"),
        "endpoint": endpoint,
        "params": params,
        "http_status": int(http_status),
        "body_path": body_name,
        "body_sha256": body_sha256,
        "body_bytes": len(body_text.encode("utf-8")),
    }
    _append_jsonl(partition / "source_index.jsonl", record)
    return record | {"body_full_path": str(body_path)}

