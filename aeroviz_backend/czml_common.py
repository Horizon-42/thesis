"""czml_common.py
===============
Shared CZML playback helpers used by both backend playback builders
(``trajectory_playback`` and ``dynamics_comparison_backend``): the fixed epoch,
ISO-8601 formatting, and the document/clock packet.

The epoch is arbitrary — a playback is standalone, so only the relative time
offsets matter; both builders just need to agree on one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Arbitrary fixed epoch shared by all playbacks.
EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    """ISO-8601 UTC, second precision (``2026-01-01T00:00:00Z``)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_ms(dt: datetime) -> str:
    """ISO-8601 UTC with millisecond precision (for sub-second availability)."""
    base = dt.astimezone(timezone.utc)
    return base.strftime("%Y-%m-%dT%H:%M:%S.") + f"{base.microsecond // 1000:03d}Z"


def document_packet(end_dt: datetime, multiplier: float, name: str) -> dict[str, Any]:
    """The CZML ``document`` packet: a LOOP_STOP clock over ``EPOCH``..``end_dt``."""
    return {
        "id": "document",
        "name": name,
        "version": "1.0",
        "clock": {
            "interval": f"{iso(EPOCH)}/{iso(end_dt)}",
            "currentTime": iso(EPOCH),
            "multiplier": multiplier,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        },
    }
