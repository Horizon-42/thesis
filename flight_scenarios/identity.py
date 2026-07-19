"""The flight identity used for record filenames and dataset split keys.

One flight = ``id_runway_icao24_landingTime``. ``id`` alone is NOT unique — it is the
callsign, the same callsign flies the same route daily, and in a multi-runway harvest the
same flight dict shape repeats per runway file. Keying anything on ``id`` alone silently
merges distinct flights: a train/val/test split leaks (three copies of "AAL1" land in
different splits), and ``predict --split test`` pulls every namesake.

Both record writers (``4dTrajectory/optimization``'s batch and ``4dTrajectory/
ts_transformer``'s export) and the ts train/val/test split key derive from THIS function,
so a record filename and a split key for the same flight cannot drift apart.
"""

from __future__ import annotations

import re
from typing import Any


def flight_key(source: dict[str, Any], index: int) -> str:
    """Unique, filename-safe identity for one flight's source dict.

    Missing fields are skipped; ``index`` (the flight's position in its input list) is the
    final fallback when there is no ``id`` at all.
    """
    parts = [
        str(source.get("id") or f"flight{index}"),
        str(source.get("runway") or ""),
        str(source.get("icao24") or ""),
        # ``2026-06-18T21:37:36Z`` -> ``20260618T213736Z`` (a filename-safe stamp).
        re.sub(r"[^0-9TZ]", "", str(source.get("landing_time_utc") or "")),
    ]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", "_".join(p for p in parts if p))
