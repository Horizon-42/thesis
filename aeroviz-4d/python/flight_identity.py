"""The flight identity ``id_runway_icao24_landingTime`` — mirror of the canonical function.

MUST match ``flight_scenarios/identity.py::flight_key`` exactly. Mirrored rather than
imported ON PURPOSE: this package is standalone frontend tooling with its own pytest
rootdir, and importing ``flight_scenarios`` executes its ``__init__``, which pulls the
whole modeling tree (aircraft/OpenAP, aerodynamic_model) — for a ten-line stdlib
function. The same pinned vector guards both copies against drift:
``tests/test_flight_identity.py`` here and
``4dTrajectory/optimization/tests/test_scenario_optimization.py`` both assert
``flight_key({EJA969, 05R, ad7f04, 2026-06-18T21:37:36Z}, 0)
== "EJA969_05R_ad7f04_20260618T213736Z"``.

This is the identity that names everything about one flight: the observed-layer CZML
entity id (``generate_czml``), the comparison group key (the record-filename stem
``build_scenario_comparison_czml._group_key`` recovers), the optimizer's and
ts_transformer's record filenames, and the ts train/val/test split key. ``id`` (the
callsign) and ``runway`` are readability prefixes; uniqueness comes from
``icao24`` + landing time.
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
