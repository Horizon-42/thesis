"""The separation rules — the ONLY import of ``ts_transformer`` in the optimizer (design §6, MD3).

FAA JO 7110.65BB's arrival minima and the loss-of-separation judge are defined once, in
``ts_transformer/inference/runway_schedule.py`` and ``separation.py`` (pure numpy + geokit, no torch).
This module imports them read-only and adds nothing to the rules; ``tests/test_traffic_rules.py``
pins their results on fixed inputs, so a change on the two-tier side fails a test here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

_TS_PARENT = Path(__file__).resolve().parents[2]      # 4dTrajectory: the ts_transformer package's parent
if str(_TS_PARENT) not in sys.path:
    sys.path.append(str(_TS_PARENT))

from ts_transformer.inference.runway_schedule import (  # noqa: E402
    FAA_VERTICAL_FT,
    Separation,
    faa_separation,
    wake_category,
)
from ts_transformer.inference.separation import (  # noqa: E402
    AT_THRESHOLD,
    DIAGONAL,
    IFR,
    IN_TRAIL,
    RADAR_OR_VERTICAL,
    VISUAL,
    Loss,
    Traffic as Scene,
    losses,
    wake_at_threshold,
)


__all__ = [
    "AT_THRESHOLD", "DIAGONAL", "FAA_VERTICAL_FT", "IFR", "IN_TRAIL", "RADAR_OR_VERTICAL", "VISUAL",
    "Loss", "Scene", "Separation", "category", "judge", "separation",
]


def separation(runway_targets: dict, speed_mps: float) -> Separation:
    """The FAA minima for an airport's runways (the manifest's targets), distances turned into times
    at ``speed_mps``."""
    return faa_separation(
        {r: {"lat": t["lat"], "lon": t["lon"], "course_deg": t["course_deg"]} for r, t in runway_targets.items()},
        speed_mps=speed_mps,
    )


def category(typecode: str | None) -> str | None:
    """The CWT category of a type, or None for no type or a type the tables do not list (the
    caller records those types); the judge then applies the radar minimum alone (``wake_known``)."""
    if typecode is None:
        return None
    try:
        return wake_category(typecode)
    except KeyError:
        return None


def judge(scene: Scene, rules: Separation, reading: str, over_threshold: Iterable[int]) -> list[Loss]:
    """Every loss at this instant: the pair rules, plus the wake check behind each aircraft that is
    over its threshold now."""
    found = losses(scene, rules, reading)
    for leader in over_threshold:
        loss = wake_at_threshold(scene, leader, rules)
        if loss is not None:
            found.append(loss)
    return found
