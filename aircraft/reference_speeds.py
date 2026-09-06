"""Published per-type approach speeds and the weights they are quoted at (stdlib only).

The threshold speed gate (``evaluation/speed_gate.py``) anchors on the approach speed
each aircraft TYPE publishes -- not on a stall model -- so every bound traces to a
document a reader can open. The table ``reference_speeds.json`` beside this module is
hand-curated from those documents; ``docs/reference_speeds/README.md`` is the
provenance index (URL, retrieval date, SHA-256, page) for every number in it, and the
downloaded sources live under ``data/reference_speeds/`` (git-ignored, re-fetched by
``docs/reference_speeds/fetch_sources.sh``).

Per type the table carries:

* ``approach_speed_kt``      -- the FAA Aircraft Characteristics Database
  ``Approach_Speed_knot``: indicated airspeed at the Maximum Allowable Landing Weight,
  the highest value over the landing flap configurations the Flight Standardization
  Board reports; ``approach_speed_min_kt`` / ``approach_speed_max_kt`` are the FAA's
  dual flap-configuration values where it gives them (else equal to the main value);
* ``malw_kg``                -- the landing weight the speed is quoted at;
* ``min_mass_kg``            -- the lowest PUBLISHED operating mass of the type
  (manufacturer minimum flight weight, OEW or BOW; ``min_mass_kind`` says which), or
  null when no document states one.

Speed scales with the square root of mass at constant lift coefficient, so the
published pair (speed at MALW) gives the reference speed at any mass:

    V_ref(m) = V_published * sqrt(m / MALW)

That is the ONE piece of physics this module applies; it is the same law
``aircraft.aero_params.stall_speed_ms`` embodies, evaluated here from a published
anchor instead of a modelled Cl_max.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

REFERENCE_SPEEDS_SCHEMA = "aircraft-reference-speeds-v1"
REFERENCE_SPEEDS_PATH = Path(__file__).with_name("reference_speeds.json")

Edge = Literal["low", "high"]


@dataclass(frozen=True)
class ReferenceSpeed:
    """One type's published approach speed and the masses that frame it."""

    typecode: str
    approach_speed_kt: float
    approach_speed_min_kt: float
    approach_speed_max_kt: float
    malw_kg: float
    min_mass_kg: float | None
    min_mass_kind: str | None
    # Source ids (keys of the table's ``sources`` block) per fact.
    approach_speed_source: str
    malw_source: str
    min_mass_source: str | None

    def vref_kt(self, mass_kg: float, *, edge: Edge) -> float:
        """The published speed at ``mass_kg``: the lower flap-configuration value
        (``edge="low"``) or the higher one (``edge="high"``), scaled by sqrt(m/MALW)."""
        if not math.isfinite(mass_kg) or mass_kg <= 0.0:
            raise ValueError(f"{self.typecode}: mass must be a positive finite number, got {mass_kg!r}")
        base = self.approach_speed_min_kt if edge == "low" else self.approach_speed_max_kt
        return base * math.sqrt(mass_kg / self.malw_kg)


def _positive(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{where} must be positive and finite, got {value!r}")
    return numeric


_ROW_KEYS = frozenset({
    "approach_speed_kt", "approach_speed_min_kt", "approach_speed_max_kt",
    "approach_speed_source", "approach_speed_note",
    "malw_kg", "malw_source", "malw_note",
    "min_mass_kg", "min_mass_kind", "min_mass_source", "min_mass_note",
    "corroboration",
})


def _entry(typecode: str, row: Mapping[str, Any], sources: Mapping[str, Any]) -> ReferenceSpeed:
    where = f"reference_speeds[{typecode}]"
    unknown = set(row) - _ROW_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    main = _positive(row.get("approach_speed_kt"), f"{where}.approach_speed_kt")
    low = _positive(row.get("approach_speed_min_kt", main), f"{where}.approach_speed_min_kt")
    high = _positive(row.get("approach_speed_max_kt", main), f"{where}.approach_speed_max_kt")
    if not low <= main <= high:
        raise ValueError(f"{where}: expected min <= main <= max, got {low}, {main}, {high}")
    malw = _positive(row.get("malw_kg"), f"{where}.malw_kg")
    min_mass = row.get("min_mass_kg")
    min_mass_kind = row.get("min_mass_kind")
    min_mass_source = row.get("min_mass_source")
    if min_mass is not None:
        min_mass = _positive(min_mass, f"{where}.min_mass_kg")
        if min_mass >= malw:
            raise ValueError(f"{where}: min_mass_kg {min_mass} must be below malw_kg {malw}")
        if not isinstance(min_mass_kind, str) or not isinstance(min_mass_source, str):
            raise ValueError(f"{where}: min_mass_kg needs min_mass_kind and min_mass_source")
    elif min_mass_kind is not None or min_mass_source is not None:
        raise ValueError(
            f"{where}: min_mass_kind/min_mass_source without min_mass_kg (a mistyped key?)"
        )
    for key in ("approach_speed_source", "malw_source"):
        if row.get(key) not in sources:
            raise ValueError(f"{where}.{key} {row.get(key)!r} is not a listed source")
    if min_mass_source is not None and min_mass_source not in sources:
        raise ValueError(f"{where}.min_mass_source {min_mass_source!r} is not a listed source")
    for item in row.get("corroboration", ()):
        if not isinstance(item, Mapping) or item.get("source") not in sources:
            raise ValueError(f"{where}.corroboration entry {item!r} names no listed source")
    return ReferenceSpeed(
        typecode=typecode,
        approach_speed_kt=main,
        approach_speed_min_kt=low,
        approach_speed_max_kt=high,
        malw_kg=malw,
        min_mass_kg=min_mass,
        min_mass_kind=min_mass_kind if min_mass is not None else None,
        approach_speed_source=str(row["approach_speed_source"]),
        malw_source=str(row["malw_source"]),
        min_mass_source=min_mass_source if min_mass is not None else None,
    )


def load_reference_speeds(path: Path = REFERENCE_SPEEDS_PATH) -> dict[str, ReferenceSpeed]:
    """Read and validate the table; every row must trace to a listed source."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != REFERENCE_SPEEDS_SCHEMA:
        raise ValueError(
            f"{path}: expected schema {REFERENCE_SPEEDS_SCHEMA!r}, got {data.get('schema')!r}"
        )
    sources = data.get("sources")
    types = data.get("types")
    if not isinstance(sources, dict) or not sources or not isinstance(types, dict) or not types:
        raise ValueError(f"{path}: needs non-empty 'sources' and 'types' objects")
    return {
        code.upper(): _entry(code.upper(), row, sources) for code, row in types.items()
    }


@lru_cache(maxsize=1)
def _table() -> dict[str, ReferenceSpeed]:
    return load_reference_speeds()


def reference_speed(typecode: str) -> ReferenceSpeed | None:
    """The packaged table's entry for an ICAO type designator, or None when the type
    has no published entry (the caller grades speed indeterminate and says so)."""
    return _table().get(typecode.upper())
