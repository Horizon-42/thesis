"""The performance index: what the model flies for a type that has no native dynamics.

A type with a hand-tuned preset or its own OpenAP model is flown as itself. Every other type the
fleet shows is decided ONCE, in ``performance_index.json`` beside this module, by the analysis in
``docs/aircraft_performance/2026-09-23_missing_performance_substitution.zh.md`` (user decisions
2026-09-23/24), instead of being flown as an A320:

* ``own``        -- the type's own landing mass, wing area and installed thrust from primary
                    documents (type-certificate data sheets, the FAA Aircraft Characteristics
                    Database, the Poll-Schumann parameter file), each with a source id; its
                    approach speed is its own published row (``aircraft/reference_speeds.json``).
* ``substitute`` -- the airframe that flies most like it (same propulsion class and FAA approach
                    category, nearest approach speed and thrust-to-weight); the model flies THAT
                    airframe, under its own code, so its mass, speed and gate all belong together.
* ``exclude``    -- no acceptable model (propeller aircraft by user decision, rotorcraft and
                    military types, no published approach speed, no same-class airframe); the
                    caller drops the flight and names the reason.

The loader validates every row once: the three decisions, positive facts that cite a listed
source, a published approach speed for every own-parameter type, and no row for an airframe
the model already flies natively (a preset or an OpenAP-direct type), whose data a row would
silently shadow; a substitute must name such a native airframe; and EVERY OpenAP synonym has a
row, since the model never flies a synonym's borrowed data under the synonym's own code.
:func:`performance_index_identity` names the file a result was built with.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

from aircraft.aircraft_sets import (
    AIRCRAFT_PRESETS,
    Aircraft,
    Approach,
    Engine,
    Geometry,
    Mass,
    class_procedure,
    published_speeds,
)
from aircraft.query_aircraft_parameters import PARAMETERS_PATH, load_json, openap_support_kind

PERFORMANCE_INDEX_SCHEMA = "aircraft-performance-index-v1"
PERFORMANCE_INDEX_PATH = Path(__file__).with_name("performance_index.json")

Decision = Literal["own", "substitute", "exclude"]
_DECISIONS = ("own", "substitute", "exclude")
_OWN_KEYS = frozenset({"decision", "name", "mtow_kg", "mlw_kg", "wing_area_m2", "engines",
                       "max_thrust_n_each", "sources"})
_SOURCE_FACTS = ("mass", "wing_area", "thrust")


@dataclass(frozen=True)
class IndexEntry:
    typecode: str
    decision: Decision
    aircraft: Aircraft | None      # the type's own airframe (decision "own")
    substitute: str | None         # the airframe it is flown as (decision "substitute")
    reason: str                    # why: the sources, the basis, or the exclusion


def _native(typecode: str) -> bool:
    return typecode in AIRCRAFT_PRESETS or openap_support_kind(typecode) == "direct"


def _positive(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{where} must be positive and finite, got {value!r}")
    return float(value)


def _own(typecode: str, row: Mapping[str, Any], sources: Mapping[str, Any]) -> IndexEntry:
    where = f"performance_index[{typecode}]"
    if set(row) != _OWN_KEYS:
        raise ValueError(f"{where}: keys {sorted(row)} != {sorted(_OWN_KEYS)}")
    mtow = _positive(row["mtow_kg"], f"{where}.mtow_kg")
    mlw = _positive(row["mlw_kg"], f"{where}.mlw_kg")
    if mlw > mtow:
        raise ValueError(f"{where}: mlw_kg {mlw} above mtow_kg {mtow}")
    engines = row["engines"]
    if isinstance(engines, bool) or not isinstance(engines, int) or engines < 1:
        raise ValueError(f"{where}.engines must be a positive integer, got {engines!r}")
    cited = row["sources"]
    if set(cited) != set(_SOURCE_FACTS) or any(cited[f] not in sources for f in _SOURCE_FACTS):
        raise ValueError(f"{where}.sources must name a listed source for {_SOURCE_FACTS}: {cited!r}")
    aircraft = Aircraft(
        code=typecode,
        name=str(row["name"]),
        category="performance_index",
        geometry=Geometry(wing_area_m2=_positive(row["wing_area_m2"], f"{where}.wing_area_m2")),
        mass=Mass(max_takeoff_kg=mtow, max_landing_kg=mlw),
        engine=Engine(count=engines,
                      max_thrust_n_each=_positive(row["max_thrust_n_each"], f"{where}.max_thrust_n_each")),
        approach=Approach(speeds=published_speeds(typecode), **class_procedure(mtow)),
    )
    return IndexEntry(typecode, "own", aircraft, None,
                      "own parameters: " + ", ".join(f"{f} {cited[f]}" for f in _SOURCE_FACTS))


def _entry(typecode: str, row: Mapping[str, Any], sources: Mapping[str, Any]) -> IndexEntry:
    where = f"performance_index[{typecode}]"
    if _native(typecode):
        raise ValueError(f"{where}: {typecode} is flown natively (a preset or OpenAP-direct type)")
    decision = row.get("decision")
    if decision not in _DECISIONS:
        raise ValueError(f"{where}.decision must be one of {_DECISIONS}, got {decision!r}")
    if decision == "own":
        return _own(typecode, row, sources)
    if decision == "substitute":
        if set(row) != {"decision", "substitute", "basis"}:
            raise ValueError(f"{where}: a substitute row has keys decision, substitute, basis")
        substitute = str(row["substitute"]).upper()
        if not _native(substitute):
            raise ValueError(f"{where}: substitute {substitute} is not a preset or OpenAP-direct type")
        return IndexEntry(typecode, "substitute", None, substitute,
                          f"substitute {substitute}: {row['basis']}")
    if set(row) != {"decision", "reason"}:
        raise ValueError(f"{where}: an exclude row has keys decision, reason")
    return IndexEntry(typecode, "exclude", None, None, str(row["reason"]))


def load_performance_index(path: Path = PERFORMANCE_INDEX_PATH) -> dict[str, IndexEntry]:
    """Read and validate the index; every decision is checked here, once."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != PERFORMANCE_INDEX_SCHEMA:
        raise ValueError(f"{path}: expected schema {PERFORMANCE_INDEX_SCHEMA!r}, got {data.get('schema')!r}")
    sources, types = data["sources"], data["types"]
    index = {code.upper(): _entry(code.upper(), row, sources) for code, row in types.items()}
    synonyms = {code for code in load_json(PARAMETERS_PATH)["typecodes"]
                if openap_support_kind(code) == "synonym"}
    undecided = sorted(synonyms - set(index))
    if undecided:
        raise ValueError(f"{path}: no decision for the OpenAP synonyms {undecided}")
    return index


@lru_cache(maxsize=1)
def _index() -> dict[str, IndexEntry]:
    return load_performance_index()


@lru_cache(maxsize=1)
def performance_index_identity() -> dict[str, Any]:
    """Which index a result was built with: the packaged file's sha256 and row count."""
    return {
        "sha256": hashlib.sha256(PERFORMANCE_INDEX_PATH.read_bytes()).hexdigest(),
        "types": len(_index()),
    }


def index_entry(typecode: str) -> IndexEntry | None:
    """The index's decision for a type, or None when the type is not in the index (a preset,
    an OpenAP-direct type, or a type the analysis never saw)."""
    return _index().get(typecode.strip().upper())
