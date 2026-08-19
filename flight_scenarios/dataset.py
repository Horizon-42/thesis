"""Choose WHICH arrivals become a scenario dataset, and record the choice.

``build.build_scenarios_from_arrivals`` is the strict builder: every flight it is handed
becomes a scenario or it raises.  That is the right contract for a library, and the wrong
one for a batch over a real harvest, where two population decisions have to be made and,
having been made, have to be *visible*:

* **Per-runway cap.**  The arrival manifests hold 42,725 flights and the runways are
  wildly unbalanced (KSJC 30L 9,603 vs 12L 14).  An uncapped batch spends most of its
  compute re-measuring the two busiest runways.  The cap is applied per runway, evenly
  spaced over landing time, so a capped runway still spans the whole harvest window
  instead of its first N days.
* **Unusable fitted approaches.**  35 of those 42,725 flights (0.08 %) have no usable
  ``final_approach`` fit, and ``build_scenario`` raises on them — which aborted the whole
  fitted-ADS-B dataset for 4 of the 5 airports.  They are dropped here, individually
  named, instead.

Both decisions are returned in a :class:`SelectionReport` and written beside the scenario
file as ``*.selection.json``: a bounded population that is not stated in the output reads
as a full population, and every rate computed from it would be quietly wrong.

The cap depends only on ``(runway, landing_time_utc, flight_key)`` — all of which come
from the arrival manifest roster and none of which depend on the target type — so the
fitted-ADS-B and runway-threshold datasets select the SAME flights and stay comparable
per flight.  The fitted-ADS-B set is then that set minus its unfittable members.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fitted_approach import UnusableFittedApproach
from .identity import flight_key

SELECTION_SCHEMA_VERSION = "flight-scenarios-selection-v1"
SELECTION_SUFFIX = ".selection.json"

# How a capped runway's flights are chosen. Named because the report repeats the claim.
SELECTION_RULE = "evenly spaced over landing time within each runway"


@dataclass
class RunwaySelection:
    """One runway's population accounting."""

    available: int = 0
    selected: int = 0
    excluded_unfittable: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "selected": self.selected,
            "excluded_unfittable": self.excluded_unfittable,
        }


@dataclass
class SelectionReport:
    """The population decisions behind one scenario dataset."""

    airport: str
    target: str
    max_per_runway: int | None
    available: int
    selected: int
    per_runway: dict[str, RunwaySelection] = field(default_factory=dict)
    excluded_unfittable: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "airport": self.airport,
            "target": self.target,
            "max_per_runway": self.max_per_runway,
            "selection_rule": SELECTION_RULE if self.max_per_runway else "all rostered arrivals",
            "available": self.available,
            "selected": self.selected,
            "per_runway": {
                runway: row.to_dict() for runway, row in sorted(self.per_runway.items())
            },
            "excluded_unfittable": self.excluded_unfittable,
        }

    def summary_line(self) -> str:
        capped = [
            runway
            for runway, row in sorted(self.per_runway.items())
            if self.max_per_runway is not None and row.available > self.max_per_runway
        ]
        parts = [f"{self.selected}/{self.available} arrival(s)"]
        if self.max_per_runway is not None:
            parts.append(
                f"cap {self.max_per_runway}/runway"
                + (f" (binds on {', '.join(capped)})" if capped else " (does not bind)")
            )
        if self.excluded_unfittable:
            parts.append(f"{len(self.excluded_unfittable)} dropped: no usable fitted approach")
        return "; ".join(parts)


def selection_path(scenario_output: str | Path) -> Path:
    """The provenance sidecar written beside a scenario JSON."""
    output = Path(scenario_output)
    return output.with_name(output.name + SELECTION_SUFFIX)


def write_selection(report: SelectionReport, scenario_output: str | Path) -> Path:
    path = selection_path(scenario_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


def _even_indices(count: int, take: int) -> list[int]:
    """``take`` indices spread evenly over ``range(count)``, endpoints included.

    ``take >= count`` returns every index.  For ``take == 1`` the single sample is the
    first, so a one-flight selection is still the earliest landing rather than a
    position that moves with the roster length.
    """
    if take >= count:
        return list(range(count))
    if take <= 1:
        return [0]
    step = (count - 1) / (take - 1)
    return sorted({round(i * step) for i in range(take)})


def select_flight_keys(
    manifest_path: str | Path,
    *,
    max_per_runway: int | None,
) -> tuple[list[str], SelectionReport, str]:
    """Choose the flight_keys for one airport's dataset from the arrival ROSTER alone.

    Reads only ``arrivals/manifest.json`` — no source track file is opened — so the cap is
    applied before ``load_arrival_flights`` pays for the tracks it would otherwise read and
    SHA-256 verify.  Returns ``(flight_keys, report, airport)``; the report's per-runway
    ``excluded_unfittable`` is filled in later by the builder.
    """
    if max_per_runway is not None and max_per_runway < 1:
        raise ValueError(f"max_per_runway must be >= 1, got {max_per_runway}")
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    airport = str(manifest.get("airport") or "").upper()
    if not airport:
        raise ValueError(f"{path} does not declare an airport")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path} lacks a records roster")

    by_runway: dict[str, list[tuple[str, str]]] = {}
    for row in records:
        runway = str(row.get("runway") or "unknown")
        # Sorted by landing time, then flight_key: the roster order is the harvest's, and
        # a cap that depends on it would move when the harvest is re-chunked.
        by_runway.setdefault(runway, []).append(
            (str(row.get("landing_time_utc") or ""), str(row["flight_key"]))
        )

    report = SelectionReport(
        airport=airport, target="", max_per_runway=max_per_runway,
        available=len(records), selected=0,
    )
    keys: list[str] = []
    for runway, rows in by_runway.items():
        rows.sort()
        take = len(rows) if max_per_runway is None else min(max_per_runway, len(rows))
        chosen = [rows[index][1] for index in _even_indices(len(rows), take)]
        report.per_runway[runway] = RunwaySelection(
            available=len(rows), selected=len(chosen)
        )
        keys.extend(chosen)
    report.selected = len(keys)
    return keys, report, airport


def build_scenario_dataset(
    manifest_path: str | Path,
    aircraft_type: str | None = None,
    *,
    target: str,
    max_per_runway: int | None = None,
    mass_kg: float | None = None,
    window_s: float | None = None,
    aircraft_provider: str = "auto",
) -> tuple[list[Any], SelectionReport]:
    """Build one airport's scenario dataset, capped and with unfittable flights dropped.

    ``target`` is ``"runway"`` (published threshold), ``"fitted-adsb"`` (the fitted OLS
    threshold crossing) or ``"track-end"``.  Only the fitted-ADS-B target can be unusable,
    and only for the individual flight — so it is the only one that ever drops anything.
    """
    # Local imports: this module is the batch policy layer, and `build` pulls in the whole
    # aircraft/aerodynamic stack that the pure roster read above does not need.
    from .build import build_scenario, load_model_arrivals
    from .start_state import DEFAULT_WINDOW_S

    targets = ("runway", "fitted-adsb", "track-end")
    if target not in targets:
        raise ValueError(f"unknown target {target!r}; expected one of {targets}")
    window = DEFAULT_WINDOW_S if window_s is None else window_s

    keys, report, airport = select_flight_keys(
        manifest_path, max_per_runway=max_per_runway
    )
    report.target = target
    flights = load_model_arrivals_subset(manifest_path, keys)

    scenarios: list[Any] = []
    for flight in flights:
        try:
            scenarios.append(build_scenario(
                flight, aircraft_type, airport=airport, mass_kg=mass_kg,
                window_s=window,
                target_from_threshold=target == "runway",
                target_from_fitted_adsb=target == "fitted-adsb",
                aircraft_provider=aircraft_provider,
            ))
        except UnusableFittedApproach as exc:
            runway = str(flight.get("runway") or "unknown")
            report.excluded_unfittable.append({
                "flight_key": flight_key(flight, len(scenarios)),
                "runway": runway,
                "reason": str(exc),
            })
            row = report.per_runway.get(runway)
            if row is not None:
                row.excluded_unfittable += 1
                row.selected -= 1
    report.selected = len(scenarios)
    return scenarios, report


def load_model_arrivals_subset(
    manifest_path: str | Path, flight_keys: list[str]
) -> list[dict[str, Any]]:
    """The selected arrivals only, through the one datum-converting loader.

    ``load_arrival_flights``'s ``include_flight_keys`` never opens an excluded track file,
    so a capped dataset does not pay to read and SHA-256 verify the tracks it discards —
    the whole point of applying the cap to the roster rather than to the built scenarios.
    """
    from trajectory_data_process.harvest.arrivals import load_arrival_flights

    from .datum import flights_to_msl

    return flights_to_msl(
        load_arrival_flights(manifest_path, include_flight_keys=set(flight_keys))
    )
