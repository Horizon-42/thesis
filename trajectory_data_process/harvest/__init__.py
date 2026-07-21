"""harvest — download, reconstruct, and assign ADS-B arrivals, one runway per track.

Pipeline order, and each stage's single responsibility::

    fetch      OpenSky history rows for a bbox around the field      (acquisition/)
      |
    tracks     rows -> ONE contiguous flight per track               (tracks.py)
      |
    classify   track -> at most one runway, via final_approach       (classify.py)
      |
    store      measured tracks + manifest                            (store.py)
      |
    arrivals   assigned + published path + final-entry crop, HAE     (arrivals.py)
      |
    observed   fitted crossings, MSL, judged                         (observed.py)

Two properties this package exists to guarantee:

**One track, one runway.** Assignment is an arg-min over every threshold at once, so
double-assignment is unrepresentable rather than guarded against. The predecessor's
guard was correct and still shipped artifacts where 72.7% of KSJC's landings sat in two
runways' files.

**Measured and inferred stay apart.** ``tracks/`` and its model-ready ``arrivals/`` view
hold measured HAE samples; ``approach/`` holds what a fit inferred in MSL. The gap between
the last measured sample and the inferred crossing is precisely why they must not be read
as the same kind of thing.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from trajectory_data_process.harvest.airports import Airport, Datum, Runway, load_airport
from trajectory_data_process.harvest.arrivals import (
    load_arrival_flights,
    resolve_arrival_manifest,
    write_arrival_records,
)
from trajectory_data_process.harvest.cifp import PathPoint, read_path_points
from trajectory_data_process.harvest.czml import RenderedObserved, render_observed_czml
from trajectory_data_process.harvest.classify import (
    ClassifiedTrack,
    classify_track,
    classify_tracks,
)
from trajectory_data_process.harvest.store import (
    HarvestPaths,
    iter_records,
    read_manifest,
    track_record,
    write_tracks,
)
from trajectory_data_process.harvest.tracks import Sample, Track, reconstruct_tracks

_RUNNER_EXPORTS = frozenset({"HarvestPlan", "HarvestResult", "harvest_airport"})


def __getattr__(name: str) -> Any:
    """Load acquisition-only exports only when a caller actually requests them.

    Importing any ``harvest.*`` submodule executes this package initializer first.  The
    runner imports the OpenSky dataframe adapter (and therefore pandas), while manifest
    readers such as the TS loader need none of that acquisition stack.  Keeping these
    three convenience exports lazy preserves ``from ...harvest import HarvestPlan``
    without making pandas an import-time dependency of every harvest consumer.
    """
    if name not in _RUNNER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("trajectory_data_process.harvest.runner"), name)
    globals()[name] = value
    return value

__all__ = [
    "Airport",
    "Runway",
    "Datum",
    "load_airport",
    "write_arrival_records",
    "load_arrival_flights",
    "resolve_arrival_manifest",
    "PathPoint",
    "read_path_points",
    "Sample",
    "Track",
    "reconstruct_tracks",
    "ClassifiedTrack",
    "classify_track",
    "classify_tracks",
    "RenderedObserved",
    "render_observed_czml",
    "HarvestPaths",
    "write_tracks",
    "read_manifest",
    "iter_records",
    "track_record",
    "HarvestPlan",
    "HarvestResult",
    "harvest_airport",
]
