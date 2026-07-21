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
    arrival    fitted crossings, MSL, judged                         (evaluation/arrival.py)

Two properties this package exists to guarantee:

**One track, one runway.** Assignment is an arg-min over every threshold at once, so
double-assignment is unrepresentable rather than guarded against. The predecessor's
guard was correct and still shipped artifacts where 72.7% of KSJC's landings sat in two
runways' files.

**Measured and inferred stay apart.** ``tracks/`` holds what the sensors said (HAE, as
broadcast, no model). ``approach/`` holds what a fit inferred (MSL, extrapolated to a
threshold the receivers never saw). The gap between them is 325 m of missing final
approach, which is precisely why they must not be read as the same kind of thing.
"""

from __future__ import annotations

from trajectory_data_process.harvest.airports import Airport, Datum, Runway, load_airport
from trajectory_data_process.harvest.cifp import PathPoint, read_path_points
from trajectory_data_process.harvest.czml import RenderedObserved, render_observed_czml
from trajectory_data_process.harvest.classify import (
    ClassifiedTrack,
    classify_track,
    classify_tracks,
)
from trajectory_data_process.harvest.runner import (
    HarvestPlan,
    HarvestResult,
    harvest_airport,
)
from trajectory_data_process.harvest.store import (
    HarvestPaths,
    iter_records,
    read_manifest,
    track_record,
    write_tracks,
)
from trajectory_data_process.harvest.tracks import Sample, Track, reconstruct_tracks

__all__ = [
    "Airport",
    "Runway",
    "Datum",
    "load_airport",
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
