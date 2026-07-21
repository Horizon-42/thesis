"""Airport + runway definitions, resolved once into both vertical datums.

THE DATUM PROBLEM THIS MODULE EXISTS TO SOLVE
---------------------------------------------
Three vertical references meet at a runway threshold and none of them is optional:

  * ADS-B ``geoaltitude`` is height above the WGS84 ELLIPSOID (HAE);
  * ``runway_thresholds.json`` elevations and CIFP altitudes are MSL (orthometric);
  * the geoid undulation N between them is about -33 m over the continental US.

The harvest keeps tracks in HAE, faithful to the sensor, because the CZML the viewer
consumes is ellipsoidal (see ``processing/czml_export.py``). The modeling plane needs
MSL. So a runway must be able to present itself in EITHER datum, and the choice must be
explicit at every call site -- a silent mix is a 33 m error that looks like nothing.

The predecessor got this wrong in the landing screen: it compared HAE track altitudes
against an MSL threshold elevation. On a 1500 m gate that is only 2.2%, but it was the
fourth appearance of the same family of bug in this project, so here the datum is a
REQUIRED argument of ``Runway.frame`` rather than a convention to remember.

N comes from ``flight_scenarios.datum.geoid_undulation_m`` — deliberately imported
rather than reimplemented. A second geoid implementation is exactly how the original
33 m bug reached five airports, and dependency tidiness is not worth a third copy of a
conversion that has already broken once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from final_approach import RunwayFrame

from flight_scenarios.datum import geoid_undulation_m
from trajectory_data_process.harvest.cifp import PathPoint, read_path_points

Datum = Literal["msl", "hae"]


@dataclass(frozen=True)
class Runway:
    """One landing threshold, with everything needed to judge approaches to it.

    ``lat``/``lon`` are the LANDING threshold (displaced where the runway has one), not
    the pavement end -- see ``acquisition/runways.py``. Six thresholds in this fleet are
    displaced, KSJC 30L/30R by 775 m, which on a 3 deg path is a 40.6 m altitude error.

    ``threshold_crossing_height_m`` is the PUBLISHED TCH, or None when the runway has no
    LPV procedure. None is not a missing value to be filled with a default: it means the
    runway cannot be judged against LPV gates at all.
    """

    airport: str
    ident: str
    lat: float
    lon: float
    elevation_msl_m: float
    course_deg: float
    geoid_undulation_m: float
    threshold_crossing_height_m: float | None
    published_glidepath_deg: float | None

    @property
    def elevation_hae_m(self) -> float:
        """h_HAE = H_MSL + N."""
        return self.elevation_msl_m + self.geoid_undulation_m

    def elevation(self, datum: Datum) -> float:
        return self.elevation_msl_m if datum == "msl" else self.elevation_hae_m

    def target_altitude(self, datum: Datum) -> float | None:
        """Where an approach should cross the threshold: elevation + published TCH."""
        if self.threshold_crossing_height_m is None:
            return None
        return self.elevation(datum) + self.threshold_crossing_height_m

    def frame(self, datum: Datum) -> RunwayFrame:
        """The runway-aligned frame, in the requested datum.

        ``datum`` is required: the caller's track altitudes decide it, and getting it
        wrong shifts every height by ~33 m without any symptom.
        """
        return RunwayFrame(
            ident=self.ident,
            lat=self.lat,
            lon=self.lon,
            elevation_m=self.elevation(datum),
            course_deg=self.course_deg,
        )


@dataclass(frozen=True)
class Airport:
    """An airport and its full threshold list."""

    code: str
    lat: float
    lon: float
    elevation_msl_m: float
    runways: tuple[Runway, ...]

    def frames(self, datum: Datum) -> list[RunwayFrame]:
        """Frames for EVERY threshold.

        Assignment must be shown all of them: ``final_approach.assign_runway`` takes an
        arg-min, and it can only rule out a competitor it was given.
        """
        return [r.frame(datum) for r in self.runways]

    def runway(self, ident: str) -> Runway:
        for r in self.runways:
            if r.ident == ident:
                return r
        raise KeyError(f"{self.code} has no threshold {ident!r}")


def load_airport(
    code: str,
    *,
    config_file: Path,
    cifp_file: Path | None = None,
) -> Airport:
    """Build one airport from ``runway_thresholds.json`` plus (optionally) the CIFP.

    Without ``cifp_file`` every runway's TCH is None, which is honest but leaves nothing
    to judge a crossing against; the pipeline always passes it.

    A threshold missing ``heading_deg`` or ``elevation_m`` RAISES. Both are structural:
    without a heading there is no runway frame, and without an elevation there is no
    height reference. Substituting the airport's field elevation would silently move the
    vertical reference by the runway's slope -- KMSY 20 is currently missing its
    elevation in the generated config, and that is a generator bug to fix at the source,
    not to paper over here.
    """
    config = json.loads(config_file.read_text(encoding="utf-8"))
    entry = config["airports"][code]
    published: dict[tuple[str, str], PathPoint] = (
        read_path_points(cifp_file) if cifp_file is not None else {}
    )

    thresholds = [t for runway in entry["runways"] for t in runway["thresholds"]]
    _require_complete(code, thresholds)

    undulations = geoid_undulation_m(
        [t["lat"] for t in thresholds], [t["lon"] for t in thresholds]
    )

    runways = tuple(
        Runway(
            airport=code,
            ident=t["ident"],
            lat=float(t["lat"]),
            lon=float(t["lon"]),
            elevation_msl_m=float(t["elevation_m"]),
            course_deg=float(t["heading_deg"]),
            geoid_undulation_m=n,
            threshold_crossing_height_m=(
                published[(code, t["ident"])].threshold_crossing_height_m
                if (code, t["ident"]) in published
                else None
            ),
            published_glidepath_deg=(
                published[(code, t["ident"])].glidepath_deg
                if (code, t["ident"]) in published
                else None
            ),
        )
        for t, n in zip(thresholds, undulations)
    )

    return Airport(
        code=code,
        lat=float(entry["lat"]),
        lon=float(entry["lon"]),
        elevation_msl_m=float(entry.get("elevation_m", 0.0)),
        runways=runways,
    )


def _require_complete(code: str, thresholds: Sequence[dict]) -> None:
    incomplete = [
        f"{t['ident']} (missing "
        + ", ".join(k for k in ("heading_deg", "elevation_m") if t.get(k) is None)
        + ")"
        for t in thresholds
        if t.get("heading_deg") is None or t.get("elevation_m") is None
    ]
    if incomplete:
        raise ValueError(
            f"{code}: {len(incomplete)} threshold(s) are unusable — {'; '.join(incomplete)}. "
            "Regenerate runway_thresholds.json with build_runway_config.py; do not fill "
            "these in by hand or substitute the field elevation."
        )
