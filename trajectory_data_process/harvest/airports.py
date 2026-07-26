"""Airport + runway definitions, resolved once into both vertical datums.

THE DATUM PROBLEM THIS MODULE EXISTS TO SOLVE
---------------------------------------------
Three vertical references meet at a runway threshold and none of them is optional:

  * ADS-B ``geoaltitude`` is height above the WGS84 ELLIPSOID (HAE);
  * CIFP publishes same-point threshold HAE and orthometric (MSL) heights;
  * their fixed difference N is carried with every assigned runway.

The harvest keeps tracks in HAE, faithful to the sensor, because the CZML the viewer
consumes is ellipsoidal (see ``harvest/czml.py``). The modeling plane needs
MSL. So a runway must be able to present itself in EITHER datum, and the choice must be
explicit at every call site -- a silent mix is a 33 m error that looks like nothing.

The predecessor got this wrong in the landing screen: it compared HAE track altitudes
against an MSL threshold elevation. On a 1500 m gate that is only 2.2%, but it was the
fourth appearance of the same family of bug in this project, so here the datum is a
REQUIRED argument of ``Runway.frame`` rather than a convention to remember.

For an LPV runway, N comes directly from its same-point CIFP HAE/MSL pair. A non-LPV
runway exists only as an assignment candidate: its configured MSL elevation is paired
with N from the nearest published Path Point at the same airport. This keeps assignment
in the airport's CIFP-derived datum without turning the fallback into a model target.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from final_approach import RunwayFrame

from trajectory_data_process.harvest.cifp import PathPoint, read_path_points

Datum = Literal["msl", "hae"]


@dataclass(frozen=True)
class Runway:
    """One landing threshold, with everything needed to judge approaches to it.

    ``lat``/``lon`` are the LANDING threshold (displaced where the runway has one), not
    the pavement end -- see ``acquisition/runways.py``. Six thresholds in this fleet are
    displaced, KSJC 30L/30R by 775 m, which on a 3 deg path is a 40.6 m altitude error.

    ``threshold_crossing_height_m`` is the PUBLISHED TCH, or None for a non-LPV runway.
    Such a runway remains available for assignment but is excluded from model arrivals.
    """

    airport: str
    ident: str
    lat: float
    lon: float
    elevation_hae_m: float
    elevation_msl_m: float
    course_deg: float
    hae_minus_msl_m: float
    threshold_crossing_height_m: float | None
    published_glidepath_deg: float | None
    # Provenance is carried because CIFP and runway geometry can differ by tens of
    # metres and that difference lands directly in the measured deviations.
    position_source: str = "faa_cifp_path_point"
    vertical_source: str = "faa_cifp_path_point"

    def __post_init__(self) -> None:
        if abs(self.elevation_hae_m - self.elevation_msl_m - self.hae_minus_msl_m) > 1e-6:
            raise ValueError(f"{self.airport} {self.ident}: inconsistent vertical datum fields")

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
    """Build one airport from ``runway_thresholds.json`` plus the required CIFP.

    ``cifp_file`` is mandatory. Configuration supplies the active roster and the
    geometry/elevation fallback needed to classify landings on non-LPV runways; CIFP
    Path Points replace that fallback wherever an LPV procedure exists.

    A threshold missing ``heading_deg`` raises because there is no runway frame without
    a heading. A non-LPV threshold also requires its configured MSL elevation.
    """
    config = json.loads(config_file.read_text(encoding="utf-8"))
    entry = config["airports"][code]
    if cifp_file is None:
        raise ValueError(f"{code}: CIFP file is required for runway vertical datum facts")
    published: dict[tuple[str, str], PathPoint] = read_path_points(cifp_file, airport=code)

    thresholds = [t for runway in entry["runways"] for t in runway["thresholds"]]
    _require_complete(code, thresholds)

    # Where a published LPV exists, its Landing Threshold Point WINS over the runway
    # geometry derived from OurAirports. The LTP is what the procedure is aimed at, so it
    # is the correct along-track origin, and the two disagree measurably: KSMF 35L by
    # 40.7 m cross-track (which showed up as that runway's entire apparent lateral error)
    # and KSTL 30L by 61.4 m along-track. Where both agree the substitution is a no-op --
    # 19 of this fleet's 23 LPV runways are within 10 m.
    airport_path_points = tuple(published.values())
    runways = tuple(
        _build_runway(
            code,
            t,
            published.get((code, t["ident"])),
            airport_path_points,
        )
        for t in thresholds
    )

    return Airport(
        code=code,
        lat=float(entry["lat"]),
        lon=float(entry["lon"]),
        elevation_msl_m=float(entry.get("elevation_m", 0.0)),
        runways=runways,
    )


def _build_runway(
    code: str,
    threshold: dict,
    point: PathPoint | None,
    airport_path_points: Sequence[PathPoint],
) -> Runway:
    if point is not None:
        hae = point.ltp_ellipsoidal_height_m
        msl = point.ltp_orthometric_height_m
        return Runway(
            airport=code, ident=threshold["ident"], lat=point.latitude, lon=point.longitude,
            elevation_hae_m=hae, elevation_msl_m=msl, hae_minus_msl_m=hae - msl,
            course_deg=float(threshold["heading_deg"]),
            threshold_crossing_height_m=point.threshold_crossing_height_m,
            published_glidepath_deg=point.glidepath_deg,
            position_source="faa_cifp_path_point",
            vertical_source="faa_cifp_path_point",
        )

    if threshold.get("elevation_m") is None:
        raise ValueError(
            f"{code} {threshold['ident']}: non-LPV runway has no configured MSL elevation"
        )
    if not airport_path_points:
        raise ValueError(
            f"{code}: no CIFP Path Point is available to establish the airport HAE/MSL offset"
        )

    lat = float(threshold["lat"])
    lon = float(threshold["lon"])
    reference = min(
        airport_path_points,
        key=lambda candidate: (
            (candidate.latitude - lat) ** 2
            + (
                (candidate.longitude - lon)
                * math.cos(math.radians(lat))
            ) ** 2
        ),
    )
    msl = float(threshold["elevation_m"])
    hae_minus_msl = (
        reference.ltp_ellipsoidal_height_m
        - reference.ltp_orthometric_height_m
    )
    return Runway(
        airport=code, ident=threshold["ident"], lat=lat, lon=lon,
        elevation_hae_m=msl + hae_minus_msl,
        elevation_msl_m=msl,
        hae_minus_msl_m=hae_minus_msl,
        course_deg=float(threshold["heading_deg"]),
        threshold_crossing_height_m=None,
        published_glidepath_deg=None,
        position_source="runway_geometry",
        vertical_source="nearest_faa_cifp_path_point_offset",
    )


def _require_complete(code: str, thresholds: Sequence[dict]) -> None:
    incomplete = [
        f"{t['ident']} (missing "
        + ", ".join(k for k in ("heading_deg",) if t.get(k) is None)
        + ")"
        for t in thresholds
        if t.get("heading_deg") is None
    ]
    if incomplete:
        raise ValueError(
            f"{code}: {len(incomplete)} threshold(s) are unusable — {'; '.join(incomplete)}. "
            "Regenerate runway_thresholds.json with build_runway_config.py; do not fill "
            "these in by hand or substitute the field elevation."
        )
