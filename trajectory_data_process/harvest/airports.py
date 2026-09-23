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
runway takes its Landing Threshold Point and elevation from its CIFP Runway record and
pairs that MSL elevation with N from the nearest published Path Point at the same
airport. Such a runway is a model target when its RNAV (GPS) approach publishes an
LNAV/VNAV path (KRDU 32, KSMF 35R), so its threshold must be the published point, not
the OurAirports runway end (39.4 m cross-track off at KSMF 35R).

Every runway's COURSE is the centreline through its own and the opposite end's CIFP
Runway records. OurAirports publishes whole degrees, up to 0.45 deg off this fleet's
centrelines (KMSY 11/29) -- ~40 m of cross-track 5 km out on final.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

from geokit import FT_M, bearing_rad

from final_approach import RunwayFrame

from trajectory_data_process.harvest.approach_minima import (
    PublishedMinima,
    RUNWAY_THRESHOLDS_SCHEMA,
)
from trajectory_data_process.harvest.cifp import (
    ApproachVertical,
    PathPoint,
    RunwayRecord,
    read_approach_verticals,
    read_path_points,
    read_runway_records,
)

Datum = Literal["msl", "hae"]
# v2 (2026-09-23): a Runway names where its course came from (``course_source``) and a
# non-LPV threshold is the CIFP Runway record's LTP; every stored event is re-derived.
RUNWAY_DATA_FINGERPRINT_SCHEMA = "harvest-runway-data-v2"
THRESHOLD_FRAME_FINGERPRINT_SCHEMA = "threshold-physical-frame-v2"
PATH_POINT_POSITION_SOURCE = "faa_cifp_path_point"
RUNWAY_RECORD_POSITION_SOURCE = "faa_cifp_runway_record"
RUNWAY_RECORD_COURSE_SOURCE = "faa_cifp_runway_records"
# The configured (OurAirports) heading is whole degrees; this fleet's centrelines sit
# within 0.45 deg of it. A CIFP centreline farther off than this means the two files do
# not describe the same pair of runway ends.
_MAX_COURSE_DISAGREEMENT_DEG = 1.0
# ``Runway.tch_source`` values: the LPV Path Point, or the RNAV (GPS) approach's runway
# leg (``cifp.ApproachVertical``) on a runway that publishes LNAV/VNAV minima only.
PATH_POINT_TCH_SOURCE = "faa_cifp_path_point"
APPROACH_LEG_TCH_SOURCE = "faa_cifp_approach_leg"
# A decoded approach-leg TCH above this means the Runway record's threshold elevation
# and the approach leg's runway altitude disagree, not a real crossing height (the fleet
# publishes 13.7-19.5 m; PANS-OPS design TCH tops out near 25 m). It guards the REPORTED
# TCH only: the judged crossing altitude is elevation + (leg - elevation) = the leg's own
# figure, so a wrong elevation moves this number and never the gate.
_MAX_PLAUSIBLE_TCH_M = 30.0


@dataclass(frozen=True)
class Runway:
    """One landing threshold, with everything needed to judge approaches to it.

    ``lat``/``lon`` are the CIFP Landing Threshold Point (displaced where the runway has
    one), not the pavement end: the Path Point's on an LPV runway, the Runway record's
    otherwise (``position_source``). Six thresholds in this fleet are displaced, KSJC
    30L/30R by 775 m, which on a 3 deg path is a 40.6 m altitude error. ``course_deg`` is
    the centreline through the two ends' Runway records (``course_source``).

    ``threshold_crossing_height_m`` / ``published_glidepath_deg`` are the PUBLISHED
    vertical path: the LPV Path Point's, or -- on a runway without LPV -- the RNAV (GPS)
    approach's runway-leg values when that approach publishes LNAV/VNAV (Baro-VNAV)
    minima (``tch_source`` says which). Both are None only when no vertically guided
    RNAV approach exists at all (KRDU 14); such a runway remains available for
    assignment but is excluded from model arrivals.
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
    width_m: float
    lpv_course_width_m: float | None
    runway_source_cycle: str
    procedure_source_cycle: str
    # What the approach PLATE publishes -- the decision altitude and the service that
    # earns it. Not in the CIFP at any price (``approach_minima``), so it is configured,
    # not decoded. It changes neither assignment nor how a threshold event is measured;
    # it is the height a go-around has to be decided at or above.
    published_minima: PublishedMinima
    # Provenance is carried because CIFP and runway geometry can differ by tens of
    # metres and that difference lands directly in the measured deviations.
    position_source: str = PATH_POINT_POSITION_SOURCE
    vertical_source: str = "faa_cifp_path_point"
    width_source: str = "faa_nasr_apt_rwy"
    course_source: str = RUNWAY_RECORD_COURSE_SOURCE
    # Where the TCH and glidepath come from -- named by whoever sets a TCH, never
    # assumed. NOT part of the physical-frame fingerprint (``threshold_frame_snapshot``):
    # it decides how a crossing is judged, not where the plane it was measured against
    # lies (it IS in ``runway_data_snapshot``, the provenance digest).
    tch_source: str | None = None
    # The runway's RNAV (GPS) approach publishes an LNAV/VNAV line of minima -- the
    # Baro-VNAV context the RNP APCH vertical bound (evaluation) is conditioned on.
    baro_vnav_minima: bool = False

    def __post_init__(self) -> None:
        if abs(self.elevation_hae_m - self.elevation_msl_m - self.hae_minus_msl_m) > 1e-6:
            raise ValueError(f"{self.airport} {self.ident}: inconsistent vertical datum fields")
        if (self.threshold_crossing_height_m is None) != (self.published_glidepath_deg is None):
            raise ValueError(
                f"{self.airport} {self.ident}: TCH and glidepath are published together or "
                "not at all"
            )
        if (self.threshold_crossing_height_m is None) != (self.tch_source is None):
            raise ValueError(
                f"{self.airport} {self.ident}: tch_source must name where the TCH came "
                "from, and be None without one"
            )
        if not math.isfinite(self.width_m) or self.width_m <= 0.0:
            raise ValueError(f"{self.airport} {self.ident}: invalid runway width {self.width_m!r}")
        if self.lpv_course_width_m is not None and (
            not math.isfinite(self.lpv_course_width_m) or self.lpv_course_width_m <= 0.0
        ):
            raise ValueError(
                f"{self.airport} {self.ident}: invalid LPV course width "
                f"{self.lpv_course_width_m!r}"
            )

    def elevation(self, datum: Datum) -> float:
        return self.elevation_msl_m if datum == "msl" else self.elevation_hae_m

    def target_altitude(self, datum: Datum) -> float | None:
        """Where an approach should cross the threshold: elevation + published TCH."""
        if self.threshold_crossing_height_m is None:
            return None
        return self.elevation(datum) + self.threshold_crossing_height_m

    @property
    def decision_height_above_threshold_m(self) -> float:
        """How high the published decision altitude sits above THIS landing threshold.

        Not the plate's own height, which is published above the TOUCHDOWN ZONE and is
        a different point -- 17 ft lower at KRDU 05L. Both sides of this subtraction are
        MSL, so the result is datum-free: it is a height above the threshold, the frame
        a trajectory is judged in.

        Raises on a runway with no vertically guided minima. There is no default to fall
        back to: without a published decision altitude the missed approach point is a
        fix rather than a height, and no altitude answers "was the go-around in time".
        """
        if not self.published_minima.vertically_guided:
            raise ValueError(
                f"{self.airport} {self.ident} publishes no vertically guided minima "
                f"({self.published_minima.note})"
            )
        return self.published_minima.decision_altitude_ft_msl * FT_M - self.elevation("msl")

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


def runway_data_snapshot(runway: Runway) -> dict:
    """Complete runway facts that can change assignment or event interpretation."""
    return {
        "schema_version": RUNWAY_DATA_FINGERPRINT_SCHEMA,
        **asdict(runway),
    }


def runway_data_fingerprint(runway: Runway) -> str:
    """Stable hash binding a threshold event to its exact runway-data cycle/frame."""
    encoded = json.dumps(
        runway_data_snapshot(runway),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_matching_runway_data(event: dict, runway: Runway) -> None:
    """Reject legacy or stale threshold events before mixing runway data cycles."""
    stored = event.get("runway_data_fingerprint")
    current = runway_data_fingerprint(runway)
    if not isinstance(stored, str) or stored != current:
        raise ValueError(
            f"track threshold event has a missing or stale runway-data fingerprint for "
            f"{runway.airport} {runway.ident}; run --reclassify-existing before "
            "--evaluate-only"
        )


def threshold_frame_snapshot(runway: Runway) -> dict:
    """Physical LTP frame facts, excluding every evaluation-policy parameter.

    TCH, glidepath scale, course width, and runway width decide how an event is
    evaluated; they do not change the physical plane where the event was estimated.
    Keeping them out allows a policy-only update to reuse the same measured event.
    """
    return {
        "schema_version": THRESHOLD_FRAME_FINGERPRINT_SCHEMA,
        "airport": runway.airport,
        "runway": runway.ident,
        "threshold_lat": runway.lat,
        "threshold_lon": runway.lon,
        "threshold_elevation_hae_m": runway.elevation_hae_m,
        "threshold_elevation_msl_m": runway.elevation_msl_m,
        "hae_minus_msl_m": runway.hae_minus_msl_m,
        "runway_course_deg": runway.course_deg,
        "runway_source_cycle": runway.runway_source_cycle,
        "procedure_source_cycle": runway.procedure_source_cycle,
        "position_source": runway.position_source,
        "vertical_source": runway.vertical_source,
        "course_source": runway.course_source,
    }


def threshold_frame_fingerprint(runway: Runway) -> str:
    """Stable hash binding a derived event to its physical threshold frame."""
    encoded = json.dumps(
        threshold_frame_snapshot(runway),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_matching_threshold_frame(event: dict, runway: Runway) -> None:
    """Reject an event estimated in a different LTP frame or data cycle."""
    stored = event.get("threshold_frame_fingerprint")
    current = threshold_frame_fingerprint(runway)
    if not isinstance(stored, str) or stored != current:
        raise ValueError(
            f"track threshold event has a missing or stale physical-frame "
            f"fingerprint for {runway.airport} {runway.ident}; run "
            "--reclassify-existing before --evaluate-only"
        )


def load_airport(
    code: str,
    *,
    config_file: Path,
    cifp_file: Path | None = None,
) -> Airport:
    """Build one airport from ``runway_thresholds.json`` plus the required CIFP.

    ``cifp_file`` is mandatory. Configuration supplies the active roster (which runway
    ends exist and pair up), the runway widths and the published minima; every threshold
    position, elevation and course comes from the CIFP -- the Path Point LTP where an LPV
    procedure exists, the Runway record's LTP otherwise, and the course from the Runway
    records of both ends.

    A threshold missing ``heading_deg`` raises: it is the cross-check that the CIFP pair
    and the configured pair are the same runway (``_MAX_COURSE_DISAGREEMENT_DEG``).
    """
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config["schema_version"] != RUNWAY_THRESHOLDS_SCHEMA:
        raise ValueError(
            f"{config_file} is {config['schema_version']}, this reader is "
            f"{RUNWAY_THRESHOLDS_SCHEMA}; run extract_approach_minima.py"
        )
    entry = config["airports"][code]
    if cifp_file is None:
        raise ValueError(f"{code}: CIFP file is required for runway vertical datum facts")
    published: dict[tuple[str, str], PathPoint] = read_path_points(cifp_file, airport=code)
    # The RNAV (GPS) approaches' runway legs, decode-pinned against the Path Points: a
    # runway with no LPV can still publish an LNAV/VNAV path (KRDU 32, KSMF 35R).
    verticals: dict[tuple[str, str], ApproachVertical] = read_approach_verticals(
        cifp_file, airport=code, path_points=published
    )
    records: dict[tuple[str, str], RunwayRecord] = read_runway_records(
        cifp_file, airport=code, path_points=published
    )

    runway_rows = [
        (threshold, runway)
        for runway in entry["runways"]
        for threshold in runway["thresholds"]
    ]
    _require_complete(code, [threshold for threshold, _runway in runway_rows])
    width_cycle = str(entry.get("runway_width_effective_date") or "")
    if not width_cycle:
        raise ValueError(f"{code}: runway_width_effective_date is required")
    procedure_cycle = _cifp_cycle(cifp_file)

    # Where a published LPV exists, its Landing Threshold Point is the threshold; any
    # other runway end takes its CIFP Runway record's LTP. Never the OurAirports runway
    # end: it disagrees measurably -- KSMF 35L/35R by 40.7/39.4 m cross-track (each
    # showed up as that runway's entire apparent lateral error) and KSTL 30L by 61.4 m
    # along-track.
    airport_path_points = tuple(published.values())
    runways = tuple(
        _build_runway(
            code,
            threshold,
            runway_row,
            published.get((code, threshold["ident"])),
            _runway_record(records, code, threshold["ident"]),
            _centreline_course_deg(records, code, threshold, runway_row),
            verticals.get((code, threshold["ident"])),
            airport_path_points,
            PublishedMinima.from_config(threshold["published_minima"]),
            runway_source_cycle=width_cycle,
            procedure_source_cycle=procedure_cycle,
        )
        for threshold, runway_row in runway_rows
    )

    return Airport(
        code=code,
        lat=float(entry["lat"]),
        lon=float(entry["lon"]),
        elevation_msl_m=float(entry.get("elevation_m", 0.0)),
        runways=runways,
    )


def _runway_record(
    records: dict[tuple[str, str], RunwayRecord], code: str, ident: str
) -> RunwayRecord:
    record = records.get((code, ident))
    if record is None:
        raise ValueError(f"{code} {ident}: the CIFP has no Runway record for this threshold")
    return record


def _centreline_course_deg(
    records: dict[tuple[str, str], RunwayRecord],
    code: str,
    threshold: dict,
    runway_row: dict,
) -> float:
    """Landing course of ``threshold``: its LTP towards the opposite end's LTP, compass
    degrees. Both LTPs lie on the centreline (displaced or not), so this is the runway's
    true centreline bearing to ~0.01 deg over a 2-3 km runway."""
    ends = [end["ident"] for end in runway_row["thresholds"]]
    if len(ends) != 2 or threshold["ident"] not in ends:
        raise ValueError(
            f"{code} {runway_row.get('name')}: a runway needs exactly its two ends to "
            f"define a centreline, got {ends}"
        )
    opposite = ends[1] if ends[0] == threshold["ident"] else ends[0]
    own = _runway_record(records, code, threshold["ident"])
    far = _runway_record(records, code, opposite)
    course = math.degrees(
        bearing_rad(own.latitude, own.longitude, far.latitude, far.longitude)
    ) % 360.0
    configured = float(threshold["heading_deg"])
    if abs((course - configured + 180.0) % 360.0 - 180.0) > _MAX_COURSE_DISAGREEMENT_DEG:
        raise ValueError(
            f"{code} {threshold['ident']}: CIFP centreline {course:.2f} deg disagrees with "
            f"the configured heading {configured:g} deg by more than "
            f"{_MAX_COURSE_DISAGREEMENT_DEG:g} deg; the two files describe different "
            "runway ends"
        )
    return course


def _build_runway(
    code: str,
    threshold: dict,
    runway_row: dict,
    point: PathPoint | None,
    record: RunwayRecord,
    course_deg: float,
    vertical: ApproachVertical | None,
    airport_path_points: Sequence[PathPoint],
    minima: PublishedMinima,
    *,
    runway_source_cycle: str,
    procedure_source_cycle: str,
) -> Runway:
    width_ft = runway_row.get("width_ft")
    if width_ft is None:
        raise ValueError(f"{code} {runway_row.get('name')}: FAA NASR runway width is required")
    width_m = float(width_ft) * 0.3048
    baro_vnav_minima = vertical is not None and vertical.baro_vnav_minima
    if point is not None:
        hae = point.ltp_ellipsoidal_height_m
        msl = point.ltp_orthometric_height_m
        return Runway(
            airport=code, ident=threshold["ident"], lat=point.latitude, lon=point.longitude,
            elevation_hae_m=hae, elevation_msl_m=msl, hae_minus_msl_m=hae - msl,
            course_deg=course_deg,
            threshold_crossing_height_m=point.threshold_crossing_height_m,
            published_glidepath_deg=point.glidepath_deg,
            width_m=width_m,
            lpv_course_width_m=point.course_width_m,
            published_minima=minima,
            runway_source_cycle=runway_source_cycle,
            procedure_source_cycle=procedure_source_cycle,
            position_source=PATH_POINT_POSITION_SOURCE,
            vertical_source="faa_cifp_path_point",
            tch_source=PATH_POINT_TCH_SOURCE,
            baro_vnav_minima=baro_vnav_minima,
        )

    if not airport_path_points:
        raise ValueError(
            f"{code}: no CIFP Path Point is available to establish the airport HAE/MSL offset"
        )

    lat = record.latitude
    lon = record.longitude
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
    msl = record.threshold_elevation_msl_m
    hae_minus_msl = (
        reference.ltp_ellipsoidal_height_m
        - reference.ltp_orthometric_height_m
    )
    # No LPV, but an RNAV (GPS) approach with LNAV/VNAV minima still publishes where the
    # path crosses the runway: its runway-leg altitude above the configured threshold
    # elevation is that approach's TCH. An LNAV-only approach (or none at all) leaves
    # the vertical path None -- never defaulted. A Baro-VNAV approach whose leg codes no
    # path, or two that disagree, is a data error and says so.
    tch = glidepath = tch_source = None
    if baro_vnav_minima:
        if vertical.conflict:
            raise ValueError(f"{code} {threshold['ident']}: {vertical.conflict}")
        if vertical.glidepath_deg is None or vertical.crossing_altitude_msl_m is None:
            raise ValueError(
                f"{code} {threshold['ident']}: approach {vertical.procedure} publishes "
                "LNAV/VNAV minima but its runway leg codes no vertical path"
            )
        tch = vertical.crossing_altitude_msl_m - msl
        if not 0.0 < tch < _MAX_PLAUSIBLE_TCH_M:
            raise ValueError(
                f"{code} {threshold['ident']}: approach {vertical.procedure} crosses the "
                f"runway at {vertical.crossing_altitude_msl_m:.1f} m MSL but the Runway "
                f"record's threshold elevation is {msl:.1f} m; the two records disagree"
            )
        glidepath = vertical.glidepath_deg
        tch_source = APPROACH_LEG_TCH_SOURCE
    return Runway(
        airport=code, ident=threshold["ident"], lat=lat, lon=lon,
        elevation_hae_m=msl + hae_minus_msl,
        elevation_msl_m=msl,
        hae_minus_msl_m=hae_minus_msl,
        course_deg=course_deg,
        threshold_crossing_height_m=tch,
        published_glidepath_deg=glidepath,
        width_m=width_m,
        lpv_course_width_m=None,
        published_minima=minima,
        runway_source_cycle=runway_source_cycle,
        procedure_source_cycle=procedure_source_cycle,
        position_source=RUNWAY_RECORD_POSITION_SOURCE,
        vertical_source="nearest_faa_cifp_path_point_offset",
        tch_source=tch_source,
        baro_vnav_minima=baro_vnav_minima,
    )


def _cifp_cycle(path: Path) -> str:
    """``.../CIFP_260806/FAACIFP18`` -> ``2026-08-06``."""
    for part in reversed(path.parts):
        match = re.fullmatch(r"CIFP_(\d{2})(\d{2})(\d{2})", part)
        if match:
            year, month, day = match.groups()
            return f"20{year}-{month}-{day}"
    raise ValueError(f"cannot determine CIFP cycle from {path}")


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
            "Fix them in acquisition/runways.landing_thresholds_from_row and rebuild the "
            "threshold block; do not fill these in by hand or substitute the field "
            "elevation. build_runway_config.py cannot rebuild this file any more — see "
            "docs/code-health-followups.md."
        )
