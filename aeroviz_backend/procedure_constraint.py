"""Canonical procedure constraint — the Python side of the front↔back contract.

This mirrors ``aeroviz-4d/src/data/procedureConstraint.ts``. The frontend builds
a :class:`ProcedureConstraint` from a procedure detail document and ships it in
the optimizer request as JSON; here we parse the SAME shape so the optimizer and
the dynamics model can read the published path, altitude windows and speed limits
directly.

It carries only what a path/altitude/speed constraint needs: an ordered list of
waypoints (position + altitude window + speed), the final-approach course and the
coded glidepath. The one canonical altitude type (:class:`AltitudeWindow`) and the
one canonical CIFP→window conversion live here too, so the backend never invents a
second interpretation of the coded data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from geokit import FT_M as FEET_TO_METERS
from geokit import WGS84_A as _EARTH_RADIUS_M

# Altitude-window kinds, matching the TypeScript ``AltitudeConstraint["kind"]``.
AT = "AT"
AT_OR_ABOVE = "AT_OR_ABOVE"
AT_OR_BELOW = "AT_OR_BELOW"
WINDOW = "WINDOW"
UNKNOWN = "UNKNOWN"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class AltitudeWindow:
    """The one canonical altitude representation (matches TS ``AltitudeConstraint``)."""

    kind: str
    min_ft_msl: float | None = None
    max_ft_msl: float | None = None
    source_text: str | None = None

    @property
    def reference_ft(self) -> float | None:
        """The single binding altitude used when a scalar target is needed."""
        if self.kind == AT_OR_BELOW:
            return self.max_ft_msl
        if self.kind == WINDOW:
            return self.min_ft_msl if self.min_ft_msl is not None else self.max_ft_msl
        return self.min_ft_msl if self.min_ft_msl is not None else self.max_ft_msl

    @classmethod
    def from_payload(cls, value: Any) -> "AltitudeWindow | None":
        if not isinstance(value, dict):
            return None
        return cls(
            kind=str(value.get("kind", UNKNOWN)),
            min_ft_msl=_as_float(value.get("minFtMsl")),
            max_ft_msl=_as_float(value.get("maxFtMsl")),
            source_text=value.get("sourceText"),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.min_ft_msl is not None:
            payload["minFtMsl"] = self.min_ft_msl
        if self.max_ft_msl is not None:
            payload["maxFtMsl"] = self.max_ft_msl
        if self.source_text is not None:
            payload["sourceText"] = self.source_text
        return payload


def altitude_window_from_cifp(altitude: Any) -> AltitudeWindow | None:
    """Canonical conversion from a coded CIFP leg altitude to an AltitudeWindow.

    Mirrors ``altitudeConstraintFromCifp`` in the frontend so an on-disk detail
    document read here yields the same windows the frontend ships.
    """
    if not isinstance(altitude, dict):
        return None
    value_ft = _as_float(altitude.get("valueFt"))
    qualifier = str(altitude.get("qualifier", "")).lower()
    raw_text = str(altitude.get("rawText", ""))

    looks_like_window = (
        "block" in qualifier
        or "window" in qualifier
        or "between" in qualifier
        or bool(re.search(r"\bbetween\b|-", raw_text))
    )
    if looks_like_window:
        bounds = _parse_window_bounds(raw_text)
        if bounds is not None:
            low, high = bounds
            return AltitudeWindow(WINDOW, low, high, raw_text)

    if "above" in qualifier:
        return AltitudeWindow(AT_OR_ABOVE, min_ft_msl=value_ft, source_text=raw_text)
    if "below" in qualifier:
        return AltitudeWindow(AT_OR_BELOW, max_ft_msl=value_ft, source_text=raw_text)
    return AltitudeWindow(
        UNKNOWN if "unknown" in qualifier else AT,
        min_ft_msl=value_ft,
        max_ft_msl=value_ft,
        source_text=raw_text,
    )


def _parse_window_bounds(raw_text: str) -> tuple[float, float] | None:
    numbers = [
        float(match.replace(",", ""))
        for match in re.findall(r"\d[\d,]*", raw_text)
    ]
    numbers = [number for number in numbers if math.isfinite(number)]
    if len(numbers) < 2:
        return None
    first, second = numbers[0], numbers[1]
    return (min(first, second), max(first, second))


@dataclass(frozen=True)
class ProcedureConstraintWaypoint:
    fix_id: str
    ident: str
    role: str
    leg_type: str
    lon_deg: float
    lat_deg: float
    altitude: AltitudeWindow | None
    altitude_ref_ft: float | None
    geometry_alt_ft: float | None
    speed_max_kt: float | None
    distance_from_start_m: float

    @classmethod
    def from_payload(cls, value: dict[str, Any]) -> "ProcedureConstraintWaypoint":
        return cls(
            fix_id=str(value.get("fixId", "")),
            ident=str(value.get("ident", "")),
            role=str(value.get("role", "")),
            leg_type=str(value.get("legType", "")),
            lon_deg=float(value["lonDeg"]),
            lat_deg=float(value["latDeg"]),
            altitude=AltitudeWindow.from_payload(value.get("altitude")),
            altitude_ref_ft=_as_float(value.get("altitudeRefFt")),
            geometry_alt_ft=_as_float(value.get("geometryAltFt")),
            speed_max_kt=_as_float(value.get("speedMaxKt")),
            distance_from_start_m=float(value.get("distanceFromStartM", 0.0)),
        )


@dataclass(frozen=True)
class Glidepath:
    angle_deg: float
    tch_ft: float | None


@dataclass(frozen=True)
class ProcedureConstraint:
    procedure_uid: str
    airport_icao: str
    runway_ident: str | None
    branch_id: str
    approach_course_deg: float | None
    glidepath: Glidepath | None
    nominal_speed_kt: float
    waypoints: tuple[ProcedureConstraintWaypoint, ...]

    # ── parsing ────────────────────────────────────────────────────────────
    @classmethod
    def from_payload(cls, payload: Any) -> "ProcedureConstraint | None":
        """Parse the JSON the frontend ships (the request ``procedureConstraint``)."""
        if not isinstance(payload, dict):
            return None
        raw_waypoints = payload.get("waypoints")
        if not isinstance(raw_waypoints, list) or not raw_waypoints:
            return None
        glidepath_payload = payload.get("glidepath")
        glidepath = None
        if isinstance(glidepath_payload, dict):
            angle = _as_float(glidepath_payload.get("angleDeg"))
            if angle is not None:
                glidepath = Glidepath(angle, _as_float(glidepath_payload.get("tchFt")))
        return cls(
            procedure_uid=str(payload.get("procedureUid", "")),
            airport_icao=str(payload.get("airportIcao", "")),
            runway_ident=payload.get("runwayIdent"),
            branch_id=str(payload.get("branchId", "")),
            approach_course_deg=_as_float(payload.get("approachCourseDeg")),
            glidepath=glidepath,
            nominal_speed_kt=float(payload.get("nominalSpeedKt", 0.0)),
            waypoints=tuple(
                ProcedureConstraintWaypoint.from_payload(item) for item in raw_waypoints
            ),
        )

    @classmethod
    def from_detail_document(
        cls,
        document: dict[str, Any],
        branch_id: str | None = None,
    ) -> "ProcedureConstraint | None":
        """Build the constraint independently from an on-disk detail document.

        Reads a single approach branch (the requested one, else the base branch,
        else the first ``final`` branch) directly from the coded legs. Unlike the
        frontend builder it does not follow cross-branch continuation — it is the
        backend's own minimal reader of the canonical fields (handy for tests and
        for understanding a bundled procedure without the frontend).
        """
        branch = _select_branch(document, branch_id)
        if branch is None:
            return None
        fixes = {fix["fixId"]: fix for fix in document.get("fixes", [])}

        waypoints: list[ProcedureConstraintWaypoint] = []
        cumulative_m = 0.0
        previous: tuple[float, float] | None = None
        for leg in sorted(branch.get("legs", []), key=lambda leg: leg.get("sequence", 0)):
            if leg.get("quality", {}).get("renderedInPlanView") is not True:
                continue
            fix = fixes.get(leg.get("path", {}).get("endFixRef"))
            position = fix.get("position") if fix else None
            if not fix or not position:
                continue
            lon, lat = float(position["lon"]), float(position["lat"])
            if previous is not None:
                cumulative_m += _distance_m(previous, (lon, lat))
            previous = (lon, lat)
            constraints = leg.get("constraints", {})
            altitude = altitude_window_from_cifp(constraints.get("altitude"))
            waypoints.append(
                ProcedureConstraintWaypoint(
                    fix_id=fix["fixId"],
                    ident=fix["ident"],
                    role=leg.get("roleAtEnd", ""),
                    leg_type=leg.get("path", {}).get("pathTerminator", ""),
                    lon_deg=lon,
                    lat_deg=lat,
                    altitude=altitude,
                    altitude_ref_ft=altitude.reference_ft if altitude else None,
                    geometry_alt_ft=_as_float(constraints.get("geometryAltitudeFt")),
                    speed_max_kt=_as_float(constraints.get("speedKt")),
                    distance_from_start_m=round(cumulative_m, 1),
                )
            )

        if len(waypoints) < 2:
            return None

        procedure = document.get("procedure", {})
        course = _bearing_deg(
            (waypoints[-2].lon_deg, waypoints[-2].lat_deg),
            (waypoints[-1].lon_deg, waypoints[-1].lat_deg),
        )
        return cls(
            procedure_uid=str(document.get("procedureUid", "")),
            airport_icao=str(document.get("airport", {}).get("icao", "")).upper(),
            runway_ident=procedure.get("runwayIdent") or document.get("runway", {}).get("ident"),
            branch_id=branch["branchId"],
            approach_course_deg=course,
            glidepath=_document_glidepath(document),
            nominal_speed_kt=float(document.get("displayHints", {}).get("nominalSpeedKt", 0.0)),
            waypoints=tuple(waypoints),
        )

    # ── accessors used by the optimizer / model ───────────────────────────
    def reference_altitudes_m(self) -> list[float | None]:
        return [
            None if wp.altitude_ref_ft is None else wp.altitude_ref_ft * FEET_TO_METERS
            for wp in self.waypoints
        ]

    def is_monotonic_descent(self) -> bool:
        """True when reference altitudes never increase from entry to runway."""
        altitudes = [wp.altitude_ref_ft for wp in self.waypoints if wp.altitude_ref_ft is not None]
        return all(earlier >= later for earlier, later in zip(altitudes, altitudes[1:]))

    def summary(self) -> dict[str, Any]:
        """The cheap sanity report echoed back in the optimizer response."""
        return {
            "waypointCount": len(self.waypoints),
            "monotonicDescent": self.is_monotonic_descent(),
            "firstFixIdent": self.waypoints[0].ident if self.waypoints else None,
            "lastFixIdent": self.waypoints[-1].ident if self.waypoints else None,
        }


def _select_branch(document: dict[str, Any], branch_id: str | None) -> dict[str, Any] | None:
    branches = document.get("branches", [])
    if not branches:
        return None
    if branch_id is not None:
        for branch in branches:
            if branch.get("branchId") == branch_id:
                return branch
    base_ident = document.get("procedure", {}).get("baseBranchIdent")
    for branch in branches:
        if branch.get("branchIdent") == base_ident:
            return branch
    for branch in branches:
        if branch.get("branchRole") == "final":
            return branch
    return branches[0]


def _document_glidepath(document: dict[str, Any]) -> Glidepath | None:
    for profile in document.get("verticalProfiles", []):
        angle = _as_float(profile.get("glidepathAngleDeg"))
        if angle is not None:
            return Glidepath(angle, _as_float(profile.get("thresholdCrossingHeightFt")))
    return None


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    mean_lat = math.radians((a[1] + b[1]) / 2.0)
    east = math.radians(b[0] - a[0]) * _EARTH_RADIUS_M * math.cos(mean_lat)
    north = math.radians(b[1] - a[1]) * _EARTH_RADIUS_M
    return math.hypot(east, north)


def _bearing_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    mean_lat = math.radians((a[1] + b[1]) / 2.0)
    east = math.radians(b[0] - a[0]) * _EARTH_RADIUS_M * math.cos(mean_lat)
    north = math.radians(b[1] - a[1]) * _EARTH_RADIUS_M
    return math.degrees(math.atan2(east, north)) % 360.0
