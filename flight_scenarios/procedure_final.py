"""The runway's coded final approach fix, read from the RNAV(GPS) procedure document.

The procedure documents are the frontend's ``procedure-details/<uid>.json`` files under
``aeroviz-4d/public/data/airports/<ICAO>/`` (schema ``rnav-procedure-runway`` 1.0.0),
indexed by ``index.json``.  Two consumers resolve them:

* the optimizer's constrained-IAF mode (``4dTrajectory/optimization/scenario_optimization``)
  builds the whole leg chain from ``aeroviz_backend.procedure_constraint``;
* the learned model (``4dTrajectory/ts_transformer``) needed ONE number per runway — how far
  back from the threshold the FAF sits — to gate a final-approach constraint at the FAF
  (that gate was deleted on 2026-09-09; the read stays for the tests that cross-check it);
  since 2026-09-10 its plan-and-guidance path reads the WHOLE coded approach as a
  :class:`ProcedureSkeleton` — the final's fixes with their altitude floors, the coded
  glidepath and TCH, and every published transition onto the final — through
  :func:`procedure_skeleton`.  (The glidepath angle and threshold crossing height also come
  from the runway target the arrival manifest carries; some documents code them as null here.)

This module owns the path resolution for both, the FAF read and the skeleton read.  The FAF
distance is taken from the document's vertical profile (``constraintSamples``: distance
along the coded final from the IF, so ``MAPt − FAF`` is the FAF's along-course distance to
the runway).  ``ProcedureConstraint.from_detail_document`` reads the same document's legs
and recomputes waypoint distances geodetically; the two agree to the coding precision of
the document, and ``tests/test_procedure_final.py`` checks them against each other on the
documents present on this machine.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from geokit import FT_M, KT_MS

DEFAULT_PROCEDURE_ROOT = (
    Path(__file__).resolve().parents[1] / "aeroviz-4d" / "public" / "data" / "airports"
)
RNAV_GPS_FAMILY = "RNAV_GPS"


def rnav_gps_procedure_path(
    airport: str, runway: str, *, root: str | Path = DEFAULT_PROCEDURE_ROOT
) -> Path:
    """Path to the runway's RNAV(GPS) procedure detail document, via the airport index."""
    airport_root = Path(root) / airport.upper() / "procedure-details"
    index = json.loads((airport_root / "index.json").read_text(encoding="utf-8"))
    runway_ident = f"RW{runway.upper()}"
    for entry in index.get("runways", []):
        if entry.get("runwayIdent") != runway_ident:
            continue
        for procedure in entry.get("procedures", []):
            if procedure.get("procedureFamily") == RNAV_GPS_FAMILY:
                return airport_root / f"{procedure['procedureUid']}.json"
    raise ValueError(f"no RNAV(GPS) procedure for {airport.upper()} {runway_ident}")


@dataclass(frozen=True)
class FinalApproachFix:
    """The coded FAF of one runway's RNAV(GPS) final."""

    procedure_uid: str
    ident: str
    distance_to_threshold_m: float   # along the coded final, FAF → MAPt (the threshold)
    crossing_altitude_m: float       # the FAF's coded crossing altitude, m MSL


@lru_cache(maxsize=None)
def final_approach_fix(
    airport: str, runway: str, *, root: str | Path = DEFAULT_PROCEDURE_ROOT
) -> FinalApproachFix:
    """Read the runway's FAF from its RNAV(GPS) document; raises when it is not coded."""
    path = rnav_gps_procedure_path(airport, runway, root=root)
    document = json.loads(path.read_text(encoding="utf-8"))
    profiles = document.get("verticalProfiles") or []
    if not profiles:
        raise ValueError(f"{path.name}: no vertical profile — the FAF distance is not coded")
    profile = profiles[0]
    samples = {sample.get("role"): sample for sample in profile.get("constraintSamples", [])}
    missing = [role for role in ("FAF", "MAPt") if role not in samples]
    if missing:
        raise ValueError(f"{path.name}: vertical profile lacks {missing} constraint samples")
    faf, mapt = samples["FAF"], samples["MAPt"]
    distance = float(mapt["distanceFromStartM"]) - float(faf["distanceFromStartM"])
    if distance <= 0.0:
        raise ValueError(f"{path.name}: FAF is not upstream of the MAPt ({distance:.0f} m)")
    return FinalApproachFix(
        procedure_uid=str(document["procedureUid"]),
        ident=str(faf["ident"]),
        distance_to_threshold_m=distance,
        crossing_altitude_m=float(faf["altitudeFt"]) * FT_M,
    )


# ── the whole coded approach: what the plan-and-guidance path flies inside ──────────────

#: ``constraints.altitude.qualifier`` values the documents on this machine code (46
#: documents, 648 constrained legs, 2026-09-10): ``atOrAbove`` 551, ``at`` 88, ``block`` 9.
#: A qualifier outside this set is refused rather than guessed at.
_ALTITUDE_QUALIFIERS = ("atOrAbove", "at", "atOrBelow", "block")
_BLOCK_RANGE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")


@dataclass(frozen=True)
class ProcedureFix:
    """One fix of a coded branch, with the constraint coded on the leg that ENDS at it."""

    ident: str
    role: str                       # ``roleAtEnd``: IF, FAF, MAPt, Route (a transition's first fix is its IAF)
    lat_deg: float
    lon_deg: float
    altitude_min_m: float | None    # at-or-above / at / block floor, m MSL
    altitude_max_m: float | None    # at-or-below / at / block ceiling, m MSL
    speed_max_mps: float | None     # ``constraints.speedKt`` — none of the documents on this machine code one


@dataclass(frozen=True)
class ProcedureBranch:
    """The final (IF → FAF → MAPt) or one transition (IAF → … → the fix where it joins the final)."""

    ident: str
    role: str                       # ``final`` | ``transition``
    transition_ident: str | None
    merge_fix: str | None           # the final's fix this transition ends on
    fixes: tuple[ProcedureFix, ...]


@dataclass(frozen=True)
class ProcedureSkeleton:
    """Everything the coded RNAV(GPS) approach fixes about a trajectory onto one runway."""

    procedure_uid: str
    airport: str
    runway: str                     # ``05L``
    threshold_lat_deg: float
    threshold_lon_deg: float
    threshold_elevation_m: float
    glidepath_angle_deg: float | None
    threshold_crossing_height_m: float | None
    final: ProcedureBranch
    transitions: tuple[ProcedureBranch, ...]
    #: Transitions the document codes but does not draw (no rendered leg), by ident.
    unrendered_transitions: tuple[str, ...] = ()

    @property
    def faf(self) -> ProcedureFix:
        return next(fix for fix in self.final.fixes if fix.role == "FAF")


def _altitude_bounds_m(constraint: dict | None) -> tuple[float | None, float | None]:
    if not constraint:
        return None, None
    qualifier = constraint.get("qualifier")
    if qualifier not in _ALTITUDE_QUALIFIERS:
        raise ValueError(f"altitude qualifier {qualifier!r} is not one this reader knows")
    value = constraint.get("valueFt")
    metres = float(value) * FT_M if value is not None else None
    if qualifier == "atOrAbove":
        return metres, None
    if qualifier == "atOrBelow":
        return None, metres
    if qualifier == "at":
        return metres, metres
    # block: the raw text carries both bounds ("3000-5000 ft"); the value alone is the floor
    match = _BLOCK_RANGE.search(str(constraint.get("rawText") or ""))
    if match:
        low, high = sorted(float(x) * FT_M for x in match.groups())
        return low, high
    return metres, None


def _branch_fixes(document: dict, branch: dict, *, stop_at: str | None) -> tuple[ProcedureFix, ...]:
    """The rendered legs of one branch, in sequence, each as the fix it ends at."""
    fixes = {fix["fixId"]: fix for fix in document.get("fixes", [])}
    out: list[ProcedureFix] = []
    for leg in sorted(branch.get("legs", []), key=lambda leg: leg.get("sequence", 0)):
        if leg.get("quality", {}).get("renderedInPlanView") is not True:
            continue
        fix = fixes.get(leg.get("path", {}).get("endFixRef"))
        position = fix.get("position") if fix else None
        if not fix or not position:
            continue
        constraints = leg.get("constraints") or {}
        low, high = _altitude_bounds_m(constraints.get("altitude"))
        speed = constraints.get("speedKt")
        out.append(ProcedureFix(
            ident=str(fix["ident"]),
            role=str(leg.get("roleAtEnd") or ""),
            lat_deg=float(position["lat"]),
            lon_deg=float(position["lon"]),
            altitude_min_m=low,
            altitude_max_m=high,
            speed_max_mps=float(speed) * KT_MS if speed is not None else None,
        ))
        # the landing threshold ends a branch whether or not the document names it: a
        # missed approach is not part of a landing
        if out[-1].role == "MAPt" or (stop_at is not None and fix["fixId"] == stop_at):
            break
    return tuple(out)


@lru_cache(maxsize=None)
def procedure_skeleton(
    airport: str, runway: str, *, root: str | Path = DEFAULT_PROCEDURE_ROOT
) -> ProcedureSkeleton:
    """Read the runway's RNAV(GPS) approach as the skeleton the plan path flies inside.

    The final branch is read up to the landing threshold (the MAPt; a document's missed
    approach is not part of a landing), every transition branch up to the fix it merges on.
    Raises when the document has no final branch, no FAF, or codes an altitude this reader
    does not know — the skeleton is a hard bound, so an unread constraint is a wrong one.
    """
    path = rnav_gps_procedure_path(airport, runway, root=root)
    document = json.loads(path.read_text(encoding="utf-8"))
    branches = document.get("branches") or []
    final = next((b for b in branches if b.get("branchRole") == "final"), None)
    if final is None:
        raise ValueError(f"{path.name}: no final branch")
    threshold_ref = (document.get("runway") or {}).get("landingThresholdFixRef")
    final_fixes = _branch_fixes(document, final, stop_at=threshold_ref)
    if not any(fix.role == "FAF" for fix in final_fixes):
        raise ValueError(f"{path.name}: the final branch codes no FAF")
    fix_ids = {fix["fixId"]: fix["ident"] for fix in document.get("fixes", [])}
    transitions = []
    unrendered = []
    for branch in branches:
        if branch.get("branchRole") != "transition":
            continue
        ident = str(branch.get("branchKey") or branch.get("branchId"))
        fixes = _branch_fixes(document, branch, stop_at=None)
        if not fixes:
            # coded but not drawable (KRDU-R32's ANOSIC: no rendered leg) — kept by name
            unrendered.append(ident)
            continue
        merge_fix = fix_ids.get(branch.get("mergeFixRef"))
        if merge_fix is not None and fixes[-1].ident != merge_fix:
            raise ValueError(
                f"{path.name}: transition {ident} ends at {fixes[-1].ident}, not at its merge "
                f"fix {merge_fix}"
            )
        transitions.append(ProcedureBranch(
            ident=ident,
            role="transition",
            transition_ident=branch.get("transitionIdent"),
            merge_fix=merge_fix if merge_fix is not None else fixes[-1].ident,
            fixes=fixes,
        ))
    threshold = (document.get("runway") or {}).get("threshold") or {}
    profiles = document.get("verticalProfiles") or []
    profile = profiles[0] if profiles else {}
    gpa = profile.get("glidepathAngleDeg")
    tch = profile.get("thresholdCrossingHeightFt")
    return ProcedureSkeleton(
        procedure_uid=str(document["procedureUid"]),
        airport=airport.upper(),
        runway=runway.upper(),
        threshold_lat_deg=float(threshold["lat"]),
        threshold_lon_deg=float(threshold["lon"]),
        threshold_elevation_m=float(threshold.get("elevationFt") or 0.0) * FT_M,
        glidepath_angle_deg=float(gpa) if gpa is not None else None,
        threshold_crossing_height_m=float(tch) * FT_M if tch is not None else None,
        final=ProcedureBranch(
            ident=str(final.get("branchKey") or final.get("branchId")),
            role="final", transition_ident=None, merge_fix=None, fixes=final_fixes,
        ),
        transitions=tuple(transitions),
        unrendered_transitions=tuple(unrendered),
    )
