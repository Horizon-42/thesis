"""The runway's coded final approach fix, read from the RNAV(GPS) procedure document.

The procedure documents are the frontend's ``procedure-details/<uid>.json`` files under
``aeroviz-4d/public/data/airports/<ICAO>/`` (schema ``rnav-procedure-runway`` 1.0.0),
indexed by ``index.json``.  Two consumers resolve them:

* the optimizer's constrained-IAF mode (``4dTrajectory/optimization/scenario_optimization``)
  builds the whole leg chain from ``aeroviz_backend.procedure_constraint``;
* the learned model (``4dTrajectory/ts_transformer``) needs ONE number per runway — how far
  back from the threshold the FAF sits — to gate a final-approach constraint at the FAF.
  (The glidepath angle and threshold crossing height come from the runway target the
  arrival manifest already carries; some documents code them as null here.)

This module owns the path resolution for both and the FAF read for the second.  The FAF
distance is taken from the document's vertical profile (``constraintSamples``: distance
along the coded final from the IF, so ``MAPt − FAF`` is the FAF's along-course distance to
the runway).  ``ProcedureConstraint.from_detail_document`` reads the same document's legs
and recomputes waypoint distances geodetically; the two agree to the coding precision of
the document, and ``tests/test_procedure_final.py`` checks them against each other on the
documents present on this machine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from geokit import FT_M

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
