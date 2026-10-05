"""The IAF→runway paths of an RNAV(GPS) procedure document, and their length for ranking.

The final branch's own entry is one IAF; each transition branch that ends at the final's first fix
is joined onto the final to form one complete IAF→runway :class:`ProcedureConstraint`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from geokit import METRES_PER_DEG_LAT, haversine_m, metres_per_deg_lon

from .constraint import ProcedureConstraint


def _recompute_distances(waypoints: list) -> list:
    """Recompute each waypoint's along-track ``distance_from_start_m`` over a merged path."""
    out: list = []
    cumulative = 0.0
    previous = None
    for wp in waypoints:
        if previous is not None:
            cumulative += haversine_m(previous.lat_deg, previous.lon_deg, wp.lat_deg, wp.lon_deg)
        out.append(replace(wp, distance_from_start_m=round(cumulative, 1)))
        previous = wp
    return out


def _identifier_match(a: str, b: str) -> bool:
    """Two waypoint identifiers name the same fix — only when BOTH are present.

    ``ProcedureConstraint`` defaults a missing ``fixId`` to ``""``, so a plain equality
    read two identifier-less waypoints as the same fix (the "two optional fields compared
    to each other" trap in the coding conventions)."""
    return bool(a) and a == b


def concat_to_runway(trans_pc: ProcedureConstraint, final_pc: ProcedureConstraint) -> ProcedureConstraint | None:
    """Join a transition (IAF→connecting fix) to the final (connecting fix→runway) into one
    IAF→runway ``ProcedureConstraint``, recomputing along-track distances. Returns ``None`` when
    the transition does not end at the final's first fix (so it doesn't feed this final)."""
    join = trans_pc.waypoints[-1]
    final_start = final_pc.waypoints[0]
    if not (_identifier_match(join.fix_id, final_start.fix_id)
            or _identifier_match(join.ident, final_start.ident)):
        return None
    merged = list(trans_pc.waypoints[:-1]) + list(final_pc.waypoints)
    return replace(
        final_pc,                              # glidepath / course / runway / nominal speed
        branch_id=trans_pc.branch_id,          # label the IAF by its transition branch
        waypoints=tuple(_recompute_distances(merged)),
    )


def iaf_paths(document: dict[str, Any]) -> list[ProcedureConstraint]:
    """All IAF→runway ``ProcedureConstraint``s for a procedure document (the final first)."""
    branches = document["branches"]
    final = next((b for b in branches if b.get("branchRole") == "final"), None)
    if final is None:
        return []
    final_pc = ProcedureConstraint.from_detail_document(document, final["branchId"])
    if final_pc is None:            # from_detail_document: None below two waypoints
        return []
    paths = [final_pc]
    for branch in branches:
        if branch.get("branchRole") != "transition":
            continue
        trans_pc = ProcedureConstraint.from_detail_document(document, branch["branchId"])
        if trans_pc is None:
            continue
        merged = concat_to_runway(trans_pc, final_pc)
        if merged is not None:
            paths.append(merged)
    return paths


def path_length_m(pc: ProcedureConstraint) -> float:
    """Horizontal polyline length of the IAF→runway waypoints, for ranking IAFs (NO NLP solve).

    Deliberately the straight polyline in a runway-anchored metric frame — NOT a fitted curve.
    The previous proxy fitted a Lagrange polynomial through the waypoints, and a high-degree fit
    oscillates on cornered routes (measured: +38% arc length on a two-corner T-arrival, +7% on a
    mild dogleg, exact on straight ones), so it could rank a genuinely shorter cornered IAF behind
    a longer straight one. A flown fly-by path only ever CUTS corners, so the polyline is the
    tighter, monotone proxy for what "shortest" claims. Horizontal only: an IAF often has no coded
    altitude, and the earlier 3D form read it as 0 ft; on the live documents of the five K-airports
    (43 with more than one IAF, 2026-10-05) the horizontal ranking orders every IAF as that 3D one did.
    """
    wps = pc.waypoints
    lat0, lon0 = wps[-1].lat_deg, wps[-1].lon_deg
    points = np.array([
        [(w.lat_deg - lat0) * METRES_PER_DEG_LAT, (w.lon_deg - lon0) * metres_per_deg_lon(lat0)]
        for w in wps
    ])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
