"""A synthetic, straight-in LPV approach — so the demo and tests have something concrete.

Everything is laid out in the ``(n, e)`` frame (frame.py), with the final approach course along
``+n`` (the aircraft flies south toward the LTP at the origin). The geometry mirrors the
8260.58D FAS data block: FPAP behind the threshold, GARP 1000 ft beyond that, a 3° glidepath.

The altitudes in :func:`feasible_segment_nodes` are placed *on* the glidepath (final) and on a
gentle step-down descent (the rest), computed here directly — so once you implement the TODOs,
``ConstraintSet.evaluate(feasible_segment_nodes())`` reports feasible, and the infeasible variant
reports a clear violation. Nothing here is a TODO.
"""

from __future__ import annotations

import math

import numpy as np

from geokit import FT_M, NM_M

from .builder import ConstraintSet
from .frame import TargetFrame
from .segments import LpvFinalSpec, SegmentKind, SegmentSpec, StepDown
from .state import STATE_DIM, GAMMA, H, MASS, N as N_, E as E_, PSI, V as V_

# Anchor (KRDU-ish); only used to demonstrate the frame — the geometry below is in (n, e).
LTP_LAT, LTP_LON = 35.8776, -78.7875

# FAS geometry (metres along +n; e = 0 on the centerline).
_L_RWY = 3000.0
_FPAP_N = -_L_RWY
_GARP_N = -(_L_RWY + 305.0)
_D_GARP_LTP = _L_RWY + 305.0
_COURSE_WIDTH = max(350.0 * FT_M, math.tan(math.radians(1.5)) * _D_GARP_LTP)  # ≈ 106.7 m

_PFAF_N = 5.0 * NM_M
_IF_N = 10.0 * NM_M
_IAF_N = 15.0 * NM_M
_START_N = 20.0 * NM_M

_TDZE = 120.0
_TCH = 50.0 * FT_M          # ≈ 15.24 m
_GPA = 3.0
_TAN_GPA = math.tan(math.radians(_GPA))

_NODES_PER_SEG = 6
SEGMENT_COUNTS = [_NODES_PER_SEG] * 4  # feeder, initial, intermediate, final


def frame() -> TargetFrame:
    return TargetFrame(LTP_LAT, LTP_LON)


def _lpv_spec() -> LpvFinalSpec:
    return LpvFinalSpec(
        ltp_ne=np.array([0.0, 0.0]),
        fpap_ne=np.array([_FPAP_N, 0.0]),
        garp_ne=np.array([_GARP_N, 0.0]),
        course_width_m=_COURSE_WIDTH,
        tdze_m=_TDZE,
        tch_m=_TCH,
        gpa_deg=_GPA,
    )


def _glidepath_alt(n: float) -> float:
    """On-glidepath altitude at along-track ``n`` from the LTP (d_to_ltp = n here)."""
    return _TDZE + _TCH + n * _TAN_GPA


def build_example_segments() -> list[SegmentSpec]:
    rnp1 = 1.0 * NM_M  # 1 × RNP corridor half-width (design-doc §3.2 "reference path" choice)
    pfaf = np.array([_PFAF_N, 0.0])
    return [
        SegmentSpec(SegmentKind.FEEDER, [_START_N, 0.0], [_IAF_N, 0.0],
                    "START", "IAF", halfwidth_m=rnp1, max_descent_deg=4.7),
        SegmentSpec(SegmentKind.INITIAL, [_IAF_N, 0.0], [_IF_N, 0.0],
                    "IAF", "IF", halfwidth_m=rnp1, max_descent_deg=4.7),
        SegmentSpec(SegmentKind.INTERMEDIATE, [_IF_N, 0.0], pfaf,
                    "IF", "PFAF", halfwidth_m=rnp1, max_descent_deg=3.0,
                    step_downs=[StepDown(s_from_start_m=4630.0, min_alt_m=700.0)]),
        SegmentSpec(SegmentKind.FINAL_LPV, pfaf, [0.0, 0.0],
                    "PFAF", "LTP", lpv=_lpv_spec()),
    ]


def build_example_constraint_set() -> ConstraintSet:
    return ConstraintSet(build_example_segments())


def _nodes(n_start, n_end, alt_start, alt_end, *, on_glidepath, e_off=0.0):
    """Build ``_NODES_PER_SEG`` state rows from (n_start,alt_start) to (n_end,alt_end)."""
    fr = np.linspace(0.0, 1.0, _NODES_PER_SEG)
    n = n_start + fr * (n_end - n_start)
    rows = np.zeros((_NODES_PER_SEG, STATE_DIM))
    rows[:, N_] = n
    rows[:, E_] = e_off
    rows[:, H] = _glidepath_alt(n) if on_glidepath else alt_start + fr * (alt_end - alt_start)
    rows[:, V_] = 90.0
    rows[:, PSI] = math.pi  # flying due south (toward the LTP along −n)
    # descent slope as the flight-path angle (negative = descending)
    span = abs(n_start - n_end)
    dh = rows[-1, H] - rows[0, H]
    rows[:, GAMMA] = math.atan2(dh, span) if span else 0.0
    rows[:, MASS] = 60000.0
    return rows


def feasible_segment_nodes() -> list[np.ndarray]:
    """Per-segment node arrays for a trajectory that should satisfy every constraint."""
    pfaf_alt = _glidepath_alt(_PFAF_N)
    return [
        _nodes(_START_N, _IAF_N, 1700.0, 1400.0, on_glidepath=False),
        _nodes(_IAF_N, _IF_N, 1400.0, 1000.0, on_glidepath=False),
        _nodes(_IF_N, _PFAF_N, 1000.0, pfaf_alt, on_glidepath=False),
        _nodes(_PFAF_N, 0.0, pfaf_alt, _glidepath_alt(0.0), on_glidepath=True),
    ]


def infeasible_segment_nodes() -> list[np.ndarray]:
    """Same trajectory, but the final leg is pushed ~400 m off the course centerline."""
    nodes = feasible_segment_nodes()
    nodes[-1][:, E_] = 400.0  # outside the narrowing LPV corridor near the threshold
    return nodes
