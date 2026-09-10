"""The eight plan parameters read off an observed track (design §3).

Every parameter has a definition, a range set by the procedure and the aircraft, and this
extractor — the supervision of the plan head and the input of the oracle-ceiling test. The
track is the OBSERVED rows from the anchor on (`FlightSeries.values`, never the fitted
supervision tail, which is a construction); the remaining path is the package's one
definition (`approach_difficulty.remaining_path_profile_m`, which does read the tail to the
threshold) so a parameter binned on distance-to-go agrees with the anytime grid; the join
is the measurement's own gate (`final_approach_geometry.truth_final_gate`), not a second
notion of "on the final".

Operating parameters (predicted): ``T_s`` the time from the anchor to the threshold;
``V_mid_mps`` the ground speed held before deceleration; ``d_decel_m`` the remaining path at
which the speed first drops below ``V_final + DECEL_MARGIN_MPS``; ``V_final_mps`` the final
approach ground speed; ``h_capture_m`` the chart height (above the threshold aim point) at
which the final is captured LATERALLY — the glidepath is usually captured later.
Route parameters (assigned, or a distribution — never a point output): ``d_join_m`` the
remaining path at which the flight becomes established; ``side`` which side the base leg
comes from; ``L_pre_m`` the path flown before the join.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from aircraft.aero_params import stall_speed_ms
from ts_transformer.config import CONTROL_SPEED_FLOOR_MARGIN_DEFAULT
from ts_transformer.data.approach_difficulty import remaining_path_profile_m
from ts_transformer.data.channels import IDX, states_from_channels
from ts_transformer.data.dataset import truth_duration_s
from ts_transformer.geometry.final_approach_geometry import truth_final_gate
from ts_transformer.outputs.plan.skeleton import RNP_HALF_WIDTH_M, RunwaySkeleton

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries

PLAN_PARAMETERS = (
    "T_s", "V_mid_mps", "d_decel_m", "V_final_mps", "h_capture_m", "d_join_m", "side", "L_pre_m",
)
OPERATING_PARAMETERS = PLAN_PARAMETERS[:5]
ROUTE_PARAMETERS = PLAN_PARAMETERS[5:]

#: `V_mid` is the MEAN ground speed over the path flown BEFORE the deceleration point —
#: its length over its time — "the speed held before deceleration" as the number the time
#: closure needs (a median over-reads a varying speed: the first readout, 2026-09-10, read a
#: 10–20 km band median and the schedule then missed the flown time by 61 s on the vectored
#: stratum). A flight already decelerating at the anchor (fewer than this many rows before
#: the point, or no point) holds what it has, its anchor ground speed.
V_MID_MIN_ROWS = 3
#: `d_decel` is where the ground speed first drops below the scenario TARGET speed (the
#: type's approach reference speed the threshold state carries) plus this — the straight-in
#: residual readout's ``target_plus_10`` threshold, `experiments.straight_in_residual_readout`.
DECEL_MARGIN_MPS = 10.0
#: The side is the sign of the widest cross-track over this window before the join (the
#: base leg; the widest row cannot cancel the way a mean over a converging intercept can),
#: and "straight-in" (0) when that width is under the base-leg floor.
SIDE_WINDOW_S = 60.0
SIDE_MIN_OFFSET_M = 500.0
#: A join is "on a published transition" when the track over this window before it stays
#: inside the RNP box of a coded pre-final leg.
JOIN_WINDOW_S = 60.0
#: The stall margin the ranges read the aircraft's speed floor at — the package's one
#: margin, at the sea-level stall speed.
STALL_MARGIN = CONTROL_SPEED_FLOOR_MARGIN_DEFAULT


@dataclass(frozen=True)
class PlanLabels:
    """The eight parameters of one flight from one anchor, and their ranges.

    A ``None`` is a parameter the track does not define (no join; never slower than the
    deceleration threshold; a join BEFORE the anchor, which censors the capture height, the
    side and the pre-final path — the window never saw them): never silently zero.
    """

    T_s: float
    V_mid_mps: float
    d_decel_m: float | None
    V_final_mps: float
    h_capture_m: float | None
    d_join_m: float | None
    side: int | None                # +1 right, −1 left, 0 straight-in; None = censored
    L_pre_m: float | None
    #: The truth gate is already open at the anchor: the flight joined before the window
    #: (59 % of KRDU val at L−1, 2026-09-10). `d_join_m` is then the anchor's remaining path
    #: and the three route/capture parameters are None.
    join_at_anchor: bool
    #: `V_mid_mps` is the anchor ground speed because the flight was already decelerating
    #: at the anchor (fewer than `V_MID_MIN_ROWS` rows before the deceleration point).
    v_mid_from_anchor: bool
    #: Whether the 60 s before the join lie inside a published pre-final leg's RNP box —
    #: a coded transition rather than a radar vector. None when the flight has no join
    #: or the procedure codes no pre-final leg.
    join_on_transition: bool | None
    #: ``(low, high)`` per parameter, from the procedure and the aircraft; None = unbounded
    #: on that side (the documents on this machine code no speed limit, so `V_mid`'s
    #: ceiling is None everywhere).
    ranges: dict[str, tuple[float | None, float | None]]
    #: Where the track sits against the skeleton at the anchor, for the readout.
    remaining_path_at_anchor_m: float
    ground_speed_at_anchor_mps: float

    def parameters(self) -> dict[str, float | int | None]:
        return {name: getattr(self, name) for name in PLAN_PARAMETERS}

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ranges"] = {name: list(bounds) for name, bounds in self.ranges.items()}
        return out


def _ground_speeds(series: FlightSeries, rows: slice) -> np.ndarray:
    """Physical horizontal ground speed (m/s) of the observed rows, from the chart velocities
    through the same inverse the export uses (never the chart derivatives themselves)."""
    states = states_from_channels(
        series.times[rows], series.values[rows], series.frame,
        mass_kg=float(series.scenario.initial.m),
    )
    return np.array([state.V * np.cos(state.gamma) for _t, state in states], dtype=np.float64)


def _arc_length_m(e: np.ndarray, n: np.ndarray) -> np.ndarray:
    steps = np.hypot(np.diff(e), np.diff(n))
    return np.concatenate(([0.0], np.cumsum(steps)))


def extract_plan(series: FlightSeries, anchor: int, skeleton: RunwaySkeleton) -> PlanLabels:
    """The eight parameters of ``series`` from ``anchor``, inside ``skeleton``."""
    rows = slice(anchor, series.n_samples)
    times = np.asarray(series.times[rows], dtype=np.float64)
    values = np.asarray(series.values[rows], dtype=np.float64)
    e, n, u = values[:, IDX["e"]], values[:, IDX["n"]], values[:, IDX["u"]]
    d, xt = skeleton.axes(e, n)
    remaining = np.asarray(remaining_path_profile_m(series)[rows], dtype=np.float64)
    speed = _ground_speeds(series, rows)
    dt = float(series.times[1] - series.times[0]) if series.n_samples > 1 else 1.0

    # the join: the measurement's gate on the observed rows
    gate = truth_final_gate(
        torch.from_numpy(d)[None], torch.from_numpy(xt)[None],
        torch.ones((1, len(d)), dtype=torch.bool),
    )[0].numpy()
    join = int(np.flatnonzero(gate)[0]) if gate.any() else None

    V_final = float(speed[-1])
    T = truth_duration_s(series, anchor)
    scenario = series.scenario
    slow = np.flatnonzero(speed < float(scenario.target.V) + DECEL_MARGIN_MPS)
    d_decel = float(remaining[slow[0]]) if slow.size else None
    held_rows = int(slow[0]) if slow.size else len(speed)
    v_mid_from_anchor = held_rows < V_MID_MIN_ROWS
    if v_mid_from_anchor:
        V_mid = float(speed[0])
    else:
        held = slice(0, held_rows)
        V_mid = float(_arc_length_m(e[held], n[held])[-1] / (times[held_rows - 1] - times[0]))

    join_at_anchor = join == 0
    if join is None or join_at_anchor:
        h_capture = L_pre = side = None
        d_join = None if join is None else float(remaining[0])
        on_transition = None
    else:
        h_capture = float(u[join])
        d_join = float(remaining[join])
        L_pre = float(_arc_length_m(e, n)[join])
        window = max(1, int(round(SIDE_WINDOW_S / dt)))
        before = xt[max(0, join - window):join]
        widest = float(before[np.argmax(np.abs(before))])
        side = 0 if abs(widest) < SIDE_MIN_OFFSET_M else (1 if widest > 0.0 else -1)
        approach = slice(max(0, join - max(1, int(round(JOIN_WINDOW_S / dt)))), join + 1)
        distance = skeleton.distance_to_published_m(e[approach], n[approach])
        on_transition = None if np.isinf(distance).all() else bool(np.max(distance) <= RNP_HALF_WIDTH_M)

    # ranges: the procedure's where coded, the aircraft's otherwise
    stall = STALL_MARGIN * stall_speed_ms(
        float(scenario.initial.m), wing_area_m2=float(scenario.aero.S),
        cl_max=float(scenario.aero.Cl_max),
    )
    # the shortest legal route at the fastest legal speed: the beeline is its lower bound,
    # the fastest legal speed the coded limit, else the type's approach maximum
    beeline = float(np.hypot(e[0] - skeleton.target_e, n[0] - skeleton.target_n))
    limit = skeleton.speed_limit_ahead_mps(float(d[0]))
    fastest = limit if limit is not None else float(scenario.aircraft.approach.max_speed_ms)
    # the capture height is bounded where the capture HAPPENS, by the constraint coded at
    # the next fix ahead of the join (its floor, and its ceiling where one is coded). NOT
    # by the glidepath: the join is the LATERAL gate, and measured on KRDU val (2026-09-10)
    # most flights are still above the glidepath there and capture it later.
    captured = join is not None and not join_at_anchor
    floor_alt = skeleton.floor_altitude_m(float(d[join])) if captured else None
    ceiling_alt = skeleton.ceiling_altitude_m(float(d[join])) if captured else None
    ranges = {
        "T_s": (beeline / fastest, None),
        "V_mid_mps": (stall, limit),
        "d_decel_m": (0.0, float(remaining[0])),
        # the published V_ref is an AIRSPEED window the evaluation reads with the METAR
        # headwind (`evaluation/docs/THRESHOLD_SPEED_GATE.md`); a ground speed has no
        # floor there but the stall margin's
        "V_final_mps": (stall, None),
        "h_capture_m": (
            None if floor_alt is None else floor_alt - skeleton.aim_altitude_m,
            None if ceiling_alt is None else ceiling_alt - skeleton.aim_altitude_m,
        ),
        "d_join_m": (0.0, float(remaining[0])),
        "side": (-1.0, 1.0),
        "L_pre_m": (0.0, float(remaining[0])),
    }
    return PlanLabels(
        T_s=T, V_mid_mps=V_mid, d_decel_m=d_decel, V_final_mps=V_final,
        h_capture_m=h_capture, d_join_m=d_join, side=side, L_pre_m=L_pre,
        join_on_transition=on_transition, ranges=ranges,
        join_at_anchor=join_at_anchor, v_mid_from_anchor=v_mid_from_anchor,
        remaining_path_at_anchor_m=float(remaining[0]),
        ground_speed_at_anchor_mps=float(speed[0]),
    )
