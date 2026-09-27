"""The executor, one flight at a time, in plain Python floats (executor design §13, the user's request of 2026-09-27).

The batched executor (`ts_transformer.autopilot`) spends ~2.7 ms a 1 s cycle on a flight flown alone: every law and
every RK4 stage is a few hundred torch calls on one-element tensors, each paying the call's overhead. This module is
the SAME cycle — the same laws, in the same order, with the same limits and the same dynamics (the control path's
point-mass model on the scaled transport chart, two RK4 steps of 0.5 s a cycle) — written for one flight: every
quantity a Python float, every torch expression mirrored operation for operation. The setup is not mirrored: the words,
the spec, the runways, the flight's physical context and the judge are the package's own.

WHAT "THE SAME" MEANS (the user's choice, 2026-09-27). Not bit for bit: torch's own elementary functions differ from
the C library's in the last bit (sin, sqrt, atan, tanh…), and torch's ``atan2`` and ``hypot`` even differ between a
flight in a batch and the flight alone, so no rewrite can be bitwise equal to the batched executor, and the backend's
single-flight torch run never was bitwise equal to the batched exports. The contract is the RESULT: flown against the
torch executor on the Training sets' flights, every segment ends with the same outcome, the same verdict for every word
and the same crossing, and the flown tracks agree to the round-off the comparison reports
(`aeroviz_backend/tests/test_single_executor.py`; the fleet check `check_single`).

PINNED TO ITS SOURCE. The executor spec's hash does not cover this module (it would make every stored spec refuse);
instead this module records the logic of every file it mirrors (`MIRRORED_SOURCE_SHA256`: the executor spec's files —
`autopilot/` and its direct imports, `spec.executor_source_files` — and the dynamics modules the rollout reaches,
`MIRRORED_DYNAMICS`), and `require_mirrored_source` refuses to fly when the code on disk is another: a change to any of
them is a change here too, and the pin is updated with it. ``spec.logic`` reads each file, so a comment or docstring
never moves the pin.

A flight whose dynamics leave the real numbers (a division by zero or a domain error in the RK4 stages: a zero speed, a
vertical path) gets a non-finite state, as the torch rollout writes NaN, and is done at that cycle (`Plant.step`).
"""

from __future__ import annotations

import bisect
import hashlib
import importlib.util
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from aerodynamic_model.common import GRAVITY_MPS2
from aerodynamic_model.torch_dynamics import (
    ISA_DENSITY_EXPONENT, ISA_LAPSE_K_PER_M, ISA_RHO0_KG_M3, ISA_T0_K,
)
from aerodynamic_model.torch_scaled_transport_chart_dynamics import SCALED_TRANSPORT_CHART_REFERENCE_UNITS
from geokit import METRES_PER_DEG_LAT, WGS84_A, WGS84_E2
from ts_transformer.autopilot.executor import LIMITS, MODES, Flown
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.inverse import LOAD_FACTOR_MAX, LOAD_FACTOR_MIN
from ts_transformer.autopilot.lateral import capture_planning_rate_deg_s
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS
from ts_transformer.autopilot.runway_data import VerticalPath
from ts_transformer.autopilot.sentence import ROW_ROUNDING, TRACK_MAX_ROWS_PER_CYCLE, TRACK_WINDOW_S, _filled
from ts_transformer.autopilot.spec import executor_source_files, logic
from ts_transformer.autopilot.speed import speed_change_mps2
from ts_transformer.autopilot.vertical import GLIDEPATH_BELOW_M, TUBE_MARGIN_SHARE
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, ANGLE_LEVEL, APPROACH, APPROACH_CLEARED, APPROACH_GO_AROUND, HEADING, RUNWAY, SPEED, Words,
)
from ts_transformer.outputs.envelope import MAX_THRUST_FRACTION, MIN_THRUST_FRACTION

#: The dynamics modules the rollout reaches beyond the executor spec's own files (which reach them through
#: `outputs.dynamics.rollout`): what `Plant` mirrors.
MIRRORED_DYNAMICS = (
    "ts_transformer.outputs.dynamics.backends",
    "aerodynamic_model.torch_piecewise_rollout",
    "aerodynamic_model.torch_scaled_transport_chart_dynamics",
    "aerodynamic_model.torch_transport_chart_dynamics",
    "aerodynamic_model.torch_dynamics",
    "aerodynamic_model.common",
)
#: The logic of every file this module mirrors, when it was written (`mirrored_source_sha256`).
MIRRORED_SOURCE_SHA256 = "aaa5156a39d0e2b976073fc6caf06dda2151d128a2bf676991264f8b882969d1"

_UNITS = SCALED_TRANSPORT_CHART_REFERENCE_UNITS
_DT_CAP = EXECUTOR_DYNAMICS.control_rollout_integrator_dt_s


def mirrored_source_sha256() -> str:
    """sha256 over the logic (`spec.logic`) of the executor spec's files (`spec.executor_source_files`, by their labels)
    and of `MIRRORED_DYNAMICS`, in that order — the digest `spec.executor_source_sha256` takes, over more files."""
    files = list(executor_source_files())
    labels = {label for label, _ in files}
    for name in MIRRORED_DYNAMICS:
        if name not in labels:
            files.append((name, importlib.util.find_spec(name).origin))
    digest = hashlib.sha256()
    for label, path in files:
        with open(path, encoding="utf-8") as handle:
            digest.update(label.encode("utf-8") + b"\0" + logic(handle.read()).encode("utf-8") + b"\0")
    return digest.hexdigest()


def require_mirrored_source() -> None:
    """Refuse to fly when the code this module mirrors is not the code it was written against."""
    found = mirrored_source_sha256()
    if found != MIRRORED_SOURCE_SHA256:
        raise RuntimeError(f"the single-flight executor mirrors executor and dynamics code {MIRRORED_SOURCE_SHA256[:12]}; "
                           f"this checkout's is {found[:12]}: port the change to aeroviz_backend/autopilot_segment/single.py "
                           f"and update its pin")


# ---- torch's semantics on floats: NaN passes through minimum / maximum / clamp, sign(0) is 0, remainder is floored

def _min(a: float, b: float) -> float:
    return a if (a != a or a < b) else (b if (b != b or b < a) else a)


def _max(a: float, b: float) -> float:
    return a if (a != a or a > b) else (b if (b != b or b > a) else a)


def _clamp(x: float, low: float = -math.inf, high: float = math.inf) -> float:
    return x if x != x else (low if x < low else (high if x > high else x))


def _sign(x: float) -> float:
    return x if x != x else (1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0))


def _remainder(x: float, m: float) -> float:
    return x % m


def _wrap180(angle_deg: float) -> float:
    return _remainder(angle_deg + 180.0, 360.0) - 180.0


# ---- the flight's state as the laws read it (`frame.read_state`)

@dataclass(frozen=True)
class Kin:
    e_m: float
    n_m: float
    height_m: float
    speed_mps: float
    track_deg: float
    gamma_rad: float
    ground_speed_mps: float
    mass_kg: float


@dataclass(frozen=True)
class Chart:
    """The airport frame the words read positions in (`frame.AirportCharts`, one flight)."""

    lat0_deg: float
    lon0_deg: float
    m_per_deg_lon: float

    def read(self, state: Sequence[float]) -> Kin:
        lat, lon, alt, speed, psi, gamma, mass = state
        return Kin(e_m=(lon - self.lon0_deg) * self.m_per_deg_lon, n_m=(lat - self.lat0_deg) * METRES_PER_DEG_LAT,
                   height_m=alt, speed_mps=speed, track_deg=_remainder(90.0 - math.degrees(psi), 360.0), gamma_rad=gamma,
                   ground_speed_mps=speed * math.cos(gamma), mass_kg=mass)


@dataclass(frozen=True)
class Runway:
    """One candidate runway as the laws read it (`lateral.Runways.pointed`)."""

    threshold_e_m: float
    threshold_n_m: float
    course_deg: float
    elevation_m: float
    crossing_height_m: float
    glidepath_tan: float


def runways_of(geometry: AirportGeometry, paths: Sequence[VerticalPath]) -> tuple[Runway, ...]:
    if len(paths) != len(geometry.candidates):
        raise ValueError("one published vertical path per candidate runway")
    return tuple(Runway(c.threshold_e_m, c.threshold_n_m, c.course_deg, c.elevation_m, path.crossing_height_m,
                        math.tan(math.radians(path.glidepath_deg))) for c, path in zip(geometry.candidates, paths))


def relative(state: Kin, runway: Runway) -> tuple[float, float, float]:
    """``(before the threshold, right of the centreline, track − course)`` (`lateral.relative`)."""
    course = math.radians(runway.course_deg)
    de, dn = state.e_m - runway.threshold_e_m, state.n_m - runway.threshold_n_m
    sin_c, cos_c = math.sin(course), math.cos(course)
    before = -(de * sin_c + dn * cos_c)
    right = de * cos_c - dn * sin_c
    return before, right, _wrap180(state.track_deg - runway.course_deg)


# ---- the dynamics (`plant.Plant` → the scaled transport chart's RK4, `aerodynamic_model`)

def _isa_density(altitude_m: float) -> float:
    temperature = ISA_T0_K - ISA_LAPSE_K_PER_M * altitude_m
    return ISA_RHO0_KG_M3 * math.pow(temperature / ISA_T0_K, ISA_DENSITY_EXPONENT)


def _polar(load: float, speed: float, mass: float, density: float, aero: Sequence[float]) -> tuple[float, bool]:
    """``(drag, stalled)`` at ``load`` (`torch_dynamics.flight_aerodynamics`)."""
    area, cl_max, cd0, induced_k, stall_threshold, k_stall = aero
    cl_required = load * mass * GRAVITY_MPS2 / (0.5 * density * area * (speed * speed))
    ratio = cl_required / cl_max
    stalled = ratio > 1.0
    cl = cl_max if stalled else cl_required
    stall_fraction = _min(ratio, 1.0)
    transition = _clamp((stall_fraction - stall_threshold) / (1.0 - stall_threshold), 0.0, 1.0)
    smooth = (transition * transition) * (3.0 - 2.0 * transition)
    stall_drag = smooth * k_stall if ratio > stall_threshold else 0.0
    cd = cd0 + induced_k * (cl * cl) + stall_drag
    return 0.5 * density * (speed * speed) * cd * area, stalled


class Plant:
    """The flight's airframe and chart: `step` integrates one cycle from a geodetic row
    ``(lat, lon, alt, V, ψ, γ, m)`` under ``(thrust fraction, bank, load factor)``."""

    def __init__(self, aero_params: Sequence[float], frame_params: Sequence[float], max_thrust_n: float) -> None:
        self.aero = tuple(aero_params)
        self.lat0_deg, self.lon0_deg, self.alt0_m, _heading = frame_params
        self.lat0_rad = math.radians(self.lat0_deg)
        self.lon0_rad = math.radians(self.lon0_deg)
        self.max_thrust_n = max_thrust_n

    def _rhs(self, s: list[float], thrust: float, bank: float, load: float) -> list[float]:
        """`scaled_transport_chart_rhs`: the physical RHS (`transport_chart_rhs`) at the scaled state, chain-ruled."""
        _east, north, up, ve, vn, vu, mass = (s[i] * _UNITS[i] for i in range(7))
        lat_rad = self.lat0_rad + north / WGS84_A
        altitude = self.alt0_m + up
        sin_lat = math.sin(lat_rad)
        denominator = 1.0 - WGS84_E2 * (sin_lat * sin_lat)
        radius_n = WGS84_A / math.sqrt(denominator)
        radius_m = WGS84_A * (1.0 - WGS84_E2) / math.pow(denominator, 1.5)
        horizontal = math.sqrt(ve * ve + vn * vn)
        speed = math.sqrt(horizontal * horizontal + vu * vu)
        cos_gamma, sin_gamma = horizontal / speed, vu / speed
        cos_psi, sin_psi = ve / horizontal, vn / horizontal
        density = _isa_density(altitude)
        drag, stalled = _polar(load, speed, mass, density, self.aero)
        area, cl_max = self.aero[0], self.aero[1]
        realized = (0.5 * density * (speed * speed) * cl_max * area / (mass * GRAVITY_MPS2)) if stalled else load
        speed_rate = (thrust - drag) / mass - GRAVITY_MPS2 * sin_gamma
        lift_up = GRAVITY_MPS2 * (realized * math.cos(bank) - cos_gamma)
        lift_side = GRAVITY_MPS2 * realized * math.sin(bank)
        force = (speed_rate * (cos_gamma * cos_psi) + lift_up * (-sin_gamma * cos_psi) + lift_side * -sin_psi,
                 speed_rate * (cos_gamma * sin_psi) + lift_up * (-sin_gamma * sin_psi) + lift_side * cos_psi,
                 speed_rate * sin_gamma + lift_up * cos_gamma + lift_side * 0.0)
        o_e = -vn / (radius_m + altitude)
        o_n = ve / (radius_n + altitude)
        o_u = ve * math.tan(lat_rad) / (radius_n + altitude)
        cross = (o_n * vu - o_u * vn, o_u * ve - o_e * vu, o_e * vn - o_n * ve)
        east_rate = ve * (WGS84_A * math.cos(self.lat0_rad) / ((radius_n + altitude) * math.cos(lat_rad)))
        north_rate = vn * WGS84_A / (radius_m + altitude)
        rate = (east_rate, north_rate, vu, force[0] - cross[0], force[1] - cross[1], force[2] - cross[2], 0.0)
        return [rate[i] / _UNITS[i] for i in range(7)]

    def _rk4(self, s: list[float], thrust: float, bank: float, load: float, dt: float) -> list[float]:
        half = 0.5 * dt
        k1 = self._rhs(s, thrust, bank, load)
        k2 = self._rhs([s[i] + half * k1[i] for i in range(7)], thrust, bank, load)
        k3 = self._rhs([s[i] + half * k2[i] for i in range(7)], thrust, bank, load)
        k4 = self._rhs([s[i] + dt * k3[i] for i in range(7)], thrust, bank, load)
        sixth = dt / 6.0
        return [s[i] + sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(7)]

    def step(self, state: Sequence[float], command: Sequence[float], duration_s: float) -> tuple[float, ...]:
        fraction, bank, load = command
        thrust = fraction * self.max_thrust_n
        try:
            lat, lon, alt, speed, psi, gamma, mass = state
            cos_gamma = math.cos(gamma)
            chart = ((math.radians(lon) - self.lon0_rad) * WGS84_A * math.cos(self.lat0_rad),
                     (math.radians(lat) - self.lat0_rad) * WGS84_A, alt - self.alt0_m,
                     speed * cos_gamma * math.cos(psi), speed * cos_gamma * math.sin(psi), speed * math.sin(gamma), mass)
            s = [chart[i] / _UNITS[i] for i in range(7)]
            # `torch_piecewise_rollout._piecewise_schedule`: ceil(duration / dt_cap) steps of min(the rest, dt_cap)
            for local in range(math.ceil(duration_s / _DT_CAP)):
                s = self._rk4(s, thrust, bank, load, _min(_clamp(duration_s - local * _DT_CAP, 0.0), _DT_CAP))
            east, north, up, ve, vn, vu, mass = (s[i] * _UNITS[i] for i in range(7))
            horizontal = math.sqrt(ve * ve + vn * vn)
            return (self.lat0_deg + math.degrees(north / WGS84_A),
                    self.lon0_deg + math.degrees(east / (WGS84_A * math.cos(self.lat0_rad))), self.alt0_m + up,
                    math.sqrt(horizontal * horizontal + vu * vu), math.atan2(vn, ve), math.atan2(vu, horizontal), mass)
        except (ArithmeticError, ValueError):
            # the dynamics left the reals: the torch rollout writes NaN, and the flight is done
            return (math.nan,) * 7


# ---- the words said, and the clocks they are said on (`sentence`)

@dataclass(frozen=True)
class WordsNow:
    """The words in force at one cycle (`sentence.WordsNow`, one flight)."""

    runway: int
    approach: int
    heading_deg: float
    altitude_m: float
    land: bool
    angle_class: int
    angle_deg: float
    speed_mps: float
    unspecified: bool
    issued_step: tuple[int, ...]


class Sentence:
    """One sentence (``grid``: the artefact's ``[N, 6]`` word grid) looked up at any sentence time (`Sentences.at`)."""

    def __init__(self, grid: np.ndarray, words: Words) -> None:
        value, issued = _filled(np.asarray(grid, dtype=np.int64))
        self.value, self.issued, self.rows = value.tolist(), issued.tolist(), len(grid)
        self.step_s = words.spec.step_s
        self.heading = [words.heading_deg(i) for i in range(words.n_heading)]
        self.altitude = [words.altitude_m(i) if i != words.altitude_land else math.nan
                         for i in range(words.altitude_land + 1)]
        self.angle = [words.angle_deg(i) for i in range(words.n_descent + 2)]
        self.speed = [words.speed_mps(i) if i != words.speed_unspecified else math.nan
                      for i in range(words.speed_unspecified + 1)]
        self.land, self.unspecified = words.altitude_land, words.speed_unspecified

    def at(self, heard_s: float) -> WordsNow:
        row = min(max(math.floor(heard_s / self.step_s + ROW_ROUNDING), 0), self.rows - 1)
        value = self.value[row]
        return WordsNow(runway=value[RUNWAY], approach=value[APPROACH], heading_deg=self.heading[value[HEADING]],
                        altitude_m=self.altitude[value[ALTITUDE]], land=value[ALTITUDE] == self.land,
                        angle_class=value[ANGLE], angle_deg=self.angle[value[ANGLE]], speed_mps=self.speed[value[SPEED]],
                        unspecified=value[SPEED] == self.unspecified, issued_step=tuple(self.issued[row]))


class TimeClock:
    """`sentence.TimeClock`: a cycle's sentence time is the time flown."""

    def __init__(self, cycle_s: float) -> None:
        self.cycle_s = cycle_s

    def now(self, cycle: int, state: Kin) -> float:
        return cycle * self.cycle_s


class DistanceClock:
    """`sentence.DistanceClock`, one flight: the observed aircraft's time at the executor's own path length."""

    def __init__(self, e_m: np.ndarray, n_m: np.ndarray, step_s: float, cycle_s: float) -> None:
        steps = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(e_m), np.diff(n_m)))))
        self.path, self.rows, self.step_s, self.cycle_s = steps.tolist(), len(e_m), step_s, cycle_s
        self.flown = 0.0
        self.last: tuple[float, float] | None = None
        self.beyond_s = 0.0

    def now(self, cycle: int, state: Kin) -> float:
        if self.last is not None:
            self.flown = self.flown + math.hypot(state.e_m - self.last[0], state.n_m - self.last[1])
        self.last = (state.e_m, state.n_m)
        index = min(max(bisect.bisect_right(self.path, self.flown) - 1, 0), len(self.path) - 2)
        low, high = self.path[index], self.path[index + 1]
        time = (index + _clamp((self.flown - low) / (high - low), 0.0, 1.0)) * self.step_s
        end = (self.rows - 1) * self.step_s
        if time >= end:
            time = end + self.beyond_s
            self.beyond_s = self.beyond_s + self.cycle_s
        return time


class TrackClock:
    """`sentence.TrackClock`, one flight: the observed track's nearest point ahead, never back, a row a cycle at most."""

    def __init__(self, e_m: np.ndarray, n_m: np.ndarray, step_s: float, cycle_s: float) -> None:
        self.e, self.n, self.rows = [float(v) for v in e_m], [float(v) for v in n_m], len(e_m)
        self.step_s, self.cycle_s = step_s, cycle_s
        self.window = int(round(TRACK_WINDOW_S / step_s))
        self.row = 0
        self.beyond_s = 0.0

    def now(self, cycle: int, state: Kin) -> float:
        nearest, best = self.row, math.inf
        for ahead in range(self.window + 1):
            index = min(self.row + ahead, self.rows - 1)
            distance = math.hypot(self.e[index] - state.e_m, self.n[index] - state.n_m)
            if distance < best:                          # the first of equal distances, as argmin
                nearest, best = index, distance
        if self.row == self.rows - 1:
            self.beyond_s = self.beyond_s + self.cycle_s
        self.row = min(nearest, self.row + TRACK_MAX_ROWS_PER_CYCLE)
        return self.row * self.step_s + self.beyond_s


def word_clock(params: ExecutorParams, e_m: np.ndarray, n_m: np.ndarray, step_s: float
               ) -> TimeClock | DistanceClock | TrackClock:
    """The spec's clock for a truth sentence whose observed rows are ``e_m`` / ``n_m`` (`replay.word_clock`)."""
    if params.word_clock == "time":
        return TimeClock(params.cycle_s)
    return (DistanceClock if params.word_clock == "distance" else TrackClock)(e_m, n_m, step_s, params.cycle_s)


# ---- the laws (`lateral`, `vertical`, `speed`, `inverse`)

class Lateral:
    """`lateral.Lateral`, one flight."""

    def __init__(self, params: ExecutorParams, words: Words) -> None:
        spec = words.spec
        self.params, self.spec = params, spec
        self.captured = self.tracking = self.cleared = False
        self.runway: int | None = None
        self.word_deg: float | None = None
        self.word_step = 0
        self.heard_s = self.track_unwrapped = self.target_unwrapped = self.last_track = 0.0
        self.planning_rad_s = math.radians(capture_planning_rate_deg_s(spec))
        self.return_gain = 2.0 * GRAVITY_MPS2 * math.radians(params.bank_rate_deg_s)
        self.tightest = GRAVITY_MPS2 * math.tan(math.radians(spec.turn_bank_max_deg))
        self.widening_tan = math.tan(math.radians(spec.corridor_widening_deg))

    def _corridor_half_width(self, before: float) -> float:
        return self.spec.corridor_half_width_m + _clamp(before, 0.0) * self.widening_tan

    def _converges(self, heading: float, course: float, right: float, before: float, tolerance: float) -> bool:
        angle = _wrap180(heading - course)
        within_angle = abs(angle) <= 90.0 + tolerance
        inside = abs(right) <= self._corridor_half_width(before)
        toward = -1.0 if right >= 0.0 else 1.0
        best = _clamp(angle + toward * tolerance, -90.0, 90.0)
        pointing = best * toward > 0.0
        slope = math.tan(math.radians(abs(best) if pointing else 1.0))
        reaches = before - abs(right) / slope > 0.0
        return within_angle and (inside or (pointing and reaches))

    def _capture_lead(self, state: Kin, off_course: float, bank: float, bank_rate: float) -> float:
        rate, speed = self.planning_rad_s, state.ground_speed_mps
        off = math.radians(off_course)
        turn_bank = math.atan(speed * rate / GRAVITY_MPS2) * (1.0 if off > 0.0 else -1.0)
        roll_s = abs(turn_bank - bank) / bank_rate
        return (speed / rate * (1.0 - math.cos(abs(off))) + speed * math.sin(abs(off)) * roll_s / 2.0
                + speed * rate * self.params.heading_time_constant_s ** 2 / 2.0)

    def _word_error(self, state: Kin, heading: float, issued: int, go_around: bool, time_s: float) -> float:
        if self.word_deg is None:
            self.track_unwrapped = state.track_deg
            self.target_unwrapped = state.track_deg + _wrap180(heading - state.track_deg)
        else:
            self.track_unwrapped = self.track_unwrapped + _wrap180(state.track_deg - self.last_track)
            new = issued != self.word_step
            if new or go_around:
                self.heard_s = time_s
            if new:
                self.target_unwrapped = self.target_unwrapped + _wrap180(heading - self.word_deg)
            if go_around:
                self.target_unwrapped = self.track_unwrapped + _wrap180(heading - state.track_deg)
        self.word_deg, self.word_step, self.last_track = heading, issued, state.track_deg
        return self.target_unwrapped - self.track_unwrapped

    def _rate_for_error(self, error: float) -> float:
        top = self.spec.turn_rate_max_deg_s
        return _clamp(error / self.params.heading_time_constant_s, -top, top)

    def _word_rate(self, error: float, to_go: float, speed: float) -> float:
        stopping = math.degrees(math.sqrt(self.return_gain / speed * math.radians(abs(error))))
        rate = _min(abs(error / _clamp(to_go, 2.0 * self.params.cycle_s)), stopping)
        return _sign(error) * _clamp(rate, high=self.spec.turn_rate_max_deg_s)

    def rate(self, state: Kin, force: WordsNow, runways: Sequence[Runway], bank: float, bank_rate: float,
             time_s: float) -> tuple[float, dict[str, bool]]:
        params, spec = self.params, self.spec
        if self.runway is not None and force.runway != self.runway and (self.cleared or self.captured):
            raise ValueError("the runway pointer changed after the clearance (executor design §4.6)")
        self.runway = force.runway
        runway = runways[force.runway]
        course = runway.course_deg
        before, right, off_course = relative(state, runway)
        go_around = force.approach == APPROACH_GO_AROUND
        cleared = force.approach == APPROACH_CLEARED
        self.cleared = (self.cleared or cleared) and not go_around
        toward_line = right * math.sin(math.radians(off_course)) < 0.0
        turn_lands_on_line = abs(right) <= self._capture_lead(state, off_course, bank, bank_rate)
        inside = abs(right) <= self._corridor_half_width(before)
        start = cleared and not self.captured and before > 0.0 and ((toward_line and turn_lands_on_line) or inside)
        self.captured = (self.captured or start) and not go_around
        on_course = abs(off_course) <= spec.corridor_course_tolerance_deg
        self.tracking = self.captured and (self.tracking or on_course or not toward_line)

        heading = force.heading_deg
        misses = not self._converges(heading, course, right, before, 0.0)
        bent_reaches = self._converges(heading, course, right, before, spec.heading_tolerance_deg)
        waiting = cleared and not self.captured and before > 0.0
        bend = waiting and misses and bent_reaches
        intercept = waiting and not bent_reaches
        side = -1.0 if right >= 0.0 else 1.0
        error = self._word_error(state, heading, force.issued_step[HEADING], go_around, time_s) + (
            side * spec.heading_tolerance_deg if bend else 0.0)
        if intercept:
            error = _wrap180(course + side * spec.intercept_angle_deg - state.track_deg)
        gain = 1.0 / (4.0 * state.ground_speed_mps * params.heading_time_constant_s)
        steer = spec.corridor_course_tolerance_deg / 2.0 if inside else spec.intercept_angle_deg
        line = course - _max(_min(math.degrees(gain * right), steer), -steer)
        if self.tracking:
            error = _wrap180(line - state.track_deg)
        if go_around:
            error = _wrap180(course - state.track_deg)
        off = abs(math.radians(off_course))
        arc = state.ground_speed_mps * (1.0 - math.cos(off)) / _clamp(abs(right), 1e-9)
        steady = _clamp(arc, math.radians(spec.turn_rate_min_deg_s))
        tightest = _clamp(self.tightest / state.ground_speed_mps, high=math.radians(spec.turn_rate_max_deg_s))
        capture_rad_s = _min(steady, tightest)
        capture = -_sign(off_course) * _min(math.degrees(capture_rad_s), abs(off_course) / params.heading_time_constant_s)
        if self.captured and not self.tracking:
            rate = capture
        elif self.tracking:
            rate = self._rate_for_error(error)
        elif intercept or go_around:
            rate = self._rate_for_error(error)
        else:
            rate = self._word_rate(error, self.heard_s + spec.heading_lead_s - time_s, state.ground_speed_mps)
        intercept_target = course + side * spec.intercept_angle_deg
        off_word = intercept and abs(_wrap180(intercept_target - heading)) > spec.heading_tolerance_deg
        return rate, {"captured": self.captured, "tracking": self.tracking, "bent": bend, "intercepting": intercept,
                      "intercepting_off_word": off_word, "go_around": go_around}


class Vertical:
    """`vertical.Vertical`, one flight."""

    def __init__(self, params: ExecutorParams, words: Words) -> None:
        spec = words.spec
        self.params, self.words = params, words
        self.tolerance_m = spec.altitude_tolerance_m
        self.landing_max_height_m = spec.landing_max_height_m
        self.steepest_low_rad = math.radians(spec.descent_angle_edges_deg[-2])
        self.steepest_low_tan = math.tan(self.steepest_low_rad)
        self.descent_max_rad = math.radians(max(spec.descent_angle_centres_deg))
        self.climb_rad = math.radians(spec.climb_angle_centre_deg)
        self.steepest_rad = math.radians(words.angle_bounds(words.n_descent)[1])
        bounds = [words.angle_bounds(index) for index in range(words.n_descent + 2)]
        self.shallow_tan = [math.tan(math.radians(low)) for low, _ in bounds]
        self.steep_tan = [math.tan(math.radians(steep)) for _, steep in bounds]
        self.captured = self.left_tube = False
        self.issued = (-1, -1)
        self.flown_m = self.anchor_m = 0.0
        self.anchor_height_m = math.nan

    def rate(self, state: Kin, force: WordsNow, to_go_m: float, runway: Runway, off_course_deg: float,
             straight_m: float, line_captured: bool, go_around: bool) -> tuple[float, float, dict[str, bool]]:
        land, angle_class = force.land, force.angle_class
        if land and not (ANGLE_LEVEL + 1 <= angle_class <= self.words.n_descent):
            raise ValueError("\"descend to land\" in force without a descent class (vocabulary §2.5, rule 3)")
        issued = (force.issued_step[ALTITUDE], force.issued_step[ANGLE])
        new_word = issued != self.issued
        self.captured = self.captured and not new_word
        self.issued = issued
        if new_word or self.anchor_height_m != self.anchor_height_m:
            self.anchor_m, self.anchor_height_m = self.flown_m, state.height_m

        params, speed = self.params, state.speed_mps
        rate_max = params.path_rate_factor * speed * self.steepest_low_rad ** 2 / (2.0 * self.tolerance_m)
        nominal = math.radians(force.angle_deg)
        height_to_go = state.height_m - force.altitude_m
        level_off = speed * (nominal * nominal) / (2.0 * rate_max)
        moving = not land and angle_class != ANGLE_LEVEL and not self.captured
        reached = moving and ((height_to_go if nominal > 0.0 else -height_to_go) <= level_off)
        self.captured = self.captured or reached or (not land and angle_class == ANGLE_LEVEL)

        hold_tau = 4.0 * params.path_time_constant_s
        hold = _clamp(-height_to_go / (speed * hold_tau), -self.descent_max_rad, self.climb_rad)
        tolerance = self.tolerance_m
        margin = TUBE_MARGIN_SHARE * tolerance
        steep_tan, shallow_tan = self.steep_tan[angle_class], self.shallow_tan[angle_class]
        crossing_height, glidepath_tan = runway.crossing_height_m, runway.glidepath_tan
        height = state.height_m - runway.elevation_m
        anchor = self.anchor_height_m - runway.elevation_m
        here = self.flown_m - self.anchor_m
        along = self.flown_m - self.anchor_m + to_go_m
        admitted_low = _clamp(crossing_height - tolerance + margin, 0.0)
        admitted_high = _clamp(crossing_height + tolerance - margin, high=self.landing_max_height_m)
        tube_low = anchor - along * steep_tan - tolerance + margin
        tube_high = anchor - along * shallow_tan + tolerance - margin
        low, high = _max(tube_low, admitted_low), _min(tube_high, admitted_high)
        nearer_edge = admitted_high if tube_low > admitted_high else admitted_low
        crossing = _min(_max(crossing_height, low), high) if low <= high else nearer_edge
        above_crossing = height - crossing
        on_line = _clamp(math.atan2(above_crossing, _clamp(to_go_m, 1.0)), 0.0, self.steepest_rad)
        shortest = _clamp(math.atan2(above_crossing, _clamp(straight_m, 1.0)), 0.0)
        toward = on_line if line_captured else _min(nominal, shortest)
        upper = anchor - here * shallow_tan + tolerance - margin
        lower = anchor - here * steep_tan - tolerance + margin
        speed_tau = speed * hold_tau
        below_glidepath = line_captured and height < crossing_height + to_go_m * glidepath_tan
        in_tube = _min(_max(0.0 if below_glidepath else toward, math.atan(shallow_tan) + (height - upper) / speed_tau),
                       math.atan(steep_tan) + (height - lower) / speed_tau)
        in_reach = (lower - to_go_m * self.steepest_low_tan <= admitted_high) and (upper >= admitted_low)
        self.left_tube = land and not go_around and line_captured and not in_reach
        crossing_floor = _max(above_crossing / speed_tau, on_line) if line_captured else shortest
        glidepath = crossing_height + to_go_m * glidepath_tan - GLIDEPATH_BELOW_M
        closing_tan = glidepath_tan * math.cos(math.radians(off_course_deg))
        on_glidepath = math.atan(closing_tan) + (height - glidepath) / speed_tau
        floor = _min(crossing_floor, on_glidepath) if to_go_m > 0.0 else crossing_floor
        wanted_aim = in_tube if in_reach else toward
        aim = _clamp(_min(wanted_aim, floor), 0.0, self.steepest_rad)
        glidepath_floor = land and not go_around and aim < _clamp(_min(wanted_aim, crossing_floor), 0.0, self.steepest_rad)
        if land:
            reference = self.climb_rad if go_around else -aim
        else:
            reference = hold if self.captured else -nominal
        wanted = (reference - state.gamma_rad) / params.path_time_constant_s
        gamma_rate = _clamp(wanted, -rate_max, rate_max)
        self.flown_m = self.flown_m + state.ground_speed_mps * params.cycle_s
        return gamma_rate, wanted, {"level_captured": self.captured, "aim_left_tube": self.left_tube,
                                    "glidepath_floor": glidepath_floor, "path_rate_limited": abs(wanted) > rate_max}


class Speed:
    """`speed.Speed`, one flight."""

    def __init__(self, approach_ias_mps: float, words: Words) -> None:
        spec = words.spec
        self.approach_ias_mps = approach_ias_mps
        self.band_mps, self.accel_max_mps2 = spec.speed_tolerance_mps, spec.speed_accel_max_mps2
        self.pace_mps2 = speed_change_mps2(spec)
        self.margin = EXECUTOR_DYNAMICS.control_speed_floor_margin

    def rate(self, state: Kin, force: WordsNow, go_around: bool, load: float, aero: Sequence[float],
             straight_m: float) -> tuple[float, float, dict[str, bool]]:
        if force.unspecified and self.approach_ias_mps != self.approach_ias_mps:
            raise ValueError("\"unspecified\" in force for a flight whose type publishes no approach speed")
        speed = state.speed_mps
        density = _isa_density(state.height_m)
        own = self.approach_ias_mps * math.sqrt(ISA_RHO0_KG_M3 / density)
        wanted = speed if go_around else (own if force.unspecified else force.speed_mps / math.cos(state.gamma_rad))
        floor = self.margin * math.sqrt(2.0 * load * state.mass_kg * GRAVITY_MPS2 / (density * aero[0] * aero[1]))
        landing = _clamp((speed * speed - own * own) / (2.0 * _clamp(straight_m, 1.0)), self.pace_mps2,
                         self.accel_max_mps2)
        slowing = landing if force.unspecified else self.pace_mps2

        def toward(reference: float) -> float:
            tau = self.band_mps / (self.pace_mps2 if reference > speed else slowing)
            return _max(_clamp((reference - speed) / tau, high=self.pace_mps2), -slowing)

        return toward(_max(wanted, floor)), toward(wanted), {"stall_floor": floor > wanted}


# ---- the cycle (`executor.Executor`)

class SingleExecutor:
    """`executor.Executor` for one flight, one cycle at a time: ``cycle`` flies the words given, ``done`` says the
    flight is over, ``flown`` hands the record over as the batch's `Flown` (a batch of one) for the judge."""

    def __init__(self, inputs: FlightInputs, geometry: AirportGeometry, vertical_paths: Sequence[VerticalPath],
                 approach_ias_mps: float, params: ExecutorParams, words: Words, *, time_limit_s: float) -> None:
        """``inputs``: the flight's physical context as the batch's executor takes it (`flights.flight_inputs`, a batch
        of one), read into floats."""
        spec = words.spec
        params.check(spec)
        if len(inputs.initial_state) != 1:
            raise ValueError(f"the single-flight executor flies one flight, got {len(inputs.initial_state)}")
        self.params, self.words, self.time_limit_s = params, words, time_limit_s
        self.chart = Chart(geometry.frame.lat0, geometry.frame.lon0, geometry.frame.m_per_deg_lon)
        self.runways = runways_of(geometry, vertical_paths)
        self.aero = tuple(inputs.aero_params[0].tolist())
        self.max_thrust_n = float(inputs.max_thrust_n[0])
        self.plant = Plant(self.aero, inputs.frame_params[0].tolist(), self.max_thrust_n)
        self.lateral, self.vertical = Lateral(params, words), Vertical(params, words)
        self.speed = Speed(approach_ias_mps, words)
        self.bank_cap_rad = math.radians(spec.turn_bank_max_deg)
        self.cycles = int(math.ceil(time_limit_s / params.cycle_s))
        self.step_rows = int(round(spec.step_s / params.cycle_s))
        self.state = tuple(inputs.initial_state[0].tolist())
        self.bank = 0.0
        self.states, self.commands, self.wanted = [self.state], [], []
        self.limits: dict[str, list[bool]] = {name: [] for name in LIMITS}
        self.modes: dict[str, list[bool]] = {name: [] for name in MODES}
        self.sentence_times: list[float] = []
        self.done, self.done_cycle, self.count = False, self.cycles - 1, 0

    def now(self) -> Kin:
        return self.chart.read(self.state)

    @property
    def runway_locked(self) -> bool:
        return self.lateral.cleared or self.lateral.captured

    def cycle(self, force: WordsNow, sentence_s: float) -> None:
        params, cycle = self.params, self.count
        now = self.now()
        self.sentence_times.append(sentence_s)
        bank_rate = math.inf if cycle == 0 else math.radians(params.bank_rate_deg_s)
        track_rate, lateral_modes = self.lateral.rate(now, force, self.runways, self.bank, bank_rate,
                                                      cycle * params.cycle_s)
        runway = self.runways[force.runway]
        before, right, off_course = relative(now, runway)
        straight = math.hypot(before, right)
        go_around = lateral_modes["go_around"]
        gamma_rate, gamma_wanted, vertical_modes = self.vertical.rate(now, force, before, runway, off_course, straight,
                                                                      self.lateral.captured, go_around)
        # `inverse.attitude`: the bank and load factor for the wanted rates, limits 1 and 2
        speed, gamma = now.speed_mps, now.gamma_rad
        cos_gamma = math.cos(gamma)
        a = -math.radians(track_rate) * speed * cos_gamma / GRAVITY_MPS2
        b = gamma_rate * speed / GRAVITY_MPS2 + cos_gamma
        wanted_bank = math.atan2(a, _clamp(b, 0.0))
        capped = _clamp(wanted_bank, -self.bank_cap_rad, self.bank_cap_rad)
        step = bank_rate * params.cycle_s
        bank = _min(_max(capped, self.bank - step), self.bank + step)
        load_wanted = b / math.cos(bank)
        load = _clamp(load_wanted, LOAD_FACTOR_MIN, LOAD_FACTOR_MAX)
        accel, accel_wanted, speed_modes = self.speed.rate(now, force, go_around, load, self.aero, straight)
        # `inverse.thrust`: the thrust fraction for the airspeed rate at that load factor, limit 3
        sin_gamma = math.sin(gamma)
        drag, stalled = _polar(load, speed, now.mass_kg, _isa_density(now.height_m), self.aero)
        thrust_wanted = (now.mass_kg * (accel + GRAVITY_MPS2 * sin_gamma) + drag) / self.max_thrust_n
        fraction = _clamp(thrust_wanted, MIN_THRUST_FRACTION, MAX_THRUST_FRACTION)
        self.bank = bank
        command = (fraction, bank, load)
        self.state = self.plant.step(self.state, command, params.cycle_s)

        self.states.append(self.state)
        self.commands.append(command)
        self.wanted.append((track_rate, gamma_wanted, accel_wanted))
        limits = {"bank_cap": capped != wanted_bank, "bank_rate": bank != capped, "load_factor": load != load_wanted,
                  "path_rate_limited": vertical_modes["path_rate_limited"], "stall_floor": speed_modes["stall_floor"],
                  "thrust_max": thrust_wanted > MAX_THRUST_FRACTION, "thrust_min": thrust_wanted < MIN_THRUST_FRACTION,
                  "stall": stalled}
        for name in LIMITS:
            self.limits[name].append(limits[name])
        modes = {**lateral_modes, "level_captured": vertical_modes["level_captured"],
                 "aim_left_tube": vertical_modes["aim_left_tube"], "glidepath_floor": vertical_modes["glidepath_floor"]}
        for name in MODES:
            self.modes[name].append(modes[name])

        finite = all(math.isfinite(value) for value in self.state)
        after = self.chart.read(self.state)
        before_after = relative(after, runway)[0] if finite else math.nan
        finished = ((self.lateral.captured and before_after <= 0.0) or (before_after > 0.0 and after.height_m < runway.elevation_m)
                    or not finite or after.speed_mps <= 0.0 or (cycle + 1) * params.cycle_s >= self.time_limit_s)
        if finished and not self.done:
            self.done_cycle = cycle
        self.done = self.done or finished
        self.count += 1

    def flown(self) -> Flown:
        def rows(values: list) -> torch.Tensor:
            return torch.tensor([values], dtype=torch.float64)

        def flags(values: list[bool]) -> torch.Tensor:
            return torch.tensor([values], dtype=torch.bool)

        return Flown(states=rows(self.states), commands=rows(self.commands), wanted=rows(self.wanted),
                     limits={name: flags(values) for name, values in self.limits.items()},
                     modes={name: flags(values) for name, values in self.modes.items()},
                     done_cycle=torch.tensor([self.done_cycle], dtype=torch.long), sentence_s=rows(self.sentence_times),
                     cycle_s=self.params.cycle_s)
