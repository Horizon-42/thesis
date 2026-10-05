"""The executor, one flight at a time, in plain Python floats — the executor's SINGLE-FLIGHT way to fly (executor design
§12.4, §13; the user's request of 2026-09-27; moved here from the backend 2026-10-01).

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
(`aeroviz_backend/tests/test_single_executor.py`; on real flights the spec's reference tracks, `conformance`).

CHECKED BY WHAT IT FLIES (executor design §12.3, the user 2026-10-01): it is one of the ways the spec's reference tracks
are flown again (`autopilot.conformance`, ``single``), so a change here or in the laws it mirrors is found by the check,
never by a pin on its source.

A flight whose dynamics leave the real numbers (a division by zero or a domain error in the RK4 stages: a zero speed, a
vertical path) gets a non-finite state, as the torch rollout writes NaN, and is done at that cycle (`Plant.step`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
import torch

from aerodynamic_model.common import GRAVITY_MPS2
from aerodynamic_model.torch_dynamics import (
    ISA_DENSITY_EXPONENT, ISA_LAPSE_K_PER_M, ISA_RHO0_KG_M3, ISA_T0_K,
)
from aerodynamic_model.torch_scaled_transport_chart_dynamics import SCALED_TRANSPORT_CHART_REFERENCE_UNITS
from geokit import METRES_PER_DEG_LAT, WGS84_A, WGS84_E2
from ts_transformer.autopilot import ends
from ts_transformer.autopilot.ends import runway_lateral_limit_m
from ts_transformer.autopilot.executor import GO_AROUND_EXTRA_S, LIMITS, MODES, Flown
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.inverse import BANK_MAX_RAD, LOAD_FACTOR_MAX, LOAD_FACTOR_MIN
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS
from ts_transformer.autopilot.sentence import ROW_ROUNDING, _filled
from ts_transformer.autopilot.speed import speed_change_mps2
from ts_transformer.autopilot.vertical import GO_AROUND_MAX_RAD, GO_AROUND_MIN_RAD
from ts_transformer.instructions.airport import AirportGeometry, landing_cross_limit_m
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, ANGLE_LEVEL, HEADING, RUNWAY, RUNWAY_GO_AROUND, SPEED, Words,
)
from ts_transformer.outputs.envelope import MAX_THRUST_FRACTION, MIN_THRUST_FRACTION

_UNITS = SCALED_TRANSPORT_CHART_REFERENCE_UNITS
_DT_CAP = EXECUTOR_DYNAMICS.control_rollout_integrator_dt_s


# ---- torch's semantics on floats: NaN passes through minimum / maximum / clamp, sign(0) is 0, remainder is floored

def _min(a: float, b: float) -> float:
    return a if (a != a or a < b) else (b if (b != b or b < a) else a)


def _max(a: float, b: float) -> float:
    return a if (a != a or a > b) else (b if (b != b or b > a) else a)


def _clamp(x: float, low: float = -math.inf, high: float = math.inf) -> float:
    return x if x != x else (low if x < low else (high if x > high else x))


def _sign(x: float) -> float:
    return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)


def _divide(a: float, b: float) -> float:
    """``a / b`` as IEEE (and torch) divide: by zero is ±inf, or NaN for 0 / 0."""
    if b != 0.0:
        return a / b
    return math.nan if a == 0.0 or a != a else math.copysign(math.inf, a) * math.copysign(1.0, b)


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
    """The airport frame the words read positions in, and its elevation E (`frame.AirportCharts`, one flight)."""

    lat0_deg: float
    lon0_deg: float
    m_per_deg_lon: float
    elevation_m: float

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
    landing_limit_m: float
    on_runway_m: float


def runways_of(geometry: AirportGeometry, spec: VocabularySpec) -> tuple[Runway, ...]:
    return tuple(Runway(c.threshold_e_m, c.threshold_n_m, c.course_deg, c.elevation_m,
                        landing_cross_limit_m(geometry, index, spec.landing_cross_limit_m, spec.parallel_course_delta_deg),
                        runway_lateral_limit_m(geometry, index, spec))
                 for index, c in enumerate(geometry.candidates))


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
        lat, lon, alt, speed, psi, gamma, mass = state
        try:
            cos_gamma = math.cos(gamma)
            chart = ((math.radians(lon) - self.lon0_rad) * WGS84_A * math.cos(self.lat0_rad),
                     (math.radians(lat) - self.lat0_rad) * WGS84_A, alt - self.alt0_m,
                     speed * cos_gamma * math.cos(psi), speed * cos_gamma * math.sin(psi), speed * math.sin(gamma), mass)
            s = [chart[i] / _UNITS[i] for i in range(7)]
            # `torch_piecewise_rollout._piecewise_schedule`: ceil(duration / dt_cap) steps of min(the rest, dt_cap)
            for local in range(math.ceil(duration_s / _DT_CAP)):
                s = self._rk4(s, thrust, bank, load, _min(_clamp(duration_s - local * _DT_CAP, 0.0), _DT_CAP))
            east, north, up, ve, vn, vu, mass_end = (s[i] * _UNITS[i] for i in range(7))
            horizontal = math.sqrt(ve * ve + vn * vn)
            return (self.lat0_deg + math.degrees(north / WGS84_A),
                    self.lon0_deg + math.degrees(east / (WGS84_A * math.cos(self.lat0_rad))), self.alt0_m + up,
                    math.sqrt(horizontal * horizontal + vu * vu), math.atan2(vn, ve), math.atan2(vu, horizontal), mass_end)
        except (ArithmeticError, ValueError):
            # the dynamics left the reals: the torch rollout writes NaN (the mass, whose rate is zero, stays), and the
            # flight is done
            return (math.nan,) * 6 + (mass,)


# ---- the words said (`sentence`)

@dataclass(frozen=True)
class WordsNow:
    """The words in force at one cycle (`sentence.WordsNow`, one flight)."""

    runway: int
    go_around: bool
    heading_rel_deg: float
    level_m: float
    no_level_off: bool
    angle_class: int
    angle_deg: float
    speed_mps: float
    unspecified: bool
    issued_step: tuple[int, ...]


class Sentence:
    """One sentence (``grid``: ``[N, 5]``, rows ``step_s`` apart) looked up at any sentence time (`Sentences.at`)."""

    def __init__(self, grid: np.ndarray, words: Words, *, step_s: float) -> None:
        value, issued, runway = _filled(np.asarray(grid, dtype=np.int64))
        self.value, self.issued, self.runway, self.rows = value.tolist(), issued.tolist(), runway.tolist(), len(grid)
        self.step_s = step_s
        self.heading = [words.heading_relative_deg(i) for i in range(words.n_heading)]
        self.level = [words.altitude_level_m(i) if i != words.altitude_no_level_off else math.nan
                      for i in range(words.altitude_no_level_off + 1)]
        self.angle = [words.angle_deg(i) for i in range(words.n_descent + 2)]
        self.speed = [words.speed_mps(i) if i != words.speed_unspecified else math.nan
                      for i in range(words.speed_unspecified + 1)]
        self.no_level_off, self.unspecified = words.altitude_no_level_off, words.speed_unspecified

    def at(self, heard_s: float) -> WordsNow:
        row = min(max(math.floor(heard_s / self.step_s + ROW_ROUNDING), 0), self.rows - 1)
        value = self.value[row]
        return WordsNow(runway=self.runway[row], go_around=value[RUNWAY] == RUNWAY_GO_AROUND,
                        heading_rel_deg=self.heading[value[HEADING]], level_m=self.level[value[ALTITUDE]],
                        no_level_off=value[ALTITUDE] == self.no_level_off, angle_class=value[ANGLE],
                        angle_deg=self.angle[value[ANGLE]], speed_mps=self.speed[value[SPEED]],
                        unspecified=value[SPEED] == self.unspecified, issued_step=tuple(self.issued[row]))


# ---- the laws (`lateral`, `vertical`, `speed`, `inverse`)

class Lateral:
    """`lateral.Lateral`, one flight."""

    def __init__(self, params: ExecutorParams, words: Words) -> None:
        spec = words.spec
        self.params, self.spec = params, spec
        self.word_deg: float | None = None
        self.word_step = 0
        self.heard_s = self.track_unwrapped = self.target_unwrapped = self.last_track = 0.0
        self.return_gain = 2.0 * GRAVITY_MPS2 * math.radians(params.bank_rate_deg_s)

    def _word_error(self, state: Kin, relative_deg: float, issued: int, course: float, time_s: float) -> float:
        heard = _remainder(course + relative_deg, 360.0)
        if self.word_deg is None:
            self.track_unwrapped = state.track_deg
            self.target_unwrapped = state.track_deg + _wrap180(heard - state.track_deg)
            word = heard
        else:
            self.track_unwrapped = self.track_unwrapped + _wrap180(state.track_deg - self.last_track)
            new = issued != self.word_step
            word = heard if new else self.word_deg
            if new:
                self.heard_s = time_s
                self.target_unwrapped = self.target_unwrapped + _wrap180(word - self.word_deg)
        self.word_deg, self.word_step, self.last_track = word, issued, state.track_deg
        return self.target_unwrapped - self.track_unwrapped

    def _word_rate(self, error: float, to_go: float, speed: float) -> float:
        stopping = math.degrees(math.sqrt(self.return_gain / speed * math.radians(abs(error))))
        rate = _min(abs(error / _clamp(to_go, 2.0 * self.params.cycle_s)), stopping)
        return _sign(error) * _clamp(rate, high=self.spec.turn_rate_max_deg_s)

    def rate(self, state: Kin, force: WordsNow, runways: Sequence[Runway], time_s: float) -> float:
        course = runways[force.runway].course_deg
        error = self._word_error(state, force.heading_rel_deg, force.issued_step[HEADING], course, time_s)
        return self._word_rate(error, self.heard_s + self.spec.heading_lead_s - time_s, state.ground_speed_mps)


class Vertical:
    """`vertical.Vertical`, one flight."""

    def __init__(self, params: ExecutorParams, words: Words) -> None:
        spec = words.spec
        self.params, self.words = params, words
        self.tolerance_m = float(words.altitude_tolerances.min())
        self.steepest_low_rad = math.radians(spec.descent_angle_edges_deg[-2])
        self.descent_max_rad = math.radians(max(spec.descent_angle_centres_deg))
        self.climb_rad = math.radians(spec.climb_angle_centre_deg)
        self.captured = False
        self.issued = (-1, -1)

    def rate(self, state: Kin, force: WordsNow, elevation_m: float, aero: Sequence[float], max_thrust_n: float
             ) -> tuple[float, float, dict[str, bool]]:
        no_level_off, angle_class, go_around = force.no_level_off, force.angle_class, force.go_around
        if no_level_off and not (ANGLE_LEVEL + 1 <= angle_class <= self.words.n_descent):
            raise ValueError("\"no level-off\" in force without a descent class (vocabulary §3.7, rule 4)")
        if no_level_off and go_around:
            raise ValueError("\"no level-off\" in force while a go-around is (vocabulary §3.7, rules 5 and 6)")
        issued = (force.issued_step[ALTITUDE], force.issued_step[ANGLE])
        new_word = issued != self.issued
        self.captured = self.captured and not new_word
        self.issued = issued

        params, speed = self.params, state.speed_mps
        rate_max = params.path_rate_factor * speed * self.steepest_low_rad ** 2 / (2.0 * self.tolerance_m)
        # `vertical.go_around_angle_rad`: the steady climb at the thrust limit, load factor 1, within its limits
        drag, _stalled = _polar(1.0, speed, state.mass_kg, _isa_density(state.height_m), aero)
        sine = _clamp((MAX_THRUST_FRACTION * max_thrust_n - drag) / (state.mass_kg * GRAVITY_MPS2), -1.0, 1.0)
        go_around_rad = _clamp(math.asin(sine), GO_AROUND_MIN_RAD, GO_AROUND_MAX_RAD)
        climb_rad = go_around_rad if go_around else self.climb_rad
        nominal = -go_around_rad if go_around and angle_class == self.words.angle_climb else math.radians(force.angle_deg)
        height_to_go = state.height_m - (force.level_m + elevation_m)
        level_off = speed * (nominal * nominal) / (2.0 * rate_max)
        moving = not no_level_off and angle_class != ANGLE_LEVEL and not self.captured
        reached = moving and ((height_to_go if nominal > 0.0 else -height_to_go) <= level_off)
        self.captured = self.captured or reached or (not no_level_off and angle_class == ANGLE_LEVEL)

        hold_tau = 4.0 * params.path_time_constant_s
        hold = _min(_max(-height_to_go / (speed * hold_tau), -self.descent_max_rad), climb_rad)
        reference = hold if self.captured and not no_level_off else -nominal
        wanted = (reference - state.gamma_rad) / params.path_time_constant_s
        gamma_rate = _clamp(wanted, -rate_max, rate_max)
        return gamma_rate, wanted, {"level_captured": self.captured, "path_rate_limited": abs(wanted) > rate_max}


class Speed:
    """`speed.Speed`, one flight."""

    def __init__(self, approach_ias_mps: float, words: Words) -> None:
        spec = words.spec
        self.approach_ias_mps = approach_ias_mps
        self.band_mps, self.accel_max_mps2 = spec.speed_tolerance_mps, spec.speed_accel_max_mps2
        self.pace_mps2 = speed_change_mps2(spec)
        self.margin = EXECUTOR_DYNAMICS.control_speed_floor_margin
        self.held_mps = math.nan

    def hear_go_around(self, state: Kin) -> None:
        self.held_mps = state.speed_mps

    def rate(self, state: Kin, force: WordsNow, load: float, aero: Sequence[float],
             straight_m: float) -> tuple[float, float, dict[str, bool]]:
        go_around = force.go_around
        if force.unspecified and not go_around and self.approach_ias_mps != self.approach_ias_mps:
            raise ValueError("\"unspecified\" in force for a flight whose type publishes no approach speed")
        speed = state.speed_mps
        density = _isa_density(state.height_m)
        own = self.approach_ias_mps * math.sqrt(ISA_RHO0_KG_M3 / density)
        if force.unspecified:
            wanted = self.held_mps if go_around else own
        else:
            wanted = force.speed_mps / math.cos(state.gamma_rad)
        floor = self.margin * math.sqrt(2.0 * load * state.mass_kg * GRAVITY_MPS2 / (density * aero[0] * aero[1]))
        landing = _clamp((speed * speed - own * own) / (2.0 * _clamp(straight_m, 1.0)), self.pace_mps2,
                         self.accel_max_mps2)
        # a speed word at a_max both ways (D43); "unspecified" at its own pace, slowing to land at the landing rate
        rising = self.pace_mps2 if force.unspecified else self.accel_max_mps2
        slowing = landing if force.unspecified and not go_around else rising

        def toward(reference: float) -> float:
            tau = self.band_mps / (rising if reference > speed else slowing)
            return _min(_max((reference - speed) / tau, -slowing), rising)

        return toward(_max(wanted, floor)), toward(wanted), {"stall_floor": floor > wanted}


# ---- the cycle (`executor.Executor`)

class SingleExecutor:
    """`executor.Executor` for one flight, one cycle at a time: ``cycle`` flies the words given, ``done`` says the
    flight is over, ``flown`` hands the record over as the batch's `Flown` (a batch of one) for the judge."""

    def __init__(self, inputs: FlightInputs, geometry: AirportGeometry, approach_ias_mps: float, params: ExecutorParams,
                 words: Words, *, step_s: float, time_limit_s: float, reserve_s: float) -> None:
        """``inputs``: the flight's physical context as the batch's executor takes it (`flights.flight_inputs`, a batch
        of one), read into floats; ``step_s`` the sentence's row interval; ``reserve_s`` the time its go-arounds may
        add."""
        spec = words.spec
        params.check(spec, step_s)
        if len(inputs.initial_state) != 1:
            raise ValueError(f"the single-flight executor flies one flight, got {len(inputs.initial_state)}")
        self.params, self.words, self.time_limit_s = params, words, time_limit_s
        self.most_s = time_limit_s + reserve_s
        self.chart = Chart(geometry.frame.lat0, geometry.frame.lon0, geometry.frame.m_per_deg_lon, geometry.elevation_m)
        self.runways = runways_of(geometry, spec)
        self.aero = tuple(inputs.aero_params[0].tolist())
        self.max_thrust_n = float(inputs.max_thrust_n[0])
        self.plant = Plant(self.aero, inputs.frame_params[0].tolist(), self.max_thrust_n)
        self.lateral, self.vertical = Lateral(params, words), Vertical(params, words)
        self.speed = Speed(approach_ias_mps, words)
        self.bank_cap_rad = math.radians(spec.turn_bank_max_deg)
        if not 0.0 < self.bank_cap_rad <= BANK_MAX_RAD:
            raise ValueError(f"bank cap {spec.turn_bank_max_deg:.1f}° outside (0, {math.degrees(BANK_MAX_RAD):.0f}°]")
        self.cycles = int(math.ceil(self.most_s / params.cycle_s))
        self.step_rows = int(round(step_s / params.cycle_s))
        self.state = tuple(inputs.initial_state[0].tolist())
        self.bank = 0.0
        self.states, self.commands, self.wanted = [self.state], [], []
        self.limits: dict[str, list[bool]] = {name: [] for name in LIMITS}
        self.modes: dict[str, list[bool]] = {name: [] for name in MODES}
        self.sentence_times: list[float] = []
        self.runway_rows: list[int] = []
        self.runway_issued = -1
        self.done, self.done_cycle, self.count = False, self.cycles - 1, 0

    def now(self) -> Kin:
        return self.chart.read(self.state)

    def cycle(self, force: WordsNow, sentence_s: float) -> None:
        params, cycle = self.params, self.count
        now = self.now()
        self.sentence_times.append(sentence_s)
        bank_rate = math.radians(params.bank_rate_deg_s)          # from the first cycle (D84)
        issued = force.issued_step[RUNWAY]
        if force.go_around and issued != self.runway_issued and not self.done:
            if self.time_limit_s + GO_AROUND_EXTRA_S > self.most_s:
                raise ValueError("a time limit extended past the executor's reserve")
            self.time_limit_s += GO_AROUND_EXTRA_S
            self.speed.hear_go_around(now)
        self.runway_issued = issued
        track_rate = self.lateral.rate(now, force, self.runways, cycle * params.cycle_s)
        runway = self.runways[force.runway]
        before, right, _off_course = relative(now, runway)
        straight = math.hypot(before, right)
        gamma_rate, gamma_wanted, vertical_modes = self.vertical.rate(now, force, self.chart.elevation_m, self.aero,
                                                                      self.max_thrust_n)
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
        accel, accel_wanted, speed_modes = self.speed.rate(now, force, load, self.aero, straight)
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
        self.runway_rows.append(force.runway)
        limits = {"bank_cap": capped != wanted_bank, "bank_rate": bank != capped, "load_factor": load != load_wanted,
                  "path_rate_limited": vertical_modes["path_rate_limited"], "stall_floor": speed_modes["stall_floor"],
                  "thrust_max": thrust_wanted > MAX_THRUST_FRACTION, "thrust_min": thrust_wanted < MIN_THRUST_FRACTION,
                  "stall": stalled}
        for name in LIMITS:
            self.limits[name].append(limits[name])
        modes = {"go_around": force.go_around, "level_captured": vertical_modes["level_captured"]}
        for name in MODES:
            self.modes[name].append(modes[name])

        after = self.chart.read(self.state)
        # `executor.Executor.cycle`'s ends, the judge's tests (`ends`, D79): a crossing of any candidate's plane
        crossed = False
        for index, candidate in enumerate(self.runways):
            was, right_was, _ = relative(now, candidate)
            past, right_after, off_after = relative(after, candidate)
            if not ends.plane_crossed(was, past):
                continue
            fraction_crossed = _clamp(_divide(was, was - past), 0.0, 1.0)
            approach, other = ends.crossing_ends(right_was + fraction_crossed * (right_after - right_was), off_after,
                                                 force.go_around, index == force.runway, candidate.landing_limit_m,
                                                 candidate.on_runway_m, self.words.spec)
            crossed = crossed or bool(approach or other)
        past = relative(after, runway)[0]
        finished = (crossed or bool(ends.ground_contact(past, after.height_m - runway.elevation_m))
                    or bool(ends.dynamics_failure(all(math.isfinite(value) for value in self.state), after.speed_mps,
                                                  stalled))
                    or (cycle + 1) * params.cycle_s >= self.time_limit_s)
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
                     runway=torch.tensor([self.runway_rows], dtype=torch.long),
                     done_cycle=torch.tensor([self.done_cycle], dtype=torch.long), sentence_s=rows(self.sentence_times),
                     cycle_s=self.params.cycle_s)


def fly(inputs: FlightInputs, geometry: AirportGeometry, approach_ias_mps: float, grid: np.ndarray,
        params: ExecutorParams, words: Words, *, step_s: float, time_limit_s: float, reserve_s: float,
        stop_cycle: int | None = None, superseded: Callable[[], bool] = lambda: False) -> tuple[Flown, bool]:
    """One flight's sentence ``grid`` (rows ``step_s`` apart) flown alone, a row's words heard on the cycle that starts
    it, to its end — or stopped before cycle ``stop_cycle``: the flown record (stopped: ``done_cycle`` the last cycle
    flown) and whether the stop ended it. ``superseded`` is asked before each cycle; when it says so, `InterruptedError`.
    The one single-flight loop: the executor's check (`conformance.fly_single`) and the live executor
    (`experiments.training_flights.fly_single`) drive it."""
    executor = SingleExecutor(inputs, geometry, approach_ias_mps, params, words, step_s=step_s,
                              time_limit_s=time_limit_s, reserve_s=reserve_s)
    sentence = Sentence(grid, words, step_s=step_s)
    stopped = False
    for cycle in range(executor.cycles):
        if stop_cycle is not None and cycle >= stop_cycle:
            stopped = True
            break
        if superseded():
            raise InterruptedError(f"superseded after {cycle} cycles")
        sentence_s = cycle * params.cycle_s
        executor.cycle(sentence.at(sentence_s), sentence_s)
        if executor.done:
            break
    flown = executor.flown()
    if stopped:
        flown = replace(flown, done_cycle=torch.full_like(flown.done_cycle, executor.count - 1))
    return flown, stopped

