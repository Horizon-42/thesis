"""The executor, one flight at a time, in plain Python floats — the executor's SINGLE-FLIGHT way to fly (executor design
§12.4, §13; the user's request of 2026-09-27; moved here from the backend 2026-10-01).

The batched executor (`ts_transformer.autopilot`) spends ~2.7 ms a 1 s cycle on a flight flown alone: every law and
every RK4 stage is a few hundred torch calls on one-element tensors, each paying the call's overhead. This module is
the SAME cycle — the same laws, in the same order, with the same limits and the same dynamics (the control path's
point-mass model on the scaled transport chart with its three first-order actuators, two RK4 steps of 0.5 s a cycle)
— written for one flight: every
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
    """The flight's airframe and chart: `step` integrates one cycle from a geodetic row ``(lat, lon, alt, V, ψ, γ, m)``
    and the actuators ``(thrust fraction, bank, load factor)`` under the commands (`plant.Plant`: the lag model,
    `aerodynamic_model.torch_lag_dynamics.lag_rhs`, its actuators unscaled)."""

    def __init__(self, aero_params: Sequence[float], frame_params: Sequence[float], max_thrust_n: float,
                 time_constants_s: Sequence[float]) -> None:
        self.aero = tuple(aero_params)
        self.lat0_deg, self.lon0_deg, self.alt0_m, _heading = frame_params
        self.lat0_rad = math.radians(self.lat0_deg)
        self.lon0_rad = math.radians(self.lon0_deg)
        self.max_thrust_n = max_thrust_n
        self.taus = tuple(time_constants_s)

    def _rhs(self, s: list[float], command: Sequence[float]) -> list[float]:
        """`lag_rhs`: the physical chart RHS (`transport_chart_rhs`) at the scaled chart state under the actuators'
        values, chain-ruled, and ``(u − a) / τ`` for each actuator."""
        _east, north, up, ve, vn, vu, mass = (s[i] * _UNITS[i] for i in range(7))
        fraction, bank, load = s[7], s[8], s[9]
        thrust = fraction * self.max_thrust_n
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
        return ([rate[i] / _UNITS[i] for i in range(7)]
                + [(command[i] - s[7 + i]) / self.taus[i] for i in range(3)])

    def _rk4(self, s: list[float], command: Sequence[float], dt: float) -> list[float]:
        half = 0.5 * dt
        k1 = self._rhs(s, command)
        k2 = self._rhs([s[i] + half * k1[i] for i in range(10)], command)
        k3 = self._rhs([s[i] + half * k2[i] for i in range(10)], command)
        k4 = self._rhs([s[i] + dt * k3[i] for i in range(10)], command)
        sixth = dt / 6.0
        return [s[i] + sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(10)]

    def step(self, state: Sequence[float], actuators: Sequence[float], command: Sequence[float],
             duration_s: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
        lat, lon, alt, speed, psi, gamma, mass = state
        try:
            cos_gamma = math.cos(gamma)
            chart = ((math.radians(lon) - self.lon0_rad) * WGS84_A * math.cos(self.lat0_rad),
                     (math.radians(lat) - self.lat0_rad) * WGS84_A, alt - self.alt0_m,
                     speed * cos_gamma * math.cos(psi), speed * cos_gamma * math.sin(psi), speed * math.sin(gamma), mass)
            s = [chart[i] / _UNITS[i] for i in range(7)] + list(actuators)
            # `torch_piecewise_rollout._piecewise_schedule`: ceil(duration / dt_cap) steps of min(the rest, dt_cap)
            for local in range(math.ceil(duration_s / _DT_CAP)):
                s = self._rk4(s, command, _min(_clamp(duration_s - local * _DT_CAP, 0.0), _DT_CAP))
            east, north, up, ve, vn, vu, mass_end = (s[i] * _UNITS[i] for i in range(7))
            horizontal = math.sqrt(ve * ve + vn * vn)
            return ((self.lat0_deg + math.degrees(north / WGS84_A),
                     self.lon0_deg + math.degrees(east / (WGS84_A * math.cos(self.lat0_rad))), self.alt0_m + up,
                     math.sqrt(horizontal * horizontal + vu * vu), math.atan2(vn, ve), math.atan2(vu, horizontal),
                     mass_end), tuple(s[7:]))
        except (ArithmeticError, ValueError):
            # the dynamics left the reals: the torch rollout writes NaN (the mass, whose rate is zero, stays), and the
            # flight is done
            return (math.nan,) * 6 + (mass,), (math.nan,) * 3


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

    def rate(self, state: Kin, force: WordsNow, runways: Sequence[Runway], time_s: float) -> tuple[float, float]:
        course = runways[force.runway].course_deg
        error = self._word_error(state, force.heading_rel_deg, force.issued_step[HEADING], course, time_s)
        to_go = self.heard_s + self.spec.heading_lead_s - time_s
        return error / _clamp(to_go, self.params.cycle_s), error


def _stop_angle(left: float, speed: float, gamma: float, climbing: bool, bank_mean: float, observed_load: float,
                ceiling: float, tau_load: float, cycle_s: float) -> float:
    """`vertical.stop_angle_rad`, one flight."""
    cos_bank = math.cos(bank_mean)
    level = (math.cos(gamma) if climbing else 1.0) / cos_bank
    limit = LOAD_FACTOR_MIN if climbing else ceiling
    reach = level - limit if climbing else limit - level
    if reach <= 0.0:                                  # the limit cannot change the path angle that way
        return 0.0
    rate = GRAVITY_MPS2 * reach * cos_bank / speed
    from_now = observed_load - limit if climbing else limit - observed_load
    delay = _clamp(tau_load * from_now / reach, 0.0) + cycle_s
    a_term, b_term = speed * delay, _divide(speed, rate)
    radius = math.hypot(a_term, b_term)
    angle = math.atan2(b_term, a_term) + math.asin(_clamp(_divide(left - b_term, radius), high=1.0))
    return angle


class Vertical:
    """`vertical.Vertical`, one flight."""

    def __init__(self, params: ExecutorParams, words: Words) -> None:
        self.params, self.words = params, words
        self.lead_s = words.spec.angle_lead_s
        self.climb_rad = math.radians(words.spec.climb_angle_centre_deg)
        self.captured = False
        self.issued = (-1, -1)
        self.heard_s = 0.0

    def rate(self, state: Kin, force: WordsNow, elevation_m: float, aero: Sequence[float], max_thrust_n: float,
             time_s: float, bank_mean: float, observed_load: float, ceiling: float) -> tuple[float, dict[str, bool]]:
        words, cycle = self.words, self.params.cycle_s
        no_level_off, angle_class, go_around = force.no_level_off, force.angle_class, force.go_around
        if no_level_off and not (ANGLE_LEVEL + 1 <= angle_class <= words.n_descent):
            raise ValueError("\"no level-off\" in force without a descent class (vocabulary §3.7, rule 4)")
        if no_level_off and go_around:
            raise ValueError("\"no level-off\" in force while a go-around is (vocabulary §3.7, rules 5 and 6)")
        issued = (force.issued_step[ALTITUDE], force.issued_step[ANGLE])
        new_word = issued != self.issued
        self.captured = self.captured and not new_word
        if new_word:
            self.heard_s = time_s
        self.issued = issued

        gamma, speed = state.gamma_rad, state.speed_mps
        if angle_class == words.angle_climb:
            if go_around:
                # `vertical.go_around_angle_rad`: the steady climb at the thrust limit, load factor 1, within its limits
                drag, _stalled = _polar(1.0, speed, state.mass_kg, _isa_density(state.height_m), aero)
                sine = _clamp((MAX_THRUST_FRACTION * max_thrust_n - drag) / (state.mass_kg * GRAVITY_MPS2), -1.0, 1.0)
                magnitude = _clamp(math.asin(sine), GO_AROUND_MIN_RAD, GO_AROUND_MAX_RAD)
            else:
                magnitude = self.climb_rad
        else:
            magnitude = abs(math.radians(force.angle_deg))
        level_word = angle_class == ANGLE_LEVEL
        above = state.height_m - (force.level_m + elevation_m)
        toward = -1.0 if no_level_off else -_sign(above)
        goal = 0.0 if level_word else toward * magnitude
        to_go = _clamp(self.heard_s + self.lead_s - time_s, cycle)
        ramp = gamma + (goal - gamma) * cycle / to_go
        if no_level_off:
            return (ramp - gamma) / cycle, {"level_captured": self.captured}
        left = _clamp(abs(above) + _sign(above) * speed * cycle * math.sin(gamma), 0.0)
        stop = _stop_angle(left, speed, gamma, toward > 0.0, bank_mean, observed_load, ceiling,
                           self.params.load_time_constant_s, cycle)
        aimed = toward * ramp > 0.0
        reached = not level_word and not self.captured and aimed and stop < abs(ramp)
        self.captured = self.captured or reached or level_word
        if level_word:
            target = 0.0
        elif self.captured:
            target = toward * _min(magnitude, stop)
        else:
            target = toward * _min(abs(ramp), stop) if aimed else ramp
        return (target - gamma) / cycle, {"level_captured": self.captured}


class Speed:
    """`speed.Speed`, one flight."""

    def __init__(self, approach_ias_mps: float, params: ExecutorParams, words: Words) -> None:
        self.approach_ias_mps = approach_ias_mps
        self.cycle_s, self.lead_s = params.cycle_s, words.spec.speed_lead_s
        self.held_mps = math.nan
        self.issued, self.heard_s = -1, 0.0

    def hear_go_around(self, state: Kin) -> None:
        self.held_mps = state.speed_mps

    def rate(self, state: Kin, force: WordsNow, time_s: float, load: float,
             aero: Sequence[float]) -> tuple[float, float, dict[str, bool]]:
        go_around = force.go_around
        if force.unspecified and not go_around and self.approach_ias_mps != self.approach_ias_mps:
            raise ValueError("\"unspecified\" in force for a flight whose type publishes no approach speed")
        issued = force.issued_step[SPEED]
        if issued != self.issued:
            self.heard_s = time_s
        self.issued = issued
        speed = state.speed_mps
        density = _isa_density(state.height_m)
        floor = math.sqrt(2.0 * load * state.mass_kg * GRAVITY_MPS2 / (density * aero[0] * aero[1]))
        if force.unspecified:
            own = self.approach_ias_mps * math.sqrt(_clamp(load, 1.0)) * math.sqrt(ISA_RHO0_KG_M3 / density)
            wanted = self.held_mps if go_around else own
        else:
            wanted = force.speed_mps / math.cos(state.gamma_rad)
        to_go = _clamp(self.heard_s + self.lead_s - time_s, self.cycle_s)
        return (_max(wanted, floor) - speed) / to_go, (wanted - speed) / to_go, {"stall_floor": floor > wanted}


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
        self.plant = Plant(self.aero, inputs.frame_params[0].tolist(), self.max_thrust_n, params.time_constants_s)
        self.lateral, self.vertical = Lateral(params, words), Vertical(params, words)
        self.speed = Speed(approach_ias_mps, params, words)
        #: `inverse.decays`
        self.decay = tuple(math.exp(-params.cycle_s / tau) for tau in params.time_constants_s)
        self.cycles = int(math.ceil(self.most_s / params.cycle_s))
        self.step_rows = int(round(step_s / params.cycle_s))
        self.state = tuple(inputs.initial_state[0].tolist())
        #: the plant's actuators and the observer's estimate of them (`executor.Executor`): the thrust the first cycle's
        self.actuators: tuple[float, ...] = (math.nan, 0.0, 1.0)
        self.observed: tuple[float, ...] = self.actuators
        self.states, self.commands, self.wanted = [self.state], [], []
        self.actuator_rows: list[tuple[float, ...]] = []
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
        issued = force.issued_step[RUNWAY]
        if force.go_around and issued != self.runway_issued and not self.done:
            if self.time_limit_s + GO_AROUND_EXTRA_S > self.most_s:
                raise ValueError("a time limit extended past the executor's reserve")
            self.time_limit_s += GO_AROUND_EXTRA_S
            self.speed.hear_go_around(now)
        self.runway_issued = issued
        time_s = cycle * params.cycle_s
        aero, (decay_thrust, decay_bank, decay_load), dt = self.aero, self.decay, params.cycle_s
        speed, gamma = now.speed_mps, now.gamma_rad
        density = _isa_density(now.height_m)
        observed_thrust, observed_bank, observed_load = self.observed
        # `inverse.load_ceiling` (the stall at the speed the cycle ends at), `inverse.bank_limit_rad`
        if observed_thrust != observed_thrust:
            end_speed = speed
        else:
            most = _min(0.5 * density * (speed * speed) * aero[0] * aero[1] / (now.mass_kg * GRAVITY_MPS2),
                        LOAD_FACTOR_MAX)
            drag_now, _ = _polar(most, speed, now.mass_kg, density, aero)
            rate_now = (observed_thrust * self.max_thrust_n - drag_now) / now.mass_kg - GRAVITY_MPS2 * math.sin(gamma)
            end_speed = speed + rate_now * dt
        ceiling = _min(0.5 * density * (end_speed * end_speed) * aero[0] * aero[1] / (now.mass_kg * GRAVITY_MPS2),
                       LOAD_FACTOR_MAX)
        limit = _clamp(math.acos(_clamp(math.cos(gamma) / ceiling, high=1.0)), high=BANK_MAX_RAD)
        # 1. `lateral`, `inverse.bank`
        track_rate, error = self.lateral.rate(now, force, self.runways, time_s)
        wanted_bank = -math.atan(speed * math.radians(track_rate) / GRAVITY_MPS2)
        capped = _min(_max(wanted_bank, -limit), limit)
        roll = math.radians(params.bank_rate_deg_s)
        p_mean = roll * params.bank_time_constant_s * (1.0 - decay_bank) / dt
        error_rad = math.radians(abs(error))
        turning = GRAVITY_MPS2 * math.tan(abs(observed_bank)) / speed * dt
        toward = _sign(-error) == _sign(observed_bank)
        left = _clamp(error_rad - turning, 0.0) if toward else error_rad + turning
        stop = math.acos(math.exp(-left * speed * p_mean / GRAVITY_MPS2))
        target = _sign(capped) * _min(abs(capped), stop)
        rate = _clamp((target - observed_bank) / dt, -roll, roll)
        rolled = observed_bank + params.bank_time_constant_s * rate
        bank = _min(_max(rolled, -limit), limit)
        bank_end = bank + (observed_bank - bank) * decay_bank
        bank_mean = 0.5 * (observed_bank + bank_end)
        # 2. `vertical`, `inverse.load`
        runway = self.runways[force.runway]
        gamma_rate, vertical_modes = self.vertical.rate(now, force, self.chart.elevation_m, aero, self.max_thrust_n,
                                                        time_s, bank_mean, observed_load, ceiling)
        solved_load = (gamma_rate * speed / GRAVITY_MPS2 + math.cos(gamma)) / math.cos(bank_mean)
        load = _min(_clamp(solved_load, LOAD_FACTOR_MIN), ceiling)
        steered_load = (load - observed_load * decay_load) / (1.0 - decay_load)
        load_command = _clamp(steered_load, LOAD_FACTOR_MIN, LOAD_FACTOR_MAX)
        # 3. `speed`; 4. `inverse.thrust`
        accel, accel_wanted, speed_modes = self.speed.rate(now, force, time_s, load, aero)
        drag, _stalled = _polar(load, speed, now.mass_kg, density, aero)
        solved = (now.mass_kg * (accel + GRAVITY_MPS2 * math.sin(gamma)) + drag) / self.max_thrust_n
        thrust = _clamp(solved, MIN_THRUST_FRACTION, MAX_THRUST_FRACTION)
        if observed_thrust != observed_thrust:               # the first cycle's thrust: in effect
            observed_thrust = thrust
            self.actuators = (thrust, *self.actuators[1:])
        steered_thrust = (thrust - observed_thrust * decay_thrust) / (1.0 - decay_thrust)
        thrust_command = _clamp(steered_thrust, MIN_THRUST_FRACTION, MAX_THRUST_FRACTION)
        command = (thrust_command, bank, load_command)
        # 5. `inverse.observe`, the plant
        self.observed = tuple(u + (a - u) * k for u, a, k in
                              zip(command, (observed_thrust, observed_bank, observed_load), self.decay))
        self.actuator_rows.append(self.actuators)
        self.state, self.actuators = self.plant.step(self.state, self.actuators, command, dt)
        after = self.chart.read(self.state)
        # `inverse.stalled` at the load factor the actuator reached
        _drag, stalled = _polar(self.actuators[2], after.speed_mps, after.mass_kg, _isa_density(after.height_m), aero)

        self.states.append(self.state)
        self.commands.append(command)
        self.wanted.append((track_rate, gamma_rate, accel_wanted))
        self.runway_rows.append(force.runway)
        limits = {"bank_cap": capped != wanted_bank or bank != rolled, "bank_rate": abs(target - observed_bank) > roll * dt,
                  "load_factor": load != solved_load or load_command != steered_load,
                  "stall_floor": speed_modes["stall_floor"],
                  "thrust_max": solved > MAX_THRUST_FRACTION or steered_thrust > MAX_THRUST_FRACTION,
                  "thrust_min": solved < MIN_THRUST_FRACTION or steered_thrust < MIN_THRUST_FRACTION,
                  "stall": stalled}
        for name in LIMITS:
            self.limits[name].append(limits[name])
        modes = {"go_around": force.go_around, "level_captured": vertical_modes["level_captured"]}
        for name in MODES:
            self.modes[name].append(modes[name])

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

        return Flown(states=rows(self.states), commands=rows(self.commands),
                     actuators=rows([*self.actuator_rows, self.actuators]), wanted=rows(self.wanted),
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

