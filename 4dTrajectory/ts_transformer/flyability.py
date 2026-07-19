"""Post-hoc flyability: what controls would a predicted trajectory have REQUIRED?

The learned predictor emits positions and velocities and nothing else — no thrust, no
bank, no load factor — so nothing in it guarantees the trajectory is physically flyable.
That is the "statistically plausible but unflyable" problem the survey in
``4dTrajectory/docs`` names. This module measures it, after the fact, without touching
training and without casadi.

**Method.** The load-factor point-mass model inverts in CLOSED FORM — no iteration, no
root-finding. Its rotational equations are (``aerodynamic_model/simulator_simple.py``,
``casadi_simulator.py``; g = 9.81 as everywhere in this repo)::

    psi_dot   = g n sin(mu) / (V cos(gamma))
    gamma_dot = g (n cos(mu) - cos(gamma)) / V

Given a trajectory we know ``V, psi, gamma`` and can difference them for the rates, so::

    A  = psi_dot   * V * cos(gamma) / g          =  n sin(mu)
    B  = gamma_dot * V / g + cos(gamma)          =  n cos(mu)
    n  = hypot(A, B)
    mu = atan2(A, B)

``n`` then fixes the lift coefficient, lift fixes drag, and the along-track equation
``V_dot = (T - D)/m - g sin(gamma)`` gives thrust last and explicitly::

    T = m (V_dot + g sin(gamma)) + D

So one pass over the samples recovers ``(n, mu, T, Cl_required)`` with no solver. The
same ``g tan(mu) = V cos(gamma) psi_dot`` relation already appears in the optimizer's
``terminal_bank_constraint_expr`` — this is the same physics, run backwards.

**Transport terms.** In the geodetic model ``psi_dot`` and ``gamma_dot`` also pick up
earth-frame rotation (the tangent plane turning under the aircraft). Those terms are
subtracted before solving, or the inversion would bill the aircraft for a bank it never
commanded. The default matches the dynamics' own default (``transport="approx"``).

**What this is NOT.** It does not project a prediction onto a flyable trajectory (that
would need casadi and is a different, larger piece of work — see the README's four
routes). It reports what the prediction demanded and whether that sits inside the
envelope.

Records in, verdicts out: the entry point takes an evaluation record's ``states`` list,
so it grades ANY producer's output — a ts_transformer prediction or an optimizer batch —
with the same code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from aircraft.aero_params import AeroParams, aero_params_for_aircraft
from aircraft.aircraft_sets import Aircraft
from geokit import WGS84_A, WGS84_E2

# Gravity: 9.81, matching every dynamics path in this repo (simulator.py, simulator_simple.py,
# casadi_exprs.py, collocation/components.py) — deliberately NOT 9.80665.
G = 9.81

# ISA density, mirroring Atmosphere.get_ISA_density / casadi_exprs.isa_density_expr.
_ISA_T0 = 288.15
_ISA_LAPSE = 0.0065
_ISA_RHO0 = 1.225
_ISA_EXP = 4.25588

# Speeds below this make the rotational equations singular (they divide by V); a predicted
# sample that slow is not a bank problem, it is a broken sample, and is reported as such.
_MIN_SPEED_MS = 1.0


@dataclass(frozen=True)
class Envelope:
    """The limits a required control is tested against.

    Values come from what the optimizer already enforces, so a trajectory called flyable
    here is one the optimizer would have been allowed to produce:

    - ``max_thrust_n`` — ``aircraft.engine.max_thrust_total_n``.
    - ``max_bank_rad`` — pi/4, the optimizer's general bound
      (``collocation/components.py`` control bounds). Its *terminal* bound is much tighter
      (5 deg) but applies only at the threshold, so it is not a whole-trajectory limit.
    - ``min_load_factor`` / ``max_load_factor`` — 0.5 / 2.0, the optimizer's bounds
      (``collocation/optimizer.py`` ``_aircraft_meta``). NOTE these carry an explicit
      "just example values" caveat upstream in ``casadi_optimizer.py``; they are this
      project's working numbers, not certified aircraft limits, and the report says so.
    - ``Cl_max`` — from ``aero_params_for_aircraft`` (A320: 2.7). This is the one genuinely
      physical limit here: ``Cl_required > Cl_max`` IS stall.
    """

    max_thrust_n: float
    cl_max: float
    max_bank_rad: float = math.pi / 4
    min_load_factor: float = 0.5
    max_load_factor: float = 2.0

    @classmethod
    def for_aircraft(cls, aircraft: Aircraft, aero: AeroParams) -> Envelope:
        return cls(max_thrust_n=aircraft.engine.max_thrust_total_n, cl_max=aero.Cl_max)


# Violations that mean the trajectory is genuinely infeasible for this airframe.
HARD_VIOLATIONS = frozenset({
    "stall", "bank", "load_factor_high", "load_factor_low", "thrust_over_max",
    "speed_below_1ms",
})

# NOT a violation, despite looking like one. Negative required thrust means the aircraft
# needs MORE drag than a clean configuration provides — i.e. speedbrakes, gear, or flaps.
# Every arrival uses them, and a single clean-configuration drag polar cannot represent
# that. Calibration on real observed KRDU tracks: median required thrust is ~0.4 kN (idle,
# exactly as expected on a descent) with excursions to -118 kN, i.e. real flown approaches
# "violate" this constantly. Counting it as unflyable would score the truth as infeasible.
SOFT_VIOLATIONS = frozenset({"thrust_negative"})


@dataclass(frozen=True)
class RequiredControl:
    """What one sample demanded, and how it sat against the envelope."""

    t: float
    load_factor: float
    bank_rad: float
    thrust_n: float
    cl_required: float
    violations: tuple[str, ...]

    @property
    def hard_violations(self) -> tuple[str, ...]:
        return tuple(v for v in self.violations if v in HARD_VIOLATIONS)

    @property
    def flyable(self) -> bool:
        """No HARD violation. Soft ones (drag augmentation) are normal on an approach."""
        return not self.hard_violations


def isa_density(altitude_m: float) -> float:
    """ISA density — the model every dynamics path in this repo uses."""
    temperature = _ISA_T0 - _ISA_LAPSE * altitude_m
    return _ISA_RHO0 * (temperature / _ISA_T0) ** _ISA_EXP


def _drag(*, load_factor: float, speed: float, rho: float, mass: float,
          aero: AeroParams) -> tuple[float, float]:
    """``(drag_N, Cl_required)`` for a commanded load factor.

    Mirrors ``casadi_exprs`` / ``simulator_simple.get_aerodynamic_coefficients`` — the
    smoothstep stall-drag blend included — but reads its constants from ``AeroParams``
    rather than from ``LoadFactorSimulator``'s hardcoded attributes. That matters: the
    simulator hardcodes ``Cl_max = 1.5`` while ``aero_params_for_aircraft`` returns 2.7 for
    an A320, an 80% disagreement on where stall begins. ``aero_params.py`` documents itself
    as the single source of truth for the stall model, so this follows it.
    """
    cl_required = load_factor * mass * G / (0.5 * rho * aero.S * speed * speed)
    ratio = cl_required / aero.Cl_max
    cl = min(cl_required, aero.Cl_max)

    cd_stall = 0.0
    if ratio > aero.stall_threshold:
        fraction = min(ratio, 1.0)
        x = min(max((fraction - aero.stall_threshold) / (1.0 - aero.stall_threshold), 0.0), 1.0)
        cd_stall = aero.k_stall * (x * x * (3.0 - 2.0 * x))

    cd = aero.Cd0 + aero.k * cl * cl + cd_stall
    return 0.5 * rho * speed * speed * cd * aero.S, cl_required


def _transport_rates(*, speed: float, psi: float, gamma: float, lat_rad: float,
                     altitude_m: float, transport: str) -> tuple[float, float]:
    """``(gamma_dot, psi_dot)`` contributed by the rotating local frame, not by forces.

    Mirrors ``casadi_simulator.make_geodetic_dynamics_model``. The forward model ADDS these
    to the force-driven rates, so the inversion subtracts them; skipping that would read
    the tangent plane's own rotation as a commanded bank.
    """
    if transport == "none":
        return 0.0, 0.0

    denom = 1.0 - WGS84_E2 * math.sin(lat_rad) ** 2
    r_n = WGS84_A / math.sqrt(denom)                    # prime vertical
    r_m = WGS84_A * (1.0 - WGS84_E2) / denom ** 1.5     # meridional

    cos_gamma = math.cos(gamma)
    gamma_dot = speed * cos_gamma * (
        math.sin(psi) ** 2 / (r_m + altitude_m) + math.cos(psi) ** 2 / (r_n + altitude_m)
    )
    psi_dot = -speed * cos_gamma * math.cos(psi) * math.tan(lat_rad) / (r_n + altitude_m)

    if transport == "full":
        psi_dot += speed * math.sin(gamma) * math.sin(psi) * math.cos(psi) * (
            1.0 / (r_n + altitude_m) - 1.0 / (r_m + altitude_m)
        )
    return gamma_dot, psi_dot


def _rates(states: Sequence[dict[str, float]], index: int) -> tuple[float, float, float]:
    """``(V_dot, psi_dot, gamma_dot)`` by central difference, one-sided at the ends.

    ``psi`` is differenced through ``math.remainder(dpsi, tau)`` — the smallest signed
    equivalent angle. A plain subtraction across the +/-pi branch cut would read a 1-degree
    heading change as 359 degrees and report an impossible bank exactly where the aircraft
    turns onto final.
    """
    previous = states[max(index - 1, 0)]
    following = states[min(index + 1, len(states) - 1)]
    dt = float(following["t"]) - float(previous["t"])
    if dt <= 0.0:
        return 0.0, 0.0, 0.0

    d_psi = math.remainder(float(following["psi"]) - float(previous["psi"]), math.tau)
    return (
        (float(following["V"]) - float(previous["V"])) / dt,
        d_psi / dt,
        (float(following["gamma"]) - float(previous["gamma"])) / dt,
    )


def required_controls(
    states: Sequence[dict[str, float]],
    aircraft: Aircraft,
    *,
    aero: AeroParams | None = None,
    envelope: Envelope | None = None,
    transport: str = "approx",
) -> list[RequiredControl]:
    """Invert one trajectory's states into the controls it would have required.

    ``states`` is an evaluation record's ``states`` list — ``{t, lat, lon, alt, V, psi,
    gamma, m}`` per sample — so this grades any producer's output.
    """
    if len(states) < 2:
        raise ValueError("need at least two states to difference a trajectory")

    aero = aero or aero_params_for_aircraft(aircraft)
    envelope = envelope or Envelope.for_aircraft(aircraft, aero)

    out: list[RequiredControl] = []
    for index, state in enumerate(states):
        speed = float(state["V"])
        gamma = float(state["gamma"])
        psi = float(state["psi"])
        mass = float(state["m"])
        altitude = float(state["alt"])
        t = float(state["t"])

        if speed < _MIN_SPEED_MS:
            out.append(RequiredControl(t=t, load_factor=math.nan, bank_rad=math.nan,
                                       thrust_n=math.nan, cl_required=math.nan,
                                       violations=("speed_below_1ms",)))
            continue

        speed_dot, psi_dot, gamma_dot = _rates(states, index)
        transport_gamma, transport_psi = _transport_rates(
            speed=speed, psi=psi, gamma=gamma, lat_rad=math.radians(float(state["lat"])),
            altitude_m=altitude, transport=transport,
        )
        psi_dot -= transport_psi
        gamma_dot -= transport_gamma

        # Closed-form inversion of the rotational equations.
        a = psi_dot * speed * math.cos(gamma) / G          # n sin(mu)
        b = gamma_dot * speed / G + math.cos(gamma)        # n cos(mu)
        load_factor = math.hypot(a, b)
        bank = math.atan2(a, b)

        rho = isa_density(altitude)
        drag, cl_required = _drag(load_factor=load_factor, speed=speed, rho=rho,
                                  mass=mass, aero=aero)
        thrust = mass * (speed_dot + G * math.sin(gamma)) + drag

        violations: list[str] = []
        if cl_required > envelope.cl_max:
            violations.append("stall")
        if abs(bank) > envelope.max_bank_rad:
            violations.append("bank")
        if load_factor > envelope.max_load_factor:
            violations.append("load_factor_high")
        elif load_factor < envelope.min_load_factor:
            violations.append("load_factor_low")
        if thrust > envelope.max_thrust_n:
            violations.append("thrust_over_max")
        elif thrust < 0.0:
            violations.append("thrust_negative")

        out.append(RequiredControl(t=t, load_factor=load_factor, bank_rad=bank,
                                   thrust_n=thrust, cl_required=cl_required,
                                   violations=tuple(violations)))
    return out


def flyability_summary(controls: Sequence[RequiredControl]) -> dict[str, Any]:
    """Per-trajectory verdict: the fraction of samples inside the envelope, and why not.

    "Flyable" counts HARD violations only — see :data:`SOFT_VIOLATIONS` for why negative
    required thrust is reported but not held against a trajectory. Soft counts are kept
    separately so the drag-augmentation demand stays visible rather than being dropped.
    """
    total = len(controls)
    unflyable = [c for c in controls if c.hard_violations]
    reasons: dict[str, int] = {}
    soft: dict[str, int] = {}
    for control in controls:
        for reason in control.violations:
            bucket = soft if reason in SOFT_VIOLATIONS else reasons
            bucket[reason] = bucket.get(reason, 0) + 1

    finite = [c for c in controls if not math.isnan(c.load_factor)]
    return {
        "samples": total,
        "flyable_samples": total - len(unflyable),
        "flyable_fraction": (total - len(unflyable)) / total if total else 0.0,
        "fully_flyable": not unflyable,
        "violations": reasons,
        "soft": soft,
        "max_abs_bank_deg": max((abs(math.degrees(c.bank_rad)) for c in finite), default=math.nan),
        "max_load_factor": max((c.load_factor for c in finite), default=math.nan),
        "min_load_factor": min((c.load_factor for c in finite), default=math.nan),
        "max_thrust_n": max((c.thrust_n for c in finite), default=math.nan),
        "max_cl_required": max((c.cl_required for c in finite), default=math.nan),
    }


def flyability_batch(
    per_trajectory: Sequence[dict[str, Any]], *, envelope: Envelope, aircraft_code: str
) -> dict[str, Any]:
    """Batch roll-up over :func:`flyability_summary` results."""
    total = len(per_trajectory)
    fully = sum(1 for s in per_trajectory if s["fully_flyable"])
    samples = sum(s["samples"] for s in per_trajectory)
    flyable_samples = sum(s["flyable_samples"] for s in per_trajectory)

    reasons: dict[str, int] = {}
    soft: dict[str, int] = {}
    for summary in per_trajectory:
        for reason, count in summary["violations"].items():
            reasons[reason] = reasons.get(reason, 0) + count
        for reason, count in summary.get("soft", {}).items():
            soft[reason] = soft.get(reason, 0) + count

    return {
        "aircraft": aircraft_code,
        "envelope": {
            "max_thrust_n": envelope.max_thrust_n,
            "cl_max": envelope.cl_max,
            "max_bank_deg": math.degrees(envelope.max_bank_rad),
            "load_factor": [envelope.min_load_factor, envelope.max_load_factor],
            # Stated, not silent: two of these are the project's working numbers rather
            # than certified aircraft limits (see Envelope's docstring).
            "note": "load-factor and bank bounds are the optimizer's working values; "
                    "Cl_max is from aero_params_for_aircraft and is the physical one",
        },
        "trajectories": total,
        "fully_flyable": fully,
        "fully_flyable_rate": fully / total if total else 0.0,
        "samples": samples,
        "flyable_samples": flyable_samples,
        "flyable_sample_rate": flyable_samples / samples if samples else 0.0,
        "violations": reasons,
        "soft": soft,
    }


def report_for_records(
    predicted_states: Sequence[Sequence[dict[str, Any]]],
    observed_states: Sequence[Sequence[dict[str, Any]]],
    aircraft: Aircraft,
    *,
    aero: AeroParams | None = None,
    transport: str = "approx",
) -> dict[str, Any]:
    """The whole check for one batch: invert both sides, summarise, calibrate.

    ``predicted_states`` / ``observed_states`` are the ``states`` lists of a batch's
    evaluation records and of their span-matched reference records.
    """
    aero = aero or aero_params_for_aircraft(aircraft)
    envelope = Envelope.for_aircraft(aircraft, aero)

    def summarise(batch: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
        return [
            flyability_summary(required_controls(states, aircraft, aero=aero,
                                                 envelope=envelope, transport=transport))
            for states in batch
            # Two samples is the minimum to difference; anything shorter carries no rates.
            if len(states) >= 3
        ]

    return calibrated_report(summarise(predicted_states), summarise(observed_states),
                             envelope=envelope, aircraft_code=aircraft.code)


def calibrated_report(
    predicted: Sequence[dict[str, Any]],
    observed: Sequence[dict[str, Any]],
    *,
    envelope: Envelope,
    aircraft_code: str,
) -> dict[str, Any]:
    """Predictions' flyability RELATIVE to the observed tracks measured the same way.

    An absolute flyability number would be misleading, because this check carries a known
    systematic bias: the point-mass model has ONE clean-configuration drag polar and a
    single ``Cl_max``, while a real arrival is flown dirty (flaps, slats, gear, speedbrakes)
    with a very different polar. Measured on real flown KRDU tracks, ``Cl_required`` reaches
    ~2.56 against a ``Cl_max`` of 2.7 near touchdown — the truth itself sits near the model's
    stall boundary. So the observed tracks are the FLOOR, not 100%.

    Reporting the pair makes the systematic part cancel: if predictions and observations
    score alike, the predictor is as flyable as the thing it was trained on; a gap is the
    predictor's own doing.
    """
    pred = flyability_batch(predicted, envelope=envelope, aircraft_code=aircraft_code)
    obs = flyability_batch(observed, envelope=envelope, aircraft_code=aircraft_code)
    return {
        "predicted": pred,
        "observed_baseline": obs,
        "delta": {
            "fully_flyable_rate": pred["fully_flyable_rate"] - obs["fully_flyable_rate"],
            "flyable_sample_rate": pred["flyable_sample_rate"] - obs["flyable_sample_rate"],
        },
        "note": "observed_baseline is the same check on the real flown tracks. It is the "
                "floor, not 100%: the clean-configuration drag polar and single Cl_max "
                "cannot represent a dirty approach, so read the delta, not the absolute.",
    }
