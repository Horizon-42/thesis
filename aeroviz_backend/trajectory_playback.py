"""trajectory_playback.py
========================
Turn an optimizer result into a *playback-ready* CZML document plus a dense
state sample series, by running the dynamics forward once.

Why this exists
---------------
Historically the frontend replayed an optimized trajectory by stepping the
backend simulator on a fixed wall-clock interval (one ``/simulation/step`` per
animation frame).  That couples playback smoothness to network latency and
keeps the optimized trajectory outside Cesium's native clock/timeline.

Instead we roll the controls forward **once** here, server-side, using the same
geodetic integrator the live simulator uses (so the replay is identical to the
old per-step playback, and consistent with the optimizer to sub-mm — see
CLAUDE.md 2026-06-21).  The dense rollout becomes:

  * a CZML document the frontend loads straight into a ``CzmlDataSource`` and
    plays on Cesium's own clock — exactly like a downloaded trajectory, and
  * a compact ``samples`` series the frontend samples at the current clock time
    to drive the live numeric readout.

Phase colour: control is piecewise-constant over the N optimizer segments, so the
trail is coloured per control segment — each segment takes one solid colour
sampled from a smooth gradient by its order, so adjacent segments are visually
continuous yet distinct.  The trail itself is built as one short polyline per
sample interval, each made available only once the aircraft has passed its far
end, so the coloured trail grows smoothly and the aircraft stays at its leading
edge.  Aircraft *orientation* (heading/pitch/bank) is computed on the frontend
from the sample series, matching the live Pilot-mode aircraft exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import sys
from typing import Any

from aeroviz_backend import paths  # noqa: F401
from aeroviz_backend.simulation_backend import (
    make_geodetic_simulator,
    read_snapshot_aero,
)
from common import Control, GeodeticState, LoadFactorControl


# Optimizers that emit LoadFactorControl-shaped controls (T, mu, n); everything
# else emits alpha-based Control (T, mu, alpha).  Mirrors the frontend's
# ``trajectoryOptimizerSimulationMode``.

# Arbitrary fixed epoch.  The optimized trajectory is a standalone playback, so
# the absolute date is irrelevant — only the relative offsets matter.
_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

_AIRCRAFT_MODEL_URI = "/models/aircraft.glb"
_AIRCRAFT_ENTITY_ID = "optimized-trajectory-aircraft"
_TRAIL_ENTITY_PREFIX = "optimized-trajectory-trail-"

# Integrate fine for accuracy; emit samples coarser so the CZML stays compact.
# The trail is one short polyline per sample interval (so it grows smoothly and
# the aircraft stays at its leading edge), so keep the sample count bounded.
_INTEGRATOR_DT_S = 0.1
_MIN_OUTPUT_DT_S = 0.5
_MAX_SAMPLES = 300


@dataclass
class TrajectorySample:
    t: float  # seconds from epoch
    lon: float
    lat: float
    alt_m: float
    speed_mps: float
    heading_deg: float
    flight_path_deg: float
    bank_deg: float
    thrust_n: float
    segment_index: int
    load_factor: float | None
    attack_deg: float | None
    lift_coefficient: float
    drag_coefficient: float
    actual_load_factor: float


def simulation_mode_for_optimizer(optimizer_name: str) -> str:
    # Covers casadiIpopt and every casadiDirectCollocation* defect-scheme
    # variant (load-factor controls); everything else is alpha-based.
    is_casadi = optimizer_name == "casadiIpopt" or optimizer_name.startswith(
        "casadiDirectCollocation"
    )
    return "casadi" if is_casadi else "alpha"


def build_optimized_trajectory_playback(
    optimizer_name: str,
    initial_state: GeodeticState,
    node_control: Any,
    final_time: float,
    aircraft: Any,
) -> dict[str, Any] | None:
    """Roll the controls forward once and return ``{czml, samples, ...}``.

    Returns ``None`` when there is nothing to play (no controls or a
    non-positive horizon), so the caller can simply omit the field.
    """
    controls = list(node_control) if node_control is not None else []
    if not controls or final_time <= 0.0:
        return None

    simulation_mode = simulation_mode_for_optimizer(optimizer_name)
    samples = _rollout(
        initial_state,
        controls,
        float(final_time),
        aircraft,
        simulation_mode,
    )
    if len(samples) < 2:
        return None

    realised_final_time = samples[-1].t
    czml = _build_czml(samples, realised_final_time, aircraft)
    return {
        "epochIso": _iso(_EPOCH),
        "multiplier": 1,
        "czml": czml,
        "samples": [_serialize_sample(sample) for sample in samples],
    }


# ── Forward rollout ───────────────────────────────────────────────────────────

def _rollout(
    initial_state: GeodeticState,
    node_control: list[Any],
    final_time: float,
    aircraft: Any,
    simulation_mode: str,
) -> list[TrajectorySample]:
    segment_count = len(node_control)
    segment_duration = final_time / segment_count
    output_dt = max(_MIN_OUTPUT_DT_S, final_time / _MAX_SAMPLES)
    integrator_dt = min(_INTEGRATOR_DT_S, output_dt)

    simulator = make_geodetic_simulator(aircraft, simulation_mode)
    controls = [_build_control(row, simulation_mode) for row in node_control]

    state = initial_state
    t = 0.0
    samples = [
        _make_sample(t, state, controls[0], 0, simulator, simulation_mode)
    ]

    for segment_index, control in enumerate(controls):
        segment_end = (segment_index + 1) * segment_duration
        next_output_t = t + output_dt
        while t < segment_end - 1e-9:
            step_dt = min(integrator_dt, segment_end - t)
            try:
                state = simulator.step(state, control, step_dt)
            except ValueError as exc:
                # The replay left the valid flight envelope (e.g. altitude
                # below ground).  A real optimized solution lands on the target
                # and never does this; truncate rather than fail the whole
                # optimize call, and surface it so it is not silent.
                sys.stderr.write(
                    "[aeroviz-backend] playback rollout truncated at "
                    f"t={t:.1f}s: {exc}\n"
                )
                return samples
            t += step_dt
            at_segment_end = t >= segment_end - 1e-9
            if t >= next_output_t - 1e-9 or at_segment_end:
                samples.append(
                    _make_sample(t, state, control, segment_index, simulator, simulation_mode)
                )
                next_output_t = t + output_dt

    return samples


def _build_control(row: Any, simulation_mode: str) -> Control | LoadFactorControl:
    thrust, bank_rad, third = (float(row[0]), float(row[1]), float(row[2]))
    if simulation_mode == "casadi":
        return LoadFactorControl(thrust=thrust, bank_rad=bank_rad, load_factor=third)
    return Control(thrust=thrust, bank_rad=bank_rad, attack_rad=third)


def _make_sample(
    t: float,
    state: GeodeticState,
    control: Control | LoadFactorControl,
    segment_index: int,
    simulator: Any,
    simulation_mode: str,
) -> TrajectorySample:
    load_factor = (
        control.load_factor if isinstance(control, LoadFactorControl) else None
    )
    attack_deg = (
        math.degrees(control.attack_rad) if isinstance(control, Control) else None
    )
    cl, cd, actual_load_factor = read_snapshot_aero(
        simulator, state, control, simulation_mode
    )
    return TrajectorySample(
        t=t,
        lon=state.longitude,
        lat=state.latitude,
        alt_m=state.altitude,
        speed_mps=state.V,
        heading_deg=math.degrees(state.psi) % 360.0,
        flight_path_deg=math.degrees(state.gamma),
        bank_deg=math.degrees(control.bank_rad),
        thrust_n=control.thrust,
        segment_index=segment_index,
        load_factor=load_factor,
        attack_deg=attack_deg,
        lift_coefficient=cl,
        drag_coefficient=cd,
        actual_load_factor=actual_load_factor,
    )


def _serialize_sample(sample: TrajectorySample) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "t": sample.t,
        "lon": sample.lon,
        "lat": sample.lat,
        "altM": sample.alt_m,
        "speedMps": sample.speed_mps,
        "headingDeg": sample.heading_deg,
        "flightPathDeg": sample.flight_path_deg,
        "bankDeg": sample.bank_deg,
        "thrustN": sample.thrust_n,
        "segmentIndex": sample.segment_index,
        "liftCoefficient": sample.lift_coefficient,
        "dragCoefficient": sample.drag_coefficient,
        "actualLoadFactor": sample.actual_load_factor,
    }
    if sample.load_factor is not None:
        payload["loadFactor"] = sample.load_factor
    if sample.attack_deg is not None:
        payload["attackDeg"] = sample.attack_deg
    return payload


# ── CZML assembly ─────────────────────────────────────────────────────────────

def _build_czml(
    samples: list[TrajectorySample],
    final_time: float,
    aircraft: Any,
) -> list[dict[str, Any]]:
    end_dt = _EPOCH + timedelta(seconds=final_time)
    packets: list[dict[str, Any]] = [
        _build_document_packet(end_dt),
        _build_aircraft_packet(samples, aircraft),
    ]
    packets.extend(_build_trail_polyline_packets(samples, end_dt))
    return packets


def _build_document_packet(end_dt: datetime) -> dict[str, Any]:
    return {
        "id": "document",
        "name": "AeroViz-4D Optimized Trajectory",
        "version": "1.0",
        "clock": {
            "interval": f"{_iso(_EPOCH)}/{_iso(end_dt)}",
            "currentTime": _iso(_EPOCH),
            "multiplier": 1,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        },
    }


def _build_aircraft_packet(
    samples: list[TrajectorySample],
    aircraft: Any,
) -> dict[str, Any]:
    position_flat: list[float] = []
    for sample in samples:
        position_flat.extend([sample.t, sample.lon, sample.lat, sample.alt_m])

    # No ``orientation`` here: the frontend sets it from the sample series so the
    # optimized aircraft banks/pitches/yaws exactly like the live Pilot aircraft.
    return {
        "id": _AIRCRAFT_ENTITY_ID,
        "name": "Optimized Trajectory",
        "description": f"<b>Optimized Trajectory</b><br/>Type: {aircraft.code}",
        "model": {
            "gltf": _AIRCRAFT_MODEL_URI,
            "scale": 3.0,
            "minimumPixelSize": 32,
            "maximumScale": 20000,
            "runAnimations": True,
        },
        "position": {
            "epoch": _iso(_EPOCH),
            "cartographicDegrees": position_flat,
            "interpolationAlgorithm": "LINEAR",
            "forwardExtrapolationType": "HOLD",
        },
        "label": {
            "text": "OPT",
            "font": "bold 12px sans-serif",
            "fillColor": {"rgba": [255, 255, 255, 255]},
            "outlineColor": {"rgba": [0, 0, 0, 255]},
            "outlineWidth": 2,
            "style": "FILL_AND_OUTLINE",
            "verticalOrigin": "BOTTOM",
            "pixelOffset": {"cartesian2": [0, -30]},
        },
    }


def _build_trail_polyline_packets(
    samples: list[TrajectorySample],
    end_dt: datetime,
) -> list[dict[str, Any]]:
    """A growing, per-control-segment-coloured trail behind the aircraft.

    One short polyline per sample interval, coloured by the interval's control
    segment, and made available only once the aircraft has passed the interval's
    far end.  So the trail grows smoothly (one short step at a time) and always
    ends just behind the aircraft — the aircraft stays at its leading edge.
    """
    segment_count = max((sample.segment_index for sample in samples), default=0) + 1
    packets: list[dict[str, Any]] = []
    for index in range(len(samples) - 1):
        a = samples[index]
        b = samples[index + 1]
        color = _segment_color_rgba(a.segment_index, segment_count)
        # Reveal this step once the aircraft has reached its far end, so the trail
        # trails the aircraft rather than jumping ahead of it.  Millisecond
        # precision keeps the sub-second growth steps distinct.
        available_from = _EPOCH + timedelta(seconds=b.t)
        packets.append({
            "id": f"{_TRAIL_ENTITY_PREFIX}{index}",
            "availability": f"{_iso_ms(available_from)}/{_iso_ms(end_dt)}",
            "polyline": {
                "positions": {
                    "cartographicDegrees": [
                        a.lon, a.lat, a.alt_m,
                        b.lon, b.lat, b.alt_m,
                    ]
                },
                "width": 3,
                "material": {
                    "solidColor": {"color": {"rgba": list(color)}}
                },
                "arcType": "NONE",
            },
        })
    return packets


# ── Per-segment colour scale ──────────────────────────────────────────────────
# Each control segment takes one solid colour, sampled from a smooth gradient by
# its order (0 → first segment, 1 → last).  Adjacent segments are neighbours on
# the gradient (continuous) but a whole step apart (distinct).  This is colour by
# *segment*, not by control magnitude.

# Smooth perceptual sweep: blue → cyan → green → yellow → red.
_SEGMENT_COLOR_STOPS = (
    (0.00, 40, 90, 255),    # blue
    (0.25, 0, 200, 255),    # cyan
    (0.50, 0, 220, 120),    # green
    (0.75, 255, 210, 0),    # yellow
    (1.00, 255, 60, 40),    # red
)
_SEGMENT_COLOR_ALPHA = 235


def _segment_color_rgba(segment_index: int, segment_count: int) -> tuple[int, int, int, int]:
    fraction = segment_index / max(1, segment_count - 1)
    fraction = min(1.0, max(0.0, fraction))
    for i in range(len(_SEGMENT_COLOR_STOPS) - 1):
        lo_t, lo_r, lo_g, lo_b = _SEGMENT_COLOR_STOPS[i]
        hi_t, hi_r, hi_g, hi_b = _SEGMENT_COLOR_STOPS[i + 1]
        if fraction <= hi_t:
            span = hi_t - lo_t
            local = 0.0 if span <= 0 else (fraction - lo_t) / span
            return (
                round(lo_r + (hi_r - lo_r) * local),
                round(lo_g + (hi_g - lo_g) * local),
                round(lo_b + (hi_b - lo_b) * local),
                _SEGMENT_COLOR_ALPHA,
            )
    _, r, g, b = _SEGMENT_COLOR_STOPS[-1]
    return (r, g, b, _SEGMENT_COLOR_ALPHA)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_ms(dt: datetime) -> str:
    """ISO 8601 with millisecond precision (for sub-second availability steps)."""
    base = dt.astimezone(timezone.utc)
    return base.strftime("%Y-%m-%dT%H:%M:%S.") + f"{base.microsecond // 1000:03d}Z"
