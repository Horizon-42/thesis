"""dynamics_comparison_backend.py
================================
Backend for the interactive *Dynamics Comparison* mode.

Flies one trajectory under one constant control four different ways (see
``4dTrajectory/optimization/dynamics_comparison.py``) and turns the result into:

  * a multi-system CZML the frontend loads into a ``CzmlDataSource`` and plays on
    Cesium's own clock — one colored, growing path per system (A/B/C/D), each a
    separate entity so the frontend can hide any one of them, and
  * a ``chart`` payload: per-sample horizontal / altitude / heading / speed error
    of A, C, D against the reference B, plus the final values — for the stats
    plots that pop up after the run.

The four systems differ only in how the dynamics are integrated (fixed vs
re-anchored tangent frame, geodetic RHS with/without the transport terms), so
the run visualises *how the choice of dynamics formulation changes the drift*.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any

from aeroviz_backend import paths  # noqa: F401
from aeroviz_backend.czml_common import EPOCH, document_packet, iso
from aeroviz_backend.dynamics_comparison_history import (
    average_history,
    clear_history,
    history_count,
    save_record,
)
from aeroviz_backend.simulation_backend import (
    DEFAULT_AIRCRAFT_TYPE,
    DEFAULT_STATE,
    clamp,
    format_geodetic_state,
    read_aircraft,
    read_float,
    read_geodetic_state,
    read_required_mapping,
)

from aerodynamic_model.casadi_simulator import aero_params_for_aircraft
from dynamics_comparison import (
    COMPARED_KEYS,
    compare_dynamics,
    error_series,
    even_sample_indices,
)


_ENTITY_PREFIX = "dyncmp-"

DEFAULT_DURATION_S = 240.0
MIN_DURATION_S = 5.0
MAX_DURATION_S = 600.0
DEFAULT_DT_S = 0.1
MIN_DT_S = 0.05
MAX_DT_S = 1.0
MAX_SAMPLES = 220
# Target playback wall-time at the doc multiplier, so a long flight does not take
# minutes to watch.
_TARGET_PLAYBACK_S = 40.0

_ERROR_FIELDS = ("horiz", "alt", "head", "speed")

# Per-system presentation: colour (rgba), label and whether it is the reference.
# Colours are distinct and read intuitively — A (worst drift) warm, B (truth)
# near-white, C/D cool — and B/C deliberately differ so the user can tell them
# apart even though C overlaps B physically.
_SYSTEMS = (
    {
        "key": "A",
        "label": "A · fixed tangent ENU (drifts)",
        "color": (244, 114, 22, 240),
        "reference": False,
    },
    {
        "key": "B",
        "label": "B · re-anchored ENU (reference)",
        "color": (226, 232, 240, 245),
        "reference": True,
    },
    {
        "key": "C",
        "label": "C · geodetic RHS, +transport",
        "color": (56, 189, 248, 240),
        "reference": False,
    },
    {
        "key": "D",
        "label": "D · geodetic RHS, no transport",
        "color": (250, 204, 21, 240),
        "reference": False,
    },
)


def _systems_payload() -> list[dict[str, Any]]:
    return [
        {
            "key": system["key"],
            "label": system["label"],
            "colorRgba": list(system["color"]),
            "isReference": system["reference"],
        }
        for system in _SYSTEMS
    ]


class DynamicsComparisonBackend:
    def run(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        initial_payload = read_required_mapping(payload, "initialState")
        control_payload = read_required_mapping(payload, "control")
        aircraft = read_aircraft(initial_payload, DEFAULT_AIRCRAFT_TYPE)
        initial_state = read_geodetic_state(initial_payload, DEFAULT_STATE, aircraft)

        thrust = clamp(read_float(control_payload, "thrustN", aircraft.approach_thrust_guess_n),
                       0.0, aircraft.max_thrust_n)
        bank_rad = math.radians(clamp(read_float(control_payload, "bankDeg", 0.0), -60.0, 60.0))
        load_factor = clamp(read_float(control_payload, "loadFactor", 1.0), 0.0, 3.0)

        duration_s = clamp(read_float(payload, "durationS", DEFAULT_DURATION_S),
                           MIN_DURATION_S, MAX_DURATION_S)
        dt_s = clamp(read_float(payload, "dtS", DEFAULT_DT_S), MIN_DT_S, MAX_DT_S)

        ap = aero_params_for_aircraft(aircraft)
        comparison = compare_dynamics(
            start=dict(
                lat=initial_state.latitude,
                lon=initial_state.longitude,
                alt=initial_state.altitude,
                V=initial_state.V,
                psi=initial_state.psi,
                gamma=initial_state.gamma,
                mass=initial_state.m,
            ),
            control=(thrust, bank_rad, load_factor),
            aero_params=[ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall],
            duration_s=duration_s,
            dt_s=dt_s,
            # Anchor system A at the start so all four share a common origin and
            # diverge as they fly away from it (clearest "different dynamics ->
            # different drift" picture for an interactive run).
            anchor_geo=None,
        )

        indices = even_sample_indices(comparison.n_samples, MAX_SAMPLES)
        realized_duration = comparison.times_s[indices[-1]]
        multiplier = max(1, round(realized_duration / _TARGET_PLAYBACK_S))

        systems = _systems_payload()
        chart = _build_chart(comparison, indices)

        # Persist this run so it can be averaged with later runs (#2/#3).
        count = save_record(
            {
                "meta": {
                    "aircraftType": aircraft.code,
                    "control": {
                        "thrustN": thrust,
                        "bankDeg": math.degrees(bank_rad),
                        "loadFactor": load_factor,
                    },
                    "durationS": realized_duration,
                    "requestedDurationS": duration_s,
                    "dtS": dt_s,
                    "initialState": format_geodetic_state(initial_state, aircraft.code),
                },
                "systems": systems,
                "chart": chart,
            },
            now_iso=iso(datetime.now(timezone.utc)),
        )

        return {
            "ok": True,
            "durationS": realized_duration,
            # The requested horizon (clamped) — differs from durationS when the
            # rollout truncated early (reached ground / left the envelope), so the
            # frontend can tell the user instead of silently showing less.
            "requestedDurationS": duration_s,
            "dtS": dt_s,
            "aircraftType": aircraft.code,
            "historyCount": count,
            "systems": systems,
            "playback": {
                "epochIso": iso(EPOCH),
                "multiplier": multiplier,
                "czml": _build_czml(comparison, indices, realized_duration, multiplier),
            },
            "chart": chart,
        }

    def history_count(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"ok": True, "historyCount": history_count()}

    def average(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        averaged = average_history()
        if averaged is None:
            return {"ok": False, "error": "No comparison history to average yet."}
        return {
            "ok": True,
            "runCount": averaged["runCount"],
            "systems": _systems_payload(),
            "chart": averaged["chart"],
        }

    def clear(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"ok": True, "historyCount": clear_history()}


# ── CZML assembly ─────────────────────────────────────────────────────────────

def _build_czml(comparison, indices, duration_s, multiplier) -> list[dict[str, Any]]:
    end_dt = EPOCH + timedelta(seconds=duration_s)
    packets: list[dict[str, Any]] = [
        document_packet(end_dt, multiplier, "AeroViz-4D Dynamics Comparison")
    ]
    for system in _SYSTEMS:
        packets.append(_system_packet(system, comparison, indices, duration_s))
    return packets


def _system_packet(system, comparison, indices, duration_s) -> dict[str, Any]:
    key = system["key"]
    path = comparison.paths[key]
    position_flat: list[float] = []
    for i in indices:
        p = path[i]
        position_flat.extend([comparison.times_s[i], p[1], p[0], p[2]])  # t, lon, lat, alt

    color = list(system["color"])
    is_ref = system["reference"]
    return {
        "id": f"{_ENTITY_PREFIX}{key}",
        "name": system["label"],
        "position": {
            "epoch": iso(EPOCH),
            "cartographicDegrees": position_flat,
            "interpolationAlgorithm": "LINEAR",
            "forwardExtrapolationType": "HOLD",
        },
        "point": {
            "pixelSize": 12 if is_ref else 9,
            "color": {"rgba": color},
            "outlineColor": {"rgba": [15, 23, 42, 255]},
            "outlineWidth": 1.6,
        },
        "path": {
            "material": {"solidColor": {"color": {"rgba": color}}},
            "width": 4 if is_ref else 3,
            "leadTime": 0,
            "trailTime": duration_s,
            "resolution": 2,
        },
        "label": {
            "text": key,
            "font": "bold 13px sans-serif",
            "fillColor": {"rgba": color},
            "outlineColor": {"rgba": [15, 23, 42, 255]},
            "outlineWidth": 2,
            "style": "FILL_AND_OUTLINE",
            "verticalOrigin": "BOTTOM",
            "pixelOffset": {"cartesian2": [0, -14]},
        },
    }


# ── Chart payload ─────────────────────────────────────────────────────────────

def _build_chart(comparison, indices) -> dict[str, Any]:
    errs = error_series(comparison, indices)
    series = {
        key: {field: [round(v, 6) for v in errs[key][field]] for field in _ERROR_FIELDS}
        for key in COMPARED_KEYS
    }
    final = {
        key: {field: round(errs[key][field][-1], 6) for field in _ERROR_FIELDS}
        for key in COMPARED_KEYS
    }
    return {
        "distanceKm": [round(comparison.dist_m[i] / 1000.0, 5) for i in indices],
        "timeS": [round(comparison.times_s[i], 4) for i in indices],
        "series": series,
        "final": final,
    }
