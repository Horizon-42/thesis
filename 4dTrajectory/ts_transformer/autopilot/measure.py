"""The executor's parameters, measured or derived (executor design §9–§10, the E7 plan in §9).

Data (train, the labeller's own reading of each flight — `flight_measurements`, pooled by
`measured_values`):

- ``turn_rate_deg_s`` (r_turn): turns of at least `TURN_MIN_DEG`, their middle half (the first and last
  quarter dropped: the 15 s velocity fit smears the roll-in and roll-out), the median row rate;
- ``bank_cap_deg`` (φ_cap): the same middle rows at ground speeds in `FAST_BAND_MPS`, the median bank,
  up to a whole degree, never past the vocabulary's bank ceiling;
- ``decel_mps2`` / ``accel_mps2`` (a_dec, a_acc): speed words with a target, from the word's row to
  where the speed enters the target's band, transitions of at least `MIN_SPAN_S`, the median mean
  acceleration of each sign;
- ``unspecified_decel_mps2`` (a_unspec): after an "unspecified" word, the first decelerating piece of
  the ground speed's piecewise fit lasting at least `MIN_SPAN_S`, the median slope;
- ``land_aim_height_m``: the height above the pointed threshold at which the flight would cross it,
  extrapolated along its last `CROSSING_FIT_ROWS` rows, the median — over the flights whose sentence ends
  within `CROSSING_MAX_BEFORE_M` of the threshold (a longer extrapolation reads the track's slope, not
  where it crosses).

Torch-free, so the runner's worker processes read the train split without loading the dynamics; method A
is `derive.py`, method B `observe.py`.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.labeller.read import Admitted, Reading, admit, read_flight
from ts_transformer.instructions.labeller.speed import span_checks
from ts_transformer.instructions.piecewise import fit_pieces
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import SPEED, Words

TURN_MIN_DEG = 90.0
FAST_BAND_MPS = (115.0, 140.0)
MIN_SPAN_S = 20.0
CROSSING_FIT_ROWS = 10
CROSSING_MAX_BEFORE_M = 1500.0
MEASUREMENTS = ("turn_mid_rate_deg_s", "turn_mid_fast_bank_deg", "transition_accel_mps2", "unspecified_slope_mps2",
                "crossing_height_m")


def flight_measurements(flight: Admitted, reading: Reading, geometry: AirportGeometry, spec: VocabularySpec,
                        words: Words) -> dict[str, list[float]]:
    """One labelled flight's contribution to every data parameter."""
    smoothed = flight.smoothed
    track, speed = smoothed.track_deg, smoothed.ground_speed_mps
    out: dict[str, list[float]] = {name: [] for name in MEASUREMENTS}
    for turn in reading.checks["turns"]:
        if turn["kind"] != "turn" or abs(turn["turn_deg"]) < TURN_MIN_DEG:
            continue
        start, stop = int(turn["departure_row"]), int(turn["arrival_row"])
        quarter = (stop - start) // 4
        rows = slice(start + quarter, stop - quarter)
        rate = np.abs(np.diff(track[rows])) / spec.step_s
        middle_speed = speed[rows][1:]
        out["turn_mid_rate_deg_s"] += rate.tolist()
        fast = (middle_speed >= FAST_BAND_MPS[0]) & (middle_speed <= FAST_BAND_MPS[1])
        out["turn_mid_fast_bank_deg"] += envelope.bank_deg_from_turn_rate(rate[fast], middle_speed[fast]).tolist()
    for check in span_checks(reading.instructions, speed, spec, words):
        seconds = check["arrival_rows"] * spec.step_s
        if not check["cut_before_arrival"] and seconds >= MIN_SPAN_S:
            row = check["row"]
            out["transition_accel_mps2"].append(float(speed[row + check["arrival_rows"]] - speed[row]) / seconds)
    time = flight.signals.time_s
    pieces = fit_pieces(time, speed, spec.speed_fit_tolerance_mps)
    for word in reading.instructions:
        if word.column != SPEED or word.value != words.speed_unspecified:
            continue
        for piece in pieces:
            lasting = (piece.stop - 1 - max(piece.start, word.row)) * spec.step_s
            if piece.stop > word.row and piece.slope < -spec.speed_flat_accel_mps2 and lasting >= MIN_SPAN_S:
                out["unspecified_slope_mps2"].append(piece.slope)
                break
    rows = len(reading.words)
    candidate = geometry.candidates[reading.runway_index]
    signals = flight.signals
    relative = relative_to_runway(signals.e_m[:rows], signals.n_m[:rows], signals.track_deg[:rows],
                                  signals.altitude_m[:rows], candidate)
    before, height = relative.before_threshold_m, relative.height_above_threshold_m
    if rows >= CROSSING_FIT_ROWS and before[-1] <= CROSSING_MAX_BEFORE_M:
        last = slice(rows - CROSSING_FIT_ROWS, rows)
        slope = np.polyfit(before[last], height[last], 1)[0]
        out["crossing_height_m"].append(float(height[-1] - slope * before[-1]))
    return out


def rounded(value: float, step: float, how) -> float:
    """``value`` to a multiple of ``step``, by ``how`` (round, math.floor, math.ceil)."""
    return round(how(round(value / step, 9)) * step, 9)


def measured_values(pooled: dict[str, Sequence[float]], spec: VocabularySpec) -> dict[str, Any]:
    """The data parameters from the pooled measurements, with the count behind each."""
    accel = np.asarray(pooled["transition_accel_mps2"])
    values = {
        "turn_rate_deg_s": rounded(float(np.median(pooled["turn_mid_rate_deg_s"])), 0.05, round),
        "bank_cap_deg": min(spec.turn_bank_max_deg, rounded(float(np.median(pooled["turn_mid_fast_bank_deg"])), 1.0,
                                                           math.ceil)),
        "decel_mps2": rounded(float(np.median(-accel[accel < 0.0])), 0.01, round),
        "accel_mps2": rounded(float(np.median(accel[accel > 0.0])), 0.01, round),
        "unspecified_decel_mps2": rounded(float(np.median(-np.asarray(pooled["unspecified_slope_mps2"]))), 0.01, round),
        "land_aim_height_m": rounded(float(np.median(pooled["crossing_height_m"])), 0.1, round),
    }
    counts = {name: len(pooled[name]) for name in MEASUREMENTS}
    counts["decelerations"], counts["accelerations"] = int((accel < 0.0).sum()), int((accel > 0.0).sum())
    return {"values": values, "counts": counts}


def measure_flights(flights: list[FlightSignals], stored: list[tuple[np.ndarray, int]], spec: VocabularySpec,
                    words: Words, geometries: dict[str, AirportGeometry]) -> dict[str, list[float]]:
    """Every flight's contribution, pooled; ``stored`` is each flight's stored sentence (its words grid and
    runway), which the re-read must reproduce — the parameters are measured on the sentences the executor flies."""
    pooled: dict[str, list[float]] = {name: [] for name in MEASUREMENTS}
    for flight, (grid, runway_index) in zip(flights, stored):
        geometry = geometries[flight.airport]
        reading = read_flight(flight, geometry, spec, words)
        if not np.array_equal(reading.words, grid) or reading.runway_index != runway_index:
            raise ValueError(f"{flight.dataset_id}: the re-read sentence differs from the stored one")
        for name, values in flight_measurements(admit(flight, geometry, spec), reading, geometry, spec, words).items():
            pooled[name] += values
    return pooled
