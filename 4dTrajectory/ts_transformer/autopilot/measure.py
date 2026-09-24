"""The executor's parameters, measured or derived (executor design §9–§10, the E7 plan in §9).

Data (train, the labeller's own reading of each flight — `flight_measurements`, pooled by
`measured_values`):

- ``decel_mps2`` / ``accel_mps2`` (a_dec, a_acc): speed words with a target, from the word's row to
  where the speed enters the target's band, transitions of at least `MIN_SPAN_S`, the median mean
  acceleration of each sign;
- ``unspecified_decel_mps2`` (a_unspec): after an "unspecified" word, the first decelerating piece of
  the ground speed's piecewise fit lasting at least `MIN_SPAN_S`, the median slope;
- ``land_aim_height_m``: the height above the pointed threshold at which the flight would cross it,
  extrapolated along its last `CROSSING_FIT_ROWS` rows, the median; ``land_window_low_m`` / ``land_window_high_m``
  the same heights' `LAND_WINDOW_PERCENTILES` — where the observed flights cross, the window the executor may
  cross in to keep inside a word's tube. (Every labelled train sentence ends
  within 1.2 km of the threshold, median 0.1 km: the extrapolation is short.)

These are the values the vocabulary does not settle yet (the vocabulary-only plan's second stage, user 2026-09-24):
the pace of a speed change and the landing aim. The turn rate and bank the executor flies at, and the word delays, are
the vocabulary's (`derive.py`; the data measurements that set them are archived: `archive/executor_vocabulary_only_2026_09/`).
Torch-free, so the runner's worker processes (`measure_chunk`) read the train split without loading the dynamics.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.labeller.read import Admitted, Reading, admit, read_flight
from ts_transformer.instructions.labeller.speed import span_checks
from ts_transformer.instructions.piecewise import fit_pieces
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import SPEED, Words

MIN_SPAN_S = 20.0
CROSSING_FIT_ROWS = 10
LAND_WINDOW_PERCENTILES = (5.0, 95.0)
MEASUREMENTS = ("transition_accel_mps2", "unspecified_slope_mps2", "crossing_height_m")


def flight_measurements(flight: Admitted, reading: Reading, geometry: AirportGeometry, spec: VocabularySpec,
                        words: Words) -> dict[str, list[float]]:
    """One labelled flight's contribution to every data parameter."""
    smoothed = flight.smoothed
    speed = smoothed.ground_speed_mps
    out: dict[str, list[float]] = {name: [] for name in MEASUREMENTS}
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
        "decel_mps2": rounded(float(np.median(-accel[accel < 0.0])), 0.01, round),
        "accel_mps2": rounded(float(np.median(accel[accel > 0.0])), 0.01, round),
        "unspecified_decel_mps2": rounded(float(np.median(-np.asarray(pooled["unspecified_slope_mps2"]))), 0.01, round),
        "land_aim_height_m": rounded(float(np.median(pooled["crossing_height_m"])), 0.1, round),
        "land_window_low_m": rounded(float(np.percentile(pooled["crossing_height_m"], LAND_WINDOW_PERCENTILES[0])),
                                     0.1, round),
        "land_window_high_m": rounded(float(np.percentile(pooled["crossing_height_m"], LAND_WINDOW_PERCENTILES[1])),
                                      0.1, round),
    }
    counts = {name: len(pooled[name]) for name in MEASUREMENTS}
    counts["decelerations"], counts["accelerations"] = int((accel < 0.0).sum()), int((accel > 0.0).sum())
    return {"values": values, "counts": counts}


def measure_chunk(flights: list[FlightSignals], stored: list[tuple[np.ndarray, int]], spec_data: dict[str, Any],
                  geometry_data: dict[str, Any]) -> dict[str, list[float]]:
    """`measure_flights` in a worker process: the spec and the geometries as their dicts."""
    spec = VocabularySpec.from_dict(spec_data)
    geometries = {code: AirportGeometry.from_dict(data) for code, data in geometry_data.items()}
    return measure_flights(flights, stored, spec, Words(spec), geometries)


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
