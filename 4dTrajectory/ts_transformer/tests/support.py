"""The fixtures the test files share (review §4.6, T4-24).

Until 2026-09-10 nine files carried a byte-identical fake data provenance, three the same
control dynamics context and two the same terminal assessment context. `tests/` is a
namespace package under `ts_transformer`, so a test imports these as
``from ts_transformer.tests.support import …`` — the same way `test_two_head_duration`
already borrowed `test_duration_quantiles`'s config.
"""

from __future__ import annotations

import torch

from evaluation.thresholds import AssessmentContext
from ts_transformer.data.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from ts_transformer.outputs.conditioning import CONDITION_WIDTH
from ts_transformer.outputs.envelope import CONTROL_LOWER, CONTROL_UPPER

AIRPORT, RUNWAY = "KRDU", "05L"


def fake_data_provenance(*airports: str) -> dict:
    """The arrival-data provenance a training fixture stamps: one manifest per airport
    (KRDU when none is named), with a placeholder digest and no source records."""
    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [
            {"airport": airport, "arrival_manifest_sha256": "a" * 64, "source_records": []}
            for airport in (airports or (AIRPORT,))
        ],
    }


def dynamics_context(batch: int, cta_s: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """A control model's per-flight context for ``batch`` synthetic flights: a random
    condition vector and the shared dimensionless control box, plus the CTA when given."""
    rows = {
        "condition": torch.randn(batch, CONDITION_WIDTH),
        "control_lower": torch.tensor(CONTROL_LOWER, dtype=torch.float32).expand(batch, -1).clone(),
        "control_upper": torch.tensor(CONTROL_UPPER, dtype=torch.float32).expand(batch, -1).clone(),
    }
    if cta_s is not None:
        rows["cta_s"] = cta_s
    return rows


def terminal_contexts() -> dict[tuple[str, str], AssessmentContext]:
    """The one terminal assessment context the synthetic KRDU 05L fixtures are graded under.

    The synthetic fixtures build their approaches on the runway_thresholds.json point
    (flight_scenarios.runway_target.find_threshold), which sits 6.7 m from the CIFP Path
    Point LTP a real KRDU context would carry; the synthetic one is pinned so this context
    describes the data it is grading.
    """
    return {(AIRPORT, RUNWAY): AssessmentContext(
        benchmark="lpv", airport=AIRPORT, runway=RUNWAY,
        threshold_lat=35.8745003, threshold_lon=-78.802002,
        runway_course_deg=45.0, runway_width_m=45.72,
        runway_source="faa_nasr_apt_rwy", runway_source_cycle="2026-08-06",
        procedure_source="faa_cifp_path_point", procedure_source_cycle="2026-08-06",
        threshold_elevation_hae_m=141.86, threshold_elevation_msl_m=111.86,
        threshold_crossing_height_m=15.0, lpv_course_width_m=106.75,
    )}


def raised_airport(geometry, elevation_m: float):
    """``geometry`` with the airport elevation E at ``elevation_m`` and every threshold raised by as much: the same
    airport, higher (D58)."""
    from ts_transformer.instructions.airport import AirportGeometry

    data = geometry.to_dict()
    rise = elevation_m - data["reference"]["elevation_m"]
    data["reference"]["elevation_m"] = elevation_m
    data["candidates"] = [{**c, "elevation_m": c["elevation_m"] + rise} for c in data["candidates"]]
    return AirportGeometry.from_dict(data)


#: `instruction_airport`'s field elevation E, MSL m.
INSTRUCTION_AIRPORT_ELEVATION_M = 60.0


def instruction_airport():
    """A synthetic airport for the instruction labeller: one candidate runway "09", threshold at
    the airport frame's origin, course 090° true, elevation 100 m MSL; the airport elevation E (the field's,
    `INSTRUCTION_AIRPORT_ELEVATION_M`) 40 m below it, so that the altitude words, heights above E (D58), differ from MSL
    by one 60 m step: a level of 1,080 m MSL is the word 1,020 m."""
    from ts_transformer.instructions.airport import AirportGeometry

    return AirportGeometry.from_dict({
        "code": "KXXX", "reference": {"lat": 35.0, "lon": -78.0, "elevation_m": INSTRUCTION_AIRPORT_ELEVATION_M},
        "candidates": [{"ident": "09", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": 90.0,
                        "elevation_m": 100.0, "length_m": 3000.0}],
        "runway_ends": [{"ident": "09", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": 90.0}],
    })


def instruction_spec(**changes):
    """A vocabulary spec at the development set's measured values, with ``changes`` applied."""
    from ts_transformer.instructions import measure
    from ts_transformer.instructions.spec import VocabularySpec

    measured = measure.MeasuredValues(
        turn_rate_max_deg_s=3.5, turn_bank_max_deg=32.0,
        corridor_half_width_m=20.0, corridor_widening_deg=0.45, corridor_course_tolerance_deg=2.0,
        descent_angle_edges_deg=(-0.5, 1.4, 2.6, 3.7, 10.0), descent_angle_centres_deg=(0.8, 2.1, 3.0, 4.4),
        climb_angle_centre_deg=1.3, speed_accel_max_mps2=2.5, **measure.D22_GRID,
    )
    data = measure.build_spec(measured).to_dict()
    data.update(changes)
    return VocabularySpec.from_dict(data)


INSTRUCTION_STEP_S = 2.0


def fly_legs(legs, track0, altitude0, end_e, end_n):
    """Integrate legs of ``(rows, turn °/row, ground speed m/s, vertical rate m/s)`` from a
    track and an altitude, 2 s a row, then shift the path so its last row sits at
    ``(end_e, end_n)``. Returns ``(e, n, altitude, track, speed)``."""
    import numpy as np

    track, altitude, speed = [track0], [altitude0], []
    for rows, turn, v, climb in legs:
        for _ in range(rows):
            speed.append(v)
            track.append(track[-1] + turn)
            altitude.append(altitude[-1] + climb * INSTRUCTION_STEP_S)
    speed.append(speed[-1])
    track, altitude, speed = np.array(track), np.array(altitude), np.array(speed)
    heading = np.radians(track)
    e = np.concatenate(([0.0], np.cumsum(speed[:-1] * np.sin(heading[:-1]) * INSTRUCTION_STEP_S)))
    n = np.concatenate(([0.0], np.cumsum(speed[:-1] * np.cos(heading[:-1]) * INSTRUCTION_STEP_S)))
    return e - e[-1] + end_e, n - n[-1] + end_n, altitude, track, speed


def fixture_days():
    """The fixtures' day split (`data.day_split`): twenty June days, dealt with the real seed —
    3 test, 3 val, 2 select, 12 train."""
    from ts_transformer.data.day_split import DAY_SPLIT_SEED, split_days

    return split_days([f"2026-06-{day:02d}" for day in range(1, 21)], DAY_SPLIT_SEED)


def landing_on(split: str) -> str:
    """A landing time (ISO UTC, midday) on the first `fixture_days` day of ``split``."""
    return f"{fixture_days().days[split][0]}T12:00:00Z"


def instruction_flight(e, n, altitude, track, speed, dataset_id="KXXX:test", split="train"):
    """`FlightSignals` for a synthetic flight onto `instruction_airport`'s runway 09, landing on a
    `fixture_days` day of ``split``."""
    import numpy as np

    from ts_transformer.instructions.signals import FlightSignals

    t = np.arange(len(e)) * INSTRUCTION_STEP_S
    landing = landing_on(split)
    return FlightSignals(dataset_id, "KXXX", "09", "A320", landing.replace("T12:", "T11:"), landing, t, e, n,
                         altitude, track, speed, np.gradient(altitude, INSTRUCTION_STEP_S))


#: `parallel_airport`'s second runway, north of 09: how far, metres.
PARALLEL_SPACING_M = 891.0


def parallel_airport():
    """`instruction_airport` with a parallel runway "09L" `PARALLEL_SPACING_M` north of "09" (candidate 0), the same
    threshold position along the course and the same elevation."""
    from ts_transformer.instructions.airport import AirportGeometry

    ends = [{"ident": "09", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": 90.0},
            {"ident": "09L", "threshold_e_m": 0.0, "threshold_n_m": PARALLEL_SPACING_M, "course_deg": 90.0}]
    return AirportGeometry.from_dict({
        "code": "KXXX", "reference": {"lat": 35.0, "lon": -78.0, "elevation_m": INSTRUCTION_AIRPORT_ELEVATION_M},
        "candidates": [{**end, "elevation_m": 100.0, "length_m": 3000.0} for end in ends], "runway_ends": ends,
    })


def signal_attitudes(_directory, signals):
    """`training_attitude.observed_attitudes` for synthetic flights, which have no arrival manifest to rebuild a series
    from: each one's heading and path angle read off its signals, its bank and attack those of a flight with no airframe
    (null) — the shape the exporters write, not the inversion (tested on its own in `test_training_attitude`)."""
    import numpy as np

    return {flight.dataset_id: {"headingDeg": np.mod(flight.track_deg, 360.0),
                                "pathAngleDeg": np.degrees(np.arctan2(flight.vertical_rate_mps, flight.ground_speed_mps)),
                                "bankRightDeg": None, "attackDeg": None} for flight in signals}


def passed_executor(spec_dir):
    """Mark a test's executor spec at ``spec_dir`` as flown within the bounds by the code on disk: placeholder reference
    files and a passed record for this code — what `replay.open_executor` asks for (`spec.require_conforming_executor`).
    The check itself is tested in `test_executor_conformance.py`."""
    import json

    from ts_transformer.autopilot import spec as executor_spec

    directory = spec_dir / executor_spec.CONFORMANCE_DIRECTORY
    directory.mkdir()
    (directory / "reference.json").write_text("{}", encoding="utf-8")
    (directory / "reference.npz").write_bytes(b"")
    code = executor_spec.executor_source_sha256()
    executor_spec.passed_path(spec_dir, code).write_text(json.dumps({
        "schema": executor_spec.PASSED_SCHEMA, "executor_source_sha256": code,
        "reference_sha256": executor_spec.reference_sha256(directory)}), encoding="utf-8")


def labelled_instruction_artefact(directory, split="train"):
    """A tmp instruction artefact at ``directory`` (created) holding one synthetic ``split`` flight onto
    `instruction_airport`'s runway 09 — a downwind, a base, a final on a 3° descent — labelled by the labeller:
    signals, candidates, spec, sentences. Returns the spec."""
    import numpy as np

    from ts_transformer.instructions.artefact import write_candidates, write_sentences, write_signals, write_spec
    from ts_transformer.instructions.conformance import labeller_code_sha256
    from ts_transformer.instructions.labeller.read import read_flight

    legs = [(60, 0.0, 100.0, 0.0), (15, -6.0, 100.0, 0.0), (20, 0.0, 90.0, 0.0), (15, -6.0, 85.0, 0.0),
            (120, 0.0, 75.0, -75.0 * np.tan(np.radians(3.0)))]
    flight = instruction_flight(*fly_legs(legs, 270.0, 1110.0, -400.0, 0.0), dataset_id="KXXX:a", split=split)
    directory.mkdir(parents=True)
    write_signals(directory, {split: [flight]},
                  {"counts": {split: {"built_usable": 1}}, "test_days": {"flights_not_opened": 0},
                   "sources": [{"airport": "KXXX", "arrival_manifest_sha256": "0" * 64}]}, fixture_days())
    write_candidates(directory, {"KXXX": instruction_airport()})
    spec = instruction_spec()
    write_spec(directory, spec, {"n": 1}, {"labeller_code_sha256": labeller_code_sha256(),
                                           "git": {"head": "test", "dirty": False}})
    write_sentences(directory, split, spec, [read_flight(flight, instruction_airport(), spec)], [0],
                    labeller_code_sha256())
    return spec


def executor_inputs(signals, geometry, row=0, mass_kg=62000.0):
    """`autopilot.flights.FlightInputs` of one A320 flown from ``signals``' 2 s row ``row`` (a synthetic flight has no
    data-plane series to rebuild): its state there, the airframe, the chart at the first candidate's threshold."""
    import math

    import torch

    from aircraft.aero_params import aero_params_for_aircraft
    from flight_scenarios.scenario import aircraft_for_code
    from ts_transformer.autopilot.flights import FlightInputs

    aircraft = aircraft_for_code("A320")
    aero = aero_params_for_aircraft(aircraft)
    lat, lon = geometry.frame.latlon_from_horizontal(signals.e_m[row], signals.n_m[row])
    gamma = math.atan2(signals.vertical_rate_mps[row], signals.ground_speed_mps[row])
    state = [lat, lon, signals.altitude_m[row], signals.ground_speed_mps[row] / math.cos(gamma),
             math.radians(90.0 - signals.track_deg[row]), gamma, mass_kg]
    candidate = geometry.candidates[0]
    tlat, tlon = geometry.frame.latlon_from_horizontal(candidate.threshold_e_m, candidate.threshold_n_m)
    f64 = torch.float64
    return FlightInputs(
        initial_state=torch.tensor([state], dtype=f64),
        aero_params=torch.tensor([[aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall]], dtype=f64),
        frame_params=torch.tensor([[tlat, tlon, candidate.elevation_m, 0.0]], dtype=f64),
        max_thrust_n=torch.tensor([aircraft.engine.max_thrust_total_n], dtype=f64))
