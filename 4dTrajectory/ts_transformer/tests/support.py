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


def instruction_airport():
    """A synthetic airport for the instruction labeller: one candidate runway "09", threshold at
    the airport frame's origin, course 090° true, elevation 100 m MSL."""
    from ts_transformer.instructions.airport import AirportGeometry

    return AirportGeometry.from_dict({
        "code": "KXXX", "reference": {"lat": 35.0, "lon": -78.0, "elevation_m": 100.0},
        "candidates": [{"ident": "09", "threshold_e_m": 0.0, "threshold_n_m": 0.0, "course_deg": 90.0,
                        "elevation_m": 100.0, "length_m": 3000.0}],
    })


def instruction_spec(**changes):
    """A vocabulary spec at the development set's measured values, with ``changes`` applied."""
    from ts_transformer.instructions import measure
    from ts_transformer.instructions.spec import VocabularySpec

    measured = measure.MeasuredValues(
        turn_bank_min_deg=6.0, turn_bank_max_deg=32.0,
        corridor_half_width_m=20.0, corridor_widening_deg=0.45, corridor_course_tolerance_deg=2.0,
        descent_angle_edges_deg=(-0.5, 1.4, 2.6, 3.7, 10.0), descent_angle_centres_deg=(0.8, 2.1, 3.0, 4.4),
        climb_angle_centre_deg=1.3, speed_accel_max_mps2=2.5,
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


def instruction_flight(e, n, altitude, track, speed, dataset_id="KXXX:test"):
    """`FlightSignals` for a synthetic flight onto `instruction_airport`'s runway 09."""
    import numpy as np

    from ts_transformer.instructions.signals import FlightSignals

    t = np.arange(len(e)) * INSTRUCTION_STEP_S
    return FlightSignals(dataset_id, "KXXX", "09", "A320", t, e, n, altitude, track, speed,
                         np.gradient(altitude, INSTRUCTION_STEP_S))
