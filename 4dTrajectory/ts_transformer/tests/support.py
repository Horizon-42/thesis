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


#: The test airports' runways' published TCH, glidepath and decision altitude (the fleet's DAs above the threshold run
#: 61–129 m), each candidate's vertical path in `candidates.json` (D61).
TEST_TCH_M, TEST_GLIDEPATH_DEG, TEST_DA_M = 15.0, 3.0, 60.0
TEST_VERTICAL_PATH = {"crossing_height_m": TEST_TCH_M, "glidepath_deg": TEST_GLIDEPATH_DEG, "decision_height_m": TEST_DA_M}


def with_vertical_path(geometry, path):
    """``geometry`` with every candidate's published vertical path ``path`` (an `instructions.airport.VerticalPath`)."""
    from dataclasses import replace

    return replace(geometry, candidates=tuple(replace(c, vertical_path=path) for c in geometry.candidates))


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
                        "elevation_m": 100.0, "length_m": 3000.0, "vertical_path": TEST_VERTICAL_PATH}],
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
    # H_final = H unless a test changes it (D66: at H_final = H the closed-loop reading is the one before it)
    data = measure.build_spec(measured, closed_loop_final_vertical_m=measure.SUGGESTED["closed_loop_vertical_m"]).to_dict()
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
        "candidates": [{**end, "elevation_m": 100.0, "length_m": 3000.0, "vertical_path": TEST_VERTICAL_PATH}
                       for end in ends], "runway_ends": ends,
    })


def signal_attitudes(_directory, signals):
    """`training_attitude.observed_attitudes` for synthetic flights, which have no arrival manifest to rebuild a series
    from: each one's heading and path angle read off its signals, its bank and attack those of a flight with no airframe
    (null) — the shape the exporters write, not the inversion (tested on its own in `test_training_attitude`)."""
    import numpy as np

    return {flight.dataset_id: {"headingDeg": np.mod(flight.track_deg, 360.0),
                                "pathAngleDeg": np.degrees(np.arctan2(flight.vertical_rate_mps, flight.ground_speed_mps)),
                                "bankRightDeg": None, "attackDeg": None} for flight in signals}


def stand_in_checks(monkeypatch):
    """Stand in for the labeller's and the executor's checks that `replay.open_executor` runs (D73), for a test whose
    synthetic artefact and spec hold no reference: both pass with nothing flown. The checks themselves are tested in
    `test_instruction_conformance.py` and `test_executor_conformance.py`, and through a real reference in
    `test_start.py`."""
    from ts_transformer.autopilot import conformance, replay
    from ts_transformer.instructions.conformance import Checked as LabellerChecked

    monkeypatch.setattr(replay, "CHECKED", {})
    monkeypatch.setattr(replay, "require_conforming_labeller", lambda directory: LabellerChecked(flights=0))
    monkeypatch.setattr(conformance, "require_conforming_executor", lambda executor_dir, instructions, **_: (
        conformance.Checked({mode: conformance.Difference(expected=0) for mode in conformance.MODES})))


def labelled_instruction_artefact(directory, split="train"):
    """A tmp instruction artefact at ``directory`` (created) holding one synthetic ``split`` flight onto
    `instruction_airport`'s runway 09 — a downwind, a base, a final on a 3° descent — labelled by the labeller:
    signals, candidates, spec, sentences. Returns the spec."""
    import numpy as np

    from ts_transformer.instructions.artefact import write_candidates, write_sentences, write_signals, write_spec
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
    write_spec(directory, spec, {"n": 1}, {"git": {"head": "test", "dirty": False}})
    write_sentences(directory, split, spec, [read_flight(flight, instruction_airport(), spec)], [0])
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


def prior_sentence(rng, words, *, candidates=3, rows=30, first_step=8, airport="KXXX", split="train",
                   variant="full", flight_key=None, interval_s=2.0):
    """A synthetic `prior.batch.SentenceRows` of ``words``' spec: random inputs, and words with a pattern a model can
    learn (row 0 without motion, D60) — at the first predicted step a word in every column (the runway: a candidate drawn at random); after it, a
    heading word where the first own-state input is positive (its class from the second), "unchanged" elsewhere, and
    now and then a word of another column. The words in force follow what was said, from the row after the first
    predicted step."""
    import numpy as np

    from ts_transformer.instructions.words import ALTITUDE, ANGLE, HEADING, RUNWAY, SPEED, UNCHANGED
    from ts_transformer.prior.batch import (
        CANDIDATE_FEATURES, CANDIDATE_MOTION_FEATURES, IN_FORCE_WORDS, OWN_FEATURES, OWN_MOTION_FEATURES, SentenceRows,
        variant_features,
    )

    counts = words.class_counts()
    width = len(variant_features(variant))
    own = rng.normal(size=(rows, len(OWN_FEATURES))).astype(np.float32)
    own[:, OWN_FEATURES.index("no_motion")] = 0.0
    own[0, [OWN_FEATURES.index(name) for name in OWN_MOTION_FEATURES]] = 0.0
    own[0, OWN_FEATURES.index("no_motion")] = 1.0
    targets = np.full((rows, 5), UNCHANGED, dtype=np.int64)
    targets[first_step] = [rng.integers(candidates), rng.integers(counts["heading"]),
                           rng.integers(counts["altitude"] - 1), rng.integers(1, counts["angle"]),
                           rng.integers(counts["speed"])]
    for r in range(first_step + 1, rows):
        if own[r, 0] > 0.0:
            targets[r, HEADING] = int(abs(own[r, 1]) * 4) % counts["heading"]
        if rng.random() < 0.1:
            column = int(rng.choice([ALTITUDE, ANGLE, SPEED]))
            targets[r, column] = rng.integers(counts[("altitude", "angle", "speed")[column - ALTITUDE]] - 1)
    runway_in_force = np.full(rows, -1, dtype=np.int64)
    heading_in_force = np.zeros((rows, 2), dtype=np.float32)
    words_in_force = np.full((rows, len(IN_FORCE_WORDS)), -1, dtype=np.int64)
    since = np.zeros((rows, 5), dtype=np.float32)
    said_at = np.full(5, np.nan)
    for r in range(first_step + 1, rows):
        before = targets[r - 1]
        runway_in_force[r] = before[RUNWAY] if before[RUNWAY] != UNCHANGED else runway_in_force[r - 1]
        heading_in_force[r] = heading_in_force[r - 1]
        if before[HEADING] != UNCHANGED:
            angle = np.radians(words.heading_relative_deg(int(before[HEADING])))
            heading_in_force[r] = (np.sin(angle), np.cos(angle))
        words_in_force[r] = np.where(before[ALTITUDE:] != UNCHANGED, before[ALTITUDE:], words_in_force[r - 1])
        said_at = np.where(before != UNCHANGED, (r - 1) * interval_s, said_at)
        since[r] = np.log1p((r * interval_s - said_at) / 2.0) / 5.0
    candidate_vectors = rng.normal(size=(rows, candidates, width)).astype(np.float32)
    candidate_vectors[0][:, [CANDIDATE_FEATURES.index(name) for name in CANDIDATE_MOTION_FEATURES]] = 0.0
    return SentenceRows(
        flight_key=flight_key or f"{airport}:{split}:{int(rng.integers(1 << 30))}", airport=airport, split=split,
        first_step=first_step, time_s=np.arange(rows, dtype=np.float32) * interval_s, own=own,
        candidates=candidate_vectors,
        runway_in_force=runway_in_force, go_around=np.zeros(rows, dtype=bool), heading_in_force=heading_in_force,
        words_in_force=words_in_force, since=since, targets=targets)


#: `prior_artefact`'s closed-loop sentences: the words said, rows from the first predicted step.
PRIOR_ARTEFACT_ROWS = 30


def prior_closed_loop_sentence(signals, words, *, interval_s, first_row=0, go_around=False, outcome="landed"):
    """A closed-loop sentence of ``signals`` (a straight-in onto `parallel_airport`'s "09", descending) at Δ =
    ``interval_s``, built by hand (`instructions.artefact.ClosedLoopSentence`): its states are the observed ones from its
    2 s row ``first_row`` (a stand-in for flown ones); its words a grammatical sentence — at the first predicted step
    "09", the course, "no level-off" with the descent class of 3°, the speed; a heading word and a speed word later;
    with ``go_around`` a go-around (a level above and the climb) and "09" again to end it; its stored outcome (D74)
    ``outcome``, by the judge's name (not judged: the states are the observed ones)."""
    import numpy as np

    from ts_transformer.instructions.artefact import ClosedLoopSentence
    from ts_transformer.instructions.labeller.interval import OBSERVATION_S, interval_rows, on_interval_rows
    from ts_transformer.instructions.words import RUNWAY_GO_AROUND, UNCHANGED

    every = interval_rows(interval_s, words.spec.step_s)
    start, said = int(round(OBSERVATION_S / interval_s)), PRIOR_ARTEFACT_ROWS
    count = (start + said - 1) * every + 1
    rows = slice(first_row, first_row + count)
    states = np.stack([signals.e_m[rows], signals.n_m[rows], signals.altitude_m[rows], signals.track_deg[rows],
                       signals.ground_speed_mps[rows], signals.vertical_rate_mps[rows]], axis=1)
    grid = np.full((said, 5), UNCHANGED, dtype=np.int16)
    grid[0] = [0, words.heading_index(0.0), words.altitude_no_level_off, words.angle_index(3.0), words.speed_index(75.0)]
    grid[5, 1] = words.heading_index(5.0)
    grid[9, 4] = words.speed_index(70.0)
    if go_around:
        height = signals.altitude_m[first_row + (start + 12) * every] - INSTRUCTION_AIRPORT_ELEVATION_M
        grid[12, [0, 2, 3]] = [RUNWAY_GO_AROUND, words.altitude_index(height + 300.0), words.angle_climb]
        grid[20, 0] = 0
    return ClosedLoopSentence(
        first_row=first_row, start=start, grid=grid, correction=np.zeros((said, 5), dtype=bool), states=states,
        on_interval=on_interval_rows(count, every), lateral_m=np.zeros(said), vertical_m=np.zeros(said),
        uncorrectable=np.zeros((said, 2), dtype=bool), observed_row=np.arange(said), matched_row=np.arange(said) * 1.0,
        timed_out=False, outcome=outcome)


def prior_artefact(directory, interval_s=2.0, airports=("KXXX",), outcomes=("landed", "landed")):
    """A tmp instruction artefact at ``directory`` (created) as the prior reads it (vocabulary §6 items 3, 4): at each of
    ``airports`` (`parallel_airport` under that code) two straight-in flights onto "09" on each development split (the
    second with a go-around), the spec, the candidates with their vertical paths, each split's sentence file (the
    labeller's reading, for the strata) and its closed-loop file at Δ = ``interval_s`` (`prior_closed_loop_sentence`;
    the first and second flight's stored outcomes ``outcomes``). Returns ``(words, roster records by airport)``: the tracks
    roster's records of the flights' landings (`prior.landings`)."""
    from dataclasses import replace

    from ts_transformer.instructions.airport import AirportGeometry
    from ts_transformer.instructions.artefact import (
        SPLITS, closed_loop_path, write_candidates, write_closed_loop, write_sentences, write_signals, write_spec,
    )
    from ts_transformer.instructions.labeller.read import read_flight
    from ts_transformer.instructions.words import Words

    spec = instruction_spec()
    words = Words(spec)
    flights = {split: [] for split in SPLITS}
    records = {code: [] for code in airports}
    for code in airports:
        for split in SPLITS:
            for i in range(2):
                legs = [(160, 0.0, 75.0 - 2.0 * i, -1.5)]
                flight = instruction_flight(*fly_legs(legs, 90.0, 750.0 + 30.0 * i, -400.0, 0.0),
                                            dataset_id=f"{code}:F{split}{i}", split=split)
                flights[split].append(replace(flight, airport=code))
                records[code].append({"flight_key": f"F{split}{i}", "outcome": "assigned", "runway": "09",
                                      "landing_time_utc": flight.landing_time_utc})
    directory.mkdir(parents=True)
    write_signals(directory, flights, {"counts": {}, "test_days": {"flights_not_opened": 0}, "sources": []},
                  fixture_days())
    geometries = {code: AirportGeometry.from_dict({**parallel_airport().to_dict(), "code": code}) for code in airports}
    write_candidates(directory, geometries)
    write_spec(directory, spec, {"n": 1}, {"git": {"head": "test", "dirty": False}})
    for split in SPLITS:
        write_sentences(directory, split, spec, [read_flight(f, geometries[f.airport], spec) for f in flights[split]],
                        range(len(flights[split])))
    (directory / "closed_loop").mkdir()
    for split in SPLITS:
        sentences = {k: prior_closed_loop_sentence(flight, words, interval_s=interval_s, go_around=k % 2 == 1,
                                                   outcome=outcomes[k % 2])
                     for k, flight in enumerate(flights[split])}
        write_closed_loop(closed_loop_path(directory, split, interval_s), spec, executor_params_sha256="test",
                          row_interval_s=interval_s, start_row=sentences[0].start, sentences=sentences)
    return words, records


def closed_loop_flight(interval_s: float = 2.0):
    """A synthetic flight on its closed-loop sentence at ``interval_s``, from its first predicted step as
    `closed_loop.replay_batch` sets it up — the core of `experiments.training_flights.closed_loop_batch`, whose drawing
    (`replay.batch_of`) needs a data-plane flight: the downwind, base and final of `test_closed_loop`, read in closed
    loop (`closed_loop.read`). Returns a namespace: ``batch``, the replay batch of
    one flight from its first predicted step — whose inputs are fixed, a synthetic flight having no data-plane series —
    ``inputs``, its executor inputs there, ``sentence``, its stored `ClosedLoopSentence`, ``params``, ``words``, and the
    observed flight as labelled: ``signals`` (from its first row), ``reading`` and ``geometry``."""
    import dataclasses
    from types import SimpleNamespace

    import torch

    from ts_transformer.autopilot import closed_loop, replay
    from ts_transformer.tests import test_closed_loop

    batch, inputs, words = test_closed_loop._batch(interval_s)
    params = test_closed_loop._params()
    (sentence,) = closed_loop.read(batch, inputs, params, words, device=torch.device("cpu"))
    replayed, missing = closed_loop.replay_batch(batch, {0: sentence}, words)
    assert not missing

    @dataclasses.dataclass
    class FixedInputs(replay.Batch):
        fixed: object = None

        def inputs(self, device):
            return self.fixed

    fixed = FixedInputs(**{f.name: getattr(replayed, f.name) for f in dataclasses.fields(replay.Batch)}, fixed=inputs)
    return SimpleNamespace(batch=fixed, inputs=inputs, sentence=sentence, params=params, words=words,
                           signals=batch.signals[0], reading=batch.readings[0], geometry=batch.geometries[0])
