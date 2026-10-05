"""The fixtures of stage C's tests (post-training §8): synthetic flights at chosen UTC times on `support.parallel_airport`,
and an artefact of them with closed-loop sentences, written to a tmp directory."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np

from ts_transformer.data.day_split import parse_utc
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.prior.procedure import Final
from ts_transformer.tests.support import (
    INSTRUCTION_STEP_S, fixture_days, fly_legs, instruction_flight, instruction_spec, parallel_airport,
    prior_closed_loop_sentence,
)

#: The FAF of the synthetic finals, m before the threshold.
FAF_M = 9_000.0


def airport() -> AirportGeometry:
    return parallel_airport()


def finals(geometry: AirportGeometry, faf_m: float = FAF_M) -> tuple[Final, ...]:
    """Every candidate's final with its FAF ``faf_m`` before the threshold (a synthetic procedure)."""
    from flight_scenarios.fas_geometry import fas_course_geometry

    return tuple(Final(geometry, k, faf_m, fas_course_geometry(c.length_m)) for k, c in enumerate(geometry.candidates))


def utc(text: str, seconds: float) -> str:
    return (parse_utc(text) + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def straight_in(key: str, entry_utc: str, *, rows: int = 160, speed_mps: float = 75.0, runway: str = "09",
                end_e: float = -400.0, offset_n: float = 0.0, typecode: str | None = "A320"):
    """`FlightSignals` of a straight-in on ``runway`` (descending at 1.5 m/s), its row 0 at ``entry_utc`` and its last
    row ``end_e`` east of the frame's origin (``offset_n`` north), landing two seconds after its last row."""
    e, n, h, track, speed = fly_legs([(rows - 1, 0.0, speed_mps, -1.5)], 90.0, 750.0, end_e, offset_n)
    flight = instruction_flight(e, n, h, track, speed, dataset_id=key)
    landing = utc(entry_utc, (rows - 1) * INSTRUCTION_STEP_S + 2.0)
    return replace(flight, airport="KXXX", runway=runway, typecode=typecode, entry_time_utc=entry_utc,
                   landing_time_utc=landing)


def train_noon(seconds: float = 0.0, split: str = "train") -> str:
    """A UTC time ``seconds`` after noon of the first `support.fixture_days` day of ``split``."""
    return utc(f"{fixture_days().days[split][0]}T12:00:00Z", seconds)


def scene_artefact(directory, flights_by_split, *, interval_s: float = 4.0, sentences=None):
    """A tmp artefact at ``directory`` (created) of ``flights_by_split`` (split → `FlightSignals`) at
    `support.parallel_airport`, with closed-loop sentences at Δ = ``interval_s`` for the flights whose places
    ``sentences[split]`` lists (every flight when None). Returns the spec."""
    from ts_transformer.instructions.artefact import (
        closed_loop_path, write_candidates, write_closed_loop, write_sentences, write_signals, write_spec,
    )
    from ts_transformer.instructions.labeller.read import read_flight
    from ts_transformer.instructions.words import Words

    spec = instruction_spec()
    words = Words(spec)
    directory.mkdir(parents=True)
    write_signals(directory, flights_by_split, {"counts": {}, "test_days": {"flights_not_opened": 0}, "sources": []},
                  fixture_days())
    geometry = airport()
    write_candidates(directory, {"KXXX": geometry})
    write_spec(directory, spec, {"n": 1}, {"git": {"head": "test", "dirty": False}})
    (directory / "closed_loop").mkdir()
    for split, flights in flights_by_split.items():
        write_sentences(directory, split, spec, [read_flight(f, geometry, spec) for f in flights], range(len(flights)))
        chosen = range(len(flights)) if sentences is None else sentences[split]
        built = {k: prior_closed_loop_sentence(flights[k], words, interval_s=interval_s) for k in chosen}
        write_closed_loop(closed_loop_path(directory, split, interval_s), spec, executor_params_sha256="test",
                          row_interval_s=interval_s, start_row=next(iter(built.values())).rows.start, sentences=built)
    return spec


def categories(typecode: str) -> str:
    """A CWT category for every synthetic type (the tables are tested in `test_runway_schedule`)."""
    return {"A320": "F", "B744": "B", "C172": "I"}[typecode]


def at_step(rng: np.random.Generator, count: int, runway: int = 0):
    """``count`` random aircraft at one step around `airport`'s finals (`scene.AircraftAt`)."""
    from ts_transformer.post.scene import AircraftAt

    at = np.column_stack((rng.uniform(-20_000, 0, count), rng.uniform(-3_000, 3_000, count), rng.uniform(300, 2_000, count)))
    before = at - np.column_stack((rng.uniform(100, 160, count), rng.uniform(-20, 20, count), rng.uniform(-6, 2, count)))
    return AircraftAt.of([(f"X{k}", tuple(at[k]), tuple(before[k]), True, runway, "F", False, False)
                          for k in range(count)])
