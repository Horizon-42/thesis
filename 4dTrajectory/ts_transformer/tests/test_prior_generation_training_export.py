"""The prior's own sentences over a Training set (`experiments/prior_generation_training_export.py`): one synthetic
flight spoken to by a small untrained prior and flown by the executor (`test_prior_free_generation._speak`), its sample
as the frontend reads it; the formal readout bound to its prior, executor spec, artefact and draw, or refused by name.
The runner's loop over a set is `prior_free_generation`'s own (`speak_and_fly`, `flight_rows`), tested there."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flight_scenarios.datum import geoid_undulation_m
from ts_transformer.autopilot import replay
from ts_transformer.autopilot.judge import outcome_of
from ts_transformer.experiments import prior_generation_training_export as export
from ts_transformer.experiments.prior_free_generation import GENERATION_SCHEMA, flight_rows
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import UNCHANGED, Words
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.test_prior_free_generation import _speak


def _sample(seed: int):
    """One flight spoken to and flown (`_speak`), its `flight_rows` row, and the rest the payload is built from."""
    flown, said, geometry, one, inputs, forbidden, signals, _ = _speak(seed)
    words = Words(one)
    batch = replay.Batch(signals=[signals], series=[None], readings=[read_flight(signals, geometry, one, words)],
                         geometries=[geometry], crossing_heights=[(15.0,)], approach_ias_mps=[70.0],
                         groups=[replay.OWN], drawn={})
    (row,) = flight_rows(batch, flown, [said[0]], words, "prior", [0], forbidden)
    return flown, said[0], geometry, words, row, inputs


@pytest.mark.parametrize("seed", [3, 11])
def test_a_sample_is_the_words_said_on_the_flights_own_steps_and_the_track_on_its_own_clock(seed):
    flown, grid, geometry, words, row, inputs = _sample(seed)
    step_s, start_s = words.spec.step_s, N_LOOK * words.spec.step_s
    payload = export.sample_payload(flown, 0, row, grid, geometry, words)
    assert payload["outcome"] == row["outcome"] and payload["sample"] == 0
    # the words said up to the flight's end, each on the flight's own step: the first predicted step says every column
    said = grid[: row["steps_said"]]
    assert payload["rows"] == N_LOOK + len(said)
    cells = [(N_LOOK + int(s), int(c), int(said[s, c])) for s, c in zip(*np.nonzero(said != UNCHANGED))]
    assert [(e["row"], e["column"], e["value"]) for e in payload["events"]] == cells
    assert sorted(e["column"] for e in payload["events"] if e["row"] == N_LOOK) == list(range(6))
    assert min(e["row"] for e in payload["events"]) == N_LOOK
    # the track: from the observed state at row N_LOOK, every step, on the flight's clock, to the outcome's row
    track = payload["track"]
    n = len(track["tS"])
    assert all(len(values) == n for values in track.values())
    assert track["tS"][0] == start_s and np.all(np.diff(track["tS"]) > 0)
    assert np.allclose(np.diff(track["tS"])[:-1], step_s)
    outcome = outcome_of(flown, 0, geometry, row["last_runway"], words.spec)
    assert track["tS"][-1] == pytest.approx(start_s + outcome.end_row * flown.cycle_s)
    assert payload["endS"] == pytest.approx(start_s + row["end_s"])
    state = inputs.initial_state[0].numpy()
    assert (track["lat"][0], track["lon"][0]) == pytest.approx((float(state[0]), float(state[1])), abs=1e-7)
    # the height Cesium draws in is the ellipsoid's: h = H + N, once
    undulation = np.asarray(geoid_undulation_m(track["lat"], track["lon"]))
    assert np.allclose(np.asarray(track["altitudeHaeM"]) - np.asarray(track["altitudeM"]), undulation, atol=0.02)
    if payload["crossing"] is not None:
        assert payload["crossing"]["atS"] == pytest.approx(start_s + outcome.crossing["at_row"] * flown.cycle_s, abs=1e-3)
    assert set(payload["forbiddenMass"]) == {"runway", "approach", "angle"}


def test_a_dynamics_failure_s_track_stops_before_the_failed_state():
    flown, _, geometry, words, _, _ = _sample(3)
    start_s = N_LOOK * words.spec.step_s
    landed = export.track_payload(flown, 0, 5, "landed", geometry, words.spec.step_s, start_s)
    failed = export.track_payload(flown, 0, 5, "dynamics_failure", geometry, words.spec.step_s, start_s)
    assert landed["tS"] == [start_s, start_s + 2.0, start_s + 4.0, start_s + 5.0]
    assert failed["tS"] == [start_s, start_s + 2.0, start_s + 4.0]


def test_a_sentence_whose_first_predicted_step_leaves_a_column_unsaid_is_refused():
    flown, grid, geometry, words, row, _ = _sample(3)
    broken = grid.copy()
    broken[0, 2] = UNCHANGED
    with pytest.raises(ValueError, match="does not say every column"):
        export.sample_payload(flown, 0, row, broken, geometry, words)


# ---- the formal val readout, bound to this prior, executor spec, artefact and draw
PRIOR = "/repo/4dTrajectory/outputs/POOLED/prior/v3_step1_20260924/full_s1337"
INSTRUCTIONS = "/repo/4dTrajectory/outputs/POOLED/instruction_language/v4_20260924"


def _cell(flights, landed):
    return {"flights": flights, "outcomes": {"landed": landed, "timeout": 1.0 - landed}}


def _generation(**changes):
    # every airport pooled, per approach kind, and per airport: KXXX drew only vectored flights
    part = {"all": _cell(80, 0.9), "straight-in": _cell(50, 0.95), "vectored": _cell(30, 0.8), "KXXX": _cell(40, 0.85),
            "KXXX vectored": _cell(40, 0.85), "KYYY": _cell(40, 0.95), "KYYY straight-in": _cell(40, 0.95)}
    generation = {
        "schema": GENERATION_SCHEMA, "written_utc": "2026-09-25T12:00:00+00:00", "seed": 1337,
        # the readout ran from a worktree: its outputs are the same tree under another checkout
        "prior": {"directory": "/repo/.claude/worktrees/prior-v3/4dTrajectory/outputs/POOLED/prior/v3_step1_20260924/"
                               "full_s1337", "variant": "full"},
        "executor": {"directory": "/somewhere/executor/v7", "sha256": "e" * 64},
        "instructions": "/repo/.claude/worktrees/prior-v3/4dTrajectory/outputs/POOLED/instruction_language/v4_20260924",
        "split": "val", "n_look": N_LOOK, "samples": 4, "temperature": 1.0,
        "drawn": {"flights": 20, "per_airport": 20},
        "readout": {"prior": part, "labelled": {"all": _cell(20, 1.0), "KXXX": _cell(10, 1.0), "KYYY": _cell(10, 1.0)}},
    }
    generation.update(changes)
    return generation


def _block(generation, **changes):
    arguments = {"prior_dir": PRIOR, "executor_sha256": "e" * 64, "instructions": INSTRUCTIONS, "samples": 4,
                 "temperature": 1.0, "airport": "KXXX", **changes}
    return export.readout_block(generation, arguments["prior_dir"], arguments["executor_sha256"],
                                arguments["instructions"], arguments["samples"], arguments["temperature"],
                                arguments["airport"])


def test_the_readout_is_this_prior_s_val_free_generation_at_the_airport_and_pooled():
    block = _block(_generation())
    assert block["prior"]["all"] == {"all": {"flights": 80, "landed": 0.9}, "straight-in": {"flights": 50, "landed": 0.95},
                                     "vectored": {"flights": 30, "landed": 0.8}}
    # the airport's own cells; a kind it drew no flight of is written as null, not left out
    assert block["prior"]["here"] == {"all": {"flights": 40, "landed": 0.85}, "straight-in": None,
                                      "vectored": {"flights": 40, "landed": 0.85}}
    assert block["labelled"]["here"] == {"all": {"flights": 10, "landed": 1.0}, "straight-in": None, "vectored": None}
    assert block["drawn"] == {"flights": 20, "perAirport": 20} and block["split"] == "val"
    assert _block(_generation(drawn={"flights": 900, "per_airport": export.EVERY_FLIGHT}))["drawn"]["perAirport"] == 0
    with pytest.raises(SystemExit, match="counted no prior flight at KZZZ"):
        _block(_generation(), airport="KZZZ")
    with pytest.raises(SystemExit, match="names 'some' flights an airport"):
        _block(_generation(drawn={"flights": 20, "per_airport": "some"}))


def test_the_every_flight_phrase_is_the_draw_s_own():
    source = Path(replay.__file__).read_text(encoding="utf-8")
    assert f'per_airport or "{export.EVERY_FLIGHT}"' in source


@pytest.mark.parametrize("change, name", [
    (dict(generation=dict(prior={"directory": PRIOR.replace("full_s1337", "full_s2024")})), "prior"),
    (dict(executor_sha256="f" * 64), "executor"),
    (dict(generation=dict(split="select")), "split"),
    (dict(samples=1), "samples"),
    (dict(temperature=0.7), "temperature"),
])
def test_a_readout_of_another_prior_spec_split_or_draw_is_refused_by_name(change, name):
    generation = _generation(**change.pop("generation", {}))
    with pytest.raises(SystemExit, match=rf"\b{name} "):
        _block(generation, **change)


def test_a_readout_of_another_schema_is_refused_by_name_before_it_is_read():
    with pytest.raises(SystemExit, match="is a ts-prior-free-generation-v0 file"):
        _block({"schema": "ts-prior-free-generation-v0"})


def test_a_path_outside_the_outputs_tree_names_itself():
    with pytest.raises(SystemExit, match="is not under 4dTrajectory/outputs/"):
        export.outputs_path("/elsewhere/prior/full_s1337")
    assert export.outputs_path(PRIOR) == "4dTrajectory/outputs/POOLED/prior/v3_step1_20260924/full_s1337"


# ---- the runner end to end, on a synthetic artefact, an untrained prior and an executor spec (every write in tmp_path)
def test_the_export_flies_the_set_s_own_dynamics_flights_and_lists_the_rest(tmp_path, monkeypatch):
    """The set's two flights: the vectored one on its own dynamics, flown ``--samples`` times; the straight-in one with no
    identified type, listed with its group and no sample. The flights' series are stand-ins (the synthetic artefact has no
    harvest to rebuild them from): the A320's physics from each flight's own signals at the first predicted row."""
    from types import SimpleNamespace

    import torch

    from ts_transformer.autopilot import spec as executor_spec
    from ts_transformer.autopilot.flights import FlightInputs
    from ts_transformer.experiments import instruction_training_export as sets
    from ts_transformer.experiments import prior_train
    from ts_transformer.instructions.artefact import labeller_source_sha256
    from ts_transformer.tests.test_autopilot import _params, _physics
    from ts_transformer.tests.test_instruction_training_export import SET_ID, _artefact, _run, _straight, _vectored
    from ts_transformer.tests.test_training_overlays import _prior_dir
    from ts_transformer.tests.support import instruction_airport, landing_on

    vectored, straight = _vectored("KXXX:V1_09_abc123_20260101T000000Z"), _straight("KXXX:S1_09_abc124_20260101T000100Z")
    one = _artefact(tmp_path / "artefact", [vectored, straight])
    assert _run(tmp_path) == 0                                           # the set, as the Training export writes it
    roster = tmp_path / "tracks.json"
    roster.write_text(json.dumps({"records": [{"outcome": "assigned", "runway": "09", "flight_key": "own",
                                               "landing_time_utc": landing_on("val")}]}), encoding="utf-8")
    monkeypatch.setattr(prior_train, "tracks_manifest_path", lambda code: roster)
    _prior_dir(tmp_path / "prior", tmp_path / "artefact", roster)
    executor_spec.write_spec(tmp_path / "executor", _params(), one.sha256, {}, {
        "executor_source_sha256": executor_spec.executor_source_sha256(), "labeller_source_sha256": labeller_source_sha256(),
        "git": {"head": "test", "dirty": False}})

    unflown = {straight.dataset_id}

    def series(signals):
        typecode = None if signals.dataset_id in unflown else "A320"
        return SimpleNamespace(dataset_id=signals.dataset_id, signals=signals, scenario=SimpleNamespace(
            source={"resolved_typecode": typecode, "dynamics_typecode": typecode}, has_dynamics=True,
            initial=SimpleNamespace(m=62000.0)))

    def inputs(items, *, device, anchor):
        rows = [_physics(replace(item.signals, **{name: getattr(item.signals, name)[anchor:] for name in (
            "time_s", "e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps", "vertical_rate_mps")}),
            instruction_airport())[0] for item in items]
        return FlightInputs(*(torch.cat([getattr(row, name) for row in rows]) for name in (
            "initial_state", "aero_params", "frame_params", "max_thrust_n")))

    signals_by_id = {flight.dataset_id: flight for flight in (vectored, straight)}
    monkeypatch.setattr(export, "rebuild_series", lambda directory, flights: [series(signals_by_id[f.dataset_id]) for f in flights])
    monkeypatch.setattr(export, "flight_inputs", inputs)
    monkeypatch.setattr(export, "published_crossing_heights", lambda geometry: (15.0,) * len(geometry.candidates))
    args = ["--prior", str(tmp_path / "prior"), "--label", "a test prior", "--instructions", str(tmp_path / "artefact"),
            "--executor", str(tmp_path / "executor"), "--airports-root", str(tmp_path / "airports"), "--set", SET_ID,
            "--airport", "KXXX", "--samples", "3"]
    assert export.main(args) == 0
    training = tmp_path / "airports" / "KXXX" / "training"
    overlay_id = f"generation_{tmp_path.name}_prior"                   # the default: the prior's parent and name
    payload = json.loads((training / overlay_id / export.PAYLOAD_FILE).read_text(encoding="utf-8"))
    sample = json.loads((training / SET_ID / "sample.json").read_text(encoding="utf-8"))
    assert payload["schema"] == export.SCHEMA and payload["base"]["setId"] == SET_ID and payload["readout"] is None
    assert payload["model"]["label"] == "a test prior" and payload["model"]["fineTuning"] is None
    assert payload["generation"]["firstPredictedRow"] == N_LOOK and payload["generation"]["samples"] == 3
    # the set's flights in its order: the own-dynamics one flown three times, the other listed with its group
    assert [f["flightKey"] for f in payload["flights"]] == [f["flightKey"] for f in sample["flights"]]
    flown = {f["datasetId"]: f for f in payload["flights"]}
    assert flown[vectored.dataset_id]["flown"] and [s["sample"] for s in flown[vectored.dataset_id]["samples"]] == [0, 1, 2]
    assert flown[straight.dataset_id] == {"flightKey": flown[straight.dataset_id]["flightKey"], "datasetId": straight.dataset_id,
                                          "group": "no identified type", "flown": False, "samples": []}
    for item in flown[vectored.dataset_id]["samples"]:
        assert item["events"][0]["row"] == N_LOOK and item["track"]["tS"][0] == N_LOOK * one.step_s
    manifest = json.loads((training / sets.OVERLAYS_FILE).read_text(encoding="utf-8"))
    assert [(o["id"], o["kind"], o["base"]) for o in manifest["overlays"]] == [(overlay_id, sets.KIND_GENERATION, SET_ID)]
    # the same seed, the same sentences: a second export under another id is identical flight for flight
    assert export.main([*args, "--overlay-id", "again"]) == 0
    again = json.loads((training / "again" / export.PAYLOAD_FILE).read_text(encoding="utf-8"))
    assert again["flights"] == payload["flights"]
    with pytest.raises(SystemExit):                                     # never overwritten
        export.main(args)
    # both flights on their own dynamics: each flight's samples are its own, flown from its own state at N_LOOK
    unflown.clear()
    assert export.main([*args, "--overlay-id", "both"]) == 0
    both = json.loads((training / "both" / export.PAYLOAD_FILE).read_text(encoding="utf-8"))
    for item in both["flights"]:
        observed = signals_by_id[item["datasetId"]]
        assert item["flown"] and len(item["samples"]) == 3
        start = inputs([series(observed)], device=None, anchor=N_LOOK).initial_state[0].numpy()
        assert {(s["track"]["lat"][0], s["track"]["lon"][0]) for s in item["samples"]} == {
            (round(float(start[0]), 7), round(float(start[1]), 7))}
