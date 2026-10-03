"""The prior's own sentences over a Training set (`experiments/prior_generation_training_export.py`): one synthetic
flight spoken to by a small untrained prior and flown by the executor (`test_prior_free_generation._speak`), its sample
as the frontend reads it; the formal readout bound to its prior, executor spec, artefact and draw, or refused by name.
The runner's loop over a set is `prior_free_generation`'s own (`speak_and_fly`, `flight_rows`), tested there."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.runway_data import VerticalPath
from ts_transformer.autopilot.judge import outcome_of
from ts_transformer.experiments import prior_generation_training_export as export
from ts_transformer.experiments.prior_free_generation import GENERATION_SCHEMA, flight_rows
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import UNCHANGED, Words
from ts_transformer.prior.augment import Augmentation, augment_signals, augment_state
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.test_instruction_training_export import HAE_MINUS_MSL_M, _offsets
from ts_transformer.tests.test_prior_free_generation import _speak
from ts_transformer.tests.test_training_attitude import AERO_ROW


def _sample(seed: int):
    """One flight spoken to and flown (`_speak`), its `flight_rows` row, and the rest the payload is built from."""
    flown, said, geometry, one, inputs, forbidden, signals, _ = _speak(seed)
    words = Words(one)
    batch = replay.Batch(signals=[signals], series=[None], readings=[read_flight(signals, geometry, one, words)],
                         geometries=[geometry], vertical_paths=[(VerticalPath(15.0, 3.0),)], approach_ias_mps=[70.0],
                         groups=[replay.OWN], drawn={})
    (row,) = flight_rows(batch, flown, [said[0]], words, "prior", [0], forbidden)
    return flown, said[0], geometry, words, row, inputs


@pytest.mark.parametrize("seed", [3, 11])
def test_a_sample_is_the_words_said_on_the_flights_own_steps_and_the_track_on_its_own_clock(seed):
    flown, grid, geometry, words, row, inputs = _sample(seed)
    step_s, start_s = words.spec.step_s, N_LOOK * words.spec.step_s
    payload = export.sample_payload(flown, 0, row, grid, geometry, words, HAE_MINUS_MSL_M, inputs.aero_params[0].numpy())
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
    assert all(len(values) == n for name, values in track.items() if name != "attitude")
    # the attitude it is drawn in, at each point: the executor's own bank there (the cycle starting at the point; at the
    # track's end, the cycle ending there)
    assert all(len(values) == n for values in track["attitude"].values())
    rows = np.round((np.asarray(track["tS"]) - start_s) / flown.cycle_s).astype(int)
    commands = flown.commands[0].numpy()
    assert track["attitude"]["bankRightDeg"] == pytest.approx(
        -np.degrees(commands[np.minimum(rows, rows[-1] - 1), 1]), abs=0.006)
    assert track["tS"][0] == start_s and np.all(np.diff(track["tS"]) > 0)
    assert np.allclose(np.diff(track["tS"])[:-1], step_s)
    outcome = outcome_of(flown, 0, geometry, row["last_runway"], words.spec)
    assert track["tS"][-1] == pytest.approx(start_s + outcome.end_row * flown.cycle_s)
    assert payload["endS"] == pytest.approx(start_s + row["end_s"])
    state = inputs.initial_state[0].numpy()
    assert (track["lat"][0], track["lon"][0]) == pytest.approx((float(state[0]), float(state[1])), abs=1e-7)
    # the height Cesium draws in: the flight's runway's HAE − MSL offset added, once (each side rounded to 0.01 m)
    assert np.allclose(np.asarray(track["altitudeHaeM"]) - np.asarray(track["altitudeM"]), HAE_MINUS_MSL_M, atol=0.011)
    if payload["crossing"] is not None:
        assert payload["crossing"]["atS"] == pytest.approx(start_s + outcome.crossing["at_row"] * flown.cycle_s, abs=1e-3)
    assert set(payload["forbiddenMass"]) == {"runway", "approach", "angle"}


def test_a_dynamics_failure_s_track_stops_before_the_failed_state():
    flown, _, geometry, words, _, _ = _sample(3)
    start_s = N_LOOK * words.spec.step_s
    step_s = words.spec.step_s
    landed = export.track_payload(flown, 0, 5, "landed", geometry, step_s, start_s, HAE_MINUS_MSL_M, AERO_ROW)
    failed = export.track_payload(flown, 0, 5, "dynamics_failure", geometry, step_s, start_s, HAE_MINUS_MSL_M, AERO_ROW)
    assert landed["tS"] == [start_s, start_s + 2.0, start_s + 4.0, start_s + 5.0]
    assert failed["tS"] == [start_s, start_s + 2.0, start_s + 4.0]


def test_a_sentence_whose_first_predicted_step_leaves_a_column_unsaid_is_refused():
    flown, grid, geometry, words, row, _ = _sample(3)
    broken = grid.copy()
    broken[0, 2] = UNCHANGED
    with pytest.raises(ValueError, match="does not say every column"):
        export.sample_payload(flown, 0, row, broken, geometry, words, HAE_MINUS_MSL_M, AERO_ROW)


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
        "split": "val", "n_look": N_LOOK, "samples": 4, "temperature": 1.0, "procedure_masks": False,
        "drawn": {"flights": 20, "per_airport": 20},
        "readout": {"prior": part, "labelled": {"all": _cell(20, 1.0), "KXXX": _cell(10, 1.0), "KYYY": _cell(10, 1.0)}},
    }
    generation.update(changes)
    return generation


def _block(generation, **changes):
    arguments = {"prior_dir": PRIOR, "executor_sha256": "e" * 64, "instructions": INSTRUCTIONS, "samples": 4,
                 "temperature": 1.0, "airport": "KXXX", "procedure_altitudes": False, **changes}
    return export.readout_block(generation, arguments["prior_dir"], arguments["executor_sha256"],
                                arguments["instructions"], arguments["samples"], arguments["temperature"],
                                arguments["airport"], arguments["procedure_altitudes"])


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
    with pytest.raises(ValueError, match="counted no prior flight at KZZZ"):
        _block(_generation(), airport="KZZZ")
    with pytest.raises(ValueError, match="names 'some' flights an airport"):
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
    # the export speaks under the model's own procedure's masks: a readout under others is another generation
    (dict(generation=dict(procedure_masks=True)), "procedure_masks"),
    (dict(procedure_altitudes=True), "procedure_masks"),
])
def test_a_readout_of_another_prior_spec_split_or_draw_is_refused_by_name(change, name):
    generation = _generation(**change.pop("generation", {}))
    with pytest.raises(ValueError, match=rf"\b{name} "):
        _block(generation, **change)


def test_a_model_trained_under_the_procedure_s_altitudes_takes_a_readout_drawn_under_them():
    assert _block(_generation(procedure_masks=True), procedure_altitudes=True)["split"] == "val"


def test_a_sentence_the_glidepath_edge_stopped_ends_at_its_stop_with_no_crossing():
    """The closed loop of `test_prior_procedure` whose untrained speaker's draws (seed 8) sink below the edge at step 65,
    read as the export reads it (`said_rows`, the formal readout's reading): its words to the stop, its track to that
    step's end state, no crossing; a stop the row does not carry is refused."""
    from ts_transformer.experiments.prior_free_generation import BELOW_GLIDEPATH, said_rows
    from ts_transformer.tests.test_prior_procedure import _altitudes, _closed_loop, _final

    final = _final(crossing_m=2_000.0, faf_d_m=12_000.0, decision_m=-1_000.0)
    one, geometry, signals, words, flown, grid, forbidden, _ = _closed_loop((final,), seed=8)
    masks = _altitudes(geometry, final)
    batch = replay.Batch(signals=[signals], series=[], readings=[read_flight(signals, geometry, one, words)],
                         geometries=[geometry], vertical_paths=[], approach_ias_mps=[], groups=[], drawn={})
    rows, grids, stops = said_rows(batch, flown, np.asarray([grid]), forbidden, words, [0], masks)
    step = int(stops.step[0])
    assert step >= 0 and rows[0]["outcome"] == BELOW_GLIDEPATH
    payload = export.sample_payload(flown, 0, rows[0], grids[0], geometry, words, HAE_MINUS_MSL_M, AERO_ROW, step)
    start_s = N_LOOK * one.step_s
    assert payload["outcome"] == BELOW_GLIDEPATH and payload["crossing"] is None
    assert payload["endS"] == pytest.approx(start_s + (step + 1) * one.step_s)
    assert payload["track"]["tS"][-1] == pytest.approx(payload["endS"])
    assert payload["rows"] == N_LOOK + step + 1 and max(e["row"] for e in payload["events"]) <= N_LOOK + step
    assert "altitude" in payload["forbiddenMass"]                      # the procedure's altitudes mask the altitude words
    with pytest.raises(ValueError, match="outcome below_glidepath with the glidepath stop at step -1"):
        export.sample_payload(flown, 0, rows[0], grids[0], geometry, words, HAE_MINUS_MSL_M, AERO_ROW)


def test_a_readout_of_another_schema_is_refused_by_name_before_it_is_read():
    with pytest.raises(ValueError, match="is a ts-prior-free-generation-v0 file"):
        _block({"schema": "ts-prior-free-generation-v0"})


def test_a_path_outside_the_outputs_tree_names_itself():
    with pytest.raises(ValueError, match="is not under 4dTrajectory/outputs/"):
        export.outputs_path("/elsewhere/prior/full_s1337")
    assert export.outputs_path(PRIOR) == "4dTrajectory/outputs/POOLED/prior/v3_step1_20260924/full_s1337"
    with pytest.raises(ValueError, match="is not under 4dTrajectory/outputs/"):
        export.in_tree_of(Path("/elsewhere/prior"), PRIOR)


# ---- which model it is, by name: base, and each post-training stage's rounds
def _tuned(schema, start, round_number):
    return {"git": {"head": "h", "dirty": False},
            "fine_tuning": {"schema": schema, "from": start, "round": round_number, "samples": 8}}


def test_the_models_are_named_by_the_method_that_trained_them_every_version_alike():
    from ts_transformer.experiments.prior_augmented_reward import AUGMENTED_REWARD_SCHEMA
    from ts_transformer.experiments.prior_landing_reward import LANDING_REWARD_SCHEMA
    from ts_transformer.experiments.prior_generation_training_export import TRAFFIC_REWARD_SCHEMA
    from ts_transformer.experiments.traffic_window_reward import SCHEMA as TRAFFIC_WINDOW_REWARD_SCHEMA

    assert export.model_identity({"git": {}}) == ("base", None)
    assert export.model_identity(_tuned(LANDING_REWARD_SCHEMA, PRIOR, 3)) == ("landing", 3)
    assert export.model_identity(_tuned(AUGMENTED_REWARD_SCHEMA, PRIOR, 2)) == ("augmented", 2)
    assert export.model_identity(_tuned(TRAFFIC_REWARD_SCHEMA, PRIOR, 5)) == ("traffic", 5)
    # M4 in windows: every aircraft of a window commanded trains "window"; one a window — M4's own setting — "traffic"
    # (the user, 2026-10-03); a round from before its runs recorded which is not named
    window_round = _tuned(TRAFFIC_WINDOW_REWARD_SCHEMA, PRIOR, 5)
    for commanded, name in (("every", "window"), ("one", "traffic")):
        window_round["fine_tuning"]["commanded"] = commanded
        assert export.model_identity(window_round) == (name, 5)
    del window_round["fine_tuning"]["commanded"]
    with pytest.raises(ValueError, match="before its rounds recorded how a window's aircraft were commanded"):
        export.model_identity(window_round)
    from ts_transformer.experiments.traffic_window import COMMANDED
    assert set(export.WINDOW_MODELS) == set(COMMANDED)
    # the adopted landing model was written by the method's first version: the same model
    assert export.model_identity(_tuned("ts-prior-landing-reward-v1", PRIOR, 1)) == ("landing", 1)
    with pytest.raises(ValueError, match="no model is named for ts-prior-closed-loop-sft-v1"):
        export.model_identity(_tuned("ts-prior-closed-loop-sft-v1", PRIOR, 1))
    for schema in ("ts-prior-landing-reward-vnext", "ts-prior-landing-reward"):
        with pytest.raises(ValueError, match="not a versioned post-training schema"):
            export.method_of(schema)
    assert set(export.METHOD_MODELS.values()) | set(export.WINDOW_MODELS.values()) | {"base"} == set(export.MODEL_NAMES)
    # the archived runner's schema, mirrored (nothing imports the archive): read from its source
    from ts_transformer.repo_layout import REPO_ROOT

    archived = (REPO_ROOT / "4dTrajectory/ts_transformer/archive/one_commanded_scene_2026_10/experiments/"
                "traffic_reward.py").read_text(encoding="utf-8")
    assert f'TRAFFIC_REWARD_SCHEMA = "{export.TRAFFIC_REWARD_SCHEMA}"' in archived
    assert [export.display_name(*pair) for pair in (("base", None), ("augmented", 3))] == ["base", "augmented r3"]


def test_a_round_names_its_run_and_the_model_it_started_from_read_in_its_own_outputs_tree(tmp_path):
    """The rounds ran from worktrees: their ``from`` paths name another checkout, read here in the round's own tree."""
    from ts_transformer.experiments.prior_augmented_reward import AUGMENTED_REWARD_SCHEMA

    prior = tmp_path / "4dTrajectory" / "outputs" / "POOLED" / "prior"
    elsewhere = "/repo/.claude/worktrees/w/4dTrajectory/outputs/POOLED/prior"
    base, landing, augmented = (prior / "step1" / "full_s1", prior / "rl" / "grpo_s1" / "round_02",
                                prior / "stage2" / "aug_s1" / "round_04")
    for directory, config in ((base, {"git": {"head": "b", "dirty": False}}),
                              (landing, _tuned("ts-prior-landing-reward-v1", f"{elsewhere}/step1/full_s1", 2)),
                              (augmented, _tuned(AUGMENTED_REWARD_SCHEMA, f"{elsewhere}/rl/grpo_s1/round_02", 4))):
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    block = export.model_block(augmented, json.loads((augmented / "config.json").read_text()), "a" * 64, "full")
    assert block == {"name": "augmented", "round": 4, "run": "4dTrajectory/outputs/POOLED/prior/stage2/aug_s1",
                     "checkpointSha256": "a" * 64, "variant": "full", "trainedAt": {"head": "h", "dirty": False},
                     "fineTuning": {"schema": AUGMENTED_REWARD_SCHEMA,
                                    "from": "4dTrajectory/outputs/POOLED/prior/rl/grpo_s1/round_02",
                                    "fromName": "landing", "fromRound": 2}}
    assert export.model_block(landing, json.loads((landing / "config.json").read_text()), "b" * 64, "full")[
        "fineTuning"]["fromName"] == "base"
    # base: its own directory is its run, and it started from nothing
    block = export.model_block(base, {"git": {"head": "b", "dirty": False}}, "c" * 64, "full")
    assert (block["name"], block["round"], block["run"], block["fineTuning"]) == (
        "base", None, "4dTrajectory/outputs/POOLED/prior/step1/full_s1", None)


# ---- the runner end to end, on a synthetic artefact, an untrained prior and an executor spec (every write in tmp_path)
def test_the_export_flies_the_set_s_own_dynamics_flights_and_lists_the_rest(tmp_path, monkeypatch):
    """The set's two flights: the vectored one on its own dynamics, flown ``--samples`` times; the straight-in one with no
    identified type, listed with its group and no sample. The flights' series are stand-ins (the synthetic artefact has no
    harvest to rebuild them from): the A320's physics from each flight's own signals at the first predicted row."""
    from types import SimpleNamespace

    import torch

    from ts_transformer.autopilot import spec as executor_spec
    from ts_transformer.autopilot.flights import FlightInputs
    from ts_transformer.instructions import training_files as sets
    from ts_transformer.experiments import prior_train
    from ts_transformer.instructions.artefact import labeller_source_sha256
    from ts_transformer.tests.test_autopilot import _params, _physics
    from ts_transformer.tests.test_instruction_training_export import SET_ID, _artefact, _run, _straight, _vectored
    from ts_transformer.tests.test_training_overlays import _prior_dir
    from ts_transformer.tests.support import instruction_airport, landing_on, passed_executor

    vectored, straight = _vectored("KXXX:V1_09_abc123_20260101T000000Z"), _straight("KXXX:S1_09_abc124_20260101T000100Z")
    one = _artefact(tmp_path / "artefact", [vectored, straight])
    assert _run(tmp_path) == 0                                           # the set, as the Training export writes it
    roster = tmp_path / "tracks.json"
    roster.write_text(json.dumps({"records": [{"outcome": "assigned", "runway": "09", "flight_key": "own",
                                               "landing_time_utc": landing_on("val")}]}), encoding="utf-8")
    monkeypatch.setattr(prior_train, "tracks_manifest_path", lambda code: roster)
    prior = tmp_path / "4dTrajectory" / "outputs" / "POOLED" / "prior" / "v_test" / "full_s1"   # every prior's tree
    prior.parent.mkdir(parents=True)
    _prior_dir(prior, tmp_path / "artefact", roster)
    executor_spec.write_spec(tmp_path / "executor", _params(), one.sha256, {}, {
        "executor_source_sha256": executor_spec.executor_source_sha256(), "python": "3", "labeller_source_sha256": labeller_source_sha256(),
        "git": {"head": "test", "dirty": False}})
    passed_executor(tmp_path / "executor")

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
    monkeypatch.setattr(export, "published_vertical_paths",
                        lambda geometry: (VerticalPath(15.0, 3.0),) * len(geometry.candidates))
    monkeypatch.setattr(export, "runway_hae_minus_msl_m", _offsets)
    args = ["--prior", str(prior), "--instructions", str(tmp_path / "artefact"),
            "--executor", str(tmp_path / "executor"), "--airports-root", str(tmp_path / "airports"), "--set", SET_ID,
            "--airport", "KXXX", "--samples", "3"]
    assert export.main(args) == 0
    training = tmp_path / "airports" / "KXXX" / "training"
    checkpoint = hashlib.sha256((prior / "checkpoint.pt").read_bytes()).hexdigest()
    overlay_id = f"generation_base_{checkpoint[:8]}"                   # the default: its name and checkpoint
    payload = json.loads((training / overlay_id / export.PAYLOAD_FILE).read_text(encoding="utf-8"))
    sample = json.loads((training / SET_ID / "sample.json").read_text(encoding="utf-8"))
    assert payload["schema"] == export.SCHEMA and payload["readout"] is None
    assert payload["base"] == {"setId": SET_ID, "specSha256": sample["vocabulary"]["specSha256"],
                               "candidatesSha256": sample["candidatesSha256"], "airportFrame": sample["airportFrame"]}
    assert payload["producedBy"]["device"] == "cpu"                     # the speaker's; the executor is always on CPU
    assert payload["model"] == {"name": "base", "round": None, "run": "4dTrajectory/outputs/POOLED/prior/v_test/full_s1",
                                "checkpointSha256": checkpoint, "variant": "full",
                                "trainedAt": {"head": "test", "dirty": False}, "fineTuning": None}
    assert payload["generation"]["firstPredictedRow"] == N_LOOK and payload["generation"]["samples"] == 3
    assert payload["generation"]["procedureMasks"] == []                 # the model's own: none (`_prior_dir`)
    # the set's flights in its order: the own-dynamics one flown three times, the other listed with its group
    assert [f["flightKey"] for f in payload["flights"]] == [f["flightKey"] for f in sample["flights"]]
    flown = {f["datasetId"]: f for f in payload["flights"]}
    assert flown[vectored.dataset_id]["flown"] and [s["sample"] for s in flown[vectored.dataset_id]["samples"]] == [0, 1, 2]
    assert flown[straight.dataset_id] == {"flightKey": flown[straight.dataset_id]["flightKey"], "datasetId": straight.dataset_id,
                                          "group": "no identified type", "flown": False, "samples": []}
    for item in flown[vectored.dataset_id]["samples"]:
        assert item["events"][0]["row"] == N_LOOK and item["track"]["tS"][0] == N_LOOK * one.step_s
        # its height the observed track's: the flight's own runway's offset added, taken per flight from the lookup
        height = np.asarray(item["track"]["altitudeHaeM"]) - np.asarray(item["track"]["altitudeM"])
        assert np.allclose(height, HAE_MINUS_MSL_M, atol=0.011)
    manifest = json.loads((training / sets.OVERLAYS_FILE).read_text(encoding="utf-8"))
    assert [(o["id"], o["kind"], o["base"]) for o in manifest["overlays"]] == [(overlay_id, sets.KIND_GENERATION, SET_ID)]
    # the same seed, the same sentences: a second export under another id is identical flight for flight
    assert export.main([*args, "--overlay-id", "again"]) == 0
    again = json.loads((training / "again" / export.PAYLOAD_FILE).read_text(encoding="utf-8"))
    assert again["flights"] == payload["flights"]
    with pytest.raises(SystemExit):                                     # never overwritten
        export.main(args)
    # heights on another datum than the set's flights: refused before anything is written
    monkeypatch.setattr(export, "runway_hae_minus_msl_m",
                        lambda *given: {ident: offset + 0.1 for ident, offset in _offsets(*given).items()})
    with pytest.raises(ValueError, match="draws it -32.00 m HAE − MSL, the overlay -31.90 m"):
        export.main([*args, "--overlay-id", "datum"])
    assert not (training / "datum").exists()
    monkeypatch.setattr(export, "runway_hae_minus_msl_m", _offsets)
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
    # from augmented starts: a kind of its own, each flight with its move and the moved rows the prior read, flown from
    # the moved state; one seed, one move a flight whatever the model or the samples' seed
    # the synthetic artefact holds no train split: a window every moved start fits (the check itself: prior_free_generation)
    monkeypatch.setattr(export, "start_altitude_windows", lambda instructions: {"KXXX": (-1e4, 1e5)})
    augmented = [*args, "--augment-seed", "7", "--overlay-id", "moved"]
    assert export.main(augmented) == 0
    moved = json.loads((training / "moved" / export.PAYLOAD_FILE).read_text(encoding="utf-8"))
    assert moved["schema"] == export.AUGMENTED_SCHEMA and "readout" not in moved
    assert moved["generation"]["augment"] == {"seed": 7, "tries": 10, "limits": {
        "rotationDeg": 15.0, "altitudeM": 150.0, "speedFraction": 0.05}, "timeoutFactor": 2.0}
    manifest = json.loads((training / sets.OVERLAYS_FILE).read_text(encoding="utf-8"))
    assert {o["id"]: o["kind"] for o in manifest["overlays"]}["moved"] == sets.KIND_AUGMENTED_GENERATION
    for item in moved["flights"]:
        observed = signals_by_id[item["datasetId"]]
        assert item["flown"] and item["augmentDraws"] >= 1 and len(item["samples"]) == 3
        move = Augmentation(item["augmentation"]["rotationDeg"], item["augmentation"]["altitudeM"],
                            item["augmentation"]["speedScale"])
        # written as drawn, not rounded to a display precision: the backend moves the start by exactly this
        assert move.rotation_deg != round(move.rotation_deg, 4) and move.speed_scale != round(move.speed_scale, 6)
        assert abs(move.rotation_deg) <= 15 and abs(move.altitude_m) <= 150 and abs(move.speed_scale - 1) <= 0.05
        # the moved rows 0 … N_LOOK − 1 the prior read, on the flight's clock; the samples start from the moved state
        rows = augment_signals(observed, move)
        assert item["observed"]["tS"] == [round(float(t), 3) for t in observed.time_s[:N_LOOK]]
        assert np.allclose(item["observed"]["altitudeM"], rows.altitude_m[:N_LOOK], atol=0.006)
        start = inputs([series(observed)], device=None, anchor=N_LOOK).initial_state[0].numpy()
        lat, lon, altitude, _, _ = augment_state(float(start[0]), float(start[1]), float(start[2]), float(start[3]),
                                                 float(start[4]), instruction_airport(), move)
        for sample in item["samples"]:                                  # each written to its display precision
            track = sample["track"]
            assert (track["lat"][0], track["lon"][0]) == pytest.approx((lat, lon), abs=1.5e-7)
            assert track["altitudeM"][0] == pytest.approx(altitude, abs=0.006)
    assert export.main([*augmented[:-1], "moved-again", "--seed", "99"]) == 0
    again = json.loads((training / "moved-again" / export.PAYLOAD_FILE).read_text(encoding="utf-8"))
    assert [f["augmentation"] for f in again["flights"]] == [f["augmentation"] for f in moved["flights"]]
    with pytest.raises(SystemExit):                                     # no readout of augmented starts
        export.main([*augmented[:-1], "moved-readout", "--readout", str(tmp_path)])
    # the speaker on the GPU (as the formal readout runs it): the same moves (drawn on the CPU), the executor still on the
    # CPU — each sample starts from the moved state — and the device recorded
    if torch.cuda.is_available():
        assert export.main([*augmented[:-1], "moved-cuda", "--device", "cuda"]) == 0
        on_gpu = json.loads((training / "moved-cuda" / export.PAYLOAD_FILE).read_text(encoding="utf-8"))
        assert on_gpu["producedBy"]["device"] == "cuda"
        assert [f["augmentation"] for f in on_gpu["flights"]] == [f["augmentation"] for f in moved["flights"]]
        for item, cpu_item in zip(on_gpu["flights"], moved["flights"]):
            assert len(item["samples"]) == 3
            assert {(s["track"]["lat"][0], s["track"]["lon"][0]) for s in item["samples"]} == {
                (s["track"]["lat"][0], s["track"]["lon"][0]) for s in cpu_item["samples"]}
    # a flight none of the draws fits is not flown and says how many draws it took; the others keep their own samples,
    # flown under the augmented time limit
    from ts_transformer.experiments.prior_free_generation import AugmentedStarts
    real_draw, real_limits, asked = export.augmented_starts, export.limits_s, []

    def first_unfitted(signals, inputs, rng, windows):
        drawn = real_draw(signals, inputs, rng, windows)
        return AugmentedStarts([None, *drawn.moves[1:]], [10, *drawn.draws[1:]])

    def limits(batch, params, step_s, *, augmented):
        asked.append(augmented)
        return real_limits(batch, params, step_s, augmented=augmented)

    monkeypatch.setattr(export, "augmented_starts", first_unfitted)
    monkeypatch.setattr(export, "limits_s", limits)
    assert export.main([*augmented[:-1], "moved-gap"]) == 0
    gap = json.loads((training / "moved-gap" / export.PAYLOAD_FILE).read_text(encoding="utf-8"))
    first, second = gap["flights"]
    assert first["flown"] is False and first["samples"] == [] and first["augmentDraws"] == 10
    assert first["augmentation"] is None and first["observed"] is None
    assert second["flown"] and second["augmentation"] == moved["flights"][1]["augmentation"] and len(second["samples"]) == 3
    assert asked == [True]
