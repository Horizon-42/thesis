"""The Training export of stage B (prior §12 B6): the prior's sentences flown again and checked, the words the procedure
masks blocked, the procedure's limits, a set written beside the other indexes and read again, and the fixtures of the
frontend's readers written by the export's own functions."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from flight_scenarios.fas_geometry import course_halfwidth_m, fas_course_geometry
from ts_transformer.experiments import prior_training_export as export
from ts_transformer.experiments.prior_free_generation import Stored
from ts_transformer.instructions.grammar import column_words
from ts_transformer.instructions.words import ALTITUDE, ANGLE, HEADING, UNCHANGED, Words
from ts_transformer.instructions import training_files as stage_a_files
from ts_transformer.instructions.artefact import load_signals
from ts_transformer.experiments import model_speed
from ts_transformer.tests.test_model_speed import speed_readout
from ts_transformer.prior import training_files as files
from ts_transformer.prior.procedure import GLIDEPATH_BELOW_M, Final
from ts_transformer.tests.support import instruction_spec, parallel_airport
from ts_transformer.tests.test_prior_free_generation import generate

CPU = torch.device("cpu")


def stored_of(generated, flight, sample=0):
    """A free sentence as the readout's reader gives it back (`prior_free_generation.read_sentences`)."""
    return Stored(generated.index, sample, generated.words, generated.states, generated.go_around_probability,
                  generated.go_around_permitted, generated.on_final, generated.blocked,
                  {"index": generated.index, "dataset_id": flight["dataset_id"], "airport": flight["airport"],
                   "sample": sample, "outcome": generated.outcome, "timed_out": generated.timed_out,
                   "go_arounds": generated.go_arounds, "rows": len(generated.words)})


@pytest.mark.parametrize("interval_s", [2.0, 4.0])
def test_a_sentence_flown_again_gives_its_readout_s_states_and_outcome(tmp_path, monkeypatch, interval_s):
    spy = {}
    generated, stored, words, geometry = generate(tmp_path, monkeypatch, interval_s=interval_s, spy=spy)
    item = stored_of(generated, spy["flights"][0])
    (payload,) = export.fly_again(spy["directory"], tmp_path / "executor", "train", interval_s, {0: stored}, [item],
                                  load_signals(spy["directory"], "train"), words, device=CPU)
    assert payload["outcome"] == generated.outcome and payload["words"] == generated.words.tolist()
    assert payload["flownFromRow"] == stored.rows.start * int(round(interval_s / words.spec.step_s))
    flown = generated.states[payload["flownFromRow"]:]
    track = payload["track"]
    assert track["rows"] == len(track["eM"]) == len(track["heightMslM"]) == len(payload["attitude"]["headingDeg"])
    shared = min(track["rows"], len(flown))
    assert np.allclose(track["heightMslM"][:shared], flown[:shared, 2], atol=0.051)
    assert set(payload["blocked"]) == {"altitude", "angle"}
    assert all(len(rows) == len(generated.words) for rows in payload["blocked"].values())
    assert len(payload["goAroundProbability"]) == len(payload["onFinal"]) == len(generated.words)
    # the block of a flown sentence (D135): the envelopes of its words on its flown track, within it (a heading word
    # whose lead runs past the end has an empty band, not judged)
    bands = payload["envelopes"]["heading"]
    assert bands and all(band["stopRow"] <= track["rows"] for band in bands if band["stopRow"] > band["firstRow"])
    # each word judged from the flown row where the executor heard it: its said row × Δ / 2 s
    every = int(round(interval_s / words.spec.step_s))
    said_rows = [t for t, row in enumerate(generated.words) if row[HEADING] != UNCHANGED]
    assert [band["row"] for band in bands] == [t * every for t in said_rows][: len(bands)]


def test_a_sentence_flown_again_is_refused_unless_it_gives_the_readout_back(tmp_path, monkeypatch):
    spy = {}
    generated, stored, words, geometry = generate(tmp_path, monkeypatch, interval_s=4.0, spy=spy)
    item = stored_of(generated, spy["flights"][0])

    def again(changed):
        return export.fly_again(spy["directory"], tmp_path / "executor", "train", 4.0, {0: stored}, [changed],
                                load_signals(spy["directory"], "train"), words, device=CPU)

    states = item.states.copy()
    states[-1, 2] += 1e-3                                              # a stored height a millimetre off
    with pytest.raises(ValueError, match="from its readout's states"):
        again(dataclasses.replace(item, states=states))
    with pytest.raises(ValueError, match="from its readout's states"):                  # a row missing
        again(dataclasses.replace(item, states=item.states[:-1]))
    other = "timeout" if generated.outcome != "timeout" else "landed"
    with pytest.raises(ValueError, match=f"the readout's {other}"):
        again(dataclasses.replace(item, row={**item.row, "outcome": other}))
    if len(item.words) > 1:
        with pytest.raises(ValueError, match="done at another row than their last word"):
            again(dataclasses.replace(item, words=item.words[:-1]))
    longer = np.concatenate((item.words, item.words[-1:]))
    with pytest.raises(ValueError, match="done at another row than their last word"):
        again(dataclasses.replace(item, words=longer))


def finals():
    geometry = parallel_airport()
    return tuple(Final(geometry, k, 9_000.0, fas_course_geometry(c.length_m)) for k, c in enumerate(geometry.candidates))


def test_the_procedure_limits_are_the_masks_own():
    """The region's outline is the FAF and the LPV cone, the lower edge the glidepath less `GLIDEPATH_BELOW_M`, the DA
    where the glidepath reaches the decision height, the entry height the glidepath at the FAF — MSL (+ E)."""
    finals_ = finals()
    block = export.procedure_block(finals_)
    assert [item["ident"] for item in block] == [c.ident for c in finals_[0].geometry.candidates]
    for final, item in zip(finals_, block):
        elevation = final.geometry.elevation_m
        region = item["region"]
        assert (region["eM"][0], region["nM"][0]) == (region["eM"][-1], region["nM"][-1])          # closed
        before, off = final.axes(np.array(region["eM"]), np.array(region["nM"]))
        assert np.allclose(off, course_halfwidth_m(before, final.cone), atol=0.2)
        assert before.min() > -0.2 and before.max() < final.faf_m + 0.2
        edge = item["glidepathLowerEdge"]
        expected = final.glidepath_m(np.array(edge["beforeThresholdM"])) - GLIDEPATH_BELOW_M + elevation
        assert np.allclose(edge["heightMslM"], expected, atol=0.06)
        da = item["decision"]
        assert abs(final.glidepath_m(np.array(da["beforeThresholdM"])) + elevation - da["heightMslM"]) < 0.2
        assert da["heightMslM"] == round(final.decision_m + elevation, 1)
        assert item["entry"]["heightMslM"] == round(final.entry_m + elevation, 1)
        e, n = final.axes(np.array(item["entry"]["eM"]), np.array(item["entry"]["nM"]))
        assert abs(e[0] - final.faf_m) < 0.2 and abs(n[0]) < 0.2


def test_the_blocked_words_are_written_as_the_vocabulary_s_values():
    words = Words(instruction_spec())
    altitude = list(column_words(ALTITUDE, words, 2))
    angle = list(column_words(ANGLE, words, 2))
    mask_altitude = np.zeros((2, len(altitude)), dtype=bool)
    mask_altitude[1, [2, 3]] = True
    mask_angle = np.zeros((2, len(angle)), dtype=bool)
    mask_angle[0, -1] = True
    out = export.blocked_payload({ALTITUDE: mask_altitude, ANGLE: mask_angle}, 2, words)
    assert out == {"altitude": [[], [int(altitude[2]), int(altitude[3])]], "angle": [[int(angle[-1])], []]}


# ---- a set written and read again; the frontend's fixtures
FIXTURES = Path(__file__).resolve().parents[3] / "aeroviz-4d" / "src" / "data" / "__tests__" / "fixtures" / "stage_b"
FIXTURE_SET = "fixture_set"
#: The set of the base's claimed validation readout (outline D109): val flights.
FIXTURE_VAL_SET = "fixture_val"


def stage_b_fixture(tmp_path, monkeypatch, *, texts=False):
    """A set of one synthetic flight (A26's artefact: `test_start._artefact`) written by the export itself (`main`, its
    set into a tmp root by `TrainingFiles.write_set`, the index by its own writer): two sentences of an untrained prior
    (seeds 0 and 1), each flown again, and the flight's head — its observed track, open-loop sentence and closed-loop
    sentence at Δ = 4 s flown again — as stage A's export gives it (`split_flights`; here built by stage A's functions:
    a synthetic flight has no harvest series, and its observed attitude is the no-airframe one). The live roots are
    replaced as in the runner's test; the paths and times written are fixed names. ``(index, sample)`` as read back,
    with ``texts`` their files' texts too."""
    from types import SimpleNamespace

    from ts_transformer.autopilot import closed_loop, replay
    from ts_transformer.experiments import training_export as stage_a
    from ts_transformer.instructions.artefact import load_candidates, load_signals
    from ts_transformer.instructions.labeller.read import read_flight
    from ts_transformer.io_utils import file_sha256
    from ts_transformer.tests import test_start

    stored_sentences, spies = [], []
    for sample in (0, 1):
        spy = {}
        generated, stored, words, geometry = generate(tmp_path / f"s{sample}", monkeypatch, interval_s=4.0, seed=sample,
                                                      spy=spy)
        stored_sentences.append(stored_of(generated, spy["flights"][0], sample))
        spies.append(spy)
    spy = spies[0]
    directory, executor = spy["directory"], tmp_path / "s0" / "executor"
    (signals,) = load_signals(directory, "train")
    reading = read_flight(signals, geometry, words.spec, words)
    attitude = {"headingDeg": signals.track_deg,
                "pathAngleDeg": np.degrees(np.arctan2(signals.vertical_rate_mps, signals.ground_speed_mps)),
                "bankRightDeg": None, "attackDeg": None}
    head = stage_a.flight_head(signals, f"{signals.dataset_id}_fixture", "train", "vectored", "without go-around",
                               "own", reading, geometry, attitude, words)
    stored = spy["stored"]
    batch, missing = closed_loop.replay_batch(spy["batch"], {0: stored}, words)        # as the formal replay flies it
    assert not missing
    flown, (verdict,) = replay.fly_batch(batch, test_start._params(), words, device=CPU)
    replayed = stage_a.replay_payload(flown, 0, verdict, batch, stored,
                                      batch.inputs(test_start._params().start_rule, CPU).aero_params[0].numpy(),
                                      words.spec, words)
    head["closedLoop"]["4"] = stage_a.closed_loop_payload(stored, replayed, 4.0, geometry, words)
    prior, readout_dir, root = tmp_path / "prior", tmp_path / "readout", tmp_path / "airports"
    prior.mkdir()
    (prior / "checkpoint.pt").write_bytes(b"a checkpoint")
    readout = fixture_readout(geometry.code, instructions=str(directory), executor=str(executor), prior=str(prior),
                              checkpoint_sha256=file_sha256(prior / "checkpoint.pt"))
    val_dir = tmp_path / "readout_val"
    readout_dir.mkdir()
    (readout_dir / "readout.json").write_text("{}")                     # the readouts written (D119, D127)
    speed = speed_readout(tmp_path / "speed")
    names = {readout_dir: "fixture/readout", Path(directory): "fixture/instruction_language", executor: "fixture/executor",
             prior: "fixture/prior", val_dir: "fixture/readout_val", speed: "fixture/speed"}
    with monkeypatch.context() as patch:
        patch.setattr(export, "read_sentences", lambda out: (readout, stored_sentences))
        patch.setattr(export, "require_conforming_closed_loop",
                      lambda *given: (test_start._params(), {"sha256": "fixture", "checks": {}}, words))
        patch.setattr(export, "open_prior", lambda prior_dir, instructions: SimpleNamespace(
            checkpoint=SimpleNamespace(identity=readout["identity"]), interval_s=4.0,
            geometries=load_candidates(directory)))
        patch.setattr(export, "airport_finals", lambda g: finals_of(g))
        patch.setattr(export, "split_flights", lambda instructions, split, ids, intervals, params, words_, device: (
            [head], load_candidates(directory)[geometry.code]))
        patch.setattr(export, "candidate_hae_minus_msl_m",
                      lambda runway_ends_from, geometry_: {c.ident: -33.0 for c in geometry_.candidates})
        patch.setattr(export, "repo_relative", lambda path: names[path])
        patch.setattr(model_speed, "repo_relative", lambda path: names[path])           # the speed readout's name
        patch.setattr(export, "git_state", lambda: {"head": "fixture", "dirty": False})
        patch.setattr(stage_a_files, "utc_now", lambda: "fixture")   # the one writer's
        build, built = export.build_airport, []
        patch.setattr(export, "build_airport", lambda *a, **k: built.append(build(*a, **k)) or built[-1])
        assert export.main(["--readout", str(readout_dir), "--set-id", FIXTURE_SET, "--root", str(root),
                            "--per-airport", "1", "--speed", str(speed), "--smoke"]) == 0
        # the base's one validation readout, claimed (D85): its set holds val flights (outline D109). Its flights are
        # the train set's, given the split val (the synthetic artefact has no val sentences to fly again).
        heads, geometry_ = built[0]
        val_heads = [{**json.loads(json.dumps(h)), "split": "val"} for h in heads]
        readout.update(split="val", smoke=False)
        (prior / "val_read_prior_free_generation.json").write_text(json.dumps({"out": str(val_dir)}))
        val_dir.mkdir(exist_ok=True)
        (val_dir / "readout.json").write_text("{}")                     # the readout written: the read spent (D119)
        patch.setattr(export, "build_airport", lambda *a, **k: (val_heads, geometry_))
        assert export.main(["--readout", str(val_dir), "--set-id", FIXTURE_VAL_SET, "--root", str(root),
                            "--per-airport", "1", "--speed", str(speed), "--smoke"]) == 0
    training = root / geometry.code / "training"
    entry, sample = files.FILES.listed_set(training, geometry.code, FIXTURE_SET)
    val_entry, _ = files.FILES.listed_set(training, geometry.code, FIXTURE_VAL_SET)
    index = json.loads((training / files.INDEX_FILE).read_text(encoding="utf-8"))
    assert index["sets"] == [entry, val_entry]
    assert sample["source"]["validationClaim"] is None
    assert sample["source"]["speed"] == {"readout": "fixture/speed", "model": "fixture/prior", "smoke": True}
    assert val_entry["source"]["validationClaim"] == {"reader": "prior_free_generation", "prior": "fixture/prior",
                                                      "readout": "fixture/readout_val"}
    if texts:
        return index, sample, {name: (training / name).read_text(encoding="utf-8")
                               for name in (files.INDEX_FILE, f"{FIXTURE_SET}/{files.SAMPLE_FILE}",
                                            f"{FIXTURE_VAL_SET}/{files.SAMPLE_FILE}")}
    return index, sample


#: A readout's identity as a fixture: a selection (`checkpoint.readable_identity` reads its counts) and a stand-in.
FIXTURE_IDENTITY = {"fixture": True, "selection": {"rule": "landed", "counts": {"train": {}, "select": {}}}}


def fixture_readout(airport, **more):
    """A free-generation readout's config as the export reads it (`prior_free_generation.main` writes it)."""
    from ts_transformer.prior.procedure import PROCEDURE_MASKS

    return {"smoke": True, "instructions": "fixture/instruction_language", "executor": "fixture/executor",
            "split": "train", "row_interval_s": 4.0, "selection": "landed", "identity": FIXTURE_IDENTITY,
            "prior": "fixture/prior", "prior_run": {"airports": [airport], "held_out": None, "sample": None},
            "checkpoint_sha256": "fixture", "procedure_masks": PROCEDURE_MASKS, "temperature": 1.0,
            "most_go_arounds": 2, "git": {"head": "fixture", "dirty": False}, "samples": 2, "seed": 1337, **more}


def test_a_set_is_written_beside_the_other_indexes_and_read_again(tmp_path, monkeypatch):
    _, sample = stage_b_fixture(tmp_path / "fixture", monkeypatch)
    airport = sample["airport"]
    training = tmp_path / "airports" / airport / "training"
    training.mkdir(parents=True)
    (training / stage_a_files.INDEX_FILE).write_text("{\"stage A\": true}")         # never read or written here
    entry = export.index_entry("one", {**sample, "setId": "one"})
    files.FILES.write_set(training, airport, entry, stage_a_files.serialise({**sample, "setId": "one"}), [])
    listed, read = files.FILES.listed_set(training, airport, "one")
    assert listed == entry and read["flights"][0]["prior"][1]["sample"] == 1
    assert (training / stage_a_files.INDEX_FILE).read_text() == "{\"stage A\": true}"
    with pytest.raises(ValueError, match="never overwritten"):
        files.FILES.read_index(training, airport, "one")
    with pytest.raises(stage_a_files.NotListed):
        files.FILES.listed_set(training, airport, "two")
    (training / "one" / files.SAMPLE_FILE).write_text(json.dumps({**read, "schema": "aeroviz-training-sample-v9"}))
    with pytest.raises(ValueError, match=files.SAMPLE_SCHEMA):                        # another format, by name
        files.FILES.listed_set(training, airport, "one")


def test_the_frontend_fixtures_are_what_the_export_writes(tmp_path, monkeypatch):
    """The fixture the frontend's readers are tested on is the export's own output today, the files its writers write
    (the sample by the export, the index by `TrainingFiles.write_set`); a change of the export moves it, and this test
    says so until it is written again (``AEROVIZ_WRITE_FIXTURES=1``)."""
    index, sample, texts = stage_b_fixture(tmp_path, monkeypatch, texts=True)
    if os.environ.get("AEROVIZ_WRITE_FIXTURES") == "1":
        for name, text in texts.items():
            (FIXTURES / name).parent.mkdir(parents=True, exist_ok=True)
            (FIXTURES / name).write_text(text, encoding="utf-8")
    for name, text in texts.items():
        assert (FIXTURES / name).read_text(encoding="utf-8") == text, (
            f"{FIXTURES / name} is not what the export writes now: AEROVIZ_WRITE_FIXTURES=1 writes it again")
    flight = sample["flights"][0]
    assert [s["sample"] for s in flight["prior"]] == [0, 1] and sorted(flight["closedLoop"]) == ["4"]
    assert len(sample["procedure"]) == len(sample["candidates"])


def test_the_flights_of_a_loop_are_told_their_own_words_and_halted_when_done():
    """Two flights done at different rows (`test_prior_free_generation.FakeLoop`): each flown to the row it was done in,
    each halted once done; a flight not done at its last word, or done before it, is refused."""
    from ts_transformer.tests.test_prior_free_generation import FakeLoop

    start = np.array([0.0, 0.0, 500.0, 90.0, 70.0, -3.0])
    words = [np.zeros((3, 5), dtype=np.int64), np.zeros((6, 5), dtype=np.int64)]
    loop = FakeLoop([start, start], [3, 6], 2, 2)
    flown = export.step_words(loop, words)
    assert [len(f) for f in flown] == [1 + 3 * 2, 1 + 6 * 2] and loop.halted.all()
    with pytest.raises(ValueError, match="done at another row than their last word"):
        export.step_words(FakeLoop([start, start], [3, 6], 2, 2), [words[0][:2], words[1]])
    with pytest.raises(ValueError, match="done at another row than their last word"):
        export.step_words(FakeLoop([start, start], [3, 6], 2, 2), [np.zeros((4, 5), dtype=np.int64), words[1]])


def test_the_state_bound_mirrors_the_executor_conformance_and_the_track_is_compared_wrapped():
    from ts_transformer.autopilot.conformance import STATE_BOUND_M
    from ts_transformer.instructions.artefact import STATE_COLUMNS

    assert export.STATE_BOUND_M == STATE_BOUND_M
    a = np.zeros((2, len(STATE_COLUMNS)))
    b = a.copy()
    a[:, export.TRACK], b[:, export.TRACK] = 359.9999999999, 0.0000000001
    assert export.apart(a, b) < 1e-9
    b[1, 0] = 1e-3
    assert export.apart(a, b) == pytest.approx(1e-3)
    assert export.apart(a, b[:1]) == float("inf")


def test_the_flights_of_an_airport_are_a_seeded_draw_of_the_readout_s():
    rows = [Stored(i, s, None, None, None, None, None, {}, {"airport": "KXXX" if i < 40 else "KYYY"})
            for i in range(60) for s in (0, 1)]
    chosen = export.chosen_flights(rows, "KXXX", 5, 1337)
    assert chosen == sorted(chosen) and len(set(chosen)) == 5 and set(chosen) <= set(range(40))
    assert chosen != list(range(5)) and chosen == export.chosen_flights(rows, "KXXX", 5, 1337)
    assert chosen != export.chosen_flights(rows, "KXXX", 5, 7)
    assert export.chosen_flights(rows, "KYYY", 50, 1337) == list(range(40, 60))


def test_the_runner_writes_a_set_and_refuses_what_it_cannot_trust(tmp_path, monkeypatch):
    """`main` on the synthetic artefact: every live root replaced (the closed-loop check, the identity, the landings, the
    CIFP finals and digests, A23's part — a head with no closed-loop sentence — and the datum); a readout of two samples;
    a set written into a tmp root, read again; refused: a smoke readout without --smoke, another identity, another
    checkpoint, an airport not in the readout, a set already written, a val readout the prior's claim does not name
    (D85)."""
    from types import SimpleNamespace

    from ts_transformer.instructions.artefact import load_candidates
    from ts_transformer.io_utils import file_sha256
    from ts_transformer.tests import test_start

    stored, spy = [], {}
    for sample in (0, 1):
        spy = {}
        generated, _, words, geometry = generate(tmp_path / "artefacts" / f"s{sample}", monkeypatch, interval_s=4.0,
                                                 seed=sample, spy=spy)
        stored.append(stored_of(generated, spy["flights"][0], sample))
    directory, executor = spy["directory"], tmp_path / "artefacts" / "s1" / "executor"
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "checkpoint.pt").write_bytes(b"a checkpoint")
    readout = fixture_readout(geometry.code, instructions=str(directory), executor=str(executor), prior=str(prior),
                              checkpoint_sha256=file_sha256(prior / "checkpoint.pt"))
    read = []
    monkeypatch.setattr(export, "read_sentences", lambda out: read.append(out) or (readout, stored))
    monkeypatch.setattr(export, "require_conforming_closed_loop",
                        lambda *given: (test_start._params(), {"sha256": "spec", "checks": {"stub": True}}, words))
    # the prior as `checkpoint.open_prior` opens it (its own checks: tests/test_prior_validation.py)
    opened = SimpleNamespace(checkpoint=SimpleNamespace(identity=FIXTURE_IDENTITY), interval_s=4.0,
                             geometries=load_candidates(directory))
    monkeypatch.setattr(export, "open_prior", lambda prior_dir, instructions: opened)
    monkeypatch.setattr(export, "airport_finals", lambda g: finals_of(g))
    monkeypatch.setattr(export, "split_flights", lambda instructions, split, ids, intervals, params, words_, device: (
        [{"datasetId": i, "runway": "09", "closedLoop": {}} for i in ids], load_candidates(directory)[geometry.code]))
    monkeypatch.setattr(export, "candidate_hae_minus_msl_m",
                        lambda runway_ends_from, geometry_: {c.ident: -33.0 for c in geometry_.candidates})
    root = tmp_path / "airports"
    speed = speed_readout(tmp_path / "speed")
    argv = ["--readout", str(tmp_path / "readout"), "--set-id", "one", "--root", str(root), "--per-airport", "1",
            "--speed", str(speed), "--smoke"]
    with pytest.raises(SystemExit):                       # a readout without its readout.json: before any file is read
        export.main(argv)
    assert read == []
    (tmp_path / "readout").mkdir()
    (tmp_path / "readout" / "readout.json").write_text("{}")
    with pytest.raises(SystemExit):                                     # a smoke readout makes only a smoke set
        export.main(argv[:-1])
    build, built = export.build_airport, []
    monkeypatch.setattr(export, "build_airport", lambda *a, **k: built.append(build(*a, **k)) or built[-1])
    assert export.main(argv) == 0
    training = root / geometry.code / "training"
    entry, sample = files.FILES.listed_set(training, geometry.code, "one")
    assert entry["sentences"] == 2 and [s["sample"] for s in sample["flights"][0]["prior"]] == [0, 1]
    assert sample["source"]["checks"] == {"stub": True} and sample["cohort"]["seed"] == 1337
    before = sorted(p.relative_to(root) for p in root.rglob("*"))
    with pytest.raises(ValueError, match="never overwritten"):
        export.main(argv)
    with pytest.raises(SystemExit):
        export.main([*argv, "--airports", "KZZZ"])
    monkeypatch.setattr(opened.checkpoint, "identity", {**FIXTURE_IDENTITY, "fixture": False})              # its data changed
    with pytest.raises(SystemExit, match="identity"):
        export.main([*argv[:3], "two", *argv[4:]])
    monkeypatch.setattr(opened.checkpoint, "identity", FIXTURE_IDENTITY)
    (prior / "checkpoint.pt").write_bytes(b"another checkpoint")                        # its prior changed
    with pytest.raises(SystemExit, match="checkpoint"):
        export.main([*argv[:3], "two", *argv[4:]])
    (prior / "checkpoint.pt").write_bytes(b"a checkpoint")
    # a readout of the val days: only the one the prior's claim of its val read names (D85)
    readout.update(split="val", smoke=False)
    with pytest.raises(SystemExit, match="D85"):
        export.main([*argv[:3], "val", *argv[4:]])
    (prior / "val_read_prior_free_generation.json").write_text(json.dumps({"out": str(tmp_path / "another")}))
    with pytest.raises(SystemExit, match="D85"):
        export.main([*argv[:3], "val", *argv[4:]])
    assert sorted(p.relative_to(root) for p in root.rglob("*")) == before                # nothing more written
    (prior / "val_read_prior_free_generation.json").write_text(json.dumps({"out": str(tmp_path / "readout")}))
    # the claimed one, written, is exported (its flights as the train readout's: the synthetic artefact has no val
    # sentences)
    monkeypatch.setattr(export, "build_airport", lambda *a, **k: built[0])
    assert export.main([*argv[:3], "val", *argv[4:]]) == 0


def finals_of(geometry):
    return tuple(Final(geometry, k, 9_000.0, fas_course_geometry(c.length_m)) for k, c in enumerate(geometry.candidates))


def test_the_set_names_the_artefact_and_the_executor_relative_to_the_repository():
    """A readout run in a worktree records its absolute paths; the set names them from the repository, so the live
    executor opens them from its own checkout."""
    from ts_transformer.repo_layout import REPO_ROOT

    readout = fixture_readout("KXXX", instructions=str(REPO_ROOT / "4dTrajectory/outputs/POOLED/instruction_language/v"),
                              executor=str(REPO_ROOT / "4dTrajectory/outputs/POOLED/executor/v"))
    source = export.source_block("r", readout, Words(instruction_spec()), {"sha256": "s", "checks": {}}, {}, smoke=True,
                                 device="cpu", claim=None, speed={"readout": "s", "model": "p", "smoke": True})
    assert (source["instructions"], source["executor"]) == ("4dTrajectory/outputs/POOLED/instruction_language/v",
                                                            "4dTrajectory/outputs/POOLED/executor/v")

