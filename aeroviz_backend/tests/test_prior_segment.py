"""The live executor on a sentence of a Training set of stage B (`aeroviz_backend.autopilot_segment.prior`; prior design
§12 B6): a word of the prior's sentence flown live is the export's flight (the readout's states within the executor
conformance's bound, the export's written track within its rounding), the sentence is chosen by the request, the request
is refused or not listed by name, the endpoint maps the errors, and the frontend reads the names the backend writes.
The set is the export's own synthetic one (`test_prior_training_export.stage_b_fixture`)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import aeroviz_backend.autopilot_segment  # noqa: F401 — puts `ts_transformer` (under 4dTrajectory/) on the path
from ts_transformer.autopilot import closed_loop
from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.autopilot.judge import flown_track
from ts_transformer.instructions.words import COLUMNS, HEADING, UNCHANGED
from ts_transformer.instructions import training_files
from ts_transformer.instructions.artefact import SEALED_READINGS
from ts_transformer.prior import training_files as prior_files
from ts_transformer.tests import test_start
from ts_transformer.tests.test_prior_free_generation import generate
from ts_transformer.tests.test_prior_training_export import FIXTURE_SET, FIXTURE_VAL_SET, stage_b_fixture

from aeroviz_backend.autopilot_segment import prior as prior_segments
from ts_transformer.repo_layout import REPO_ROOT

from aeroviz_backend.autopilot_segment.backend import AutopilotSegmentBackend, SetFlown, stage_a_service
from aeroviz_backend.autopilot_segment.errors import NotListed, RequestRefused, Superseded
from aeroviz_backend.autopilot_segment.fly import fly_segment
from aeroviz_backend.http_server import AeroVizBackendApp

CPU = torch.device("cpu")
#: The export writes e, n and height to 0.1 m: a distance of two of them is at most 0.05 m · √2 off.
ROUNDING_M = 0.05 * 2**0.5 + 1e-3
FRONTEND = Path(__file__).resolve().parents[2] / "aeroviz-4d" / "src" / "data"
FIXTURES = FRONTEND / "__tests__" / "fixtures" / "stage_b"


class SyntheticBackend(AutopilotSegmentBackend):
    """The service on the export's synthetic set: its flight is the artefact's synthetic one (no artefact on disk)."""

    def __init__(self, root, flown):
        super().__init__(splits=training_files.SPLITS, airports_root=root)       # stage A's sets' splits (D109)
        self.flown = flown
        val = SyntheticValBackend(root, flown)
        self.prior.val_service = lambda: val        # the validation service is a synthetic one too

    def executor_for(self, sample):
        params, words = test_start._params(), self.flown.words
        return Path("fixture/instruction_language"), Path("fixture/executor"), params, {"sha256": "fixture"}, words

    def set_flown(self, sample, split, interval_s, instructions, params, words, opened=None):
        return self.flown.set


class SyntheticValBackend(SyntheticBackend):
    def __init__(self, root, flown):
        AutopilotSegmentBackend.__init__(self, splits=SEALED_READINGS, airports_root=root)
        self.flown = flown


class Flown:
    pass


@pytest.fixture(autouse=True)
def claim_on_disk(monkeypatch):
    """The prior's run holds the claim the fixture's val set names (the fixture's paths are not on disk)."""
    held = {REPO_ROOT / "fixture/prior": "fixture/readout_val"}
    monkeypatch.setattr(prior_segments, "validation_claim",
                        lambda prior_dir, reader: held[prior_dir] if prior_dir in held else None)
    return held


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The synthetic set written under a tmp airports root, its flight set up on its stored closed-loop sentence, and the
    readouts' unrounded states of the two sentences the set holds (seeds 0 and 1: the same synthetic flight)."""
    tmp = tmp_path_factory.mktemp("prior")
    monkeypatch = pytest.MonkeyPatch()
    index, sample, texts = stage_b_fixture(tmp / "fixture", monkeypatch, texts=True)
    val_sample = json.loads(texts[f"{FIXTURE_VAL_SET}/{prior_files.SAMPLE_FILE}"])
    root = tmp / "airports"
    training = root / sample["airport"] / "training"
    training.mkdir(parents=True)
    prior_files.write_set(training, sample["airport"], index["sets"][0], prior_files.serialise(sample), [])
    prior_files.write_set(training, sample["airport"], index["sets"][1], prior_files.serialise(val_sample),
                          [index["sets"][0]])
    readouts = []
    for seed in (0, 1):
        spy = {}
        generated, stored, words, geometry = generate(tmp / f"again{seed}", monkeypatch, interval_s=4.0, seed=seed, spy=spy)
        readouts.append(generated)
    batch, missing = closed_loop.replay_batch(spy["batch"], {0: stored}, words)
    assert not missing
    flown = Flown()
    flown.words, flown.set = words, SetFlown(batch, [stored], test_start._params())
    monkeypatch.undo()
    return {"root": root, "entry": index["sets"][0], "sample": sample, "val_sample": val_sample, "readouts": readouts, "flown": flown, "geometry": geometry}


def request(world, **changes):
    sample = world["sample"]
    said = sample["flights"][0]["prior"][0]
    event = [e for e in said["events"] if e["column"] == HEADING][0]
    return {"clientId": "page", "seq": 1, "airport": sample["airport"], "setId": FIXTURE_SET,
            "flightKey": sample["flights"][0]["flightKey"], "sentence": 0, "column": COLUMNS[HEADING],
            "row": event["row"], **changes}


def test_every_word_of_a_prior_sentence_flown_live_is_the_exported_flight(world):
    """Outline §6 item 6: the live segment of each word of each sentence of the set is the flight the export flew — the
    readout's states from the first predicted step within the executor conformance's bound (the live flight is the
    unrounded one), the written track within its rounding — and the word flown to its outcome ends as the export says."""
    sample, flown = world["sample"], world["flown"]
    batch, inputs = flown.set.batch, flown.set.inputs
    params = test_start._params()
    for said, readout in zip(sample["flights"][0]["prior"], world["readouts"]):
        reference = readout.states[said["flownFromRow"]:]
        grid = np.array(said["words"], dtype=np.int16)
        told, sentence = prior_segments.on_words(batch, 0, flown.set.sentences[0], grid, flown.words)
        last = {int(column): int(row) for row, column in zip(*np.nonzero(grid != UNCHANGED))}
        for row, column in zip(*np.nonzero(grid != UNCHANGED)):
            result = fly_segment(told, inputs, 0, sentence, int(column), int(row), params, flown.words, lambda: False)
            end = int(result.flown.done_cycle[0]) + 2
            live = flown_track(result.flown.states[0, :end].numpy(), world["geometry"])
            rows = min(len(reference), (end - 1) // every_cycles(flown, params) + 1)
            cycles = np.arange(rows) * every_cycles(flown, params)
            for name, index in (("e", 0), ("n", 1), ("height", 2)):
                assert float(np.abs(live[name][cycles] - reference[:rows, index]).max()) < STATE_BOUND_M, (said["sample"], row, column, name)
            apart = prior_segments.apart_from_exported(result, said["track"], told, 0, flown.words.spec.step_s)
            assert max(apart["horizontalM"], apart["verticalM"]) < ROUNDING_M
            # a column's last word has no stop: it is flown on to the judge's outcome (an earlier word is too, when its
            # stop lies past the flight's end)
            assert last[int(column)] != int(row) or result.verdict is not None, (said["sample"], row, column)
            if result.verdict is not None:
                assert result.verdict.outcome == said["outcome"] and int(result.verdict.end_row) == said["endCycle"]


def every_cycles(flown, params):
    return int(round(flown.words.spec.step_s / params.cycle_s))


def test_a_request_flies_the_sentence_it_names_and_says_what_it_flew(world):
    backend = SyntheticBackend(world["root"], world["flown"])
    answer = backend.prior.fly(request(world))
    assert (answer["schema"], answer["setId"], answer["sentence"], answer["rowIntervalS"]) == (
        prior_segments.SCHEMA, FIXTURE_SET, 0, 4.0)
    assert answer["segment"]["column"] == HEADING and answer["stored"]["horizontalM"] < ROUNDING_M
    other = backend.prior.fly(request(world, sentence=1, seq=2, row=world["sample"]["flights"][0]["prior"][1]["events"][1]["row"],
                                      column=COLUMNS[world["sample"]["flights"][0]["prior"][1]["events"][1]["column"]]))
    assert other["sentence"] == 1 and other["stored"]["verticalM"] < ROUNDING_M
    closed = world["sample"]["flights"][0]["closedLoop"]["4"]["events"]
    event = [e for e in closed if e["row"] > 0][0]
    answer = backend.prior.fly(request(world, sentence=prior_segments.CLOSED_LOOP, seq=3, row=event["row"],
                                       column=COLUMNS[event["column"]]))
    assert answer["sentence"] == prior_segments.CLOSED_LOOP and answer["stored"]["horizontalM"] < STATE_BOUND_M


class SetOpened(Exception):
    """What opening the set would raise: here, the sign that it was reached."""


class CheckedBackend(SyntheticBackend):
    def executor_for(self, sample):
        raise SetOpened("the set's executor spec was opened")


def write_extra_set(world, name, sample):
    """``sample`` as set ``name`` of its own airports root (beside the world's, whose files stay as they are)."""
    root = world["root"].parent / name
    training = root / sample["airport"] / "training"
    training.mkdir(parents=True)
    entry = {**world["entry"], "id": name, "file": f"{name}/sample.json",
             "cohort": sample["cohort"], "source": sample["source"]}
    prior_files.write_set(training, sample["airport"], entry, prior_files.serialise({**sample, "setId": name}), [])
    return root


def test_the_claimed_validation_set_flies_its_val_flights(world):
    """D109: the set exported from the base's one validation readout holds val flights and the service, built with
    stage A's splits, flies them; the set's splits are the sealed readings alone."""
    backend = SyntheticBackend(world["root"], world["flown"])
    val = world["val_sample"]
    assert val["source"]["validationClaim"] is not None and val["cohort"]["split"] in SEALED_READINGS
    assert {item["split"] for item in val["flights"]} == set(SEALED_READINGS)
    assert backend.prior.splits_of(val) == SEALED_READINGS and not set(SEALED_READINGS) & set(backend.splits)
    assert set(backend.prior.splits_of(world["sample"])) == set(backend.splits)
    answer = backend.prior.fly(request(world, setId=FIXTURE_VAL_SET, flightKey=val["flights"][0]["flightKey"]))
    assert answer["setId"] == FIXTURE_VAL_SET and answer["segment"]["column"] == HEADING


def test_a_val_flight_of_an_unclaimed_set_is_refused_before_any_executor_check(world):
    """An unclaimed set's val flight, and a claim on a set that is not val (and a val set with no claim), are refused by
    name; the split is refused before the set is opened (`executor_for`: this backend's raises)."""
    train = world["sample"]
    flight = {**train["flights"][0], "split": SEALED_READINGS[0]}
    stray = write_extra_set(world, "stray", {**train, "flights": [flight]})
    backend = CheckedBackend(stray, world["flown"])
    with pytest.raises(RequestRefused, match="'val'"):
        backend.prior.fly(request(world, setId="stray"))
    with pytest.raises(SetOpened):             # a train flight of the unclaimed set does reach the check
        CheckedBackend(world["root"], world["flown"]).prior.fly(request(world, clientId="train"))
    claim = world["val_sample"]["source"]["validationClaim"]
    claimed = write_extra_set(world, "claimed_train", {**train, "source": {**train["source"], "validationClaim": claim}})
    with pytest.raises(RequestRefused, match="holds a validation claim but its cohort.split is 'train'"):
        CheckedBackend(claimed, world["flown"]).prior.fly(request(world, setId="claimed_train"))
    val = world["val_sample"]
    bare = write_extra_set(world, "bare_val", {**val, "source": {**val["source"], "validationClaim": None}})
    with pytest.raises(RequestRefused, match="holds no validationClaim"):
        CheckedBackend(bare, world["flown"]).prior.fly(request(world, setId="bare_val", flightKey=val["flights"][0]["flightKey"]))


def test_a_claim_the_disk_does_not_hold_or_a_forged_claim_is_refused_by_name(world, claim_on_disk):
    val = world["val_sample"]
    flight = val["flights"][0]["flightKey"]
    backend = CheckedBackend(world["root"], world["flown"])
    held = dict(claim_on_disk)
    claim_on_disk.clear()                                            # the prior's run holds no claim
    with pytest.raises(RequestRefused, match="holds no claim of"):
        backend.prior.fly(request(world, setId=FIXTURE_VAL_SET, flightKey=flight, clientId="disk"))
    claim_on_disk[REPO_ROOT / "fixture/prior"] = REPO_ROOT / "fixture/another_readout"      # another readout's claim
    with pytest.raises(RequestRefused, match="holds no claim of"):
        backend.prior.fly(request(world, setId=FIXTURE_VAL_SET, flightKey=flight, clientId="other"))
    claim_on_disk.update(held)
    for field, message in (("reader", "validationClaim.reader"), ("prior", "validationClaim.prior"),
                           ("readout", "validationClaim.readout")):
        forged = {**val, "source": {**val["source"], "validationClaim": {**val["source"]["validationClaim"], field: "forged"}}}
        root = write_extra_set(world, f"forged_{field}", forged)
        with pytest.raises(RequestRefused, match=message):
            CheckedBackend(root, world["flown"]).prior.fly(request(world, setId=f"forged_{field}", flightKey=flight, clientId=field))


def test_the_production_validation_service_flies_the_sealed_readings_and_is_built_once(tmp_path):
    prior = stage_a_service(tmp_path).prior
    service = prior.val_service()
    assert tuple(service.splits) == SEALED_READINGS and prior.val_service() is service


def test_a_request_the_view_cannot_make_is_refused_and_a_set_flight_or_sentence_not_listed_is_not_listed(world):
    backend = SyntheticBackend(world["root"], world["flown"])
    for changes, message in [({"column": "flaps"}, "none of"), ({"row": -1}, "whole number"),
                             ({"sentence": -1}, "whole number"), ({"sentence": "x"}, "whole number"),
                             ({"airport": "../x"}, "not an airport code")]:
        with pytest.raises(RequestRefused, match=message):
            backend.prior.fly({**request(world, clientId=str(changes)), **changes})
    with pytest.raises(RequestRefused, match="no 'sentence'"):
        asked = request(world, clientId="a")
        del asked["sentence"]
        backend.prior.fly(asked)
    said = world["sample"]["flights"][0]["prior"][0]
    with pytest.raises(RequestRefused, match="after the flight's outcome"):       # a word said after the outcome has no segment
        backend.prior.fly(request(world, clientId="f", row=said["track"]["lastCycle"] // 4 + 1))
    # an in-sentence word said after the outcome: the set read with the sentence's outcome at cycle 40
    late = {**world["sample"], "setId": "late"}
    late["flights"] = [{**late["flights"][0], "prior": [
        {**said, "track": {**said["track"], "lastCycle": 40}}, *late["flights"][0]["prior"][1:]]}]
    training = world["root"].parent / "late_airports" / late["airport"] / "training"
    if not training.exists():
        training.mkdir(parents=True)
        prior_files.write_set(training, late["airport"], {**world["entry"], "id": "late", "file": "late/sample.json"},
                              prior_files.serialise(late), [])
    behind = SyntheticBackend(world["root"].parent / "late_airports", world["flown"])
    after = next(e for e in said["events"] if e["row"] * 4 > 40 and e["column"] == HEADING)
    with pytest.raises(RequestRefused, match="after the flight's outcome at cycle 40"):
        behind.prior.fly(request(world, setId="late", clientId="g", row=after["row"]))
    with pytest.raises(NotListed, match="has no flight"):
        backend.prior.fly(request(world, flightKey="nobody", clientId="b"))
    with pytest.raises(NotListed, match="no prior sentence 7"):
        backend.prior.fly(request(world, sentence=7, clientId="c"))
    with pytest.raises(NotListed, match="lists no set"):
        backend.prior.fly(request(world, setId="another", clientId="d"))
    backend.prior.fly(request(world, clientId="e", seq=2))
    with pytest.raises(Superseded):
        backend.prior.fly(request(world, clientId="e", seq=1))


def test_the_warm_up_opens_every_listed_prior_set_and_says_what_it_skipped(world, tmp_path):
    lines = []
    SyntheticBackend(world["root"], world["flown"]).prior.warm_up(lines.append)
    assert any("fixture_set" in line and "flights opened" in line for line in lines) and "2 sets ready" in lines[-1]
    broken = tmp_path / "KXXX" / "training"
    broken.mkdir(parents=True)
    (broken / prior_files.INDEX_FILE).write_text(json.dumps({"schema": "aeroviz-training-prior-index-v0", "airport": "KXXX",
                                                             "sets": []}))
    lines = []
    SyntheticBackend(tmp_path, world["flown"]).prior.warm_up(lines.append)
    assert "skipped" in lines[0] and prior_files.INDEX_SCHEMA in lines[0] and "0 sets ready" in lines[-1]


class LockWatched(SyntheticBackend):
    """Records, at each opening of a set, whether the request lock is held."""

    def set_flown(self, sample, split, interval_s, instructions, params, words, opened=None):
        self.held.append(self._lock.locked())
        return super().set_flown(sample, split, interval_s, instructions, params, words, opened)


def test_the_backend_warm_up_opens_the_prior_sets_without_the_request_lock(world):
    """A43: a set is opened under its own lock, never the request lock, so a request waits only for the set it needs —
    the prior's sets too; the backend's warm-up opens them after stage A's."""
    backend = LockWatched(world["root"], world["flown"])
    backend.held = []
    lines = []
    backend.warm_up(lines.append)
    assert backend.held and not any(backend.held)
    assert any(line.startswith("prior warm-up:") and "sets ready" in line for line in lines)


class FakeAutopilot:
    def __init__(self, error=None):
        self.error, self.calls = error, []
        self.prior = self

    def fly(self, payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return {"ok": True}


def test_the_prior_endpoint_delegates_and_maps_the_errors_as_the_segment_endpoint_does():
    def post(autopilot):
        app = AeroVizBackendApp(simulation_backend=object(), optimization_backend=object(),
                                dynamics_comparison_backend=object(), observed_trajectory_backend=object(),
                                autopilot_segment_backend=autopilot)
        return app.handle_post("/autopilot/prior-segment", {"row": 3})[:2]

    ok = FakeAutopilot()
    assert post(ok) == (200, {"ok": True}) and ok.calls == [{"row": 3}]
    assert post(FakeAutopilot(RequestRefused("no 'row'"))) == (400, {"ok": False, "error": "no 'row'"})
    assert post(FakeAutopilot(NotListed("no set"))) == (404, {"ok": False, "error": "no set"})
    assert post(FakeAutopilot(Superseded("newer"))) == (409, {"ok": False, "error": "newer"})
    assert post(FakeAutopilot(ValueError("x"))) == (500, {"ok": False, "error": "ValueError: x"})


def ts_constant(path, name):
    import re
    match = re.search(rf'export const {name} = "([^"]*)"', path.read_text(encoding="utf-8"))
    assert match is not None, f"{path.name} has no string constant {name}"
    return match.group(1)


def test_the_frontend_reads_the_names_the_backend_and_the_export_write():
    sample, autopilot = FRONTEND / "trainingPriorSample.ts", FRONTEND / "trainingPriorAutopilot.ts"
    assert ts_constant(sample, "TRAINING_PRIOR_INDEX_SCHEMA") == prior_files.INDEX_SCHEMA
    assert ts_constant(sample, "TRAINING_PRIOR_INDEX_FILE") == prior_files.INDEX_FILE
    assert ts_constant(sample, "TRAINING_PRIOR_SAMPLE_SCHEMA") == prior_files.SAMPLE_SCHEMA
    assert ts_constant(sample, "TRAINING_PRIOR_SET_KIND") == prior_files.SET_KIND
    assert ts_constant(autopilot, "TRAINING_PRIOR_AUTOPILOT_SCHEMA") == prior_segments.SCHEMA
    assert ts_constant(autopilot, "TRAINING_PRIOR_CLOSED_LOOP") == prior_segments.CLOSED_LOOP
    # the columns the procedure masks rule (`ProcedureMasks.columns`), by name
    from ts_transformer.prior.procedure import ProcedureMasks
    import re
    listed = re.search(r"export const TRAINING_PRIOR_BLOCKED_COLUMNS = \[([^\]]*)\]", sample.read_text(encoding="utf-8"))
    assert re.findall(r'"([^"]*)"', listed.group(1)) == [COLUMNS[c] for c in ProcedureMasks.columns]


def test_the_frontend_fixture_of_an_answer_is_what_the_service_answers(world):
    """Two answers on the set's first prior sentence, written by the service: a heading word stopped at its stop and the
    column's last heading word flown to its outcome — wall-clock fields fixed. ``AEROVIZ_WRITE_FIXTURES=1`` writes it."""
    backend = SyntheticBackend(world["root"], world["flown"])
    sample = world["sample"]
    heading = [e["row"] for e in sample["flights"][0]["prior"][0]["events"] if e["column"] == HEADING]
    answers = []
    for seq, row in enumerate((heading[1], heading[-1]), start=1):
        answer = backend.prior.fly(request(world, clientId="fixture", seq=seq, row=row))
        answer["computedUtc"] = "fixture"
        answer["timing"] = {name: 0 for name in answer["timing"]}
        answers.append(answer)
    text = json.dumps(answers, separators=(",", ":"), allow_nan=False) + "\n"
    path = FIXTURES / "autopilot_prior_segment.json"
    if os.environ.get("AEROVIZ_WRITE_FIXTURES") == "1":
        path.write_text(text, encoding="utf-8")
    assert path.read_text(encoding="utf-8") == text, (
        f"{path} is not what the service answers now: AEROVIZ_WRITE_FIXTURES=1 writes it again")
