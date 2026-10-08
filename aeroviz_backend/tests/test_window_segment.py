"""The live executor on a window of a Training set of stage C (`aeroviz_backend.autopilot_segment.window`; post-training
§8 C11): a word of the commanded aircraft's sentence flown live is the export's flight — window B's from its moved
start — (the export's unrounded states within the executor conformance's bound, its written track within its rounding),
the window, its commanded aircraft and the round are chosen by the request (an aircraft the window does not command is
refused by name), a word after the window's end is refused, the endpoint maps the errors,
and the frontend reads the names the backend writes. The set is the export's own synthetic one
(`test_post_training_export.stage_c_fixture`)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import aeroviz_backend.autopilot_segment  # noqa: F401 — puts `ts_transformer` (under 4dTrajectory/) on the path
from ts_transformer.autopilot import closed_loop
from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.autopilot.judge import flown_track
from ts_transformer.instructions import training_files
from ts_transformer.instructions.words import COLUMNS, UNCHANGED
from ts_transformer.post import training_files as post_files
from ts_transformer.post.scene import INSERTED_SUFFIX
from ts_transformer.tests import test_start
from ts_transformer.tests.test_post_training_export import FIXTURE_SET, stage_c_fixture

from aeroviz_backend.autopilot_segment import window as window_segments
from aeroviz_backend.autopilot_segment.backend import AutopilotSegmentBackend, SetFlown
from aeroviz_backend.autopilot_segment.errors import NotListed, RequestRefused, Superseded
from aeroviz_backend.autopilot_segment.fly import last_state
from aeroviz_backend.autopilot_segment.payload import SEGMENT_END
from aeroviz_backend.autopilot_segment.prior import apart_from_exported, on_words
from aeroviz_backend.http_server import AeroVizBackendApp

FRONTEND = Path(__file__).resolve().parents[2] / "aeroviz-4d" / "src" / "data"


class SyntheticBackend(AutopilotSegmentBackend):
    """The service on the export's synthetic set: its flight is the artefact's synthetic one (no artefact on disk)."""

    def __init__(self, root, flown, words):
        super().__init__(splits=training_files.SPLITS, airports_root=root)       # stage A's sets' splits (D109)
        self.flown, self.words = flown, words

    def executor_for(self, sample):
        return (Path("fixture/instruction_language"), Path("fixture/executor"), test_start._params(),
                {"sha256": "fixture"}, self.words)

    def set_flown(self, sample, split, interval_s, instructions, params, words, opened=None):
        return self.flown


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The synthetic window set written under a tmp airports root, its flight set up on its stored closed-loop sentence,
    and each window's unrounded states. The synthetic flight's stand-ins (`window_setup`) stay in place while the
    module's tests fly: a synthetic flight has no data-plane series to start from."""
    tmp = tmp_path_factory.mktemp("window")
    monkeypatch = pytest.MonkeyPatch()
    fixture = stage_c_fixture(tmp / "fixture", monkeypatch)
    s = fixture["setup"]
    batch, missing = closed_loop.replay_batch(s["batch"], {0: s["stored"]}, s["words"])
    assert not missing
    every = int(round(4.0 / s["words"].spec.step_s))
    assert batch.sentences[0].first_row == s["stored"].rows.first_row + s["stored"].rows.start * every   # its first step
    fixture["flown"] = SetFlown(batch, [s["stored"]], test_start._params())
    # the live executor's start of a moved flight (`window.moved_start`: `Batch.inputs` on its moved rows): A26's
    # stand-in gives the artefact's flight whatever it is given, so here it reads the batch's own rows
    from ts_transformer.autopilot import replay
    from ts_transformer.tests.support import executor_inputs

    monkeypatch.setattr(replay.Batch, "inputs", lambda self, rule, device:
                        executor_inputs(self.observed[0], self.geometries[0], self.sentences[0].first_row, rule=rule))
    yield fixture
    monkeypatch.undo()


def aircraft_of(world, window):
    """The window's one commanded aircraft (stage C's windows command one, frontend §5.7)."""
    (aircraft,) = world["sample"]["windows"][window]["commanded"]
    return aircraft


def request(world, window=0, **changes):
    aircraft = aircraft_of(world, window)
    said = aircraft["rounds"][0]
    event = said["events"][0] if said["events"] else {"column": 0, "row": 0}
    return {"clientId": "page", "seq": 1, "stage": "C", "airport": world["sample"]["airport"], "setId": FIXTURE_SET,
            "window": window, "aircraft": aircraft["datasetId"], "round": "start", "column": COLUMNS[event["column"]],
            "row": event["row"], **changes}


def every_cycles(params, words):
    return int(round(words.spec.step_s / params.cycle_s))


def test_every_word_of_every_window_flown_live_is_the_exported_flight(world):
    """Outline §6 item 6: the live segment of each word of each window's sentence — window B's from its moved start —
    is the flight the export flew: its unrounded states from the first predicted step within the executor conformance's
    bound, the written track within its rounding, on the rows of the window (to its end: its outcome, or the end of the
    row of its loss)."""
    s, flown = world["setup"], world["flown"]
    params, words = test_start._params(), s["words"]
    kinds = []
    for window, states in zip(world["sample"]["windows"], world["states"]):
        kinds.append(window["kind"])
        (aircraft,) = window["commanded"]
        (said,) = aircraft["rounds"]
        reference = states[said["flownFromRow"]:]
        batch, inputs = flown.batch, flown.inputs
        if window_segments.is_moved(aircraft["startMove"]):
            batch, inputs = window_segments.moved_start(batch, 0, aircraft["startMove"], params.start_rule)
        grid = np.array(said["words"], dtype=np.int16)
        told, sentence = on_words(batch, 0, flown.sentences[0], grid, words)
        for row, column in zip(*np.nonzero(grid != UNCHANGED)):
            lost = said["outcome"] == window_segments.LOST_SEPARATION
            result = window_segments.fly_window_segment(told, inputs, 0, sentence, int(column), int(row), params, words,
                                                        lambda: False, said["track"]["lastCycle"] if lost else None)
            if lost:                                   # ended at the window's end, its loss: never past it, unjudged
                assert result.verdict is None and last_state(result) <= said["track"]["lastCycle"]
            end = int(result.flown.done_cycle[0]) + 2
            live = flown_track(result.flown.states[0, :end].numpy(), s["geometry"])
            rows = min(len(reference), (end - 1) // every_cycles(params, words) + 1)
            cycles = np.arange(rows) * every_cycles(params, words)
            for name, index in (("e", 0), ("n", 1), ("height", 2)):
                assert float(np.abs(live[name][cycles] - reference[:rows, index]).max()) < STATE_BOUND_M, (
                    window["kind"], row, column, name)
            apart = apart_from_exported(result, said, told, 0, words.spec.step_s)        # refused past the bound
            assert max(apart["horizontalM"], apart["verticalM"]) < STATE_BOUND_M
    assert kinds == ["real", "A", "B"]
    # window B's set carries its moved observed rows (the view draws them before the first predicted step)
    moved = aircraft_of(world, 2)["movedStart"]
    stored = s["stored"].rows.states[: moved["rows"]]
    assert moved["rows"] == aircraft_of(world, 2)["rounds"][0]["flownFromRow"]
    assert not np.allclose(moved["eM"], stored[:, 0], atol=1.0)
    assert np.allclose(moved["heightMslM"], world["states"][2][: moved["rows"], 2], atol=0.051)


def test_a_request_flies_the_window_and_round_it_names(world):
    backend = SyntheticBackend(world["root"], world["flown"], world["setup"]["words"])
    for place in range(len(world["sample"]["windows"])):
        answer = backend.window.fly(request(world, window=place, seq=place + 1))
        assert (answer["schema"], answer["setId"], answer["window"], answer["round"], answer["rowIntervalS"]) == (
            window_segments.SCHEMA, FIXTURE_SET, place, "start", 4.0)
        assert answer["aircraft"] == answer["datasetId"] == aircraft_of(world, place)["datasetId"]
        assert max(answer["stored"]["horizontalM"], answer["stored"]["verticalM"]) < STATE_BOUND_M


def test_a_request_is_refused_or_not_listed_by_name(world):
    backend = SyntheticBackend(world["root"], world["flown"], world["setup"]["words"])
    with pytest.raises(NotListed, match="no window 9"):
        backend.window.fly({**request(world), "window": 9})
    with pytest.raises(NotListed, match="no round 3"):
        backend.window.fly(request(world, round=3, seq=2))
    with pytest.raises(RequestRefused, match="whole number"):
        backend.window.fly(request(world, round="first", seq=3))
    with pytest.raises(NotListed):
        backend.window.fly(request(world, setId="other", seq=4))
    with pytest.raises(NotListed, match="commands no aircraft 'KXXX:nobody'"):    # not one of the window's
        backend.window.fly(request(world, aircraft="KXXX:nobody", seq=6))
    with pytest.raises(RequestRefused, match="aircraft"):                            # the request must name it
        backend.window.fly({key: value for key, value in request(world, seq=7).items() if key != "aircraft"})
    lost = aircraft_of(world, 1)["rounds"][0]
    with pytest.raises(RequestRefused, match="after the window's end"):           # the lost window ends at its loss
        backend.window.fly(request(world, window=1, seq=8, row=lost["track"]["lastCycle"] // 4 + 1))
    with pytest.raises(RequestRefused, match="stage"):                               # and the stage whose index lists it
        backend.window.fly({key: value for key, value in request(world, seq=9).items() if key != "stage"})
    with pytest.raises(RequestRefused, match="stage 'B' is none of"):
        backend.window.fly(request(world, stage="B", seq=10))
    with pytest.raises(NotListed):                                                   # stage D's index lists no such set
        backend.window.fly(request(world, stage="D", seq=11))


class FakeAutopilot:
    def __init__(self, error=None):
        self.error, self.calls = error, []
        self.window = self

    def fly(self, payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return {"ok": True}


def test_the_window_endpoint_delegates_and_maps_the_errors_as_the_segment_endpoint_does():
    def post(autopilot):
        app = AeroVizBackendApp(simulation_backend=object(), optimization_backend=object(),
                                dynamics_comparison_backend=object(), observed_trajectory_backend=object(),
                                autopilot_segment_backend=autopilot)
        return app.handle_post("/autopilot/window-segment", {"row": 3})[:2]

    ok = FakeAutopilot()
    assert post(ok) == (200, {"ok": True}) and ok.calls == [{"row": 3}]
    assert post(FakeAutopilot(RequestRefused("no 'row'"))) == (400, {"ok": False, "error": "no 'row'"})
    assert post(FakeAutopilot(NotListed("no set"))) == (404, {"ok": False, "error": "no set"})
    assert post(FakeAutopilot(Superseded("newer"))) == (409, {"ok": False, "error": "newer"})
    assert post(FakeAutopilot(ValueError("x"))) == (500, {"ok": False, "error": "ValueError: x"})


def test_the_warm_up_opens_every_listed_window_set(world):
    backend = SyntheticBackend(world["root"], world["flown"], world["setup"]["words"])
    lines = []
    backend.window.warm_up(lines.append)
    assert any("1 sets ready" in line for line in lines) and post_files.INDEX_FILE == "index_post_v3.json"


def test_a_set_of_stage_d_is_found_under_stage_d_s_index_and_warmed_up(world, tmp_path):
    """Frontend §5.5, §5.7: stage D's sets are listed in ``index_multi_v1.json``, one format with stage C's; a request
    naming stage D flies a set that index lists (here stage C's set listed there too), and the warm-up opens both."""
    import shutil

    root = tmp_path / "airports"
    shutil.copytree(world["root"], root)
    training = root / world["sample"]["airport"] / "training"
    shutil.copy(training / post_files.INDEX_FILE, training / post_files.MULTI_INDEX_FILE)
    backend = SyntheticBackend(root, world["flown"], world["setup"]["words"])
    answer = backend.window.fly(request(world, stage="D"))
    assert answer["schema"] == window_segments.SCHEMA and answer["setId"] == FIXTURE_SET
    lines = []
    backend.window.warm_up(lines.append)
    assert any("2 sets ready" in line for line in lines)


def test_a_lost_window_s_last_word_ends_at_the_loss(world):
    """A column's last word is flown to the outcome in a window the executor ended; in a window ended at a loss of
    separation it stops at the window's last state, unjudged, and the answer carries the window's end."""
    backend = SyntheticBackend(world["root"], world["flown"], world["setup"]["words"])
    lost = aircraft_of(world, 1)["rounds"][0]
    grid = np.array(lost["words"])
    said = [(int(r), int(c)) for r, c in zip(*np.nonzero(grid != UNCHANGED))]
    if not said:
        pytest.skip("the lost window's one row said no word")
    row, column = said[-1]
    answer = backend.window.fly(request(world, window=1, seq=9, row=row, column=COLUMNS[column]))
    (loss,) = answer["windowEnd"]["losses"]
    assert loss["aircraft"][1].endswith(INSERTED_SUFFIX) and loss["answering"] == [answer["aircraft"]]
    assert answer["reward"] == 0.0 and answer["crossing"] is None
    assert answer["segment"]["end"] == SEGMENT_END and answer["segment"]["endCycle"] == lost["track"]["lastCycle"]


FIXTURES = FRONTEND / "__tests__" / "fixtures" / "stage_c"


def window_answers(world):
    """The live answers the frontend's reader is tested on: the real window's first word (its segment stopped at the next
    word of its column, or flown to its outcome) and the lost window's last word (stopped at its loss), as the service
    answers them (the times written as fixed names)."""
    backend = SyntheticBackend(world["root"], world["flown"], world["setup"]["words"])
    out = []
    for seq, place in enumerate((0, 1), start=1):
        said = aircraft_of(world, place)["rounds"][0]
        grid = np.array(said["words"])
        cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(grid != UNCHANGED))]
        if not cells:
            continue
        row, column = cells[0] if place == 0 else cells[-1]
        answer = backend.window.fly(request(world, window=place, seq=seq, row=row, column=COLUMNS[column]))
        out.append({**answer, "computedUtc": "fixture", "timing": {key: 0 for key in answer["timing"]}})
    return out


def test_the_frontend_fixtures_of_the_live_answers_are_what_the_service_answers(world):
    """The answers the frontend's reader is tested on are the service's own today (``AEROVIZ_WRITE_FIXTURES=1`` writes them
    again after a change)."""
    text = json.dumps(window_answers(world), separators=(",", ":"), allow_nan=False)
    path = FIXTURES / "autopilot_window_segment.json"
    if os.environ.get("AEROVIZ_WRITE_FIXTURES") == "1":
        path.write_text(text, encoding="utf-8")
    assert path.read_text(encoding="utf-8") == text, f"{path} is not what the service answers now: AEROVIZ_WRITE_FIXTURES=1"


def ts_constant(path, name):
    import re
    match = re.search(rf'export const {name} = "([^"]*)"', path.read_text(encoding="utf-8"))
    assert match is not None, f"{path.name} has no string constant {name}"
    return match.group(1)


def test_the_frontend_reads_the_names_the_backend_and_the_export_write():
    import re

    from ts_transformer.experiments import post_training_export
    from ts_transformer.experiments.post_window_loop import LOST_SEPARATION
    from ts_transformer.post.scene import WINDOW_KINDS

    sample, autopilot = FRONTEND / "trainingWindowSample.ts", FRONTEND / "trainingWindowAutopilot.ts"
    assert ts_constant(sample, "TRAINING_WINDOW_INDEX_SCHEMA") == post_files.INDEX_SCHEMA
    files = re.search(r"TRAINING_WINDOW_INDEX_FILES: Record<TrainingWindowStage, string> = \{([^}]*)\}",
                      sample.read_text(encoding="utf-8"))
    assert dict(re.findall(r'(\w+): "([^"]*)"', files.group(1))) == {
        stage: files_of.index_file for stage, files_of in window_segments.STAGE_FILES.items()}
    assert ts_constant(sample, "TRAINING_WINDOW_SAMPLE_SCHEMA") == post_files.SAMPLE_SCHEMA
    assert ts_constant(sample, "TRAINING_WINDOW_SET_KIND") == post_files.SET_KIND
    assert ts_constant(sample, "TRAINING_WINDOW_START") == post_training_export.START == window_segments.START
    assert ts_constant(autopilot, "TRAINING_WINDOW_AUTOPILOT_SCHEMA") == window_segments.SCHEMA
    assert ts_constant(FRONTEND / "trainingSample.ts", "TRAINING_LOST_SEPARATION") == LOST_SEPARATION
    kinds = re.search(r"export const TRAINING_WINDOW_KINDS = \[([^\]]*)\]", sample.read_text(encoding="utf-8"))
    assert re.findall(r'"([^"]*)"', kinds.group(1)) == list(WINDOW_KINDS)
