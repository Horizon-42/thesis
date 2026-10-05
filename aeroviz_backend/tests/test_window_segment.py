"""The live executor on a window of a Training set of stage C (`aeroviz_backend.autopilot_segment.window`; post-training
§8 C11): a word of the commanded aircraft's sentence flown live is the export's flight — window B's from its moved
start — (the export's unrounded states within the executor conformance's bound, its written track within its rounding),
the window and round are chosen by the request, a word after the window's end is refused, the endpoint maps the errors,
and the frontend reads the names the backend writes. The set is the export's own synthetic one
(`test_post_training_export.stage_c_fixture`)."""

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

#: The export writes e, n and height to 0.1 m: a distance of two of them is at most 0.05 m · √2 off.
ROUNDING_M = 0.05 * 2**0.5 + 1e-3
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


def request(world, window=0, **changes):
    said = world["sample"]["windows"][window]["rounds"][0]
    event = said["events"][0] if said["events"] else {"column": 0, "row": 0}
    return {"clientId": "page", "seq": 1, "airport": world["sample"]["airport"], "setId": FIXTURE_SET,
            "window": window, "round": "start", "column": COLUMNS[event["column"]], "row": event["row"], **changes}


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
        (said,) = window["rounds"]
        reference = states[said["flownFromRow"]:]
        batch, inputs = flown.batch, flown.inputs
        if window_segments.is_moved(window["startMove"]):
            batch, inputs = window_segments.moved_start(batch, 0, window["startMove"], params.start_rule)
        grid = np.array(said["words"], dtype=np.int16)
        told, sentence = on_words(batch, 0, flown.sentences[0], grid, words)
        for row, column in zip(*np.nonzero(grid != UNCHANGED)):
            lost = said["end"]["loss"] is not None
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
            apart = apart_from_exported(result, said["track"], told, 0, words.spec.step_s)
            assert max(apart["horizontalM"], apart["verticalM"]) < ROUNDING_M
    assert kinds == ["real", "A", "B"]
    # window B's set carries its moved observed rows (the view draws them before the first predicted step)
    moved = world["sample"]["windows"][2]["movedStart"]
    stored = s["stored"].rows.states[: moved["rows"]]
    assert moved["rows"] == world["sample"]["windows"][2]["rounds"][0]["flownFromRow"]
    assert not np.allclose(moved["eM"], stored[:, 0], atol=1.0)
    assert np.allclose(moved["heightMslM"], world["states"][2][: moved["rows"], 2], atol=0.051)


def test_a_request_flies_the_window_and_round_it_names(world):
    backend = SyntheticBackend(world["root"], world["flown"], world["setup"]["words"])
    for place in range(len(world["sample"]["windows"])):
        answer = backend.window.fly(request(world, window=place, seq=place + 1))
        assert (answer["schema"], answer["setId"], answer["window"], answer["round"], answer["rowIntervalS"]) == (
            window_segments.SCHEMA, FIXTURE_SET, place, "start", 4.0)
        assert max(answer["stored"]["horizontalM"], answer["stored"]["verticalM"]) < ROUNDING_M


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
    lost = world["sample"]["windows"][1]["rounds"][0]
    with pytest.raises(RequestRefused, match="after the window's end"):           # the lost window ends at its loss
        backend.window.fly(request(world, window=1, seq=5, row=lost["track"]["lastCycle"] // 4 + 1))


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
    assert any("1 sets ready" in line for line in lines) and post_files.INDEX_FILE == "index_post_v1.json"


def test_a_lost_window_s_last_word_ends_at_the_loss(world):
    """A column's last word is flown to the outcome in a window the executor ended; in a window ended at a loss of
    separation it stops at the window's last state, unjudged, and the answer carries the window's end."""
    backend = SyntheticBackend(world["root"], world["flown"], world["setup"]["words"])
    lost = world["sample"]["windows"][1]["rounds"][0]
    grid = np.array(lost["words"])
    said = [(int(r), int(c)) for r, c in zip(*np.nonzero(grid != UNCHANGED))]
    if not said:
        pytest.skip("the lost window's one row said no word")
    row, column = said[-1]
    answer = backend.window.fly(request(world, window=1, seq=9, row=row, column=COLUMNS[column]))
    assert answer["windowEnd"]["loss"]["other"].endswith(INSERTED_SUFFIX) and answer["crossing"] is None
    assert answer["segment"]["end"] == SEGMENT_END and answer["segment"]["endCycle"] == lost["track"]["lastCycle"]
