"""Stage C, C4: the window loop (post-training §2 items 1–3, §8 C4; D29–D31, D91, D93, D105, D110) — on the synthetic
artefact of one flight that free generation's tests use (A26's test artefact)."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from flight_scenarios.fas_geometry import fas_course_geometry
from ts_transformer.autopilot.judge import OUTCOMES
from ts_transformer.autopilot.start import start
from ts_transformer.experiments import post_window_loop
from ts_transformer.experiments.post_window_loop import LOST_SEPARATION, WindowLoop
from ts_transformer.experiments.prior_free_generation import speak_and_fly
from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop, flight_numbers
from ts_transformer.instructions.artefact import load_candidates, load_day_split, load_spec, signals_flights
from ts_transformer.instructions.words import COLUMNS
from ts_transformer.post.conformance import reference_steps, write_edge_reference
from ts_transformer.post.landings import roster_key, window_landings
from ts_transformer.post.reward import GO_AROUND_FACTOR, LANDED, present_runways, reward
from ts_transformer.post.scene import INSERTED, INSERTED_SUFFIX, MovedScene, airport_scenes, real_windows
from ts_transformer.post.traffic_attention import TrafficConfig, add_traffic_attention, traffic_modules
from ts_transformer.prior.batch import variant_features
from ts_transformer.prior.inputs import LANDINGS_SCALE
from ts_transformer.prior.landings import Landing, LandingIndex, utc_s
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.procedure import Final
from ts_transformer.prior.speaker import MOST_GO_AROUNDS
from ts_transformer.tests import test_start
from ts_transformer.tests.support import fixture_days, parallel_airport

CPU = torch.device("cpu")
DELTA = 4.0


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A26's one-flight artefact: its window (no other aircraft), its roster (the flight's landing and another one), the
    finals, a prior and an edge reference written for the test."""
    directory, words, _, stored, _ = test_start._artefact(tmp_path, monkeypatch, DELTA)
    flights = {0: signals_flights(directory, "train")[0]}
    geometries = load_candidates(directory)
    geometry = geometries[flights[0]["airport"]]
    key = roster_key(flights[0]["dataset_id"], flights[0]["airport"])
    roster = LandingIndex(tuple(c.ident for c in geometry.candidates),
                          (Landing(utc_s(flights[0]["entry_time_utc"]) + 30.0, geometry.candidates[0].ident, "OTHER"),
                           Landing(utc_s(flights[0]["landing_time_utc"]), geometry.candidates[0].ident, key)), 0,
                          load_day_split(directory))
    finals = {geometry.code: tuple(Final(geometry, k, 9_000.0, fas_course_geometry(c.length_m))
                                   for k, c in enumerate(geometry.candidates))}
    spec = load_spec(directory)
    scenes, signals = airport_scenes(directory, "train", spec, DELTA, geometries)
    windows = real_windows(directory, "train", spec, DELTA, scenes, signals)
    reference = tmp_path / "edges.npz"
    write_edge_reference(reference, reference_steps(windows, np.random.default_rng(1337)), geometries, spec.step_s)
    torch.manual_seed(0)
    base = Prior(PriorConfig.from_words(words, "full", d_model=32, layers=1, heads=4, feedforward=64)).eval()

    def loop():
        return start(directory, "train", DELTA, {0: stored}, tmp_path / "executor", most_go_arounds=MOST_GO_AROUNDS,
                     device=CPU)

    return dict(directory=directory, words=words, stored=stored, flights=flights, geometries=geometries,
                geometry=geometry, roster=roster, finals=finals, windows=windows, reference=reference, base=base,
                loop=loop, key=key)


def _with_module(base, weights=True):
    model = copy.deepcopy(base)
    add_traffic_attention(model, TrafficConfig(hidden=16, heads=4))
    if weights:
        generator = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for module in traffic_modules(model):
                for p in module.parameters():
                    p.copy_(torch.randn(p.shape, generator=generator) * 0.3)
    return model.eval()


def _window_loop(s, model, windows):
    loop, order = s["loop"]()
    return WindowLoop(model, loop, order, windows, {0: s["stored"]}, s["flights"], s["geometries"],
                      {s["geometry"].code: s["roster"]}, s["finals"], s["words"], interval_s=DELTA, variant="full",
                      edges_reference=s["reference"], device=CPU)


def test_a_window_without_other_aircraft_is_free_generation_bit_for_bit(setup):
    """D29, §2 item 1: the same flight, the same random numbers, the same bound — the same words and states, bit for bit,
    whatever the traffic module's weights."""
    s = setup
    assert len(s["windows"]) == 1 and not len(s["windows"][0].others_at(10))
    loop, order = s["loop"]()
    (free,) = speak_and_fly(s["base"], loop, order, {0: s["stored"]}, s["flights"], s["geometries"],
                            {s["geometry"].code: s["roster"]}, s["finals"], s["words"], interval_s=DELTA,
                            variant="full", numbers=[flight_numbers(7, 0, 0)], device=CPU)
    windows = _window_loop(s, _with_module(s["base"]), s["windows"])
    (window,) = windows.run([flight_numbers(7, 0, 0)])
    assert np.array_equal(window.words, free.words) and np.array_equal(window.states, free.states)
    assert window.outcome == free.outcome and window.go_arounds == free.go_arounds and window.loss is None
    assert window.speed_mask_rows == 0
    # the speaker's probabilities too, bit for bit: free generation's loop as `speak_and_fly` runs it
    loop, order = s["loop"]()
    speaking = SpeakingLoop(s["base"], loop, order, {0: s["stored"]}, s["flights"], s["geometries"], [s["roster"]],
                            s["finals"], s["words"], interval_s=DELTA, variant="full", device=CPU)
    numbers = flight_numbers(7, 0, 0)
    while speaking.observing:
        speaking.observe()
    while speaking.alive.any():
        speaking.step(np.stack([numbers.random(len(COLUMNS))]))
    assert np.array_equal(np.stack(speaking.speaker.drawn_probability),
                          np.stack(windows.speaking.speaker.drawn_probability))


def test_a_loss_of_separation_that_the_commanded_aircraft_answers_for_ends_its_window(setup):
    """D93, D30: the flight itself, inserted 8 s ahead (window A), is lost separation with at the first row flown; the
    window ends there with reward 0, and the other aircraft is named."""
    s = setup
    (window,) = s["windows"]
    own = window.scene.flights[0]
    key = own.key + INSERTED_SUFFIX
    ahead = replace(window, kind=INSERTED, moved=((key, -8.0),),
                    scene=MovedScene(window.scene, added=(own.shifted(-8.0, DELTA, key=key),)))
    loop = _window_loop(s, _with_module(s["base"]), [ahead])
    (result,) = loop.run([flight_numbers(7, 0, 0)])
    start_row = s["stored"].rows.start
    assert result.outcome == LOST_SEPARATION and result.reward == 0.0 and result.other == key
    assert result.loss_step == start_row + 1 and 0 in result.loss.responsible
    assert len(result.words) == 1                                   # the one row said before the loss
    assert result.states.shape[0] == (start_row + 1) * loop.every + 1


def test_the_landing_context_counts_the_windows_landings_never_the_commanded_aircrafts_own(setup):
    """D31, D105: at every row said, the commanded aircraft's landings input is its window's landings (an inserted
    aircraft's at its shifted time) without its own landing."""
    s = setup
    (window,) = s["windows"]
    own = window.scene.flights[0]
    key = own.key + INSERTED_SUFFIX
    shift = round((window.row0_s + 30.0 - own.landing_s) / DELTA) * DELTA          # it lands 30 s into the window
    inserted = replace(window, kind=INSERTED, moved=((key, shift),),
                       scene=MovedScene(window.scene, added=(own.shifted(shift, DELTA, key=key),)))
    loop = _window_loop(s, _with_module(s["base"], weights=False), [inserted])
    loop.run([flight_numbers(7, 0, 0)])
    (rows,) = loop.speaking.sentences("train")
    times = window.row0_s + np.arange(len(rows.time_s)) * DELTA
    counts = window_landings(inserted, s["roster"]).counts_before(times, without=s["key"])[:, 0]
    feature = variant_features("full").index("landings_30min")
    assert np.allclose(rows.candidates[:, 0, feature], counts / LANDINGS_SCALE)
    without_insert = s["roster"].counts_before(times, without=s["key"])[:, 0]
    assert (counts == without_insert + (times > own.landing_s + shift)).all()     # the inserted landing, from its time
    assert (counts > without_insert).any()


def test_the_reward_table():
    """§2 item 2, D30."""
    present = np.array([True, False])
    assert reward(LANDED, 0, 0, present, False) == 1.0
    assert reward(LANDED, 0, 2, present, False) == pytest.approx(GO_AROUND_FACTOR ** 2)
    assert reward(LANDED, 1, 0, present, False) == 0.0                   # a runway of another direction
    assert reward(LANDED, 0, 0, present, True) == 0.0                    # a loss of separation
    for other in OUTCOMES[1:]:
        assert reward(other, None, 0, present, False) == 0.0
    assert LANDED == OUTCOMES[0]                                         # the mirror of the judge's literal
    with pytest.raises(ValueError, match="names the runway"):
        reward(LANDED, None, 0, present, False)


def test_the_present_landing_direction():
    """§2 item 2: a runway within 90° of a runway with a landing in the 30 min before the first predicted step, without
    the commanded aircraft's own landing."""
    geometry = parallel_airport()                                        # 09 and 09L, both 090°
    days = fixture_days()
    noon = utc_s(f"{days.days['train'][0]}T12:00:00Z")
    index = LandingIndex(("09", "09L"), (Landing(noon - 600.0, "09L", "x"), Landing(noon + 60.0, "09", "own")), 0, days)
    assert present_runways(index, geometry, noon, "own").tolist() == [True, True]
    assert present_runways(index, geometry, noon - 700.0, "own").tolist() == [False, False]   # none in the 30 min
    alone = LandingIndex(("09", "09L"), (Landing(noon - 60.0, "09", "own"),), 0, days)
    assert present_runways(alone, geometry, noon, "own").tolist() == [False, False]           # its own: not counted


def test_the_window_loop_reads_no_time_limit():
    """Vocabulary D90: the loop's end is the executor's; nothing here names the time limit."""
    source = open(post_window_loop.__file__, encoding="utf-8").read()
    assert "time_limit" not in source and "timed_out" not in source
