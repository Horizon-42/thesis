"""The multi-aircraft post-training in windows (`experiments/traffic_window_tuner`, multi-aircraft design §6.6 step 7,
the 7.6 plan): a window's words scored whole give back the distributions they were sampled from — several commanded
aircraft, replayed ones, a traffic attention that reads the others — and each aircraft's rows are rebuilt with the landing
context each was encoded with. On the scene-data fixture."""

from __future__ import annotations

import numpy as np
import torch

from ts_transformer.instructions.words import Words
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.test_traffic_window import STEP_S, _airport, _keys, _physics_of, _pool

CPU = torch.device("cpu")


def _reading_model(spec, variant="no-context"):
    """A traffic prior whose traffic attention reads the others (its output layer off zero)."""
    from ts_transformer.inference.scene_edges import EDGE_FEATURES
    from ts_transformer.prior.model import with_traffic
    from ts_transformer.tests.test_prior_speaker import _model

    torch.manual_seed(1)
    model = with_traffic(_model(Words(spec), variant=variant), EDGE_FEATURES).eval()
    for layer in model.layers:
        torch.nn.init.normal_(layer.traffic.out.weight, std=0.2)
    return model


def _loop(model, airport, signals, spec, commanded, limits, landings=None, seed=4):
    from ts_transformer.experiments.traffic_window import WindowLoop, window_of
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params

    windows = [window_of(airport, 0.0, keys, [limit] * len(keys), STEP_S) for keys, limit in zip(commanded, limits)]
    flights = [signals[k] for window in windows for k in window.commanded]
    geometry = airport.flights.geometry
    per_aircraft = [limit for keys, limit in zip(commanded, limits) for _ in keys]
    return WindowLoop(model, windows, flights, [geometry] * len(flights), *_physics_of(flights, geometry), per_aircraft,
                      Words(spec), _params(), landings, generator=torch.Generator().manual_seed(seed), temperature=1.0,
                      procedure_masks=ProcedureMasks.none())


def _flights(loop, signals, landings, spec):
    from ts_transformer.experiments.traffic_window_tuner import window_flight

    return [window_flight(r, loop.windows[r.window], signals[r.key], loop.geometries[i], landings, 0, 0, len(r.grid),
                          spec.step_s) for i, r in enumerate(loop.records())]


def test_a_window_scored_whole_gives_back_what_each_of_its_words_was_sampled_from(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_window_tuner import window_layout, window_logits
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior.window_speaker import WindowSpeaker

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    model = _reading_model(spec)
    # f3 and f4 commanded with f1, f2 and f5 replayed around them; f7 alone at the end of the segment; batch-wide
    # pre-roll longer than f7's own
    loop = _loop(model, airport, signals, spec, [_keys(3, 4), _keys(7)], [60.0, 40.0])
    assert loop.windows[0].others and loop.pre > 0
    calls, sampled = [], []
    logits, sample = model.logits, WindowSpeaker._sample

    def spy_logits(*args):
        out = logits(*args)
        calls.append([logit[:, 0, 0].detach().clone() for logit in out])
        return out

    def spy_sample(speaker, h, tokens, valid, now, opening, row, runway_locked):
        first = len(calls)
        chosen = sample(speaker, h, tokens, valid, now, opening, row, runway_locked)
        sampled.append((now.copy(), row.copy(), calls[first: first + 6]))
        return chosen

    model.logits = spy_logits
    monkeypatch.setattr(WindowSpeaker, "_sample", spy_sample)
    try:
        while loop.running:
            loop.step()
    finally:
        del model.logits
    flights = _flights(loop, signals, None, spec)
    with torch.no_grad():
        got = window_logits(model, window_layout(model, loop.windows, flights, loop.records(), spec.step_s, CPU))
    compared = set()
    for now, row, columns in sampled:
        for i in np.flatnonzero(now):
            step = int(row[i])
            for c in range(6):
                want, have = columns[c][c][i], got[c][i, 0, step]
                finite = torch.isfinite(want)
                assert torch.equal(finite, torch.isfinite(have)), (i, step, c)
                assert torch.allclose(have[finite], want[finite].to(have.dtype), atol=1e-4), (i, step, c)
            compared.add((int(i), step))
    assert {i for i, _ in compared} == {0, 1, 2} and len(compared) > 40
    # the others did change the words: the same layout read with the traffic attention at zero
    for layer in model.layers:
        torch.nn.init.zeros_(layer.traffic.out.weight)
    with torch.no_grad():
        deaf = window_logits(model, window_layout(model, loop.windows, flights, loop.records(), spec.step_s, CPU))
    finite = torch.isfinite(got[3][:2])
    assert float((got[3][:2][finite] - deaf[3][:2][finite]).abs().max()) > 1e-2


def test_each_row_is_rebuilt_with_the_landing_context_it_was_encoded_with(tmp_path, monkeypatch):
    import dataclasses

    from ts_transformer.experiments.traffic_window_tuner import window_flight
    from ts_transformer.instructions.artefact import load_signals

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    model = _reading_model(spec, variant="full")
    landings = {"KXXX": _pool(airport)}
    loop = _loop(model, airport, signals, spec, [_keys(0, 1)], [400.0], landings)          # 200 s apart
    speaker = loop.speaker
    while speaker.step < int(loop.start[1]) + N_LOOK + 6:
        loop.step()
    # f0 lands in the loop a second before the last row encoded (as a landing found at a step comes after its row), and
    # the loop runs on
    encoded = speaker.steps_encoded
    landing_s = loop.window_time_s(0, encoded - 1) - 1.0
    loop.landing_s[0] = landing_s
    loop._landed(0, 0)
    for _ in range(4):
        loop.step()
    records = loop.records()
    switch = encoded - int(loop.start[1])
    assert not records[0].context_changes and records[1].context_changes == [(switch, landing_s, "09")]
    flights = _flights(loop, signals, landings, spec)
    for i, flight in enumerate(flights):
        rows = int(speaker.rows[i])
        assert flight.rows == rows
        assert torch.equal(torch.as_tensor(flight.features), speaker.features[i, :rows]), i
        assert torch.equal(torch.as_tensor(flight.relative), speaker.relative[i, :rows, : flight.relative.shape[1]]), i
        assert torch.equal(torch.as_tensor(flight.in_force, dtype=torch.long), speaker.in_force[i, :rows]), i
        assert torch.allclose(torch.as_tensor(flight.since), speaker.since[i, :rows]), i
    # read with the landing from row 0, f1's last row encoded before it would read it: the switch row matters
    whole = window_flight(dataclasses.replace(records[1], context_changes=[(0, landing_s, "09")]), loop.windows[0],
                          signals["KXXX:f1"], loop.geometries[1], landings, 0, 0, len(records[1].grid), spec.step_s)
    before = switch - 1
    assert not torch.equal(torch.as_tensor(whole.relative[before]), speaker.relative[1, before, : whole.relative.shape[1]])
    assert torch.equal(torch.as_tensor(whole.relative[switch:]), speaker.relative[1, switch: flights[1].rows,
                                                                                  : whole.relative.shape[1]])
