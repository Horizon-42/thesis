"""Augmented windows (`experiments/traffic_window_augment`, multi-aircraft design §6.6 step 7 item 6): a flight's times
moved by whole seconds, the flow compressed, a start moved, a flight inserted and commanded, the qualification and the
cap on the aircraft at once — and the M3 window runner reading augmented windows. On the scene-data fixture."""

from __future__ import annotations

import numpy as np
import pytest

from ts_transformer.instructions.words import Words
from ts_transformer.tests.test_traffic_window import (
    LIMIT_S, STEP_S, _airport, _as_drawn, _keys, _physics_of, _pool, _traffic_model,
)

ANY_ALTITUDE = {"KXXX": (-1e9, 1e9)}
MANY = 99
ROWS = 2_048


def _signals(tmp_path):
    from ts_transformer.instructions.artefact import load_signals

    return {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}


def _window_and_batch(tmp_path, monkeypatch, entries, commanded, pool=()):
    """A window commanding ``commanded`` (opening at the first one's first step) and a replay batch of its flights, then
    the ``pool`` ones (the draw's other flights at the airport), the executor's inputs patched to the fixture's."""
    from ts_transformer.experiments import traffic_window_augment
    from ts_transformer.experiments.traffic_window import window_of
    from ts_transformer.tests.test_traffic_speaking import _batch

    _, airports, spec = _airport(tmp_path, monkeypatch, entries, tuple(range(len(entries))))
    airport = airports["KXXX"]
    signals = _signals(tmp_path)
    keys = [*commanded, *pool]
    batch = _batch(airport, signals, spec, keys)
    opens = airport.tracks[commanded[0]].first_step_s
    window = window_of(airport, opens, commanded, [LIMIT_S] * len(commanded), STEP_S)
    monkeypatch.setattr(traffic_window_augment, "flight_inputs",
                        lambda series, device, anchor: _physics_of([signals[k] for k in series],
                                                                   airport.flights.geometry)[0])
    return airport, signals, spec, window, batch, keys


def test_a_flight_shifted_by_whole_seconds_keeps_its_rows_and_its_landing_leaves_its_context(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_augment import moved
    from ts_transformer.experiments.traffic_speaking import scene_landings, scene_of
    from ts_transformer.experiments.traffic_window_augment import shifted
    from ts_transformer.prior.data import own_context
    from ts_transformer.prior.scene import presence, utc_s

    _, airports, spec = _airport(tmp_path, monkeypatch, [0.0, 400.0], (0, 1))
    airport = airports["KXXX"]
    signals = _signals(tmp_path)["KXXX:f1"]
    later = shifted(signals, -37, "KXXX:f1")
    assert utc_s(later.entry_time_utc) == utc_s(signals.entry_time_utc) - 37.0
    assert utc_s(later.landing_time_utc) == utc_s(signals.landing_time_utc) - 37.0
    assert later.e_m is signals.e_m and later.time_s is signals.time_s and later.dataset_id == "KXXX:f1"
    rows = airport.flights.flights["KXXX:f1"]
    new_rows, new_track = moved(rows, airport.tracks["KXXX:f1"], -37.0, "KXXX:f1", STEP_S)
    geometry = airport.flights.geometry
    assert np.array_equal(presence(later, len(rows.presence.times_s), geometry).times_s, new_rows.presence.times_s)
    # its moved landing is in the scene's landings, and its own leaves them by its exact time
    scene = scene_of(airport, "KXXX:f1", LIMIT_S, STEP_S)
    context = scene_landings(_pool(airport), type(scene)(airport, "KXXX:f1", scene.first_step_s, scene.others,
                                                         ((new_rows, new_track),)))
    assert len(own_context(later, context).times_s) == len(context.times_s) - 1
    assert shifted(signals, 5, "KXXX:f1+x").dataset_id == "KXXX:f1+x"


def test_the_flow_compressed_moves_the_commanded_toward_the_opening_and_the_replayed_stay(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_window_augment import COMPRESSION, augment_window

    commanded = _keys(3, 4, 5)
    airport, signals, spec, window, batch, _ = _window_and_batch(
        tmp_path, monkeypatch, [0.0, 200.0, 400.0, 600.0, 800.0, 1_000.0, 1_200.0, 1_400.0], commanded)
    got, kind, refused = augment_window(window, batch, [0, 1, 2], {}, [LIMIT_S] * 3, [2 * LIMIT_S] * 3, MANY, ROWS,
                                        np.random.default_rng(2), ANY_ALTITUDE, spec, kinds=("C",))
    assert got is not None and got.kind == kind == "C" and got.draws == 1 + sum(refused.values())
    c = got.drawn["compression"]
    assert COMPRESSION[0] <= c <= COMPRESSION[1]
    assert got.window.commanded == commanded and got.places == (0, 1, 2) and got.roles == ("shifted",) * 3
    for key, flight in zip(commanded, got.signals):
        start = airport.flights.flights[key].presence.start_s
        shift = got.drawn["shift_s"][key]
        assert shift == round((c - 1.0) * (start - window.opens_s)) and shift <= 0
        assert got.window.rows(key).presence.start_s == start + shift and flight.dataset_id == key
    assert got.limits == (LIMIT_S,) * 3 and got.moves == (None,) * 3
    # the replayed ones as they were: none of them moved
    assert all(got.window.track(k) is airport.tracks[k] for k in got.window.others)


def test_only_what_an_augmentation_adds_is_capped_and_a_window_lost_before_the_prior_speaks_is_left_out(
        tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_augment import TRIES
    from ts_transformer.experiments.traffic_window_augment import at_once, augment_window, qualifies

    # f1 commanded with f0 in the air before it (two at once as drawn), f3 an hour later to insert
    _, signals, spec, window, batch, _ = _window_and_batch(
        tmp_path, monkeypatch, [0.0, 200.0, 400.0, 3_600.0], _keys(1), pool=_keys(3))
    drawn_most = at_once([window.track(k).presence for k in (*window.commanded, *window.others)], STEP_S)
    assert drawn_most == 2 and qualifies(window, {k: signals[k] for k in window.commanded}, STEP_S)
    # the window as drawn is busier than a cap of 1, and compressing one aircraft adds none: kept
    got, _, refused = augment_window(window, batch, [0], {}, [LIMIT_S] * 2, [LIMIT_S] * 2, 1, ROWS,
                                     np.random.default_rng(2), ANY_ALTITUDE, spec, kinds=("C",))
    assert got is not None and not refused
    # an inserted aircraft in the air with both is one more than the window has had
    got, kind, refused = augment_window(window, batch, [0], {"KXXX:f3": 1}, [LIMIT_S] * 2, [LIMIT_S] * 2, 1, ROWS,
                                        np.random.default_rng(2), ANY_ALTITUDE, spec, kinds=("A",))
    assert got is None and kind == "A" and refused == {"busier_than_the_data": TRIES}
    # f1 enters 20 s behind f0 on the same approach: lost before it speaks, however the flow is compressed
    (tmp_path / "close").mkdir()
    _, signals, spec, close, batch, _ = _window_and_batch(tmp_path / "close", monkeypatch, [0.0, 20.0], _keys(0, 1))
    assert not qualifies(close, {k: signals[k] for k in close.commanded}, STEP_S)
    got, _, refused = augment_window(close, batch, [0, 1], {}, [LIMIT_S] * 2, [LIMIT_S] * 2, MANY, ROWS,
                                     np.random.default_rng(2), ANY_ALTITUDE, spec, kinds=("C",))
    assert got is None and refused == {"lost_before_the_model_speaks": TRIES}


def test_the_busiest_training_step_counts_background_flights(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_window_augment import busiest

    # f0, f2 (background) and f1 entering 0, 10 and 200 s apart, each some 460 s in the scene; f3 an hour later
    from ts_transformer.tests.test_traffic_scene_data import _artefact

    directory, _, spec, _ = _artefact(tmp_path, [0.0, 200.0, 10.0, 3_600.0], [0, 1, 3])
    assert busiest(directory, spec, ("KXXX",), STEP_S) == {"KXXX": 3}
    (tmp_path / "labelled").mkdir()
    directory, _, spec, _ = _artefact(tmp_path / "labelled", [0.0, 200.0, 3_600.0], [0, 1, 2])
    assert busiest(directory, spec, ("KXXX",), STEP_S) == {"KXXX": 2}


def test_an_inserted_flight_is_commanded_the_gap_ahead_of_its_follower_and_its_source_is_not_replayed(tmp_path,
                                                                                                       monkeypatch):
    from ts_transformer.experiments.traffic_speaking import INSERTED
    from ts_transformer.experiments.traffic_window import window_of
    from ts_transformer.experiments.traffic_window_augment import augment_window, qualifies
    from ts_transformer.prior.scene import utc_s

    # f1 commanded, f0 in the air before it, f2 entering 300 s after f1 (the pool) and flying 900 s once inserted: the
    # window it opens reaches f2's own record
    airport, signals, spec, window, batch, _ = _window_and_batch(
        tmp_path, monkeypatch, [0.0, 400.0, 700.0], _keys(1), pool=_keys(2))
    assert "KXXX:f2" not in window.others
    got, kind, refused = augment_window(window, batch, [0], {"KXXX:f2": 1}, [LIMIT_S, 900.0], [LIMIT_S] * 2, MANY,
                                        ROWS, np.random.default_rng(1), ANY_ALTITUDE, spec, kinds=("A",))
    assert got is not None and got.kind == kind == "A" and set(refused) <= {"lost_before_the_model_speaks"}
    key = "KXXX:f2" + INSERTED
    assert got.drawn["inserted"] == "KXXX:f2" and got.drawn["follower"] == "KXXX:f1"
    # ahead of f1 (landing first), commanded in the order of the first steps, over its own flight's time limit
    assert got.window.commanded == (key, "KXXX:f1") and got.places == (1, 0) and got.limits == (900.0, LIMIT_S)
    assert got.roles == ("inserted", None) and got.moves == (None, None)
    separation = airport.flights.separation
    follower = airport.tracks["KXXX:f1"]
    inserted = got.window.track(key)
    gap_s = separation.gap_s("09", inserted.category, "09", follower.category)
    assert abs((follower.presence.landing_s - inserted.presence.landing_s) - got.drawn["gap"] * gap_s) <= 0.5
    assert got.drawn["shift_s"] == int(got.drawn["shift_s"])
    flight = got.signals[0]
    assert flight.dataset_id == key and utc_s(flight.landing_time_utc) == inserted.presence.landing_s
    # the flight it came from is not replayed beside it — it would be, left in
    assert "KXXX:f2" not in got.window.others and "KXXX:f0" in got.window.others
    parts = [(got.window.rows(key), inserted)]
    assert "KXXX:f2" in window_of(airport, window.opens_s, got.window.commanded, list(got.limits), STEP_S, parts).others
    assert qualifies(got.window, dict(zip(got.window.commanded, got.signals)), STEP_S)
    # nothing to insert: A never applies
    alone, kind, refused = augment_window(window, batch, [0], {}, [LIMIT_S, 900.0], [LIMIT_S] * 2, MANY, ROWS,
                                          np.random.default_rng(1), ANY_ALTITUDE, spec, kinds=("A",))
    assert alone is None and kind == "A" and refused == {"no_flight_to_insert": 10}


def test_a_moved_start_moves_one_commanded_aircraft_and_what_the_others_read_of_it(tmp_path, monkeypatch):
    import math

    from ts_transformer.experiments.traffic_window_augment import augment_window
    from ts_transformer.prior.generate import rows_for

    airport, signals, spec, window, batch, _ = _window_and_batch(
        tmp_path, monkeypatch, [0.0, 200.0, 400.0], _keys(1, 2))
    got, _, _ = augment_window(window, batch, [0, 1], {}, [LIMIT_S] * 2, [2 * LIMIT_S] * 2, MANY, ROWS,
                               np.random.default_rng(3), ANY_ALTITUDE, spec, kinds=("B",))
    assert got is not None and got.kind == "B"
    key = got.drawn["moved"]
    m = got.window.commanded.index(key)
    assert got.roles[m] == "moved" and got.moves[m] is not None and got.limits[m] == 2 * LIMIT_S
    assert [r for n, r in enumerate(got.roles) if n != m] == [None] and got.window.commanded == window.commanded
    assert not np.allclose(got.signals[m].e_m, signals[key].e_m)
    record = got.window.track(key)
    assert math.isinf(record.captured_s) and np.array_equal(record.e_m, got.signals[m].e_m[: len(record.e_m)])
    assert got.window.rows(key).presence is record.presence
    assert np.array_equal(record.presence.times_s, airport.tracks[key].presence.times_s)
    assert got.window.others == window.others
    # a stage-2 time limit past the model's positions is never drawn
    rows = rows_for(LIMIT_S + STEP_S, STEP_S)
    got, _, refused = augment_window(window, batch, [0, 1], {}, [LIMIT_S] * 2, [2 * LIMIT_S] * 2, MANY, rows,
                                     np.random.default_rng(3), ANY_ALTITUDE, spec, kinds=("B",))
    assert got is None and refused == {"longer_than_the_model": 10}


def test_the_window_runner_reads_augmented_windows_by_the_model_s_sources_with_their_kinds(tmp_path, monkeypatch):
    """Inserted and compressed windows through the loop — a model reading the landing context too (each aircraft's own
    landing, moved, leaves its context by its exact time) — and the readout by kind and part."""
    import torch

    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments.traffic_speaking import INSERTED, scene_airports
    from ts_transformer.experiments.traffic_window import window_of
    from ts_transformer.inference.scene_edges import EDGE_FEATURES
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.prior.model import with_traffic
    from ts_transformer.tests.test_autopilot import _params
    from ts_transformer.tests.test_prior_speaker import _model
    from ts_transformer.tests.test_traffic_window import _patch_runner_physics

    airport, signals, spec, _, batch, keys = _window_and_batch(
        tmp_path, monkeypatch, [0.0, 400.0, 3_600.0], _keys(1), pool=_keys(2))
    _patch_runner_physics(monkeypatch, signals, airport.flights.geometry, keys)
    every = {"KXXX": _pool(airport)}
    # the airport's rows as a model reading the landing context has them
    context_airport = scene_airports(tmp_path / "artefact", "train", spec, ("KXXX",), every, 2_048)[0]["KXXX"]
    torch.manual_seed(1)
    full = with_traffic(_model(Words(spec), variant="full"), EDGE_FEATURES).eval()

    def drawn_at(at):
        """f1 in a window opening with f0 (400 s before it), f2 in its own."""
        windows = [window_of(at, at.tracks[o].first_step_s, (k,), [LIMIT_S], STEP_S)
                   for o, k in (("KXXX:f0", "KXXX:f1"), ("KXXX:f2", "KXXX:f2"))]
        return _as_drawn(windows, [range(0, 1), range(1, 2)], batch, [LIMIT_S] * 2)

    rows = []
    for kind in ("A", "C"):
        augmented, counts = runner.augmented_windows(drawn_at(airport), _params(), {"KXXX": MANY}, ROWS, 1,
                                                     ANY_ALTITUDE, spec, kinds=(kind,))
        assert counts["kinds"][kind] == len(augmented.windows) == 2 and counts["left_out"] == {}
        assert [s.dataset_id for s in augmented.batch.signals] == [k for w in augmented.windows for k in w.commanded]
        chunk = list(range(len(augmented.windows)))
        for source in ("scene", "alone"):
            rows += runner.model_rows(_traffic_model(spec), augmented, chunk, source, Words(spec), _params(), None,
                                      every, 2, generator=torch.Generator().manual_seed(4), temperature=1.0,
                                      procedure_masks=ProcedureMasks.none())
        in_context, _ = runner.augmented_windows(drawn_at(context_airport), _params(), {"KXXX": MANY}, ROWS, 1,
                                                 ANY_ALTITUDE, spec, kinds=(kind,))
        read = runner.model_rows(full, in_context, chunk, "scene", Words(spec), _params(), every, every, 1,
                                 generator=torch.Generator().manual_seed(4), temperature=1.0,
                                 procedure_masks=ProcedureMasks.none())
        assert len(read) == len(augmented.batch.signals)
    compressed = next(w for w, a in zip(augmented.windows, augmented.augmented) if a["shift_s"]["KXXX:f1"])
    assert compressed.rows("KXXX:f1").presence.start_s < airport.flights.flights["KXXX:f1"].presence.start_s
    inserted = [r for r in rows if r["role"] == "inserted"]
    assert len(inserted) == 2 * 2 * 2 and all(r["dataset_id"].endswith(INSERTED) and r["augmented"]["kind"] == "A"
                                              for r in inserted)
    assert all(r["reward"] in (0.0, 1.0) and r["outcome"] for r in rows)
    readout = runner.summaries(rows, augmented=True)
    assert set(readout["kinds"]) == {"A", "C"} and set(readout["roles"]) == {"inserted", "shifted", "as drawn"}
    with pytest.raises(ValueError, match="model's sources only"):
        runner.fixed_rows(augmented, chunk, "recorded", Words(spec), _params(), every, ProcedureMasks.none())


def test_a_moved_start_flies_from_its_moved_state_through_the_window_runner(tmp_path, monkeypatch):
    """B through the loop: the moved aircraft's executor starts where its moved rows end (`augmented_inputs`), over
    stage 2's time limit, read by a model with the landing context too; the readout by part."""
    import torch

    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments.traffic_speaking import scene_airports
    from ts_transformer.experiments.traffic_window import window_of
    from ts_transformer.inference.scene_edges import EDGE_FEATURES
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.prior.model import with_traffic
    from ts_transformer.tests.test_autopilot import _params
    from ts_transformer.tests.test_prior_speaker import _model
    from ts_transformer.tests.test_traffic_window import _patch_runner_physics

    airport, signals, spec, _, batch, keys = _window_and_batch(tmp_path, monkeypatch, [0.0, 200.0, 400.0], _keys(1, 2))
    _patch_runner_physics(monkeypatch, signals, airport.flights.geometry, keys)
    every = {"KXXX": _pool(airport)}
    context_airport = scene_airports(tmp_path / "artefact", "train", spec, ("KXXX",), every, 2_048)[0]["KXXX"]
    torch.manual_seed(1)
    full = with_traffic(_model(Words(spec), variant="full"), EDGE_FEATURES).eval()

    def drawn_at(at):
        window = window_of(at, at.tracks["KXXX:f0"].first_step_s, _keys(1, 2), [LIMIT_S] * 2, STEP_S)
        return _as_drawn([window], [range(0, 2)], batch, [LIMIT_S] * 2)

    augmented, counts = runner.augmented_windows(drawn_at(airport), _params(), {"KXXX": MANY}, ROWS, 3, ANY_ALTITUDE,
                                                 spec, kinds=("B",))
    assert counts["kinds"]["B"] == 1 and counts["left_out"] == {}
    m = next(j for j, move in enumerate(augmented.moves) if move is not None)
    assert augmented.roles[m] == "moved" and augmented.limits[m] > LIMIT_S
    # the executor's first state is the moved start's (`augmented_inputs` over the fixture's physics)
    from ts_transformer.experiments.prior_free_generation import augmented_inputs
    from ts_transformer.tests.test_traffic_window import _physics_of

    geometry = airport.flights.geometry
    own = _physics_of([signals[keys[m]]], geometry)[0]
    moved = augmented_inputs(own, [geometry], [augmented.moves[m]])
    assert not torch.equal(moved.initial_state, own.initial_state)
    rows = []
    for source in ("scene", "alone"):
        rows += runner.model_rows(_traffic_model(spec), augmented, [0], source, Words(spec), _params(), None, every, 2,
                                  generator=torch.Generator().manual_seed(4), temperature=1.0,
                                  procedure_masks=ProcedureMasks.none())
    in_context, _ = runner.augmented_windows(drawn_at(context_airport), _params(), {"KXXX": MANY}, ROWS, 3,
                                             ANY_ALTITUDE, spec, kinds=("B",))
    rows += runner.model_rows(full, in_context, [0], "scene", Words(spec), _params(), every, every, 1,
                              generator=torch.Generator().manual_seed(4), temperature=1.0,
                              procedure_masks=ProcedureMasks.none())
    assert sorted(str(r["role"]) for r in rows) == ["None"] * 5 + ["moved"] * 5
    readout = runner.summaries(rows, augmented=True)
    assert set(readout["kinds"]) == {"B"} and set(readout["roles"]) == {"moved", "as drawn"}
