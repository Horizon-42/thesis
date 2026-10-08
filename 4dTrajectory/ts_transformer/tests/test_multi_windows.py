"""Stage D, MC1: the windows of several commanded aircraft and their census (multi-aircraft control D146, §3.1, §11
MC1; `multi/windows.py`, `multi/census.py`) — on the synthetic flights of stage C's scene tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ts_transformer.instructions.artefact import load_candidates
from ts_transformer.multi.census import summary, window_count
from ts_transformer.multi.separation import PAIRS, classify
from ts_transformer.multi.windows import COMPRESSED, REAL_KIND, Anchors, Drawn, compressed, left_out, opening_loss
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import airport_scenes, real_windows
from ts_transformer.tests.post_support import categories, finals, scene_artefact, straight_in, train_noon
from ts_transformer.tests.support import INSTRUCTION_STEP_S

DELTA = 4.0


@pytest.fixture
def built(tmp_path):
    """Three train flights onto 09 in a stream 120 s apart and one far later; their real windows."""
    flights = {"train": [straight_in("KXXX:a", train_noon(0.0)), straight_in("KXXX:b", train_noon(120.0)),
                         straight_in("KXXX:c", train_noon(242.0)), straight_in("KXXX:far", train_noon(7_200.0))]}
    spec = scene_artefact(tmp_path / "art", flights, interval_s=DELTA)
    geometries = load_candidates(tmp_path / "art")
    scenes, signals = airport_scenes(tmp_path / "art", "train", spec, DELTA, geometries, categories)
    windows = real_windows(tmp_path / "art", "train", spec, DELTA, scenes, signals)
    geometry = geometries["KXXX"]
    return windows, airport_separation(geometry), finals(geometry)


def test_a_window_commands_every_flight_whose_row_0_is_within_its_span(built):
    """D146: the anchor and every other flight of its airport and split with a sentence whose row 0 is in [the anchor's
    row 0, + L), in the order of their row 0; L = 0 is stage C's real window."""
    windows, _, _ = built
    anchors = Anchors(windows)
    a, b, c, far = windows
    assert anchors.window_of(a, 0.0) == a and a.span_s == 0.0
    five = anchors.window_of(a, 300.0)
    assert five.span_s == 300.0                                        # its span, which its batch reads (D146)
    with pytest.raises(ValueError, match="span is not negative"):
        replace(a, span_s=-1.0)
    assert [r.key for r in five.commanded_all] == ["KXXX:a", "KXXX:b", "KXXX:c"]
    assert five.signal_indices == (a.signal_index, b.signal_index, c.signal_index)
    assert np.allclose(five.join_steps() * DELTA, [0.0, b.row0_s - a.row0_s, c.row0_s - a.row0_s])
    assert [r.key for r in anchors.window_of(a, 200.0).commanded_all] == ["KXXX:a", "KXXX:b"]
    assert [r.key for r in anchors.window_of(b, 300.0).commanded_all] == ["KXXX:b", "KXXX:c"]
    assert anchors.window_of(far, 1200.0).joined == ()
    span = c.row0_s - a.row0_s                                         # the span's end is open
    assert [r.key for r in anchors.window_of(a, span).commanded_all] == ["KXXX:a", "KXXX:b"]
    for step in range(0, 150, 10):
        assert not set(five.others_at(step).keys) & {"KXXX:a", "KXXX:b", "KXXX:c"}
    with pytest.raises(ValueError, match="stage C's real windows"):
        Anchors([five])


def test_a_compressed_window_moves_each_later_aircraft_toward_the_anchor(built):
    """D146: each commanded aircraft after the anchor moved to its offset × c (c uniform in [c_min, 1)), rounded to a
    whole Δ; its record moves with it; the anchor and the recorded aircraft do not move."""
    windows, _, _ = built
    window = Anchors(windows).window_of(windows[0], 300.0)
    rng = np.random.default_rng(1337)
    for _ in range(20):
        drawn = compressed(window, rng, 0.6)
        assert drawn.kind == COMPRESSED and 0.6 <= drawn.c < 1.0
        moved = drawn.window
        assert moved.commanded == window.commanded
        for before, after in zip(window.joined, moved.joined):
            offset = round((before.record.first_step_s - window.first_step_s) / DELTA)
            now = round((after.record.first_step_s - window.first_step_s) / DELTA)
            assert now == int(round(offset * drawn.c)) and after.shift_s == (now - offset) * DELTA <= 0.0
            assert after.record.start_s == before.record.start_s + after.shift_s
            assert np.array_equal(after.record.e_m, before.record.e_m)
    assert Drawn(window, REAL_KIND).c == 1.0
    for wrong in ((REAL_KIND, 0.5), (COMPRESSED, 1.0), ("other", 1.0)):
        with pytest.raises(ValueError, match="a window of stage D"):
            Drawn(window, *wrong)
    with pytest.raises(ValueError, match="c_min"):
        compressed(window, rng, 1.0)


def test_a_window_whose_aircraft_opens_inside_a_loss_is_left_out_and_by_which(built):
    """D146 (post-training D113 for each commanded aircraft): compressed so that the second aircraft joins 8 s behind the
    anchor, each answers for a loss at its first predicted step on the records (no runway in force; neither established:
    both responsible) — the anchor against the second, already in the air, and the second against the anchor — and the
    window is left out by both; the real window of the stream, 120 s apart, is not."""
    windows, separation, fin = built
    window = Anchors(windows).window_of(windows[0], 200.0)                 # a and b
    assert left_out(window, separation, fin, INSTRUCTION_STEP_S) == []
    close = compressed(window, np.random.default_rng(0), 0.02)
    while round((close.window.joined[0].record.first_step_s - window.first_step_s) / DELTA) != 2:
        close = compressed(window, np.random.default_rng(int(close.c * 1e6)), 0.02)
    assert left_out(close.window, separation, fin, INSTRUCTION_STEP_S) == [0, 1]
    loss = opening_loss(close.window, 1, separation, fin, INSTRUCTION_STEP_S)
    assert loss is not None and 0 in loss.responsible
    count = window_count(close, separation, fin, INSTRUCTION_STEP_S)
    assert count.left_out_by == [0, 1] and summary([count])["all"]["left_out"] == {
        "windows": 1, "by_anchor": 1, "by_a_later_aircraft": 1}
    assert count.loss_steps["commanded_commanded"] > 0                 # 8 s apart on the records: a loss of the pair
    kept = window_count(Drawn(window, REAL_KIND), separation, fin, INSTRUCTION_STEP_S)
    both = summary([count, kept])["all"]
    assert both["losses"]["commanded_commanded"]["windows"] == 0       # the window left out is not counted
    assert both["steps_judged"] == kept.steps


def test_the_census_counts_the_losses_on_the_records_by_pair(built):
    """MC1: the losses on the records of a window's steps, by pair — two commanded aircraft, a commanded one answering,
    only a recorded one responsible (D145) — and the windows, the commanded and the recorded aircraft by airport."""
    windows, separation, fin = built
    anchors = Anchors(windows)
    counts = [window_count(Drawn(anchors.window_of(w, 300.0), REAL_KIND), separation, fin, INSTRUCTION_STEP_S)
              for w in windows]
    assert [c.commanded for c in counts] == [3, 2, 1, 1]
    alone = window_count(Drawn(windows[1], REAL_KIND), separation, fin, INSTRUCTION_STEP_S)
    assert alone.commanded == 1 and alone.steps > 0
    closer = compressed(anchors.window_of(windows[0], 300.0), np.random.default_rng(3), 0.6)
    squeezed = window_count(closer, separation, fin, INSTRUCTION_STEP_S)
    assert squeezed.steps > 0 and set(squeezed.loss_steps) == set(PAIRS)
    out = summary(counts + [squeezed])
    assert out["all"]["windows"] == 5 and out["KXXX"]["commanded"]["max"] == 3.0
    assert set(out["all"]["losses"]) == set(PAIRS)


def test_a_loss_is_classified_by_the_commanded_aircraft_it_holds():
    assert classify(0, 1, (1,), 2) == "commanded_commanded"
    assert classify(0, 3, (0,), 2) == "commanded_answers"
    assert classify(1, 3, (1, 3), 2) == "commanded_answers"
    assert classify(1, 3, (3,), 2) == "recorded_only"
    assert classify(2, 3, (3,), 2) is None


def test_the_census_runner_writes_each_split_span_and_kind_and_never_under_the_outputs(tmp_path, monkeypatch):
    """MC1 (multi-aircraft control §11): the census of train and select, for each span and kind, on the records and on
    the closed-loop states (the baseline), written to a new directory outside `4dTrajectory/outputs/`; a sample is
    stated; the closed loop's checks run first (a stand-in here: the synthetic artefact has no executor spec)."""
    import json

    from ts_transformer.experiments import multi_windows
    from ts_transformer.instructions.words import Words
    from ts_transformer.post import scene as post_scene
    from ts_transformer.repo_layout import REPO_ROOT

    flights = {"train": [straight_in("KXXX:a", train_noon(0.0)), straight_in("KXXX:b", train_noon(120.0)),
                         straight_in("KXXX:c", train_noon(242.0)), straight_in("KXXX:far", train_noon(7_200.0))],
               "select": [straight_in("KXXX:s", train_noon(60.0, split="select"))]}
    spec = scene_artefact(tmp_path / "art", flights, interval_s=DELTA)
    checked = []
    monkeypatch.setattr(multi_windows, "require_conforming_closed_loop",
                        lambda instructions, executor: checked.append(executor) or (None, {"stand-in": True},
                                                                                      Words(spec)))
    monkeypatch.setattr(multi_windows, "airport_scenes",
                        lambda *a, **k: post_scene.airport_scenes(*a, **k, category_of=categories))
    monkeypatch.setattr(multi_windows, "airport_finals", lambda geometry, root: finals(geometry))
    out = tmp_path / "census"
    args = ["--instructions", str(tmp_path / "art"), "--executor", str(tmp_path / "executor"), "--interval-s", "4"]
    assert multi_windows.main(args + ["--out", str(out), "--spans-min", "0", "5"]) == 0
    assert checked == [tmp_path / "executor"]
    record = json.loads((out / "census.json").read_text())
    assert record["schema"] == multi_windows.CENSUS_SCHEMA and set(record["splits"]) == {"train", "select"}
    train = record["splits"]["train"]
    assert train["anchors"] == 4 and set(train["spans"]) == {"0min", "5min"}
    assert set(train["spans"]["0min"]) == {"real"}                         # L = 0: no compressed window
    five = train["spans"]["5min"]
    assert set(five) == {"real", "compressed_0.6", "compressed_0.8"}
    assert five["real"]["windows"] == 4 and five["compressed_0.6"]["windows"] == 2     # the windows with a later aircraft
    assert five["real"]["records"]["KXXX"]["commanded"]["max"] == 3.0
    assert set(five["real"]["baseline"]["all"]["losses"]) == set(PAIRS)
    assert five["real"]["baseline"]["all"]["steps_judged"] > 0
    assert record["sample"] is None and record["checks"] == {"stand-in": True}
    sampled = tmp_path / "sampled"
    multi_windows.main(args + ["--out", str(sampled), "--splits", "train", "--sample", "2", "--spans-min", "0"])
    again = json.loads((sampled / "census.json").read_text())
    assert again["splits"]["train"]["anchors"] == 2 and again["sample"] == 2 and again["sample_seed"] == 1337
    with pytest.raises(SystemExit):
        multi_windows.main(args + ["--out", str(out)])                       # an existing directory
    forbidden = REPO_ROOT / "4dTrajectory" / "outputs" / "multi_windows_test_never_written"
    with pytest.raises(SystemExit):
        multi_windows.main(args + ["--out", str(forbidden)])
    assert not forbidden.exists()
    with pytest.raises(SystemExit):
        multi_windows.main(args + ["--out", str(tmp_path / "v"), "--splits", "val"])


def _closed_loop(tmp_path):
    """The artefact of `built` with its closed-loop sentences read back; its words."""
    from ts_transformer.instructions.artefact import closed_loop_sentences, load_spec
    from ts_transformer.instructions.words import Words

    spec = load_spec(tmp_path / "art")
    return closed_loop_sentences(tmp_path / "art", "train", DELTA, spec), Words(spec)


def test_the_baselines_positions_are_the_closed_loop_states_with_r_after_the_first_predicted_step(built, tmp_path):
    """§5 item 4, D23: an aircraft of the baseline is at its closed-loop sentence's states, at its own row 0 in the
    window (its moved row 0, in a compressed window); its runway in force is none through its first predicted step and
    its first word's from the next Δ row; the baseline is judged to the end of those states."""
    from ts_transformer.multi.census import closed_loop_positions

    windows, _, _ = built
    sentences, words = _closed_loop(tmp_path)
    window = Anchors(windows).window_of(windows[0], 200.0)
    positions = closed_loop_positions(window, sentences, words)
    for member, record in enumerate(window.commanded_all):
        rows = sentences[window.signal_indices[member]].rows
        first = record.first_step_s
        assert positions.at(member, first)[4] == -1                     # D23: no runway in force at its first step
        assert positions.at(member, first + DELTA)[4] == int(rows.grid[0][0])
        row0 = first - 16.0
        assert positions.at(member, row0)[1] == tuple(rows.states[0, :3])
        assert positions.at(member, row0 - DELTA) is None
    assert positions.end_s == max(r.first_step_s - 16.0 + (len(sentences[i].rows.states) - 1) * INSTRUCTION_STEP_S
                                  for r, i in zip(window.commanded_all, window.signal_indices))
    moved = compressed(window, np.random.default_rng(3), 0.6).window
    shifted = closed_loop_positions(moved, sentences, words)
    record = moved.commanded_all[1]
    rows = sentences[moved.signal_indices[1]].rows
    assert shifted.at(1, record.first_step_s - 16.0)[1] == tuple(rows.states[0, :3])   # at its moved row 0


def test_a_loss_that_only_a_recorded_aircraft_answers_for_is_charged_when_the_records_kept_their_separation(built):
    """D145 on the baseline: a commanded aircraft placed just ahead of a recorded follower (the follower responsible,
    in trail) where their records are 120 s apart is counted ``records_kept`` (the loop charges it), never
    ``recorded_only``; on the records themselves no such loss is found."""
    from ts_transformer.multi.separation import Positions

    windows, separation, fin = built
    b, c = windows[1], windows[2]                                       # the window of b alone: c behind it, recorded
    follower = b.scene.flight("KXXX:c")
    interval = b.scene.interval_s

    def ahead_of_c(member, time_s):                                     # b flown 8 s ahead of c's record
        later = time_s + 8.0
        if not follower.start_s <= later <= follower.end_s:
            return None
        _, here, before, known, runway, category, _, go_around = follower.at_step(later, interval)
        return b.commanded.key, here, before, known, runway, category, False, go_around

    placed = Positions(ahead_of_c, follower.end_s - 8.0, False)
    charged = window_count(Drawn(b, REAL_KIND), separation, fin, INSTRUCTION_STEP_S, positions=placed)
    assert charged.loss_steps["records_kept"] > 0 and charged.loss_steps["recorded_only"] == 0
    plain = window_count(Drawn(b, REAL_KIND), separation, fin, INSTRUCTION_STEP_S)
    assert plain.loss_steps["records_kept"] == 0


def test_a_loss_that_the_records_also_have_stays_recorded_only(built):
    """D145's other branch on the baseline: a recorded follower 8 s behind a commanded aircraft on the records (its own
    flight inserted behind it), the follower responsible: judged on positions that are the records but taken as a
    baseline, the same pair is lost on the records at that step, so it stays ``recorded_only`` — never
    ``records_kept``."""
    from dataclasses import replace

    from ts_transformer.multi.separation import Positions, on_records
    from ts_transformer.post.scene import INSERTED, INSERTED_SUFFIX, MovedScene

    windows, separation, fin = built
    window = windows[0]
    own = window.scene.flight(window.commanded.key)
    key = own.key + INSERTED_SUFFIX
    behind = replace(window, kind=INSERTED, moved=((key, 8.0),),
                     scene=MovedScene(window.scene, added=(own.shifted(8.0, DELTA, key=key),)))
    records = on_records(behind)
    as_baseline = Positions(records.at, records.end_s, False)
    count = window_count(Drawn(behind, REAL_KIND), separation, fin, INSTRUCTION_STEP_S, positions=as_baseline)
    assert count.loss_steps["recorded_only"] > 0 and count.loss_steps["records_kept"] == 0


def _counts_before(window, positions, separation, fin, step_s, reached):
    """The census's step loop as it was before the generator (`multi.census.window_losses` at 246eca69, frontend
    D177 (12)), kept here as the reference the new loop must count bit for bit like: (steps judged, loss steps by pair).
    ``reached`` counts the branches the inputs reach (a landed commanded aircraft over its threshold, a loss holding
    none, two recorded-only losses in a step, a recorded-only loss on the records)."""
    from ts_transformer.multi.census import first_step_rows
    from ts_transformer.multi.separation import judged_step, losses_on_records

    steps, loss_steps = 0, {pair: 0 for pair in PAIRS}
    interval = window.scene.interval_s
    first = int(first_step_rows(window)[0])
    last = int(round((positions.end_s - window.row0_s) / interval))
    for step in range(first, last + 1):
        judged = judged_step(window, positions, step, separation, fin, step_s)
        if judged is None:
            continue
        aircraft, commanded, losses = judged
        steps += 1
        found, kept_on_records = set(), None
        over = frozenset(int(k) for k in np.flatnonzero(aircraft.last_step[:commanded]))
        reached["over"] += bool(over)
        pairs = [classify(loss.i, loss.j, loss.responsible, commanded, over) for loss in losses]
        reached["none"] += pairs.count(None)
        reached["two_recorded_only"] += pairs.count("recorded_only") > 1
        reached["recorded_only_on_records"] += positions.records and "recorded_only" in pairs
        for loss in losses:
            pair = classify(loss.i, loss.j, loss.responsible, commanded, over)
            if pair == "recorded_only" and not positions.records:
                if kept_on_records is None:
                    kept_on_records = losses_on_records(window, step, separation, fin, step_s)
                if frozenset((aircraft.keys[loss.i], aircraft.keys[loss.j])) not in kept_on_records:
                    pair = "records_kept"
            if pair is not None:
                found.add(pair)
        for pair in found:
            loss_steps[pair] += 1
    return steps, loss_steps


def _cases(windows):
    """Fixed inputs of the census (the tests above): each window of a 300 s span and two compressed (the second close
    enough for its commanded aircraft to lose separation from each other), on the records;
    a commanded aircraft placed 8 s ahead of a recorded follower (records_kept); a follower 8 s behind on the records
    taken as a baseline (recorded_only)."""
    from ts_transformer.multi.separation import Positions, on_records
    from ts_transformer.post.scene import INSERTED, INSERTED_SUFFIX, MovedScene

    anchors = Anchors(windows)
    out = [(anchors.window_of(w, 300.0), None) for w in windows]
    out.append((compressed(anchors.window_of(windows[0], 300.0), np.random.default_rng(3), 0.6).window, None))
    out.append((compressed(anchors.window_of(windows[0], 300.0), np.random.default_rng(2), 0.3).window, None))
    b = windows[1]
    follower = b.scene.flight("KXXX:c")

    def ahead_of_c(member, time_s):
        later = time_s + 8.0
        if not follower.start_s <= later <= follower.end_s:
            return None
        _, here, before, known, runway, category, _, go_around = follower.at_step(later, b.scene.interval_s)
        return b.commanded.key, here, before, known, runway, category, False, go_around

    out.append((b, Positions(ahead_of_c, follower.end_s - 8.0, False)))
    own = windows[0].scene.flight(windows[0].commanded.key)
    key = own.key + INSERTED_SUFFIX
    behind = replace(windows[0], kind=INSERTED, moved=((key, 8.0),),
                     scene=MovedScene(windows[0].scene, added=(own.shifted(8.0, DELTA, key=key),)))
    records = on_records(behind)
    as_baseline = Positions(records.at, records.end_s, False)
    out.append((behind, as_baseline))
    out.append((behind, None))                                          # the same on the records themselves
    key2 = key + "2"                                                    # a second copy, 16 s behind
    two = replace(windows[0], kind=INSERTED, moved=((key, 8.0), (key2, 16.0)),
                  scene=MovedScene(windows[0].scene, added=(own.shifted(8.0, DELTA, key=key),
                                                            own.shifted(16.0, DELTA, key=key2))))
    records = on_records(two)
    out += [(two, None), (two, Positions(records.at, records.end_s, False))]

    def over_threshold(member, time_s):                                 # the commanded aircraft judged as landed
        item = as_baseline.at(member, time_s)
        return None if item is None else (*item[:6], True, item[7])

    out.append((behind, Positions(over_threshold, as_baseline.end_s, False)))
    return out


def test_the_census_counts_bit_for_bit_as_before_the_generator(built):
    """Frontend D177 (12): the readout's census (`window_count`, through `judged_steps`) gives on fixed inputs exactly
    the counts of the loop it replaced (`_counts_before`): every pair, records_kept and recorded_only among them."""
    from ts_transformer.multi.separation import on_records

    windows, separation, fin = built
    seen = {pair: 0 for pair in PAIRS}
    reached = dict.fromkeys(("over", "none", "two_recorded_only", "recorded_only_on_records"), 0)
    for window, positions in _cases(windows):
        placed = on_records(window) if positions is None else positions
        count = window_count(Drawn(window, REAL_KIND), separation, fin, INSTRUCTION_STEP_S, positions=positions)
        assert (count.steps, count.loss_steps) == _counts_before(window, placed, separation, fin, INSTRUCTION_STEP_S,
                                                                 reached)
        for pair, n in count.loss_steps.items():
            seen[pair] += n
    assert all(seen[pair] > 0 for pair in PAIRS)                         # every pair counted somewhere
    assert all(n > 0 for n in reached.values()), reached                 # and every branch of the loop reached


def test_the_step_loop_gives_each_judged_step_and_its_losses(built):
    """`judged_steps` on two synthetic windows: a flight alone gives every step from its first predicted step to its
    end, none with a loss; with its own flight inserted 8 s behind on the records (a baseline, the stream's flights
    around them), each loss holds the commanded aircraft and is recorded_only exactly where the commanded aircraft is
    not among the responsible (before the records are read), commanded_answers where it is; the steps with each, and
    every judged step, are the census's counts."""
    from ts_transformer.multi.census import first_step_rows, judged_steps
    from ts_transformer.multi.separation import on_records

    windows, separation, fin = built
    alone = windows[3]
    steps = list(judged_steps(alone, on_records(alone), separation, fin, INSTRUCTION_STEP_S))
    first = int(first_step_rows(alone)[0])
    assert [s.step for s in steps] == list(range(first, first + len(steps))) and len(steps) > 10
    assert all(s.commanded == 1 and s.losses == () for s in steps)
    window, positions = _cases(windows)[-5]                            # the follower 8 s behind, as a baseline
    steps = list(judged_steps(window, positions, separation, fin, INSTRUCTION_STEP_S))
    losses = [x for s in steps for x in s.losses]
    assert any(frozenset(x.keys) == frozenset({window.commanded.key, window.moved[0][0]}) for x in losses)
    assert all(window.commanded.key in x.keys for x in losses)
    assert all((x.pair == "recorded_only") == (0 not in x.loss.responsible) for x in losses)
    count = window_count(Drawn(window, REAL_KIND), separation, fin, INSTRUCTION_STEP_S, positions=positions)
    for name in ("recorded_only", "commanded_answers"):
        assert sum(any(x.pair == name for x in s.losses) for s in steps) == count.loss_steps[name] > 0
    assert len(steps) == count.steps
