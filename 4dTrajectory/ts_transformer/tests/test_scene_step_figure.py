"""The multi-aircraft design's §2.1 figure of how rows hang on the scene's steps (`experiments/scene_step_figure`): the
file in the docs is what the runner draws, the rows hang as the scenes hang them (`prior.scene.hang`), and the row of
the other aircraft each row reads, and how far it is moved, are the ones `inference.scene_edges` uses."""

from itertools import permutations

import numpy as np
import pytest

from ts_transformer.experiments import scene_step_figure as figure
from ts_transformer.inference.runway_schedule import Separation
from ts_transformer.inference.scene_edges import EDGE_FEATURES, HEIGHT_SCALE_M, SceneRows, scene_edges
from ts_transformer.prior.scene import hang, scene_steps


def test_the_figure_in_the_docs_is_what_the_runner_draws():
    """Change the example or the drawing, then run `python run_ts.py scene_step_figure` and commit the SVG with it."""
    assert figure.FIGURE.read_text(encoding="utf-8") == figure.render()


def test_every_row_hangs_on_the_nearest_even_second_one_offset_an_aircraft():
    assert figure.hung("A") == [100.0, 102.0, 104.0, 106.0] and figure.hung("B") == [102.0, 104.0, 106.0, 108.0]
    for name, _, _ in figure.AIRCRAFT:
        offsets = np.array(figure.hung(name)) - np.array(figure.row_times(name))
        assert np.allclose(offsets, offsets[0]) and abs(offsets[0]) <= figure.STEP_S / 2
    assert float(hang(101.0, figure.STEP_S)) == 102.0           # half a step rounds up, as the legend says


def height_m(time_s: float, name: str) -> float:
    """A height not linear in the row number, so reading the row before gives another answer."""
    k = figure.row_times(name).index(time_s)
    return 100.0 * k * k + (500.0 if name == "B" else 0.0)


def edges_of_the_example() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """`scene_edges` on the example's aircraft: each row on the step it hangs on, position 0, no runway."""
    names = [name for name, _, _ in figure.AIRCRAFT]
    steps = scene_steps(min(min(figure.hung(n)) for n in names), max(max(figure.hung(n)) for n in names),
                        figure.STEP_S)
    time = np.full((len(names), len(steps)), np.nan)
    height = np.full_like(time, np.nan)
    for a, name in enumerate(names):
        for t, s in zip(figure.row_times(name), figure.hung(name)):
            time[a, list(steps).index(s)] = t
            height[a, list(steps).index(s)] = height_m(t, name)
    zero = np.where(np.isnan(time), np.nan, 0.0)
    rows = SceneRows(time_s=time, e_m=zero, n_m=zero, height_m=height, along_m=np.full_like(time, np.nan),
                     runway=[[None] * len(steps) for _ in names], category=[None] * len(names))
    return scene_edges(rows, Separation(same_nm=3.0, speed_mps=70.0)), steps, names


def check_against_scene_edges() -> int:
    """Every step two aircraft share, both ways: the height `scene_edges` gives is the read row's height moved on by
    its climb (from the row before it; 0 on a first row) for `seen_from`'s time, minus the reader's."""
    edges, steps, names = edges_of_the_example()
    column = EDGE_FEATURES.index("height")
    checked = 0
    for seer, seen in permutations(names, 2):
        for step in sorted(set(figure.hung(seer)) & set(figure.hung(seen))):
            source, moved = figure.seen_from(seer, seen, step)
            times = figure.row_times(seen)
            k = times.index(source)
            climb = (height_m(source, seen) - height_m(times[k - 1], seen)) / figure.STEP_S if k else 0.0
            expected = height_m(source, seen) + climb * moved - height_m(figure.row_on(seer, step), seer)
            t = list(steps).index(step)
            assert edges[t, names.index(seer), names.index(seen), column] == pytest.approx(
                expected / HEIGHT_SCALE_M, abs=1e-6), (seer, seen, step)
            assert abs(moved) < figure.STEP_S
            checked += 1
    return checked


def test_inside_a_step_each_row_reads_the_others_row_as_scene_edges_does():
    """The figure's step: A (104.37 s) reads B's 103.20 s row moved 1.17 s on, B (103.20 s) reads A's 102.37 s row
    moved 0.83 s on — and on every step they share, `scene_edges` reads the same rows."""
    assert figure.seen_from("A", "B", figure.FOCUS_STEP_S) == (103.20, pytest.approx(1.17))
    assert figure.seen_from("B", "A", figure.FOCUS_STEP_S) == (102.37, pytest.approx(0.83))
    assert check_against_scene_edges() == 6                    # steps 102, 104, 106, both ways


def test_a_first_row_later_than_the_readers_is_carried_back_as_scene_edges_does(monkeypatch):
    """C enters at 102.90 s: its first row hangs on 102 s, after A's 102.37 s row there, so A reads it carried back
    0.53 s (no step before it to take instead)."""
    monkeypatch.setattr(figure, "AIRCRAFT", (*figure.AIRCRAFT, ("C", 102.90, 3)))
    assert figure.hung("C") == [102.0, 104.0, 106.0]
    assert figure.seen_from("A", "C", 102.0) == (102.90, pytest.approx(-0.53))
    assert check_against_scene_edges() == 6 + 2 * 3 + 2 * 3   # A–C on 102–106, B–C on 102–106
