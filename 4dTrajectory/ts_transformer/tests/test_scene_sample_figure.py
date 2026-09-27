"""The multi-aircraft design's §2.3 figure (`experiments/scene_sample_figure`): the file in the docs is what the runner
draws, and the runner's cuts follow the design's rule."""

from ts_transformer.experiments import scene_sample_figure as figure
from ts_transformer.experiments.scene_sample_figure import minutes


def test_the_figure_in_the_docs_is_what_the_runner_draws():
    """Change the example or the drawing, then run `python run_ts.py scene_sample_figure` and commit the SVG with it."""
    assert figure.FIGURE.read_text(encoding="utf-8") == figure.render()


def test_the_example_is_cut_where_the_rule_says():
    cuts = figure.cut_times(figure.EXAMPLE)
    # sample 1: the fewest aircraft between minute 15 and 20 is one (D alone), first at 15:44; sample 2 likewise from there
    assert [figure.mmss(cut) for cut in cuts] == ["15:44", "31:50"]
    assert [[f.name for f in figure.carried(figure.EXAMPLE, cut)] for cut in cuts] == [["D"], ["H"]]


def test_a_segment_of_at_most_twenty_minutes_is_one_sample():
    assert figure.cut_times((minutes(0.0, 9.0, "A"), minutes(5.0, 19.0, "B"))) == []
    assert figure.cut_times((minutes(0.0, 20.0, "A"),)) == []


def test_the_window_holds_both_its_ends_and_ties_take_the_earliest_step():
    # 30 minutes, one aircraft all the way: every step of the window ties, the earliest (minute 15) is taken
    alone = (minutes(0.0, 30.0, "A"),)
    assert figure.cut_times(alone) == [figure.SEARCH_FROM_S]
    # two aircraft until just before minute 20, one at minute 20: the window's last step is taken
    assert figure.cut_times((minutes(0.0, 40.0, "A"), minutes(10.0, 19.99, "B"))) == [figure.SAMPLE_MAX_S]


def test_a_background_aircraft_counts_in_the_scene():
    flights = (minutes(0.0, 35.0, "A"), minutes(5.0, 16.0, "B"))
    assert figure.cut_times(flights) == [16 * 60.0 + 2.0]                  # the first step after B leaves
    background = minutes(16.02, 17.0, "X", speaking=False)
    assert figure.cut_times((*flights, background)) == [17 * 60.0 + 2.0]   # the first step after X leaves


def test_the_step_at_a_cut_is_the_later_samples():
    """A flight whose last row is at the cut's step is split (that step's loss is in the later sample); one whose
    first row is at the cut's step is not (it has no row before the cut)."""
    cut = figure.SEARCH_FROM_S
    leaves, enters = minutes(0.0, 15.0, "leaves"), minutes(15.0, 30.0, "enters")
    assert figure.carried((leaves, enters), cut) == [leaves]
