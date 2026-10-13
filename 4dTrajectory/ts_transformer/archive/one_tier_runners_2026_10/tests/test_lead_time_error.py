"""The lead-time error readout: displacement against lead time, per stratum.

The readout is arithmetic over stored records, so what can go wrong is arithmetic: a
flight counted at a lead its record never reaches (or held at its last node there), a
stratum holding the wrong flights, the short-horizon ADE averaged over the wrong span, a
growth exponent read across two different cohorts, a lead silently rounded onto its
neighbour's key. Each is asserted against hand-computed numbers on records whose
prediction and truth differ in HEIGHT only, so the displacement is the height offset
exactly, whatever the chart.
"""

from __future__ import annotations

import json

import pytest

import ts_transformer.experiments.lead_time_error as runner
from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED

LAT, LON = 35.8, -78.8
LOOKBACK_S = (-10.0, -8.0, -6.0, -4.0, -2.0)


def _state(t: float, alt: float) -> dict:
    return {"t": t, "lat": LAT, "lon": LON, "alt": alt, "V": 80.0, "psi": 0.0,
            "gamma": 0.0, "m": 60_000.0}


def _flight(identifier: str, tortuosity: float, established: bool, remaining: float,
            t_end: float, offset_per_s: float, ade_m: float, *,
            truth_end: float | None = None, lookback_alt: float = 1000.0) -> tuple[dict, dict]:
    """A record whose predicted height runs ``offset_per_s`` metres per second ABOVE the
    truth (both on a 2 s clock from the anchor), so the displacement at lead h is exactly
    ``offset_per_s * h``. The prediction ends at ``t_end``, the truth at ``truth_end``
    (default: the same); the truth's lookback rows sit at ``lookback_alt``."""
    row = {
        "id": identifier, "runway": "05L", "icao24": identifier.lower(),
        "landing_time_utc": f"2026-01-01T00:{len(identifier):02d}:00Z",
        "route_tortuosity": tortuosity, "established_at_anchor": established,
        "remaining_path_m": remaining, "ade_m": ade_m,
        "states_file": f"{identifier}_states.json",
    }
    truth_end = t_end if truth_end is None else truth_end
    pred_clock = [2.0 * i for i in range(int(t_end // 2) + 1)]
    truth_clock = [2.0 * i for i in range(int(truth_end // 2) + 1)]
    states = {
        "predicted_states": [_state(t, 1000.0 + offset_per_s * t) for t in pred_clock],
        "observed_states": [_state(t, lookback_alt) for t in LOOKBACK_S]
        + [_state(t, 1000.0) for t in truth_clock],
    }
    return row, states


def _write(tmp_path, flights, name: str = "arm"):
    directory = tmp_path / name
    directory.mkdir()
    rows = []
    for row, states in flights:
        rows.append(row)
        (directory / row["states_file"]).write_text(json.dumps(states))
    summary = {"split": "val", "config": {"prediction_output": "control"}, "results": rows}
    (directory / "summary.json").write_text(json.dumps(summary))
    return directory


def test_displacement_short_horizon_ade_and_coverage_per_lead(tmp_path) -> None:
    # Two straight-in flights (1 and 3 m/s of height drift, 120 s and 40 s long) and one
    # vectored flight (10 m/s, 200 s). At lead 60 s the 40 s flight is absent, not zero.
    arm = _write(tmp_path, [
        _flight("STR1", 1.00, True, 9_000.0, 120.0, 1.0, 60.0),
        _flight("STR2", 1.02, True, 8_000.0, 40.0, 3.0, 60.0),
        _flight("VEC1", 1.60, False, 30_000.0, 200.0, 10.0, 1000.0),
    ])
    result = runner.readout("A", arm, leads=(10.0, 30.0, 60.0, 150.0))

    assert result["flights"] == 3
    assert result["coverage"] == {"summary_rows": 3, "scored_rows": 3, "dropped_unscored_rows": 0}
    straight = result["strata"][STRATUM_STRAIGHT_IN]
    assert straight["n"] == 2
    at30 = straight["at_lead"]["30"]
    assert at30["n"] == 2
    assert at30["mean"] == pytest.approx((30.0 + 90.0) / 2)
    assert at30["p50"] == pytest.approx(60.0)
    at60 = straight["at_lead"]["60"]
    assert at60["n"] == 1                       # STR2 ends at 40 s: absent, never 0
    assert at60["mean"] == pytest.approx(60.0)
    assert "150" not in straight["at_lead"]     # nobody reaches it
    # ADE over [0, h] is the mean of the 1 s grid 0..h inclusive: a ramp a*t averages to
    # a*h/2 exactly on that grid.
    assert straight["ade_to"]["30"]["mean"] == pytest.approx((15.0 + 45.0) / 2)
    assert straight["ade_to"]["60"]["mean"] == pytest.approx(30.0)

    everyone = result["strata"][STRATUM_ALL]
    # p90 on a three-flight lead: numpy's linear interpolation between 90 and 300.
    assert everyone["at_lead"]["30"]["p90"] == pytest.approx(90.0 + 0.8 * (300.0 - 90.0))
    assert everyone["whole_record_ade"]["p50"] == pytest.approx(60.0)

    vectored = result["strata"][STRATUM_VECTORED]
    assert vectored["n"] == 1
    assert vectored["at_lead"]["150"] == {"n": 1, "mean": pytest.approx(1500.0),
                                          "p50": pytest.approx(1500.0),
                                          "p90": pytest.approx(1500.0)}
    assert vectored["whole_record_ade"]["mean"] == pytest.approx(1000.0)


def test_the_curve_stops_at_the_earlier_of_the_two_records(tmp_path) -> None:
    # PRED_LONG's prediction runs to 120 s over a truth that ends at 40 s; TRUTH_LONG's
    # truth runs to 120 s under a prediction that ends at 40 s. Both are present at 30 s
    # and absent at 60 s — neither series is held past its own end.
    arm = _write(tmp_path, [
        _flight("PREDLONG", 1.00, True, 9_000.0, 120.0, 1.0, 30.0, truth_end=40.0),
        _flight("TRUTHLONG", 1.00, True, 9_000.0, 40.0, 1.0, 30.0, truth_end=120.0),
    ])
    result = runner.readout("A", arm, leads=(30.0, 60.0))
    block = result["strata"][STRATUM_ALL]
    assert block["at_lead"]["30"]["n"] == 2
    assert block["at_lead"]["30"]["mean"] == pytest.approx(30.0)
    assert "60" not in block["at_lead"]


def test_the_growth_exponent_is_paired_over_the_flights_present_at_both_leads(tmp_path) -> None:
    # Two flights with LINEAR growth (exponent 1 each): 1 m/s over 200 s and 3 m/s over
    # 40 s. Unpaired medians would read 20 m at 10 s and 60 m at 60 s — a slope of
    # ln 3 / ln 6 = 0.61 that is true of no flight. Paired over the flight present at
    # both leads the slope is 1, with n = 1 said.
    arm = _write(tmp_path, [
        _flight("LONG", 1.00, True, 9_000.0, 200.0, 1.0, 100.0),
        _flight("SHORT", 1.02, True, 9_000.0, 40.0, 3.0, 100.0),
    ])
    result = runner.readout("A", arm, leads=(10.0, 60.0, 120.0))
    slopes = result["strata"][STRATUM_ALL]["paired_p50_log_log_slope"]
    assert slopes["10-60"] == {"n": 1, "p50_short": pytest.approx(10.0),
                               "p50_long": pytest.approx(60.0), "slope": pytest.approx(1.0)}
    assert slopes["60-120"]["slope"] == pytest.approx(1.0)
    # A quadratic law reads 2: the exponent is read off the medians, not assumed.
    assert runner.log_log_slope(10.0, 100.0, 60.0, 3600.0) == pytest.approx(2.0)
    # A pair whose leads were not asked for is simply not read.
    only_ten = runner.readout("A", arm, leads=(10.0, 30.0))
    assert only_ten["strata"][STRATUM_ALL]["paired_p50_log_log_slope"] == {}


def test_the_lookback_is_not_the_truth_and_a_thin_record_refuses_the_arm(tmp_path) -> None:
    # Five lookback rows at another height plus ONE row from the anchor: without the
    # t >= 0 filter this record would pass as six rows and interpolate over the lookback.
    row, states = _flight("STR1", 1.00, True, 9_000.0, 60.0, 1.0, 30.0, lookback_alt=5000.0)
    states["observed_states"] = states["observed_states"][: len(LOOKBACK_S) + 1]
    arm = _write(tmp_path, [(row, states)])
    with pytest.raises(SystemExit, match="fewer than two rows from the anchor"):
        runner.readout("A", arm, leads=(10.0,))
    # ...and with two rows from the anchor the lookback's height never enters.
    row, states = _flight("STR2", 1.00, True, 9_000.0, 2.0, 1.0, 30.0, lookback_alt=5000.0)
    arm = _write(tmp_path, [(row, states)], name="arm2")
    result = runner.readout("A", arm, leads=(1.0,))
    assert result["strata"][STRATUM_ALL]["at_lead"]["1"]["mean"] == pytest.approx(1.0)


def test_leads_off_the_grid_are_refused_not_rounded() -> None:
    with pytest.raises(SystemExit, match="not a positive whole multiple"):
        runner.validated_leads((10.5,))
    with pytest.raises(SystemExit, match="not a positive whole multiple"):
        runner.validated_leads((-10.0,))
    with pytest.raises(SystemExit, match="given twice"):
        runner.validated_leads((10.0, 10.0))
    assert runner.validated_leads((10, 60.0)) == (10.0, 60.0)


def test_an_unscored_row_is_dropped_and_counted_and_an_empty_stratum_is_absent(tmp_path) -> None:
    good, states = _flight("STR1", 1.00, True, 9_000.0, 60.0, 1.0, 30.0)
    bad, bad_states = _flight("STR2", 1.00, True, 9_000.0, 60.0, 1.0, 30.0)
    bad["established_at_anchor"] = None        # present-but-null: dropped, never False
    arm = _write(tmp_path, [(good, states), (bad, bad_states)])
    result = runner.readout("A", arm, leads=(10.0,))
    assert result["coverage"] == {"summary_rows": 2, "scored_rows": 1, "dropped_unscored_rows": 1}
    assert result["strata"][STRATUM_ALL]["n"] == 1
    assert STRATUM_VECTORED not in result["strata"]


def test_main_renders_two_arms_and_refuses_to_overwrite_either_artifact(tmp_path, capsys) -> None:
    arm_a = _write(tmp_path, [_flight("STR1", 1.00, True, 9_000.0, 60.0, 1.0, 30.0)], "a")
    arm_b = _write(tmp_path, [_flight("VEC1", 1.60, False, 30_000.0, 60.0, 2.0, 60.0)], "b")
    out = tmp_path / "block.json"
    assert runner.main([f"A={arm_a}", f"B={arm_b}", "--leads", "10,30", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "── A —" in text and "── B —" in text
    assert "      10     1         10      10      10" in text
    assert "      30     1         60      60      60" in text
    payload = json.loads(out.read_text())
    assert payload["schema"] == runner.RESULT_SCHEMA
    assert [arm["label"] for arm in payload["arms"]] == ["A", "B"]
    assert out.with_suffix(".txt").read_text() == runner.render(payload)
    # Immutable: the JSON, the rendering, and a --json that names the rendering's suffix.
    with pytest.raises(FileExistsError, match="immutable"):
        runner.main([f"A={arm_a}", "--json", str(out)])
    out.unlink()
    with pytest.raises(FileExistsError, match="immutable"):
        runner.main([f"A={arm_a}", "--json", str(out)])
    with pytest.raises(SystemExit):
        runner.main([f"A={arm_a}", "--json", str(tmp_path / "other.txt")])
    with pytest.raises(SystemExit):
        runner.main([f"A={arm_a}", f"A={arm_b}"])
    with pytest.raises(SystemExit, match="not a positive whole multiple"):
        runner.main([f"A={arm_a}", "--leads", "10.5"])


def test_render_refuses_another_schema() -> None:
    with pytest.raises(SystemExit, match="schema"):
        runner.render({"schema": "something-else", "grid_dt_s": 1.0, "arms": []})
