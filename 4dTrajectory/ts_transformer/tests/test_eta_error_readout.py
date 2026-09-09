"""B0: the arrival-time error distribution read out of a scored batch.

The readout is arithmetic over `summary.json`, so what can go wrong is arithmetic: a
stratum that quietly holds the wrong flights, an absolute quantile reported as a signed one
(a symmetric interval around a biased head is the wrong shape), or an unscored row averaged
in as a zero. Each of those is asserted here against hand-computed numbers.
"""

from __future__ import annotations

import json

import pytest

import run_ts_eta_error_readout as runner
from ts_transformer.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED

# id, tortuosity, established, remaining path, final-time error, FDE. Three straight-in
# flights and two vectored ones, with errors chosen so every quantile is hand-computable.
_FLIGHTS = [
    ("STR1", 1.00, True, 9_000.0, 10.0, 100.0),
    ("STR2", 1.02, True, 8_000.0, -20.0, 200.0),
    ("STR3", 1.04, True, 7_000.0, 60.0, 600.0),
    ("VEC1", 1.60, False, 30_000.0, -30.0, 3_000.0),
    ("VEC2", 2.40, False, 40_000.0, -90.0, 9_000.0),
]


def _summary(rows: list[dict]) -> dict:
    return {"split": "val", "config": {"prediction_output": "control"}, "results": rows}


def _row(flight, **overrides) -> dict:
    identifier, tortuosity, established, remaining, error, fde = flight
    row = {
        "id": identifier, "runway": "05L", "icao24": identifier.lower(),
        "landing_time_utc": f"2026-01-01T00:{len(identifier):02d}:00Z",
        "route_tortuosity": tortuosity, "established_at_anchor": established,
        "remaining_path_m": remaining, "final_time_error_s": error, "fde_m": fde,
    }
    row.update(overrides)
    return row


def _write(tmp_path, rows: list[dict], name: str = "arm"):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps(_summary(rows)))
    return directory


def test_the_strata_carry_their_own_absolute_and_signed_quantiles(tmp_path) -> None:
    arm = _write(tmp_path, [_row(flight) for flight in _FLIGHTS])
    result = runner.readout("A", arm)

    assert result["flights"] == 5
    assert result["coverage"] == {
        "summary_rows": 5, "scored_rows": 5, "dropped_unscored_rows": 0
    }
    assert result["split"] == "val"

    straight = result["strata"][STRATUM_STRAIGHT_IN]["final_time_error_s"]
    assert straight["abs_p50"] == pytest.approx(20.0)     # |10|, |-20|, |60|
    assert straight["abs_p80"] == pytest.approx(44.0)
    assert straight["abs_p90"] == pytest.approx(52.0)
    # The SIGNED quantiles say which way the head is wrong; the absolute ones cannot.
    assert straight["signed_p10"] == pytest.approx(-14.0)
    assert straight["signed_p50"] == pytest.approx(10.0)
    assert straight["signed_p90"] == pytest.approx(50.0)
    assert straight["mean_signed"] == pytest.approx(50.0 / 3.0)

    vectored = result["strata"][STRATUM_VECTORED]["final_time_error_s"]
    assert result["strata"][STRATUM_VECTORED]["n"] == 2
    assert vectored["abs_p50"] == pytest.approx(60.0)     # |-30|, |-90|
    assert vectored["abs_p80"] == pytest.approx(78.0)
    # Both vectored flights are EARLY: a symmetric interval around this point estimate
    # would be centred on the wrong value, which is the whole reason both are reported.
    assert vectored["signed_p50"] == pytest.approx(-60.0)
    assert vectored["signed_p90"] == pytest.approx(-36.0)

    fde = result["strata"][STRATUM_STRAIGHT_IN]["fde_m"]
    assert fde["abs_p50"] == pytest.approx(200.0)
    assert fde["abs_p80"] == pytest.approx(440.0)
    # FDE is non-negative, so its signed quantiles are its absolute ones.
    assert fde["signed_p50"] == pytest.approx(fde["abs_p50"])


def test_an_unscored_row_is_dropped_and_counted(tmp_path) -> None:
    """A null metric is a flight that was not scored, never a zero error."""
    rows = [_row(flight) for flight in _FLIGHTS]
    rows.append(_row(("NULL1", 1.0, True, 5_000.0, None, None), fde_m=None))
    arm = _write(tmp_path, rows)
    result = runner.readout("A", arm)

    assert result["coverage"] == {
        "summary_rows": 6, "scored_rows": 5, "dropped_unscored_rows": 1
    }
    assert result["strata"][STRATUM_ALL]["n"] == 5


@pytest.mark.parametrize("covariate", runner.STRATA_COVARIATES)
def test_a_row_missing_any_stratum_covariate_cannot_be_stratified(tmp_path, covariate) -> None:
    """Every covariate the strata are cut on, not just the numeric ones.

    A present-but-NULL `established_at_anchor` would pass a "tortuosity is not None" filter
    and then read as False — quietly moving that flight into the vectored stratum, where it
    would widen exactly the interval B is trying to size.
    """
    rows = [_row(flight) for flight in _FLIGHTS]
    rows.append(_row(("NOMIX", 1.6, False, 5_000.0, 5.0, 50.0), **{covariate: None}))
    result = runner.readout("A", _write(tmp_path, rows))
    assert result["coverage"] == {
        "summary_rows": 6, "scored_rows": 5, "dropped_unscored_rows": 1
    }
    assert result["strata"][STRATUM_VECTORED]["n"] == 2       # NOT 3


def test_the_json_is_immutable(tmp_path) -> None:
    arm = _write(tmp_path, [_row(flight) for flight in _FLIGHTS])
    out = tmp_path / "block" / "b0.json"
    assert runner.main([f"one={arm}", "--json", str(out)]) == 0
    with pytest.raises(FileExistsError):
        runner.main([f"one={arm}", "--json", str(out)])


def test_every_arm_is_reported_and_the_json_carries_the_schema(tmp_path) -> None:
    first = _write(tmp_path, [_row(flight) for flight in _FLIGHTS], name="first")
    second = _write(tmp_path, [_row(flight) for flight in _FLIGHTS[:3]], name="second")
    out = tmp_path / "b0.json"
    assert runner.main([f"one={first}", f"two={second}", "--json", str(out)]) == 0

    payload = json.loads(out.read_text())
    assert payload["schema"] == runner.RESULT_SCHEMA
    assert payload["absolute_quantiles"] == [50, 80, 90]
    assert payload["signed_quantiles"] == [10, 50, 90]
    assert [arm["label"] for arm in payload["arms"]] == ["one", "two"]
    assert payload["arms"][1]["flights"] == 3
    assert STRATUM_VECTORED not in payload["arms"][1]["strata"]     # none in that arm


def test_the_text_states_coverage_and_every_stratum(tmp_path, capsys) -> None:
    arm = _write(tmp_path, [_row(flight) for flight in _FLIGHTS])
    runner.main([f"one={arm}"])
    text = capsys.readouterr().out
    assert "coverage: 5 of 5 summary rows" in text
    assert STRATUM_VECTORED in text and STRATUM_STRAIGHT_IN in text
    assert "|dt|p50" in text and "dt p10" in text and "FDE p90" in text


@pytest.mark.parametrize("argv, message", [
    (["nolabel"], "LABEL=PRED_DIR"),
    (["a=x", "a=y"], "used twice"),
])
def test_the_command_line_is_refused(tmp_path, capsys, argv, message) -> None:
    with pytest.raises(SystemExit):
        runner.main(argv)
    assert message in capsys.readouterr().err
