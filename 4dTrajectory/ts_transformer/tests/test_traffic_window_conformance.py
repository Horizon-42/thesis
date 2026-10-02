"""Does a window readout read the same under today's code (`experiments/traffic_window_conformance`, multi-aircraft
design §6.6 step 9.9.2): a readout passes under the code that read it; another reading or another judging constant fails
at the first difference; rows that are not the draw's fail; a dirty checkout writes no record; a commit's record is
written once. On the window loop's fixture, a readout written by the readout program itself (`test_traffic_window_readout`)."""

from __future__ import annotations

import dataclasses
import json
import shutil

import pytest

from ts_transformer.tests.test_traffic_window_readout import _raw, prepared_fixture

CLEAN = {"head": "a" * 40, "dirty": False}


def _readout(tmp_path, monkeypatch):
    """A two-batch readout written by `traffic_window_generation.main` on the fixture (2 samples, seed 5, on the CPU),
    and the stand-in for `prepare` both programs use (it builds the fixture's windows for the configuration given)."""
    from ts_transformer.experiments import traffic_window_conformance as conformance
    from ts_transformer.experiments import traffic_window_generation as runner

    config = runner.load_config(_raw(samples=2, seed=5))
    prepared = prepared_fixture(tmp_path, monkeypatch, config)

    def prepare(given):
        return dataclasses.replace(prepared, config=dataclasses.replace(given, prior_checkpoint_sha256="p",
                                                                         executor_sha256="e"))

    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(conformance, "prepare", prepare)
    monkeypatch.setattr(runner, "git_state", lambda: dict(CLEAN))
    monkeypatch.setattr(conformance, "git_state", lambda: dict(CLEAN))
    (tmp_path / "config.json").write_text(json.dumps(_raw(samples=2, seed=5)))
    readout = tmp_path / "readout"
    assert runner.main(["--config", str(tmp_path / "config.json"), "--out", str(readout), "--workers", "1",
                        "--device", "cpu"]) == 0
    return readout, prepare


def _check(readout, *more):
    from ts_transformer.experiments import traffic_window_conformance as conformance

    return conformance.main(["--readout", str(readout), "--device", "cpu", "--workers", "2", *more])


def test_the_batches_read_again_run_evenly_from_the_first_to_the_last():
    from ts_transformer.experiments.traffic_window_conformance import BATCHES, chosen_batches

    assert BATCHES == 24
    assert chosen_batches(1, 24) == [0] and chosen_batches(2, 24) == [0, 1] and chosen_batches(5, 3) == [0, 2, 4]
    many = chosen_batches(100, 24)
    assert len(many) == 24 and many[0] == 0 and many[-1] == 99 and many == sorted(set(many))


def test_a_readout_passes_under_its_own_code_and_the_record_is_written_once_beside_it(tmp_path, monkeypatch, capsys):
    from ts_transformer.experiments import traffic_window_conformance as conformance
    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments.code_version import KEYS

    readout, _ = _readout(tmp_path, monkeypatch)
    assert _check(readout) == 0
    record = tmp_path / "readout.conformance" / "passed-aaaaaaaaaaaa.json"
    assert conformance.record_path(readout, CLEAN["head"]) == record
    written = json.loads(record.read_text())
    rows = runner.read_aircraft(readout)
    assert (written["batches"], written["checked_batches"], written["rows_compared"], written["rows"]) == \
        (2, [0, 1], len(rows), len(rows))
    assert list(written["code"]) == list(KEYS) and written["code"]["commit"] == CLEAN["head"]
    # the code version is the one a readout read under it carries: its code.json
    assert written["code"] == json.loads((readout / runner.CODE_FILE).read_text())
    assert conformance.passed_records(readout) == [(record, written)]
    assert "the same: " in capsys.readouterr().out
    # never written over
    with pytest.raises(SystemExit):
        _check(readout)
    assert json.loads(record.read_text()) == written


def test_a_dirty_checkout_is_checked_and_nothing_is_written(tmp_path, monkeypatch, capsys):
    from ts_transformer.experiments import traffic_window_conformance as conformance

    readout, _ = _readout(tmp_path, monkeypatch)
    monkeypatch.setattr(conformance, "git_state", lambda: {"head": "b" * 40, "dirty": True})
    assert _check(readout) == 0
    assert "dirty: nothing written" in capsys.readouterr().out
    assert conformance.passed_records(readout) == []


def test_another_reading_fails_at_the_first_difference_and_writes_nothing(tmp_path, monkeypatch, capsys):
    """Today's code reading another temperature than the configuration's: the same draw, other rows."""
    from ts_transformer.experiments import traffic_window_conformance as conformance

    readout, prepare = _readout(tmp_path, monkeypatch)

    def other_reading(given):
        prepared = prepare(given)
        return dataclasses.replace(prepared, config=dataclasses.replace(prepared.config, temperature=0.25))

    monkeypatch.setattr(conformance, "prepare", other_reading)
    assert _check(readout) == 1
    out = capsys.readouterr().out
    assert "NOT THE SAME: batch 0, row (0, 'KXXX:f0', 0, 'scene'): fields [" in out
    assert conformance.passed_records(readout) == []


def test_another_judging_constant_fails(tmp_path, monkeypatch, capsys):
    """Today's judge with another radar minimum (the separation judge's `FAA_RADAR_NM`): the losses move."""
    from ts_transformer.inference import separation

    readout, _ = _readout(tmp_path, monkeypatch)
    monkeypatch.setattr(separation, "FAA_RADAR_NM", 0.01)
    assert _check(readout) == 1
    assert "NOT THE SAME: batch 0, row" in capsys.readouterr().out


def _edited(readout, tmp_path, edit):
    """A writable copy of ``readout`` whose rows ``edit`` changed (a list of dicts in, one out)."""
    from ts_transformer.experiments import traffic_window_generation as runner

    copy = tmp_path / "edited"
    shutil.copytree(readout, copy)
    rows = edit(runner.read_aircraft(copy))
    (copy / runner.AIRCRAFT_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return copy


def test_rows_that_are_not_the_draw_s_fail(tmp_path, monkeypatch, capsys):
    readout, _ = _readout(tmp_path, monkeypatch)

    def moved(rows):                                    # one row of batch 0 filed under batch 1
        first = next(i for i, r in enumerate(rows) if r["batch"] == 0)
        rows[first]["batch"] = 1
        return rows

    assert _check(_edited(readout, tmp_path, moved)) == 1
    assert "NOT THE SAME: batch 1 holds other (window, flight) pairs" in capsys.readouterr().out
    shutil.rmtree(tmp_path / "edited")

    def extra(rows):                                    # a batch today's draw does not make
        rows[-1]["batch"] = 5
        return rows

    assert _check(_edited(readout, tmp_path, extra)) == 1
    assert "today's draw makes 2" in capsys.readouterr().out
    shutil.rmtree(tmp_path / "edited")

    def dropped(rows):                                  # a sample's row gone: the pairs stay, the rows do not
        return [r for r in rows if not (r["batch"] == 0 and r["source"] == "scene" and r["sample"] == 1
                                        and r["dataset_id"] == "KXXX:f0")]

    assert _check(_edited(readout, tmp_path, dropped)) == 1
    assert "NOT THE SAME: batch 0: " in capsys.readouterr().out


def test_a_row_with_other_fields_is_named_as_such(tmp_path, monkeypatch, capsys):
    readout, _ = _readout(tmp_path, monkeypatch)

    def renamed(rows):
        rows[0]["note"] = rows[0].pop("reward")
        return rows

    assert _check(_edited(readout, tmp_path, renamed)) == 1
    assert "other fields — ['note'] only in the readout, ['reward'] only today" in capsys.readouterr().out
