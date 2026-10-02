"""A window readout's configuration and files (`experiments/traffic_window_generation`, multi-aircraft design §6.6 step
9.9.1, 9.9.5): the one loader of a configuration (unknown and missing required keys refused, defaults filled in, values
checked), the checksums against the disk, and a readout run from its configuration file — the same rows in one and in
two processes, five files of one kind each. On the window loop's fixture (`test_traffic_window`)."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from ts_transformer.instructions.words import Words
from ts_transformer.tests.test_traffic_window import (
    LIMIT_S, STEP_S, _airport, _as_drawn, _patch_runner_physics, _pool, _traffic_model,
)


def _raw(**overrides):
    """A configuration's required keys (and ``overrides``)."""
    from ts_transformer.experiments.traffic_window_generation import PROGRAM

    return {"program": PROGRAM, "prior": "4dTrajectory/outputs/POOLED/prior/run/round_05",
            "executor": "4dTrajectory/outputs/POOLED/executor/spec", "split": "val",
            "instructions": "4dTrajectory/outputs/POOLED/instruction_language/v6", **overrides}


def test_a_configuration_takes_the_defaults_of_the_keys_it_lacks_and_names_its_paths_from_the_repository():
    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.repo_layout import REPO_ROOT

    config = runner.load_config(_raw())
    assert config == runner.ReadoutConfig(**_raw())
    assert (config.commanded, config.windows_per_airport, config.samples, config.temperature, config.seed,
            config.augment_seed, config.model_sources, config.probe_samples, config.probe_margin,
            config.aircraft_steps, config.prior_checkpoint_sha256) == \
        ("every", runner.WINDOWS_PER_AIRPORT, 4, 1.0, 1337, None, runner.MODEL_SOURCES, 0, runner.PROBE_MARGIN,
         runner.AIRCRAFT_STEPS, None)
    # every key written, in the design's order; a completed configuration loads as itself
    assert set(runner.CONFIG_KEYS) == {f.name for f in dataclasses.fields(runner.ReadoutConfig)}
    written = config.as_json()
    assert list(written) == list(runner.CONFIG_KEYS) and written["model_sources"] == list(runner.MODEL_SOURCES)
    assert runner.load_config(json.loads(json.dumps(written))) == config
    assert runner.REQUIRED == ("program", "prior", "executor", "instructions", "split")
    # a path given from the root is named from the repository; a whole-number temperature is a number
    named = runner.load_config(_raw(prior=str(REPO_ROOT / "4dTrajectory/outputs/POOLED/prior/run/round_05"),
                                    temperature=2))
    assert named.prior == config.prior and named.temperature == 2.0 and isinstance(named.temperature, float)
    assert runner.load_config(_raw(model_sources=["scene"], augment_seed=7919)).model_sources == ("scene",)


@pytest.mark.parametrize("change, refusal", [
    ({"window": 3}, r"unknown configuration keys \['window'\]"),
    ({"split": None}, r"'split' is None, not a text"),
    ({"samples": None}, r"'samples' is None, not a count"),
    ({"samples": True}, r"'samples' is True, not a count"),
    ({"temperature": "1"}, r"'temperature' is '1', not a number"),
    ({"model_sources": "scene"}, r"'model_sources' is 'scene', not a names"),
    ({"prior_checkpoint_sha256": 3}, r"not a sha"),
    ({"program": "traffic_window_rewind"}, r"program 'traffic_window_rewind' is not"),
    ({"split": "test"}, r"split 'test' is not one of"),
    ({"commanded": "two"}, r"commanded 'two' is not one of"),
    ({"model_sources": ["alone", "scene"]}, r"in that order, each once"),
    ({"model_sources": ["scene", "scene"]}, r"in that order, each once"),
    ({"model_sources": []}, r"in that order, each once"),
    ({"model_sources": ["labelled"]}, r"in that order, each once"),
    ({"samples": 0}, r"at least 1"),
    ({"windows_per_airport": 0}, r"at least 1"),
    ({"aircraft_steps": 0}, r"at least 1"),
    ({"probe_samples": 5}, r"probe_samples 5 of 4 samples"),
    ({"probe_samples": -1}, r"probe_samples -1 of 4 samples"),
    ({"temperature": 0.0}, r"temperature 0.0 is not finite and positive"),
    ({"temperature": math.inf}, r"temperature inf is not finite and positive"),
    ({"probe_margin": math.nan}, r"probe_margin nan is not finite"),
])
def test_a_configuration_with_an_unknown_key_a_mistyped_value_or_one_out_of_range_is_refused(change, refusal):
    from ts_transformer.experiments import traffic_window_generation as runner

    with pytest.raises(ValueError, match=refusal):
        runner.load_config(_raw(**change))


@pytest.mark.parametrize("name", ["program", "prior", "executor", "instructions", "split"])
def test_a_configuration_lacking_a_key_without_a_default_is_refused(name):
    from ts_transformer.experiments import traffic_window_generation as runner

    raw = _raw()
    del raw[name]
    with pytest.raises(ValueError, match=rf"the configuration lacks \['{name}'\]"):
        runner.load_config(raw)


class _Built(Exception):
    """`prepare` got past the checksums (the test's stand-in prior raises it)."""


def test_a_checksum_the_configuration_names_must_be_the_disk_s(monkeypatch):
    from ts_transformer.experiments import traffic_window_generation as runner

    monkeypatch.setattr(runner.replay, "open_executor", lambda executor, instructions: (None, {"sha256": "e"}, None))
    monkeypatch.setattr(runner, "file_sha256", lambda path: "p")

    def built(*args):
        raise _Built

    monkeypatch.setattr(runner, "window_prior", built)
    for named in ({}, {"prior_checkpoint_sha256": "p", "executor_sha256": "e"}):
        with pytest.raises(_Built):
            runner.prepare(runner.load_config(_raw(**named)))
    with pytest.raises(ValueError, match="prior_checkpoint_sha256 is q, the disk's p: the data changed"):
        runner.prepare(runner.load_config(_raw(prior_checkpoint_sha256="q")))
    with pytest.raises(ValueError, match="executor_sha256 is f, the disk's e: the data changed"):
        runner.prepare(runner.load_config(_raw(executor_sha256="f")))


def prepared_fixture(tmp_path, monkeypatch, config):
    """A `Prepared` of ``config`` (its checksums "p" and "e") on the window loop's fixture: two windows — f0 and f1
    together, f2 alone an hour later — a batch each, read by the fixture's traffic model on the CPU."""
    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments.traffic_window import window_of
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params
    from ts_transformer.tests.test_traffic_speaking import _batch

    _, airports, spec = _airport(tmp_path, monkeypatch, [0.0, 40.0, 3_600.0], (0, 1, 2))
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    commanded = [("KXXX:f0", "KXXX:f1"), ("KXXX:f2",)]
    keys = [k for keys in commanded for k in keys]
    windows = [window_of(airport, 0.0, c, [LIMIT_S] * len(c), STEP_S) for c in commanded]
    drawn = _as_drawn(windows, [range(0, 2), range(2, 3)], _batch(airport, signals, spec, keys), [LIMIT_S] * 3)
    _patch_runner_physics(monkeypatch, signals, airport.flights.geometry, keys)
    return runner.Prepared(dataclasses.replace(config, prior_checkpoint_sha256="p", executor_sha256="e"),
                           _traffic_model(spec), "zero", ProcedureMasks.none(), _params(), Words(spec), None,
                           {"KXXX": _pool(airport)}, {"KXXX": 3}, {"KXXX": {"drawn": 2}}, drawn, None, [[0], [1]])


def test_a_readout_runs_from_its_configuration_reads_the_same_rows_in_one_and_two_processes_and_writes_five_files(
        tmp_path, monkeypatch):
    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments.code_version import KEYS

    config = runner.load_config(_raw(samples=2, seed=5))
    prepared = prepared_fixture(tmp_path, monkeypatch, config)
    asked = []

    def prepare(given):
        asked.append(given)
        return dataclasses.replace(prepared, config=dataclasses.replace(given, prior_checkpoint_sha256="p",
                                                                         executor_sha256="e"))

    monkeypatch.setattr(runner, "prepare", prepare)
    (tmp_path / "config.json").write_text(json.dumps(_raw(samples=2, seed=5)))
    for workers in (1, 2):
        assert runner.main(["--config", str(tmp_path / "config.json"), "--out", str(tmp_path / f"r{workers}"),
                            "--workers", str(workers), "--device", "cpu"]) == 0
    assert asked == [config, config]
    one, two = tmp_path / "r1", tmp_path / "r2"
    assert sorted(p.name for p in one.iterdir()) == sorted(
        (runner.CONFIG_FILE, runner.CODE_FILE, runner.AIRCRAFT_FILE, runner.SUMMARY_FILE, runner.RUN_FILE))
    assert (one / runner.AIRCRAFT_FILE).read_bytes() == (two / runner.AIRCRAFT_FILE).read_bytes()
    rows = runner.read_aircraft(one)
    assert [r["batch"] for r in rows] == sorted(r["batch"] for r in rows) and {r["batch"] for r in rows} == {0, 1}
    # the configuration completed with both checksums
    written = json.loads((one / runner.CONFIG_FILE).read_text())
    assert list(written) == list(runner.CONFIG_KEYS)
    assert runner.load_config(written) == dataclasses.replace(config, prior_checkpoint_sha256="p", executor_sha256="e")
    code = json.loads((one / runner.CODE_FILE).read_text())
    assert list(code) == list(KEYS) and (code["device"], code["gpu"]) == ("cpu", None)
    assert code["constants"] == json.loads(json.dumps(runner.READING_CONSTANTS))
    assert code == json.loads((two / runner.CODE_FILE).read_text())
    summary = json.loads((one / runner.SUMMARY_FILE).read_text())
    assert list(summary) == ["schema", "model", "drawn", "augmenting", "scenes", "batches", "readout"]
    assert summary["schema"] == runner.SUMMARY_SCHEMA and summary["batches"] == 2 and summary["augmenting"] is None
    assert summary == json.loads((two / runner.SUMMARY_FILE).read_text())
    run = json.loads((two / runner.RUN_FILE).read_text())
    assert list(run) == ["finished_utc", "elapsed_s", "workers", "gpu_peak_gb_a_process"] and run["workers"] == 2


def test_a_readout_is_never_written_over(tmp_path):
    from ts_transformer.experiments import traffic_window_generation as runner

    (tmp_path / "config.json").write_text(json.dumps(_raw()))
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit):
        runner.main(["--config", str(tmp_path / "config.json"), "--out", str(tmp_path / "out")])
