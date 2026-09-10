"""The evaluation protocol: the sealed test split, its one-shot release, the development cohort, the CLI refusals.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_TS_DIR = Path(__file__).resolve().parents[1]

_CLI_SPEC = importlib.util.spec_from_file_location(
    "ts_transformer_cli_test", _TS_DIR / "__main__.py"
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
ts_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(ts_cli)

import ts_transformer.cli.common as cli_common  # noqa: E402
import ts_transformer.cli.train as cli_train  # noqa: E402
import ts_transformer.evaluation_protocol as evaluation_protocol  # noqa: E402
import ts_transformer.experiment_index as experiment_index  # noqa: E402
from ts_transformer.config import (
    AIRCRAFT_FILTER_OPENAP_DIRECT,
    CONTROL_RECIPE_SIMPLE_V3,
    PREDICTION_CONTROL,
    TSConfig,
)
from ts_transformer.dataset import build_series
from ts_transformer.development_cohorts import DEVELOPMENT_COHORT_SCHEMA, DevelopmentCohort  # noqa: E402
from ts_transformer.synthetic import synthetic_arrivals  # noqa: E402
from ts_transformer.tests.support import fake_data_provenance

AIRPORT, RUNWAY = "KRDU", "05L"


def _run_development_cohort_train_cli(
    monkeypatch, tmp_path, *, built_ids
):
    cohort = DevelopmentCohort(
        name="KRDU-05L-cluster-0",
        train_flight_ids=("KRDU:train",),
        val_flight_ids=("KRDU:val",),
        selection={"kind": "approach-cluster"},
    )
    outer_splits = {
        "train": ["KRDU:train"],
        "val": ["KRDU:val"],
        "test": ["KRDU:test"],
    }
    captured = {}
    monkeypatch.setattr(cli_common, "load_development_cohort", lambda _path: cohort)
    monkeypatch.setattr(
        cli_common, "arrival_data_provenance", lambda _data: fake_data_provenance()
    )
    monkeypatch.setattr(
        cli_common, "flight_keys_by_split", lambda _provenance, _config: outer_splits
    )
    monkeypatch.setattr(
        cli_common, "load_flight_dicts", lambda _data, include_flight_keys: [{}]
    )
    monkeypatch.setattr(
        cli_common,
        "build_series_or_exit",
        lambda *_args: (
            [SimpleNamespace(dataset_id=dataset_id) for dataset_id in built_ids],
            SimpleNamespace(to_dict=lambda: {"built": len(built_ids)}),
        ),
    )
    monkeypatch.setattr(cli_common, "data_selection_audit", lambda *_args: {})
    monkeypatch.setattr(cli_common, "development_cohort_audit", lambda *_args: {})

    def capture_train(_series, _config, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(cli_train, "run_training", capture_train)
    result = ts_cli.main([
        "train",
        "--data", str(tmp_path / "manifest.json"),
        "--output-dir", str(tmp_path / "run"),
        "--development-cohort", str(tmp_path / "cohort.json"),
    ])
    return result, captured


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config
def test_test_release_is_checkpoint_bound_and_one_shot_per_flight(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    provenance = fake_data_provenance()
    payload = {
        evaluation_protocol.TEST_RELEASE_PROTOCOL_FIELD:
            evaluation_protocol.TEST_RELEASE_SCHEMA,
        "data_provenance": provenance,
        "split": {"test": ["KRDU:flight-1", "KRDU:flight-2"]},
    }

    release = evaluation_protocol.create_test_release(
        checkpoint, payload, provenance
    )
    claim = evaluation_protocol.begin_test_evaluation(
        checkpoint,
        payload,
        provenance,
        ["KRDU:flight-1"],
        output_dir=tmp_path / "prediction",
    )
    evaluation_protocol.complete_test_evaluation(checkpoint, claim)

    recorded = json.loads(release.read_text(encoding="utf-8"))
    assert recorded["status"] == "partially_evaluated"
    assert recorded["claims"][0]["status"] == "complete"
    with pytest.raises(evaluation_protocol.TestReleaseError, match="already exposed"):
        evaluation_protocol.begin_test_evaluation(
            checkpoint,
            payload,
            provenance,
            ["KRDU:flight-1"],
            output_dir=tmp_path / "repeat",
        )

    second_claim = evaluation_protocol.begin_test_evaluation(
        checkpoint,
        payload,
        provenance,
        ["KRDU:flight-2"],
        output_dir=tmp_path / "second",
    )
    evaluation_protocol.complete_test_evaluation(checkpoint, second_claim)
    recorded = json.loads(release.read_text(encoding="utf-8"))
    assert recorded["status"] == "complete"

    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(evaluation_protocol.TestReleaseError, match="checkpoint"):
        evaluation_protocol.begin_test_evaluation(
            checkpoint,
            payload,
            provenance,
            ["KRDU:flight-2"],
            output_dir=tmp_path / "changed",
        )


def test_test_evaluation_requires_a_frozen_release(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    provenance = fake_data_provenance()
    payload = {
        evaluation_protocol.TEST_RELEASE_PROTOCOL_FIELD:
            evaluation_protocol.TEST_RELEASE_SCHEMA,
        "data_provenance": provenance,
        "split": {"test": ["KRDU:flight-1"]},
    }

    with pytest.raises(evaluation_protocol.TestReleaseError, match="freeze-test"):
        evaluation_protocol.begin_test_evaluation(
            checkpoint,
            payload,
            provenance,
            ["KRDU:flight-1"],
            output_dir=tmp_path / "prediction",
        )


def test_legacy_checkpoint_cannot_be_released_as_a_fresh_blind_test(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"legacy checkpoint")
    provenance = fake_data_provenance()
    payload = {
        "data_provenance": provenance,
        "split": {"test": ["KRDU:flight-1"]},
    }

    with pytest.raises(evaluation_protocol.TestReleaseError, match="predates"):
        evaluation_protocol.create_test_release(checkpoint, payload, provenance)


def test_development_cohort_checkpoint_cannot_be_frozen_for_test(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"development-only checkpoint")
    provenance = fake_data_provenance()
    payload = {
        evaluation_protocol.TEST_RELEASE_PROTOCOL_FIELD:
            evaluation_protocol.TEST_RELEASE_SCHEMA,
        "data_provenance": provenance,
        "data_selection": {
            "development_cohort": {
                "schema_version": DEVELOPMENT_COHORT_SCHEMA,
                "name": "KRDU-05L-cluster-0",
            },
        },
        "split": {"test": ["KRDU:flight-1"]},
    }

    with pytest.raises(
        evaluation_protocol.TestReleaseError, match="development-only cohort"
    ):
        evaluation_protocol.create_test_release(checkpoint, payload, provenance)


def test_development_cohort_checkpoint_has_no_outer_test_roster(
    monkeypatch, tmp_path
):
    result, captured = _run_development_cohort_train_cli(
        monkeypatch,
        tmp_path,
        built_ids=("KRDU:train", "KRDU:val"),
    )

    assert result == 0
    assert captured["reserved_test_keys"] == []


def test_development_cohort_rejects_incomplete_rebuild(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run_development_cohort_train_cli(
            monkeypatch,
            tmp_path,
            built_ids=("KRDU:train",),
        )


@pytest.mark.parametrize(
    ("field", "value", "extra_argv"),
    [
        # The hook needs a control run to be a legal config at all, so the override is
        # applied to one: the refusal under test is "not available", not "not applicable".
        ("control_command_hook", "nominal-residual",
         ["--prediction-output", PREDICTION_CONTROL, "--control-recipe-name", CONTROL_RECIPE_SIMPLE_V3]),
        ("state_position_reference", "anchor-relative", []),
    ],
)
def test_a_value_a_new_run_may_not_select_does_not_begin_a_formal_run(
    monkeypatch, tmp_path, capsys, field, value, extra_argv
):
    """`--config-overrides` is the second door into a field whose flag already refuses it.

    Both values are legal in a STORED config (their artifacts and checkpoints must keep
    loading) and neither may be selected for a new run — one is archived code, one was
    vetoed by its own campaign. Without the check at the config boundary the run gets a
    dataset build and a formal experiment manifest before dying deep in the training loop,
    which leaves a half-open run in the experiment index.
    """
    began_run = False
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({field: value}), encoding="utf-8")
    monkeypatch.setattr(
        cli_common, "arrival_data_provenance", lambda _data: fake_data_provenance()
    )
    monkeypatch.setattr(
        cli_common,
        "flight_keys_by_split",
        lambda _provenance, _config: {
            "train": ["KRDU:train"], "val": ["KRDU:val"], "test": ["KRDU:test"]
        },
    )
    monkeypatch.setattr(cli_common, "load_flight_dicts", lambda *_args, **_kwargs: [{}])
    monkeypatch.setattr(
        cli_common,
        "build_series_or_exit",
        lambda *_args: (
            [SimpleNamespace(dataset_id="KRDU:train")],
            SimpleNamespace(to_dict=lambda: {}),
        ),
    )
    monkeypatch.setattr(cli_common, "data_selection_audit", lambda *_args: {})

    def record_begin(*_args, **_kwargs):
        nonlocal began_run
        began_run = True
        return tmp_path / "run" / experiment_index.RUN_MANIFEST_NAME

    monkeypatch.setattr(cli_common, "begin_run", record_begin)
    monkeypatch.setattr(
        cli_train, "run_training",
        lambda *_args, **_kwargs: pytest.fail("train must not start"),
    )

    with pytest.raises(SystemExit):
        ts_cli.main([
            "train",
            "--data", str(tmp_path / "manifest.json"),
            "--output-dir", str(tmp_path / "run"),
            "--config-overrides", str(overrides),
            "--campaign-id", "retired-vocabulary",
            "--experiment-id", field,
            *extra_argv,
        ])

    assert f"{field}={value!r} cannot be selected" in capsys.readouterr().err
    assert not began_run


def test_predict_refuses_the_archived_hook_at_the_parser(tmp_path, capsys):
    """The predict flag's own choices: `--command-hook` names what can still be flown."""
    with pytest.raises(SystemExit) as info:
        ts_cli.main([
            "predict",
            "--checkpoint", str(tmp_path / "checkpoint.pt"),
            "--data", str(tmp_path / "manifest.json"),
            "--output-dir", str(tmp_path / "prediction"),
            "--command-hook", "nominal-residual",
            "--hook-saturation", "soft",
        ])
    assert info.value.code == 2
    assert "invalid choice: 'nominal-residual'" in capsys.readouterr().err


def test_predict_cli_refuses_test_without_explicit_release(tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        ts_cli.main([
            "predict",
            "--checkpoint", str(tmp_path / "checkpoint.pt"),
            "--data", str(tmp_path / "manifest.json"),
            "--output-dir", str(tmp_path / "prediction"),
            "--split", "test",
        ])
    assert info.value.code == 2 and "--split test is sealed" in capsys.readouterr().err


def test_openap_direct_filter_rejects_synonym_before_scenario_fallback():
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=2, seed=3)
    flights[1] = {**flights[1], "type": "A306"}
    config = TSConfig(aircraft_filter=AIRCRAFT_FILTER_OPENAP_DIRECT)

    series, report = build_series(flights, config, airport=AIRPORT)

    assert [item.scenario.aircraft.code for item in series] == ["A320"]
    assert series[0].scenario.source["dynamics_source"].startswith("openap-")
    assert series[0].scenario.source["dynamics_surrogate_typecode"] == "A320"
    assert report.selected_typecodes == {"A320": 1}
    assert report.skipped["aircraft filter rejected"] == 1
    assert report.rejected_aircraft == {"OpenAP synonym model (A306)": 1}
