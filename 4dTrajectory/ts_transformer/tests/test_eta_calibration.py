"""Split-conformal calibration of the ETA interval (anytime design §三 3.2, B2).

CQR on the VALIDATION split, halved: half A fits the DEPLOYED δ and half B measures what it
covered; the mirror is a stability check, never averaged in. The ``deployed`` block measures
what flights actually GET — including the ones whose stratum refused and fell through to the
pooled δ — and that is the number the design's gate reads. The table is a sidecar of
``checkpoint_metadata.json`` bound to the checkpoint's digest AND its quantile levels, and
`predict` stamps the interval on every record — or says ``calibrated: false``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import calibration
from approach_difficulty import (
    STRATUM_ALL,
    STRATUM_ESTABLISHED,
    STRATUM_STRAIGHT_IN,
    STRATUM_VECTORED,
)
from calibration import (
    CALIBRATION_HALF_RULE,
    CONFORMAL_ALPHAS,
    CONFORMAL_METADATA_KEY,
    CONFORMAL_SCHEMA,
    MIN_CALIBRATION_FLIGHTS,
    CalibrationSample,
    calibrate,
    calibrated_interval,
    calibration_halves,
    conformal_delta,
    conformity_scores,
    interval_stratum,
    load_conformal_table,
    quantile_pair_indices,
    render,
    write_conformal_table,
)
from config import DURATION_HEAD_POINT, DURATION_HEAD_QUANTILE, DURATION_QUANTILES

from tests.test_duration_quantiles import _config  # the tiny quantile-head config

CHECKPOINT_SHA = "b" * 64


def _covariates(tortuosity: float = 1.0, established: bool = False) -> dict:  # noqa: D401
    return {
        "route_tortuosity": tortuosity,
        "established_at_anchor": established,
        "remaining_path_m": 12_000.0,
        "anchor_range_m": 12_000.0,
        "anchor_cross_track_m": 0.0,
    }


def _cohort(
    count: int, *, narrow_s: float, seed: int = 0, tortuosity: float = 1.0,
    spread: float = 60.0, key_prefix: str = "flight",
) -> list[CalibrationSample]:
    """Flights whose head emits the TRUE quantiles of a N(400, 60) truth, narrowed by
    ``narrow_s``, while their OWN truth is drawn with standard deviation ``spread``.

    With ``spread == 60`` the head is right and the score of a narrowed interval is the
    true-interval score plus the narrowing, whose (1 − α) quantile is 0 by definition — so δ
    must come back as ``narrow_s``, whatever else changes. A larger ``spread`` makes the
    same head wrong for those flights, which is how a stratum that needs a much wider
    interval than the pooled one is built.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(400.0, spread, size=count)
    exact = np.array([400.0 + 60.0 * _z(tau) for tau in DURATION_QUANTILES])
    narrowing = np.array([+narrow_s, +narrow_s, 0.0, -narrow_s, -narrow_s])
    return [
        CalibrationSample(
            key=f"KRDU:{key_prefix}{index:04d}",
            quantiles_s=exact + narrowing,
            truth_final_time_s=float(value),
            covariates=_covariates(tortuosity),
        )
        for index, value in enumerate(truth)
    ]


def replace_covariates(sample: CalibrationSample, **overrides) -> CalibrationSample:
    """The same flight in a different stratum."""
    return CalibrationSample(
        key=sample.key,
        quantiles_s=sample.quantiles_s,
        truth_final_time_s=sample.truth_final_time_s,
        covariates=_covariates(**overrides),
    )


def _z(tau: float) -> float:
    """The standard normal quantile, without scipy."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(tau)


# ── the arithmetic ──────────────────────────────────────────────────────────

def test_the_conformity_score_is_the_distance_outside_the_interval():
    lo = np.array([100.0, 100.0, 100.0])
    hi = np.array([200.0, 200.0, 200.0])
    truth = np.array([150.0, 80.0, 260.0])
    assert list(conformity_scores(lo, hi, truth)) == [-50.0, 20.0, 60.0]


def test_the_conformal_quantile_uses_the_finite_sample_level():
    scores = np.arange(100, dtype=np.float64)          # 0 … 99
    # ceil((100 + 1) * 0.8) / 100 = 0.81 -> the 81st order statistic, not the 80th.
    assert conformal_delta(scores, 0.2) == 81.0
    with pytest.raises(ValueError, match="alpha must be in"):
        conformal_delta(scores, 1.0)


def test_a_level_above_one_is_infinite_not_the_largest_score():
    """Four scores cannot answer alpha=0.2: ceil(5 * 0.8) / 4 = 1.0 is admissible, three
    cannot (ceil(4 * 0.8) / 3 = 1.33) and the conformal delta is +inf — raised, never
    silently clamped to the maximum score."""
    assert conformal_delta(np.arange(4, dtype=np.float64), 0.2) == 3.0
    with pytest.raises(ValueError, match=r"conformal delta is \+inf"):
        conformal_delta(np.arange(3, dtype=np.float64), 0.2)
    # ...and MIN_CALIBRATION_FLIGHTS keeps both alphas well clear of that edge.
    for alpha in CONFORMAL_ALPHAS:
        conformal_delta(np.arange(MIN_CALIBRATION_FLIGHTS, dtype=np.float64), alpha)


def test_an_inverted_interval_is_refused_not_collapsed():
    """A delta that crosses the interval is a calibration failure; the midpoint would also
    disagree with the width and coverage numbers, which use the raw arithmetic."""
    assert calibrated_interval(100.0, 200.0, -10.0) == (110.0, 190.0)
    with pytest.raises(ValueError, match="inverts the interval"):
        calibrated_interval(100.0, 200.0, -60.0)


def test_the_halves_are_deterministic_equal_and_disjoint():
    keys = [f"KRDU:f{i}" for i in range(101)]
    first, second = calibration_halves(keys, 7)
    assert sorted([*first, *second]) == list(range(101))
    assert abs(len(first) - len(second)) <= 1
    assert (calibration_halves(keys, 7)[0] == first).all()
    assert not (calibration_halves(keys, 8)[0] == first).all()   # the seed decides
    # Two probe seeds cut two different cohorts — which is the whole point of being able to
    # pass one: a coverage that survives the re-cut is a property of the flights.
    assert set(calibration_halves(keys, 2024)[0]) != set(calibration_halves(keys, 99)[0])


# ── the table ───────────────────────────────────────────────────────────────

def test_delta_recovers_a_known_miscalibration():
    """The head is narrowed by exactly 25 s on each side; δ must be 25 s."""
    table = calibrate(
        _cohort(1600, narrow_s=25.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    block = table["alphas"]["0.2"]["strata"][STRATUM_ALL]
    assert block["delta_s"] == pytest.approx(25.0, abs=6.0)
    # The published coverage is measured on the half the DEPLOYED delta was not fitted on.
    assert block["coverage"] == pytest.approx(0.8, abs=0.05)
    # The mirror is a stability check kept beside it, never averaged into the deployed one.
    assert block["stability_delta_s"] == pytest.approx(25.0, abs=6.0)
    assert block["stability_coverage"] == pytest.approx(0.8, abs=0.06)
    assert "delta_halves_s" not in block and "coverage_halves" not in block
    half = table["alphas"]["0.5"]["strata"][STRATUM_ALL]
    assert half["coverage"] == pytest.approx(0.5, abs=0.06)


def test_the_deployed_block_measures_what_flights_actually_get():
    """HIGH-1: per-stratum δ are measured on their own members, but deployment falls
    through — so the deployed block assigns each held-out flight the δ it would really get
    and measures THAT. With every flight straight-in there is no fall-through, and the
    deployed coverage equals the straight-in row's."""
    table = calibrate(
        _cohort(1600, narrow_s=25.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    block = table["alphas"]["0.2"]
    deployed = block["deployed"]
    assert deployed["flights"] == table["half_flights"][1]
    assert deployed["coverage"] == pytest.approx(0.8, abs=0.05)
    assert list(deployed["groups"]) == [STRATUM_STRAIGHT_IN]
    assert deployed["groups"][STRATUM_STRAIGHT_IN]["fell_through"] is False
    assert deployed["coverage"] == pytest.approx(
        block["strata"][STRATUM_STRAIGHT_IN]["coverage"], abs=1e-9
    )


def test_the_deployed_block_catches_a_fall_through_the_per_stratum_rows_hide():
    """The failure HIGH-1 was raised for: a thin vectored stratum refuses, its flights take
    the POOLED δ, and the pooled row's own coverage says nothing about them. Here the
    vectored flights need a far wider interval than the pooled δ provides, so the pooled row
    looks fine while the fall-through group is badly under-covered — and the deployed block
    is what shows it."""
    straight = _cohort(1200, narrow_s=0.0, seed=11)
    vectored = [
        replace_covariates(sample, tortuosity=1.6)
        for sample in _cohort(40, narrow_s=0.0, seed=12, spread=400.0, key_prefix="v")
    ]
    table = calibrate(
        [*straight, *vectored], split_seed=4, split="val", checkpoint_sha256=CHECKPOINT_SHA,
    )
    block = table["alphas"]["0.2"]
    assert STRATUM_VECTORED in block["refused_strata"]          # 20 per half, below 30
    fell_through = f"{STRATUM_VECTORED} -> {STRATUM_ALL}"
    groups = block["deployed"]["groups"]
    assert groups[fell_through]["fell_through"] is True
    assert groups[fell_through]["flights"] > 0
    # The pooled per-stratum row is healthy; the flights that fell through to it are not.
    assert block["strata"][STRATUM_ALL]["coverage"] > 0.7
    assert groups[fell_through]["coverage"] < 0.5
    # ...and the deployed pooled number sits between them, which is the point of publishing it.
    assert block["deployed"]["coverage"] < block["strata"][STRATUM_ALL]["coverage"]


#: The alphas block of `_cohort(1600, narrow_s=25.0)` calibrated at ``split_seed=1``, as it
#: stood before ``--half-seed`` existed (2026-09-08, commit 0fedf08). The half seed is now a
#: parameter, and the ONE thing that must not change is the answer when nobody passes it. If a
#: deliberate change to the alphas block moves this, re-pin it — what it guards is the CUT.
DEPLOYED_ALPHAS_SHA256 = "5ac3de6c026a8b36c815c2a63a7351d97d681b2c0db97d59f4e26b0a228f3ba1"


def _alphas_digest(table: dict) -> str:
    return hashlib.sha256(json.dumps(table["alphas"], sort_keys=True).encode()).hexdigest()


def test_the_default_half_seed_is_the_split_seed_and_the_table_is_unchanged():
    """The deployed cut is the checkpoint's own ``split_seed``, and passing it explicitly is
    the same table to the byte — so ``--half-seed 1337`` on a run whose split_seed is 1337 is
    a CONTROL that reproduces the deployed numbers rather than a second calibration."""
    samples = _cohort(1600, narrow_s=25.0)
    default = calibrate(samples, split_seed=1, split="val", checkpoint_sha256=CHECKPOINT_SHA)
    explicit = calibrate(samples, split_seed=1, split="val",
                         checkpoint_sha256=CHECKPOINT_SHA, half_seed=1)
    assert default == explicit
    assert _alphas_digest(default) == DEPLOYED_ALPHAS_SHA256
    # ...and the numbers that digest stands for, spelled out so a failure is readable.
    block = default["alphas"]["0.2"]["strata"][STRATUM_ALL]
    assert block["delta_s"] == 24.220483062742744
    assert block["coverage"] == 0.80625 and block["stability_coverage"] == 0.7975
    assert default["half_seed"] == 1 and default["deployed_half_rule"] is True
    assert default["half_rule"] == CALIBRATION_HALF_RULE


def test_a_probe_half_seed_cuts_different_halves_and_says_so():
    """A different seed is a different pair of halves, so different deltas and different
    coverages — the measurement the flag exists to make. The table carries the seed it was
    cut with, and its rule text stops it being read as the deployed one."""
    samples = _cohort(1600, narrow_s=25.0)
    deployed = calibrate(samples, split_seed=1, split="val", checkpoint_sha256=CHECKPOINT_SHA)
    probes = [
        calibrate(samples, split_seed=1, split="val", checkpoint_sha256=CHECKPOINT_SHA,
                  half_seed=seed)
        for seed in (2024, 7)
    ]
    for probe, seed in zip(probes, (2024, 7)):
        assert probe["half_seed"] == seed and probe["deployed_half_rule"] is False
        assert "NOT THE DEPLOYED RULE" in probe["half_rule"]
        assert f"conformal:{seed}:" in probe["half_rule"]
        assert _alphas_digest(probe) != DEPLOYED_ALPHAS_SHA256
        assert "PROBE HALF RULE" in render(probe)
        # The cohort, the head and the split are identical — only the cut moved.
        assert probe["split_seed"] == deployed["split_seed"]
        assert probe["half_flights"] == deployed["half_flights"]
    assert _alphas_digest(probes[0]) != _alphas_digest(probes[1])


def test_a_head_that_is_already_calibrated_needs_no_widening():
    table = calibrate(
        _cohort(1600, narrow_s=0.0, seed=5), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    assert table["alphas"]["0.2"]["strata"][STRATUM_ALL]["delta_s"] == pytest.approx(0.0, abs=6.0)


def test_a_thin_stratum_refuses_and_the_flight_falls_through_to_the_pooled_one():
    # Every flight straight-in: the vectored stratum is empty.
    table = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=3, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    block = table["alphas"]["0.2"]
    assert STRATUM_STRAIGHT_IN in block["strata"] and STRATUM_ALL in block["strata"]
    assert block["refused_strata"][STRATUM_VECTORED] == [0, 0]
    # Only the strata the precedence can DEPLOY are fitted at all: a delta for `established`
    # or a remaining-path band is a number in the artifact that nothing would ever read.
    assert STRATUM_ESTABLISHED not in block["strata"]
    assert STRATUM_ESTABLISHED not in block["refused_strata"]
    assert set(block["strata"]) | set(block["refused_strata"]) == set(
        table["interval_stratum_precedence"]
    )
    # A vectored flight has no delta of its own, so it reads the pooled one.
    assert interval_stratum(table, _covariates(tortuosity=2.4)) == STRATUM_ALL
    assert interval_stratum(table, _covariates(tortuosity=1.0)) == STRATUM_STRAIGHT_IN


def test_a_cohort_too_small_to_calibrate_at_all_is_refused():
    with pytest.raises(ValueError, match="nothing to calibrate"):
        calibrate(
            _cohort(20, narrow_s=0.0), split_seed=1, split="val",
            checkpoint_sha256=CHECKPOINT_SHA,
        )
    with pytest.raises(ValueError, match="at least one flight"):
        calibrate([], split_seed=1, split="val", checkpoint_sha256=CHECKPOINT_SHA)


def test_the_interval_is_the_quantile_pair_widened_by_its_own_delta():
    table = calibrate(
        _cohort(800, narrow_s=15.0, seed=2), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    quantiles = [200.0, 300.0, 400.0, 500.0, 600.0]
    intervals = calibration.conformal_intervals(quantiles, table, STRATUM_ALL)
    assert [entry["alpha"] for entry in intervals] == list(CONFORMAL_ALPHAS)
    for entry in intervals:
        low_index, high_index = quantile_pair_indices(entry["alpha"])
        delta = table["alphas"][f"{entry['alpha']:g}"]["strata"][STRATUM_ALL]["delta_s"]
        assert entry["lo"] == pytest.approx(quantiles[low_index] - delta)
        assert entry["hi"] == pytest.approx(quantiles[high_index] + delta)
        assert entry["lo"] <= entry["hi"]


def test_the_readout_prints_both_halves_coverage():
    table = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    text = render(table)
    assert "coverage" in text and "stab cov" in text and "refused" in text
    assert "alpha 0.2" in text and "alpha 0.5" in text
    # The deployed block, and the sentence that stops it being read as a guarantee.
    assert "DEPLOYED" in text and "GATE 3.4-2 READS THIS COVERAGE" in text
    assert "NOT a finite-sample guarantee" in text
    assert "half rule:" in text


# ── the sidecar ─────────────────────────────────────────────────────────────

def _metadata(tmp_path: Path, sha: str = CHECKPOINT_SHA) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "checkpoint_metadata.json"
    path.write_text(json.dumps({"checkpoint_sha256": sha, "schema_version": "x"}))
    return path


def test_the_table_round_trips_through_the_checkpoint_metadata(tmp_path: Path):
    table = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    metadata_path = _metadata(tmp_path)
    write_conformal_table(metadata_path, table)
    stored = json.loads(metadata_path.read_text())
    assert stored["schema_version"] == "x"          # merged, not replaced
    assert stored[CONFORMAL_METADATA_KEY]["schema"] == CONFORMAL_SCHEMA
    loaded = load_conformal_table(tmp_path / "checkpoint.pt", CHECKPOINT_SHA)
    assert loaded == table


def test_a_table_from_another_checkpoint_is_refused_not_applied(tmp_path: Path):
    table = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    metadata_path = _metadata(tmp_path)
    write_conformal_table(metadata_path, table)
    with pytest.raises(ValueError, match="was fitted on checkpoint"):
        load_conformal_table(tmp_path / "checkpoint.pt", "c" * 64)
    with pytest.raises(ValueError, match="belongs to checkpoint"):
        write_conformal_table(_metadata(tmp_path / "other", sha="d" * 64), table)


def test_a_smoke_table_is_refused_at_the_sidecar_unless_it_is_asked_for(tmp_path: Path):
    """`--limit` fits a delta on a PREFIX of the split; deploying it silently would widen
    every record of a full run with nothing at the write site saying so."""
    smoke = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA, airports=("KRDU",), limit=400,
    )
    assert smoke["smoke_test"] is True and smoke["limit"] == 400
    assert smoke["airports"] == ["KRDU"]
    assert "SMOKE TEST" in render(smoke)
    metadata_path = _metadata(tmp_path)
    with pytest.raises(ValueError, match="smoke test"):
        write_conformal_table(metadata_path, smoke)
    write_conformal_table(metadata_path, smoke, allow_smoke=True)
    assert load_conformal_table(tmp_path / "checkpoint.pt", CHECKPOINT_SHA)["smoke_test"]


def test_a_probe_table_can_never_reach_the_sidecar(tmp_path: Path):
    """A deployed delta comes from the documented half rule alone: the refusal is at the
    write site as well as at the flag, and unlike the smoke table it has no escape hatch."""
    probe = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA, half_seed=2024,
    )
    metadata_path = _metadata(tmp_path)
    for permission in ({}, {"allow_smoke": True}):
        with pytest.raises(ValueError, match="PROBE of the half rule"):
            write_conformal_table(metadata_path, probe, **permission)
    assert CONFORMAL_METADATA_KEY not in json.loads(metadata_path.read_text())


def test_a_full_table_carries_its_cohort_and_needs_no_permission(tmp_path: Path):
    table = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA, airports=("KRDU", "KSJC"),
    )
    assert table["smoke_test"] is False and table["limit"] == 0
    assert table["airports"] == ["KRDU", "KSJC"]
    assert table["half_rule"] and table["coverage_claim"]
    write_conformal_table(_metadata(tmp_path), table)


def test_a_table_written_under_other_quantile_levels_is_refused(tmp_path: Path):
    """The levels are a mirror of DURATION_QUANTILES inside the artifact; a mirror that is
    written and never checked is a version the artifact cannot verify."""
    table = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    table["quantiles"] = [0.05, 0.25, 0.5, 0.75, 0.95]
    metadata_path = _metadata(tmp_path)
    write_conformal_table(metadata_path, table)
    with pytest.raises(ValueError, match="index different columns"):
        load_conformal_table(tmp_path / "checkpoint.pt", CHECKPOINT_SHA)


def test_an_unknown_schema_is_refused(tmp_path: Path):
    metadata_path = _metadata(tmp_path)
    metadata_path.write_text(json.dumps({
        "checkpoint_sha256": CHECKPOINT_SHA,
        CONFORMAL_METADATA_KEY: {"schema": "ts-conformal-cqr-v0",
                                 "checkpoint_sha256": CHECKPOINT_SHA},
    }))
    with pytest.raises(ValueError, match="conformal schema"):
        load_conformal_table(tmp_path / "checkpoint.pt", CHECKPOINT_SHA)


def test_no_metadata_and_no_table_both_read_as_uncalibrated(tmp_path: Path):
    assert load_conformal_table(tmp_path / "checkpoint.pt", CHECKPOINT_SHA) is None
    _metadata(tmp_path)
    assert load_conformal_table(tmp_path / "checkpoint.pt", CHECKPOINT_SHA) is None


# ── the runner's refusals ───────────────────────────────────────────────────

def _runner():
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "run_ts_eta_calibration.py"
    spec = importlib.util.spec_from_file_location("run_ts_eta_calibration_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_runner_refuses_every_split_but_val(tmp_path: Path):
    runner = _runner()
    for split in ("test", "train"):
        with pytest.raises(SystemExit):
            runner.main(["--checkpoint", str(tmp_path / "checkpoint.pt"),
                         "--out", str(tmp_path / "out"), "--split", split])


def test_the_runner_only_deploys_a_smoke_table_when_asked():
    """`--limit` alone cannot deploy: the escape hatch is a separate, explicit flag."""
    parser = _runner().build_parser()
    plain = parser.parse_args(["--checkpoint", "c.pt", "--out", "o", "--limit", "50"])
    assert plain.limit == 50 and plain.allow_smoke_table is False
    asked = parser.parse_args(
        ["--checkpoint", "c.pt", "--out", "o", "--limit", "50", "--allow-smoke-table"]
    )
    assert asked.allow_smoke_table is True
    assert parser.parse_args(["--checkpoint", "c.pt", "--out", "o"]).limit == 0


def test_the_runner_refuses_a_probe_half_seed_unless_it_is_a_readout(
    tmp_path: Path, capsys
):
    """`--half-seed` re-cuts the halves, and a DEPLOYED table comes from the documented rule
    alone — so the flag is refused on its own, before any checkpoint is read, and the refusal
    names the flag that makes it admissible."""
    runner = _runner()
    default = runner.build_parser().parse_args(["--checkpoint", "c.pt", "--out", "o"])
    assert default.half_seed is None and default.readout_only is False
    with pytest.raises(SystemExit):
        runner.main(["--checkpoint", str(tmp_path / "checkpoint.pt"),
                     "--out", str(tmp_path / "out"), "--half-seed", "2024"])
    message = capsys.readouterr().err
    assert "--half-seed" in message and "--readout-only" in message


def _stubbed_runner(monkeypatch, samples: list[CalibrationSample], split_seed: int):
    """The runner with its checkpoint load, cohort rebuild and forward pass replaced.

    Those three are what make the real thing need a trained model and the arrival manifests,
    and they decide none of what is under test here: which seed cuts the halves, and whether
    the checkpoint's sidecar is written, are `main`'s own control flow.
    """
    runner = _runner()
    truths = {sample.key: sample.truth_final_time_s for sample in samples}
    covariates = {sample.key: sample.covariates for sample in samples}
    arm = SimpleNamespace(
        config=SimpleNamespace(duration_head=DURATION_HEAD_QUANTILE,
                               resolved_split_seed=split_seed),
        airports=("KRDU",), model=object(), normalizer=object(),
    )
    monkeypatch.setattr(runner, "resolve_device", lambda _name: "cpu")
    monkeypatch.setattr(runner, "load_arm", lambda *args, **kwargs: arm)
    monkeypatch.setattr(runner, "cohort_series", lambda _arm, _grid: [
        SimpleNamespace(dataset_id=sample.key) for sample in samples
    ])
    monkeypatch.setattr(runner, "default_anchor", lambda _config: 0)
    monkeypatch.setattr(runner, "duration_quantile_predictions", lambda *args, **kwargs:
                        np.stack([sample.quantiles_s for sample in samples]))
    monkeypatch.setattr(runner, "truth_duration_s",
                        lambda item, _anchor: truths[item.dataset_id])
    monkeypatch.setattr(runner, "approach_difficulty", lambda item, _anchor:
                        SimpleNamespace(to_dict=lambda: covariates[item.dataset_id]))
    return runner


def test_readout_only_writes_the_table_and_leaves_the_sidecar_byte_for_byte(
    tmp_path: Path, monkeypatch
):
    """The probe may read a checkpoint the campaign is deploying, so `--readout-only` must
    not write one byte of its metadata — and the deployed run below is what proves that
    assertion could have failed."""
    from io_utils import file_sha256

    samples = _cohort(400, narrow_s=10.0)
    runner = _stubbed_runner(monkeypatch, samples, split_seed=1)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    metadata_path = _metadata(tmp_path, sha=file_sha256(checkpoint))
    before = metadata_path.read_bytes()

    def run(out: str, *flags: str) -> dict:
        assert runner.main(["--checkpoint", str(checkpoint), "--out", str(tmp_path / out),
                            "--device", "cpu", *flags]) == 0
        return json.loads((tmp_path / out / runner.JSON_NAME).read_text())["conformal"]

    probe = run("probe", "--readout-only", "--half-seed", "2024")
    assert metadata_path.read_bytes() == before
    assert probe["half_seed"] == 2024 and probe["deployed_half_rule"] is False
    assert "PROBE HALF RULE" in (tmp_path / "probe" / runner.TEXT_NAME).read_text()

    # The default seed under --readout-only is the deployed cut, read and not written.
    control = run("control", "--readout-only")
    assert metadata_path.read_bytes() == before
    assert control["half_seed"] == 1 and control["deployed_half_rule"] is True
    assert control["alphas"] != probe["alphas"]

    # ...and without the flag the same command deploys, so "untouched" above is a difference.
    deployed = run("deployed")
    assert metadata_path.read_bytes() != before
    stored = json.loads(metadata_path.read_text())[CONFORMAL_METADATA_KEY]
    assert stored == deployed and stored["half_seed"] == 1
    assert deployed["alphas"] == control["alphas"]      # the flag changes only the writing


# ── predict reads it ────────────────────────────────────────────────────────

def test_predict_stamps_the_calibrated_interval_on_every_record(tmp_path: Path, monkeypatch):
    """End to end on a tiny checkpoint: uncalibrated records say so, and after a table is
    written beside the checkpoint the same command writes an interval per alpha."""
    import importlib.util

    import cli.predict as predict_module
    from data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
    from dataset import build_series
    from synthetic import synthetic_arrivals
    from train import load_checkpoint, train

    cli_path = Path(__file__).resolve().parents[1] / "__main__.py"
    spec = importlib.util.spec_from_file_location("ts_cli_calibration_test", cli_path)
    ts_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts_cli)

    config = _config()
    flights = synthetic_arrivals("KRDU", "05L", n_flights=12, seed=3)
    series, _report = build_series(flights, config, airport="KRDU")
    provenance = {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
                  "manifests": [{"airport": "KRDU", "arrival_manifest_sha256": "a" * 64,
                                 "source_records": []}]}
    run = tmp_path / "run"
    train(series, config, output_dir=run, data_provenance=provenance, verbose=False)
    monkeypatch.setattr(predict_module, "provenance_from_args", lambda _args: provenance)
    monkeypatch.setattr(
        predict_module, "load_flight_dicts", lambda _path, include_flight_keys=None: flights
    )

    def predict(out: Path) -> int:
        return ts_cli.main([
            "predict", "--checkpoint", str(run / "checkpoint.pt"),
            "--data", str(tmp_path / "manifest.json"), "--airport", "KRDU",
            "--output-dir", str(out), "--split", "val", "--device", "cpu",
        ])

    assert predict(tmp_path / "raw") == 0
    raw = json.loads((tmp_path / "raw" / "summary.json").read_text())["results"][0]
    assert raw["calibrated"] is False and raw["duration_interval_s"] is None
    assert len(raw["duration_quantiles_s"]) == len(DURATION_QUANTILES)

    # A table for THIS checkpoint, fitted on a synthetic cohort (the runner's own path
    # needs the real arrival manifests; the table it writes is this object).
    _model, loaded, _normalizer, _payload = load_checkpoint(run / "checkpoint.pt")
    from io_utils import file_sha256

    table = calibrate(
        _cohort(400, narrow_s=12.0), split_seed=loaded.resolved_split_seed, split="val",
        checkpoint_sha256=file_sha256(run / "checkpoint.pt"),
    )
    write_conformal_table(run / "checkpoint_metadata.json", table)

    assert predict(tmp_path / "calibrated") == 0
    row = json.loads((tmp_path / "calibrated" / "summary.json").read_text())["results"][0]
    assert row["calibrated"] is True
    assert [entry["alpha"] for entry in row["duration_interval_s"]] == list(CONFORMAL_ALPHAS)
    assert row["duration_interval_stratum"] in (
        STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED
    )
    states = json.loads((tmp_path / "calibrated" / row["states_file"]).read_text())
    assert states["source"]["durationIntervalS"] == row["duration_interval_s"]
    # The interval brackets the quantile pair it was built from.
    quantiles = row["duration_quantiles_s"]
    for entry in row["duration_interval_s"]:
        low_index, high_index = quantile_pair_indices(entry["alpha"])
        assert entry["lo"] <= entry["hi"]
        assert entry["hi"] - entry["lo"] >= 0.0
        assert entry["lo"] == pytest.approx(
            quantiles[low_index]
            - table["alphas"][f"{entry['alpha']:g}"]["strata"][
                row["duration_interval_stratum"]]["delta_s"]
        )


def test_neither_half_of_the_pair_alone_produces_an_interval():
    """The interval needs BOTH a quantile forecast and a table: a point-head forecast has
    nothing to widen, and an uncalibrated quantile forecast has nothing to widen it by."""
    from forecast import _calibrated_interval_fields

    table = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    assert _config(duration_head=DURATION_HEAD_POINT).duration_head == DURATION_HEAD_POINT
    assert _calibrated_interval_fields(_UNUSED_SERIES, 0, None, table) == {}
    assert _calibrated_interval_fields(_UNUSED_SERIES, 0, np.zeros(5), None) == {}


#: Neither branch above reaches the series, and building one would hide that.
_UNUSED_SERIES = object()
