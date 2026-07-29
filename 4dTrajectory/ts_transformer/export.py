"""Predicted approaches -> evaluation records the existing `evaluation` package can judge.

A prediction batch writes the same directory shape ``4dTrajectory/optimization`` writes, so
``python -m evaluation --input <dir>`` grades a learned predictor against the identical regulatory
gates (lateral <= 106.75 m, vertical in [-3.05, +6.10] m) that grade the optimizer::

    <output-dir>/
      <flight>_states.json                     canonical predicted + observed arrays
      <flight>_eval.json                       metadata + predicted states_ref
      references/<flight>_reference_eval.json  metadata + observed states_ref
      summary.json                             the manifest — load_records reads ONLY this

State-output records are reference-shaped (``controls == []``). Control-output records use
the optimizer-shaped contract: rollout states plus their aligned active controls.

Record construction goes through ``4dTrajectory/optimization/evaluation_export.py`` rather
than building the JSON here, so there is one definition of the record shape. That module is
casadi-free, which is why importing it into the torch env works.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from aerodynamic_model.common import GeodeticState, LoadFactorControl
from aerodynamic_model.rollout import RolloutSample

# evaluation_export lives beside the optimizer; it is a pure mapping module with no casadi
# dependency, so it imports cleanly into the torch env.
_OPT_DIR = Path(__file__).resolve().parents[1] / "optimization"
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))

# The filename suffixes are single-sourced there too — shared with the optimizer batch's
# writers and globs, so the two record families keep the same directory shape by import,
# not by mirror comment.
from evaluation_export import (  # noqa: E402
    EVAL_SUFFIX as _EVAL_SUFFIX,
    REFERENCE_EVAL_SUFFIX as _REFERENCE_EVAL_SUFFIX,
    REFERENCES_DIR,
    STATES_SUFFIX as _STATES_SUFFIX,
    evaluation_record,
    reference_evaluation_record,
    state_dict,
    summary_row,
)

from channels import states_from_channels  # noqa: E402
from dataset import FlightSeries, flight_key  # noqa: E402
from forecast import Forecast  # noqa: E402
from metrics import (  # noqa: E402
    RAW_KINEMATIC_METRIC_KEYS, raw_kinematic_metrics, trajectory_metrics,
)

# The filename stem IS the flight identity — flight_scenarios.identity.flight_key, the
# same function that keys the train/val/test split and the optimizer's record filenames;
# re-exported here under the name that reads right at a write site.
record_stem = flight_key


@dataclass
class PredictionRecord:
    """One flight's prediction, ready to write."""

    stem: str
    eval_record: dict[str, Any]
    states_payload: dict[str, Any]
    reference_record: dict[str, Any]

    # Views, not stored copies: the summary rows read these, and a second copy could
    # disagree with the record it points at (final_time_s == states[-1].t is a contract
    # the evaluation reader rejects violations of).
    @property
    def source(self) -> dict[str, Any]:
        return self.states_payload["source"]

    @property
    def final_time_s(self) -> float:
        return float(self.eval_record["final_time_s"])


def build_prediction_record(
    series: FlightSeries,
    forecast: Forecast,
    *,
    index: int,
    model_name: str,
    horizon_mode: str,
    split: str = "test",
) -> PredictionRecord:
    """Assemble the record set for one predicted approach.

    Three alignment rules make the record mean what it says:

    1. **``t = 0`` is the anchor** — the last observed sample the model was shown. Times are
       rebased there, so ``states[0]`` IS ``initial_state``, exactly as in an optimizer
       record, and the prediction proper starts at ``final_time_s / N``.
    2. **``initial_state`` is the observed state AT the anchor**, not at the start of the
       track. The model was conditioned on the anchor; scoring it from a state 60 s earlier
       would charge it for a span it never predicted.
    3. **The reference covers the SAME span** — the observed track from the anchor onward,
       not the whole arrival. ``evaluation.reference.compare_to_reference`` resamples both
       paths at 101 fractions of *their own* arc length, so a whole-track reference against
       an anchor-to-threshold prediction compares mismatched segments and reports a path
       deviation of kilometres that is pure span mismatch.

    ``final_time_s`` comes from the learned duration and therefore from the last predicted
    sample's ``t``. ``evaluation.records`` requires these to agree within 1e-6.
    """
    scenario = series.scenario
    mass_kg = scenario.initial.m
    anchor_time = float(series.times[forecast.anchor])

    # Rebase everything so t=0 sits at the anchor. ONE conversion covers the whole
    # observed track; the anchor sample and the anchor-onward reference span are slices of
    # it (times are rebased identically, and full[anchor] lands at exactly t = 0.0 by
    # float x - x), so the per-sample state construction runs once, not three times.
    full_observed_states = states_from_channels(
        series.times - anchor_time, series.values, series.frame, mass_kg=mass_kg
    )
    anchor_state = full_observed_states[forecast.anchor]
    initial_state = anchor_state[1]

    observed_states = full_observed_states[forecast.anchor:]

    source = dict(scenario.source)
    prediction_output = "control" if forecast.controls is not None else "state"
    source.update({
        "predictor": model_name,
        "predictionOutput": prediction_output,
        "horizonMode": horizon_mode,
        "forecastPasses": forecast.passes,
        "predictedFinalTimeS": forecast.final_time_s,
        "durationHeadFinalTimeS": forecast.predicted_final_time_s,
        "truncatedAtThreshold": forecast.truncated_at_threshold,
        "horizonCapped": forecast.horizon_capped,
        "anchorIndex": forecast.anchor,
        "anchorTimeS": anchor_time,
        "predictionSplit": split,
    })

    if forecast.controls is None:
        relative_offsets = np.cumsum(forecast.segment_durations_s)
        predicted_states = [anchor_state] + states_from_channels(
            relative_offsets, forecast.values, series.frame, mass_kg=mass_kg
        )
        eval_record = reference_evaluation_record(
            initial_state, scenario.target, predicted_states, source
        )
        predicted_state_rows = [{"t": t, **state_dict(s)} for t, s in predicted_states]
    else:
        if forecast.geodetic_values is None or forecast.segment_durations_s is None:
            raise ValueError("control forecast must carry geodetic states and segment durations")
        controls = [LoadFactorControl(*map(float, row)) for row in forecast.controls]
        endpoint_states = [GeodeticState(*map(float, row)) for row in forecast.geodetic_values]
        offsets = np.cumsum(forecast.segment_durations_s)
        samples = [RolloutSample(0.0, initial_state, controls[0], 0)] + [
            RolloutSample(float(t), state, controls[index], index)
            for index, (t, state) in enumerate(zip(offsets, endpoint_states))
        ]
        eval_record = evaluation_record(
            initial_state, scenario.target, samples, source
        )
        predicted_state_rows = list(eval_record["states"])
    eval_record["reference_file"] = f"{REFERENCES_DIR}/{record_stem(scenario.source, index)}{_REFERENCE_EVAL_SUFFIX}"

    reference_record = reference_evaluation_record(
        initial_state, scenario.target, observed_states, dict(scenario.source)
    )

    states_payload = {
        "source": source,
        "final_time_s": eval_record["final_time_s"],
        # Named for what they are. The optimizer's analogue writes optimizer_states /
        # simulator_states; there is no simulator here, so the pairing is prediction
        # against the observed truth it is supposed to reproduce. observed_states is the
        # WHOLE track (negative t before the anchor) so a viewer can show the lookback the
        # model was given; the reference RECORD is span-matched instead.
        "predicted_states": predicted_state_rows,
        "observed_states": [{"t": t, **state_dict(s)} for t, s in full_observed_states],
    }

    return PredictionRecord(
        stem=record_stem(scenario.source, index),
        eval_record=eval_record,
        states_payload=states_payload,
        reference_record=reference_record,
    )


def clear_stale_records(output_dir: Path) -> None:
    """Delete a previous run's records before writing new ones.

    Same hygiene rule as the optimizer batch: a shrinking flight set would otherwise leave
    orphans that the manifest does not list but a human reading the directory would count.
    The glob is non-recursive, so ``references/`` survives and is cleared separately.
    """
    for path in list(output_dir.glob(f"*{_STATES_SUFFIX}")) + list(output_dir.glob(f"*{_EVAL_SUFFIX}")):
        path.unlink()
    references = output_dir / REFERENCES_DIR
    if references.is_dir():
        for path in references.glob(f"*{_REFERENCE_EVAL_SUFFIX}"):
            path.unlink()


def accuracy_block(overlap: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-flight overlap errors up into the batch's headline accuracy numbers.

    Flights whose forecast and observed track do not overlap at all (``n_steps == 0``) carry
    NaN errors and are excluded from the statistics AND counted, rather than being quietly
    averaged in as zero.

    Median, mean, p95 and max are reported because raw derivatives have a deliberately
    unbounded physical scale: one catastrophically short predicted duration can dominate
    the mean while the fleet p95 still describes the tail experienced by a typical batch.
    """
    time_errors = np.array([m["final_time_error_s"] for m in overlap], dtype=np.float64)
    time_block = {
        "mae": float(np.abs(time_errors).mean()),
        "p95_abs": float(np.percentile(np.abs(time_errors), 95)),
        "mean_signed": float(time_errors.mean()),
    }

    def raw_role(role: str) -> dict[str, dict[str, float]]:
        result = {}
        for key in RAW_KINEMATIC_METRIC_KEYS:
            values = np.array(
                [row["raw_kinematics"][role][key] for row in overlap], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            if not len(values):
                result[key] = {
                    "count": 0,
                    "median": float("nan"),
                    "mean": float("nan"),
                    "p95": float("nan"),
                    "max": float("nan"),
                }
            else:
                result[key] = {
                    "count": int(len(values)),
                    "median": float(np.median(values)),
                    "mean": float(values.mean()),
                    "p95": float(np.percentile(values, 95)),
                    "max": float(values.max()),
                }
        return result

    predicted_raw = raw_role("predicted")
    observed_raw = raw_role("observed_baseline")
    raw_block = {
        "predicted": predicted_raw,
        "observed_baseline": observed_raw,
        # Signed difference of fleet p95 values. Positive means the prediction's physical
        # tail is rougher or less self-consistent than the observed tracks under the same
        # raw-node metric. Means remain available above for outlier audits.
        "delta": {
            key: predicted_raw[key]["p95"] - observed_raw[key]["p95"]
            for key in RAW_KINEMATIC_METRIC_KEYS
        },
    }
    finite = [m for m in overlap if m["n_steps"]]
    if not finite:
        return {
            "flights": 0,
            "flights_without_overlap": len(overlap),
            "final_time_s": time_block,
            "raw_kinematics": raw_block,
        }

    def stats(key: str) -> dict[str, float]:
        values = np.array([m[key] for m in finite], dtype=np.float64)
        return {"median": float(np.median(values)),
                "mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
                "max": float(values.max())}

    return {
        "flights": len(finite),
        "flights_without_overlap": len(overlap) - len(finite),
        "final_time_s": time_block,
        "ade_m": stats("ade_m"),
        "fde_m": stats("fde_m"),
        "cross_track_p95_m": stats("cross_track_p95_m"),
        "altitude_p95_m": stats("altitude_p95_m"),
        "raw_kinematics": raw_block,
    }


def write_batch(
    records: Sequence[PredictionRecord],
    *,
    output_dir: str | Path,
    config_dict: dict[str, Any],
    overlap: Sequence[dict[str, float]],
    checkpoint: str | None = None,
    split: str = "test",
) -> list[Path]:
    """Write every record plus the ``summary.json`` manifest. Returns the eval-file paths.

    ``overlap`` is ``observed_series_metrics`` per record, positionally aligned. It is
    required, not optional: a prediction batch's error against the observed track is its
    headline result, and leaving it to the caller to print made it live in terminal
    scrollback only — invisible to any comparison across batches.
    """
    if len(overlap) != len(records):
        raise ValueError(
            f"overlap has {len(overlap)} entries for {len(records)} records — "
            "observed_series_metrics must be collected once per record, in order"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / REFERENCES_DIR).mkdir(exist_ok=True)
    clear_stale_records(out)

    written: list[Path] = []
    rows: list[dict[str, Any]] = []
    for record, metrics in zip(records, overlap):
        states_path = out / f"{record.stem}{_STATES_SUFFIX}"
        eval_path = out / f"{record.stem}{_EVAL_SUFFIX}"
        reference_path = out / REFERENCES_DIR / f"{record.stem}{_REFERENCE_EVAL_SUFFIX}"

        source = dict(record.source)
        source["predictionSplit"] = split
        states_payload = dict(record.states_payload)
        states_payload["source"] = source
        states_path.write_text(
            json.dumps(states_payload, separators=(",", ":")), encoding="utf-8"
        )
        eval_payload = dict(record.eval_record)
        eval_payload["source"] = source
        eval_payload["states_ref"] = {"file": states_path.name, "key": "predicted_states"}
        eval_payload["states"] = []
        reference_payload = dict(record.reference_record)
        reference_payload["states_ref"] = {
            "file": f"../{states_path.name}",
            "key": "observed_states",
            "start_index": int(record.source["anchorIndex"]),
        }
        reference_payload["states"] = []
        eval_path.write_text(
            json.dumps(eval_payload, separators=(",", ":")), encoding="utf-8"
        )
        reference_path.write_text(
            json.dumps(reference_payload, separators=(",", ":")), encoding="utf-8"
        )
        written.append(eval_path)

        row = summary_row(
            source,
        # A state forecast is solved in one forward pass under its checkpoint clock.
            status="solved",
            states_file=states_path.name,
            eval_file=eval_path.name,
            final_time_s=record.final_time_s,
            reason=None,
        )
        # ts_transformer additions beyond the optimizer's row. The optimizer has no
        # equivalent of ade/fde — it solves toward a target rather than reproducing a track,
        # so these are graded against the flight's OWN observed samples over the overlap.
        row.update({
            "predictor": source.get("predictor"),
            "prediction_output": source.get("predictionOutput"),
            "horizon_mode": source.get("horizonMode"),
            "forecast_passes": source.get("forecastPasses"),
            "horizon_capped": source.get("horizonCapped"),
            "predicted_final_time_s": source.get("predictedFinalTimeS"),
            "true_final_time_s": metrics["true_final_time_s"],
            "final_time_error_s": metrics["final_time_error_s"],
            "split": split,
            "ade_m": metrics["ade_m"],
            "fde_m": metrics["fde_m"],
            "overlap_steps": metrics["n_steps"],
            "raw_kinematics": metrics["raw_kinematics"],
        })
        rows.append(row)

    summary = {
        "scenarios": None,
        "mode": (
            f"tsTransformer:{config_dict.get('model')}:"
            f"{config_dict.get('horizon_mode')}:"
            f"{config_dict.get('prediction_output', 'state')}:{split}"
        ),
        "split": split,
        "checkpoint": checkpoint,
        "config": config_dict,
        "total": len(rows),
        "solved": len(rows),
        "failed": 0,
        "failure_rate": 0.0,
        "accuracy": accuracy_block(overlap),
        "results": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return written


def observed_series_metrics(series: FlightSeries, forecast: Forecast) -> dict[str, Any]:
    """Error of one predicted approach against its own observed track, metres.

    The forecast carries explicit physical timestamps reconstructed from its checkpoint's
    normalized or fixed-horizon clock. Observations are interpolated at those timestamps over
    their overlap.
    """
    valid = forecast.times <= series.times[-1]
    n = int(valid.sum())
    true_final_time_s = float(
        series.supervision_times[-1] - series.times[forecast.anchor]
    )
    time_metrics = {
        "true_final_time_s": true_final_time_s,
        "final_time_error_s": forecast.final_time_s - true_final_time_s,
    }
    anchor_values = series.values[forecast.anchor][None, ...]
    predicted_durations = forecast.segment_durations_s[None, ...]
    observed_values = series.values[forecast.anchor + 1 :]
    observed_durations = np.diff(series.times[forecast.anchor:])[None, ...]
    raw_metrics = {
        "predicted": raw_kinematic_metrics(
            anchor_values, forecast.values[None, ...], predicted_durations
        ),
        "observed_baseline": raw_kinematic_metrics(
            anchor_values, observed_values[None, ...], observed_durations
        ),
    }
    if n == 0:
        return {
            "ade_m": float("nan"),
            "fde_m": float("nan"),
            "n_steps": 0,
            "raw_kinematics": raw_metrics,
            **time_metrics,
        }

    query_times = forecast.times[valid]
    observed = np.column_stack([
        np.interp(query_times, series.times, series.values[:, channel])
        for channel in range(series.values.shape[1])
    ])
    predicted = forecast.values[valid][None, ...]
    truth = observed[None, ...]
    mask = np.ones((1, n), dtype=np.float64)
    block = trajectory_metrics(predicted, truth, mask)
    return {
        "ade_m": block["ade_m"],
        "fde_m": block["fde_m"],
        "cross_track_p95_m": block["cross_track_m"]["p95_abs"],
        "altitude_p95_m": block["altitude_m"]["p95_abs"],
        "n_steps": n,
        "raw_kinematics": raw_metrics,
        **time_metrics,
    }
