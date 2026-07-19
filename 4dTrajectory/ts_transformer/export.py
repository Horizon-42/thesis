"""Predicted approaches -> evaluation records the existing `evaluation` package can judge.

A prediction batch writes the same directory shape ``4dTrajectory/optimization`` writes, so
``python -m evaluation --input <dir>`` grades a learned predictor against the identical regulatory
gates (lateral <= 106.75 m, vertical in [-3.05, +6.10] m) that grade the optimizer::

    <output-dir>/
      <flight>_states.json                     predicted + observed, side by side
      <flight>_eval.json                       the evaluation record
      references/<flight>_reference_eval.json  the observed track, same contract
      summary.json                             the manifest — load_records reads ONLY this

The records are **reference-shaped**: ``controls == []``. That is not a shortcut, it is the
contract — a learned predictor emits no control schedule, and ``evaluation.records`` treats
an empty control list as exactly that case.

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
    reference_evaluation_record,
    state_dict,
    summary_row,
)

from channels import states_from_channels  # noqa: E402
from dataset import FlightSeries, flight_key  # noqa: E402
from forecast import Forecast  # noqa: E402
from metrics import trajectory_metrics  # noqa: E402

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
) -> PredictionRecord:
    """Assemble the record set for one predicted approach.

    Three alignment rules make the record mean what it says:

    1. **``t = 0`` is the anchor** — the last observed sample the model was shown. Times are
       rebased there, so ``states[0]`` IS ``initial_state``, exactly as in an optimizer
       record, and the prediction proper starts at ``t = dt``.
    2. **``initial_state`` is the observed state AT the anchor**, not at the start of the
       track. The model was conditioned on the anchor; scoring it from a state 60 s earlier
       would charge it for a span it never predicted.
    3. **The reference covers the SAME span** — the observed track from the anchor onward,
       not the whole arrival. ``evaluation.reference.compare_to_reference`` resamples both
       paths at 101 fractions of *their own* arc length, so a whole-track reference against
       an anchor-to-threshold prediction compares mismatched segments and reports a path
       deviation of kilometres that is pure span mismatch.

    ``final_time_s`` comes from the last predicted sample's ``t``, never from the model's
    horizon length — ``evaluation.records`` requires ``final_time_s == states[-1]["t"]`` to
    within 1e-6, and truncation at the threshold makes those two differ.
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

    predicted_states = [anchor_state] + states_from_channels(
        forecast.times - anchor_time, forecast.values, series.frame, mass_kg=mass_kg
    )
    observed_states = full_observed_states[forecast.anchor:]

    source = dict(scenario.source)
    source.update({
        "predictor": model_name,
        "horizonMode": horizon_mode,
        "forecastPasses": forecast.passes,
        "anchorIndex": forecast.anchor,
        "anchorTimeS": anchor_time,
        "truncatedAtThreshold": forecast.truncated_at_threshold,
        # True = the fixed horizon ended this forecast short of the threshold; its final
        # state (what the gates judge) is a cap artifact, not a prediction.
        "horizonCapped": forecast.horizon_capped,
    })

    eval_record = reference_evaluation_record(
        initial_state, scenario.target, predicted_states, source
    )
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
        "predicted_states": [{"t": t, **state_dict(s)} for t, s in predicted_states],
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


def accuracy_block(overlap: Sequence[dict[str, float]]) -> dict[str, Any]:
    """Roll per-flight overlap errors up into the batch's headline accuracy numbers.

    Flights whose forecast and observed track do not overlap at all (``n_steps == 0``) carry
    NaN errors and are excluded from the statistics AND counted, rather than being quietly
    averaged in as zero.

    Both a mean and a p95 are reported because they answer different questions: chained
    window-mode forecasts compound their error into the TAIL, so a mean alone reads as a much
    smaller gap to one-pass full mode than the distribution actually shows.
    """
    finite = [m for m in overlap if m["n_steps"]]
    if not finite:
        return {"flights": 0, "flights_without_overlap": len(overlap)}

    def stats(key: str) -> dict[str, float]:
        values = np.array([m[key] for m in finite], dtype=np.float64)
        return {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
                "max": float(values.max())}

    return {
        "flights": len(finite),
        "flights_without_overlap": len(overlap) - len(finite),
        "ade_m": stats("ade_m"),
        "fde_m": stats("fde_m"),
        "cross_track_p95_m": stats("cross_track_p95_m"),
        "altitude_p95_m": stats("altitude_p95_m"),
    }


def write_batch(
    records: Sequence[PredictionRecord],
    *,
    output_dir: str | Path,
    config_dict: dict[str, Any],
    overlap: Sequence[dict[str, float]],
    checkpoint: str | None = None,
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

        states_path.write_text(json.dumps(record.states_payload, indent=2), encoding="utf-8")
        eval_path.write_text(json.dumps(record.eval_record, indent=2), encoding="utf-8")
        reference_path.write_text(json.dumps(record.reference_record, indent=2), encoding="utf-8")
        written.append(eval_path)

        source = record.source
        row = summary_row(
            source,
            # A forecast always produces states, so the row is "solved" — but a
            # horizon-capped flight's final state is a cap artifact, and its gate verdict
            # must be read with horizon_capped below.
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
            "horizon_mode": source.get("horizonMode"),
            "forecast_passes": source.get("forecastPasses"),
            "horizon_capped": source.get("horizonCapped"),
            "ade_m": metrics["ade_m"],
            "fde_m": metrics["fde_m"],
            "overlap_steps": metrics["n_steps"],
        })
        rows.append(row)

    summary = {
        "scenarios": None,
        "mode": f"tsTransformer:{config_dict.get('model')}:{config_dict.get('horizon_mode')}",
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


def observed_series_metrics(series: FlightSeries, forecast: Forecast) -> dict[str, float]:
    """Error of one predicted approach against its own observed track, metres.

    Compares only over the overlap: the forecast is truncated at the threshold and the
    observed track ends at touchdown, so the two rarely have the same length.
    """
    start = forecast.anchor + 1
    observed = series.values[start : start + forecast.n_steps]
    n = len(observed)   # the slice already bounds it by forecast.n_steps
    if n == 0:
        return {"ade_m": float("nan"), "fde_m": float("nan"), "n_steps": 0}

    predicted = forecast.values[None, :n]
    truth = observed[None, :n]
    mask = np.ones((1, n), dtype=np.float64)
    block = trajectory_metrics(predicted, truth, mask)
    return {
        "ade_m": block["ade_m"],
        "fde_m": block["fde_m"],
        "cross_track_p95_m": block["cross_track_m"]["p95_abs"],
        "altitude_p95_m": block["altitude_m"]["p95_abs"],
        "n_steps": n,
    }
