"""CLI: train a trajectory predictor, or predict a batch of approaches with a trained one.

SCOPE: this package is a purely kinematic, single-aircraft BASELINE. No aerodynamic or
dynamics model is connected — channels in, channels out, no equations of motion anywhere.
That is a deliberate decision, not an unfinished TODO: it is what lets the learned
component be measured on its own. Predicted trajectories therefore carry no flyability
guarantee. Read README "Inputs, outputs, and the deliberate absence of dynamics" (which
also lists the four routes for adding dynamics later) before changing that.

    TS=4dTrajectory/ts_transformer/__main__.py

    # train iTransformer on KRDU arrivals, short horizon (chained at predict time)
    python $TS train \
        --data trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json \
        --airport KRDU --model itransformer --horizon-mode window \
        --output-dir 4dTrajectory/outputs/KRDU/ts_itransformer_window

    # the same data, PatchTST, whole-approach horizon in one pass
    python $TS train --data ... --model patchtst --horizon-mode full ...

    # predict, then grade with the same gates the optimizer is graded by
    python $TS predict \
        --checkpoint 4dTrajectory/outputs/KRDU/ts_itransformer_window/checkpoint.pt \
        --data ... --airport KRDU --output-dir 4dTrajectory/outputs/KRDU/ts_pred
    python -m evaluation --input 4dTrajectory/outputs/KRDU/ts_pred

Invoked by script path, like ``4dTrajectory/optimization/scenario_optimization.py`` — the
bootstrap below makes it work from any working directory. (``python -m ts_transformer``
also works, but only from inside ``4dTrajectory/``, so the path form is what the docs use.)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

# Same bootstrap as 4dTrajectory/optimization/*.py: the repo root for the shared packages
# (flight_scenarios, aerodynamic_model, geokit, evaluation), and this directory so sibling
# modules import flat.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TS_DIR = Path(__file__).resolve().parent
if str(_TS_DIR) not in sys.path:
    sys.path.insert(0, str(_TS_DIR))

from config import (  # noqa: E402
    COORDINATE_FRAMES,
    DEFAULT_AIRCRAFT_TYPE,
    EVAL_ANCHOR_POLICIES,
    HORIZON_MODES,
    MODELS,
    SAMPLING_STRATEGIES,
    TSConfig,
    config_for_mode,
)
from cross_validation import cross_validate  # noqa: E402
from dataset import (  # noqa: E402
    arrival_data_provenance,
    build_series,
    dataset_flight_key,
    load_flight_dicts,
    require_matching_data_provenance,
)
from export import (  # noqa: E402
    accuracy_block, build_prediction_record, observed_series_metrics, write_batch,
)
from flyability import report_for_records  # noqa: E402
from forecast import forecast_approach  # noqa: E402
from models import resolve_device  # noqa: E402
from train import load_checkpoint, train  # noqa: E402


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data", required=True, action="append",
        help="airport harvest directory or arrivals/manifest.json; repeat for pooled training",
    )
    parser.add_argument("--airport", default=None,
                        help="ICAO code, when the flight dicts do not carry arr_airport")
    parser.add_argument("--aircraft-type", default=None,
                        help="fallback aircraft when the flight dict has no resolvable type "
                             f"(train default: {DEFAULT_AIRCRAFT_TYPE}; predict default: the "
                             "checkpoint's train-time value). Every harvested arrival is "
                             "currently 'UNK', so this applies to ALL of them and sets the "
                             "target Vref / threshold-crossing height the gates measure against")
    parser.add_argument("--output-dir", required=True)


def _batch_size(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch size must be a positive integer or 'auto'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch size must be positive")
    return parsed


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--horizon-mode", choices=HORIZON_MODES, default="window",
                        help="window: short H, chained at predict time. "
                             "full: H covers the whole approach in one pass (padded + masked)")
    parser.add_argument("--seq-len", type=int, default=None, help="lookback L, in steps")
    parser.add_argument("--pred-len", type=int, default=None,
                        help="horizon H, in steps (defaults to the horizon mode's own default)")
    parser.add_argument("--dt", type=float, default=None, help="resample step, seconds")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=_batch_size, default=None,
                        help="positive integer, or 'auto' to probe the active CUDA GPU")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--fitted-tail-weight", type=float, default=None,
                        help="position-only weight for fitted ADS-B tail rows (default: 0.25)")
    parser.add_argument("--fitted-terminal-weight", type=float, default=None,
                        help="additional position-only weight at the fitted crossing (default: 1.0)")
    parser.add_argument("--patience", type=int, default=None, help="early-stopping patience")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--e-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help='"auto" (default), "cpu", "cuda"')
    parser.add_argument("--coordinate-frame", choices=COORDINATE_FRAMES, default=None)
    parser.add_argument("--sampling-strategy", choices=SAMPLING_STRATEGIES, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--eval-anchor-policy", choices=EVAL_ANCHOR_POLICIES, default=None)
    parser.add_argument("--config-overrides", default=None,
                        help="JSON object of TSConfig overrides, e.g. CV best_config.json")
    parser.add_argument("--instance-norm", dest="instance_norm", action="store_true", default=None,
                        help="per-window de/normalisation; OFF by default because absolute "
                             "threshold-relative position is signal")
    parser.add_argument("--no-instance-norm", dest="instance_norm", action="store_false")


def _config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[TSConfig, bool]:
    overrides: dict[str, object] = {}
    if args.config_overrides:
        try:
            loaded = json.loads(Path(args.config_overrides).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --config-overrides: {exc}")
        allowed = {field.name for field in fields(TSConfig)}
        if not isinstance(loaded, dict) or any(key not in allowed for key in loaded):
            parser.error("--config-overrides must be a JSON object containing TSConfig fields")
        overrides.update(loaded)

    batch_auto = args.batch_size == "auto"
    cli_values = (
        ("model", args.model), ("seq_len", args.seq_len), ("pred_len", args.pred_len),
        ("dt_s", args.dt), ("epochs", args.epochs),
        ("batch_size", args.batch_size if isinstance(args.batch_size, int) else None),
        ("learning_rate", args.learning_rate), ("patience", args.patience),
        ("fitted_tail_position_weight", args.fitted_tail_weight),
        ("fitted_terminal_position_weight", args.fitted_terminal_weight),
        ("d_model", args.d_model), ("e_layers", args.e_layers), ("n_heads", args.n_heads),
        ("seed", args.seed), ("device", args.device),
        ("aircraft_type", args.aircraft_type), ("coordinate_frame", args.coordinate_frame),
        ("sampling_strategy", args.sampling_strategy),
        ("train_samples_per_epoch", args.samples_per_epoch),
        ("eval_anchor_policy", args.eval_anchor_policy),
        ("use_norm", args.instance_norm), ("revin", args.instance_norm),
    )
    overrides.update({key: value for key, value in cli_values if value is not None})
    return config_for_mode(args.horizon_mode, **overrides), batch_auto


def _build_series_or_exit(args: argparse.Namespace, config: TSConfig,
                          parser: argparse.ArgumentParser, flights: list[dict]):
    # config.aircraft_type is the train-time value (checkpoint-carried on predict); an
    # explicit --aircraft-type wins, with the mismatch warned about at the predict site.
    aircraft_type = args.aircraft_type or config.aircraft_type
    series, report = build_series(flights, config, airport=args.airport,
                                  aircraft_type=aircraft_type)
    print(f"  aircraft   {aircraft_type} (fallback for unresolvable types)")
    print(report.format())
    if not series:
        parser.error(f"no usable series built from {args.data} — see the skip reasons above")
    return series


def split_keys_for_current_data(
    checkpoint_split_keys: list[str],
    current_provenance: dict[str, object],
) -> list[str]:
    """Restrict a pooled checkpoint split to the supplied manifest airport subset."""
    manifests = current_provenance.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("current arrival-data provenance has no manifest entries")
    airports = {
        entry.get("airport")
        for entry in manifests
        if isinstance(entry, dict) and isinstance(entry.get("airport"), str)
    }
    if not airports:
        raise ValueError("current arrival-data provenance has no airport identities")
    prefixes = tuple(f"{airport}:" for airport in sorted(airports))
    return [key for key in checkpoint_split_keys if key.startswith(prefixes)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ts_transformer",
        description="Learned 4D trajectory prediction (iTransformer / PatchTST)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train / cross validation ─────────────────────────────────────────────
    p_train = sub.add_parser("train", help="train a predictor and write a checkpoint")
    _add_data_args(p_train)
    _add_training_args(p_train)

    p_cv = sub.add_parser(
        "cross-validate", help="select hyperparameters using outer-train folds only"
    )
    _add_data_args(p_cv)
    _add_training_args(p_cv)
    p_cv.add_argument("--folds", type=int, default=3)
    p_cv.add_argument("--trials", type=int, default=4)
    p_cv.add_argument("--cv-epochs", type=int, default=12)
    p_cv.add_argument("--cv-patience", type=int, default=4)

    # ── predict ──────────────────────────────────────────────────────────────
    p_predict = sub.add_parser(
        "predict", help="forecast approaches with a checkpoint; write evaluation records")
    _add_data_args(p_predict)
    p_predict.add_argument("--checkpoint", required=True)
    p_predict.add_argument("--no-truncate", action="store_true",
                           help="keep the forecast past the threshold instead of cutting it "
                                "at closest approach")
    p_predict.add_argument("--split", choices=("test", "val", "train", "all"), default="test",
                           help="which of the checkpoint's flight splits to predict "
                                "(default: test — the only one the model never saw)")
    p_predict.add_argument("--device", default="auto",
                           help='"auto" (default: cuda when available), "cpu", "cuda". '
                                "The device is a runtime property, deliberately NOT taken "
                                "from the checkpoint — a cuda-trained checkpoint must stay "
                                "predictable on a CPU-only machine")

    args = parser.parse_args(argv)

    if args.command in ("train", "cross-validate"):
        if len(args.data) > 1 and args.airport:
            parser.error("--airport cannot override flights when multiple --data manifests are used")
        config, batch_auto = _config_from_args(args, parser)
        data_provenance = arrival_data_provenance(args.data)
        series = _build_series_or_exit(args, config, parser, load_flight_dicts(args.data))
        if args.command == "cross-validate":
            cross_validate(
                series,
                config,
                output_dir=args.output_dir,
                data_provenance=data_provenance,
                n_splits=args.folds,
                max_trials=args.trials,
                cv_epochs=args.cv_epochs,
                cv_patience=args.cv_patience,
                auto_batch_size=batch_auto,
            )
            return 0
        train(
            series,
            config,
            output_dir=args.output_dir,
            data_provenance=data_provenance,
            auto_batch_size=batch_auto,
        )
        return 0

    # ── predict ──────────────────────────────────────────────────────────────
    model, config, normalizer, payload = load_checkpoint(args.checkpoint)
    current_provenance = arrival_data_provenance(args.data)
    require_matching_data_provenance(payload, current_provenance, allow_subset=True)
    device = resolve_device(args.device)
    model = model.to(device)

    if args.aircraft_type and args.aircraft_type != config.aircraft_type:
        print(f"  WARNING: predicting with --aircraft-type {args.aircraft_type}, but the "
              f"checkpoint was trained with {config.aircraft_type} — the ENU frames and "
              f"gate targets will differ from the ones the normalizer was fit under")
    flights = load_flight_dicts(args.data)
    if args.split != "all":
        # Filter the RAW dicts by the checkpoint's airport-qualified identity, before the
        # expensive per-flight build (aircraft resolution +
        # least-squares velocity fits) never runs for the ~85% of flights a default
        # test-split predict would discard.
        split_keys = split_keys_for_current_data(
            payload["split"][args.split], current_provenance
        )
        indexed = {
            dataset_flight_key(flight, index): flight for index, flight in enumerate(flights)
        }
        missing = [key for key in split_keys if key not in indexed]
        if missing:
            parser.error(
                f"{len(missing)} flight(s) from the checkpoint's {args.split!r} split "
                f"are absent from {args.data}; first missing key: {missing[0]!r}"
            )
        flights = [indexed[key] for key in split_keys]
    series = _build_series_or_exit(args, config, parser, flights)
    print(f"predicting {len(series)} flight(s) from the {args.split!r} split")

    records, overlap = [], []
    for index, s in enumerate(series):
        forecast = forecast_approach(model, s, config, normalizer, device=device,
                                     truncate=not args.no_truncate)
        records.append(build_prediction_record(
            s,
            forecast,
            index=index,
            model_name=config.model,
            horizon_mode=config.horizon_mode,
            split=args.split,
        ))
        overlap.append(observed_series_metrics(s, forecast))

    paths = write_batch(records, output_dir=args.output_dir, config_dict=config.to_dict(),
                        overlap=overlap, checkpoint=str(args.checkpoint), split=args.split)

    capped = sum(1 for r in records if r.source.get("horizonCapped"))
    if capped:
        print(f"  WARNING: {capped} flight(s) were still short of the threshold when the "
              f"forecast horizon ran out — their final states (and gate verdicts) are "
              f"horizon artifacts, marked horizon_capped in summary.json")

    # Printed from the block that was just persisted, so the terminal and summary.json
    # cannot report different numbers for the same run.
    accuracy = accuracy_block(overlap)
    if accuracy["flights"]:
        ade, fde = accuracy["ade_m"], accuracy["fde_m"]
        print(f"  vs observed track: ADE {ade['mean']:.1f} m (p95 {ade['p95']:.1f})   "
              f"FDE {fde['mean']:.1f} m (p95 {fde['p95']:.1f})   "
              f"over {accuracy['flights']} flight(s)")

    # Flyability: what controls would these trajectories have REQUIRED, and does that sit
    # inside the airframe's envelope? Reported against the observed tracks measured the same
    # way — the check carries a known systematic bias (one clean-configuration drag polar
    # against approaches actually flown dirty), so the delta is the meaningful number and
    # the observed baseline is the floor, not 100%.
    # Each flight is judged against its OWN airframe's envelope: the KRDU harvest spans 14
    # types, so one shared envelope would grade an E170 or a CRJ9 by an A320's Cl_max and
    # max thrust.
    flyability = report_for_records(
        [record.eval_record["states"] for record in records],
        [record.reference_record["states"] for record in records],
        [s.scenario.aircraft for s in series],
    )
    (Path(args.output_dir) / "flyability_report.json").write_text(
        json.dumps(flyability, indent=2), encoding="utf-8")
    predicted, observed = flyability["predicted"], flyability["observed_baseline"]
    print(f"  flyability: {predicted['fully_flyable_rate'] * 100:.1f}% of predictions fully "
          f"flyable vs {observed['fully_flyable_rate'] * 100:.1f}% of the observed tracks "
          f"({flyability['delta']['fully_flyable_rate'] * 100:+.1f} pp)")

    print(f"✓ wrote {len(paths)} evaluation record(s) to {args.output_dir}")
    print(f"  grade them with:  python -m evaluation --input {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
