"""``predict``: forecast a checkpoint's split and write the evaluation records.

One dense rollout batch at a time, so the latent arms (K prior samples, the N(0, I)
control, the shuffle) decode the same flights the top-1 record set does. The outer-test
split is sealed: it needs a `freeze-test` ledger and `--test-release`, and the claim is
written BEFORE the first row is read, so a crash still counts as exposure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace

from config import (
    TSConfig,
    CONTROL_HOOKS_AVAILABLE,
    CONTROL_HOOK_OFF,
    CORRIDOR_GATES,
    CTA_CONDITIONING_GIVEN,
    HOOK_SATURATIONS,
    PREDICTION_CLOSURE,
)
from data_provenance import require_matching_data_provenance
from closure_output import load_labels
from dataset import dataset_flight_key, load_flight_dicts
from evaluation_protocol import (
    TestReleaseError,
    begin_test_evaluation,
    complete_test_evaluation,
)
from export import (
    accuracy_block, build_prediction_record, observed_series_metrics, write_batch,
)
from flyability import report_for_records
from forecast import (
    forecast_approaches,
    forecast_closure_from_labels,
    latent_mode_forecasts,
    posterior_latent_forecasts,
    random_latent_forecasts,
    shuffled_latent_forecasts,
)
from models import resolve_device
from train import load_checkpoint

from .common import add_data_args, build_series_or_exit, provenance_from_args, split_keys_for_current_data

HELP = "forecast approaches with a checkpoint; write evaluation records"

#: `TSConfig` field -> the predict flag that OVERRIDES the checkpoint's value for this run.
#: Predict's config comes from the checkpoint, so these are overrides rather than settings
#: of a new run — which is why they are not in `common.CLI_CONFIG_FIELDS`. They still obey
#: the same rule, with two deliberate exceptions: `--command-hook` and `--hook-saturation`
#: keep their shorter names because `CLAUDE.md` names
#: `predict --command-hook barrier --hook-saturation soft` as THE adopted delivery form and
#: two arm files (`control_hooks{,_v2}_arms.json`) spell them in still-re-runnable
#: `predict_args`; renaming would rewrite a completed campaign's record for no measurement.
PREDICT_CONFIG_FLAGS: dict[str, str] = {
    "control_command_hook": "--command-hook",
    "control_hook_saturation": "--hook-saturation",
    "control_barrier_alpha": "--control-barrier-alpha",
    "control_barrier_heading_gain": "--control-barrier-heading-gain",
}
_unknown = [name for name in PREDICT_CONFIG_FLAGS if name not in {f.name for f in fields(TSConfig)}]
if _unknown:  # fail at import, like cli.common's list: a renamed field must rename here too
    raise AssertionError(f"predict overrides unknown TSConfig fields: {_unknown}")

#: The N(0, I) control draws from its own stream, never the treatment arm's.
LATENT_RANDOM_SEED_OFFSET = 1_000_003


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    add_data_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="keep full/window forecasts past closest threshold approach",
    )
    parser.add_argument(
        "--command-hook",
        choices=[hook for hook in CONTROL_HOOKS_AVAILABLE if hook != CONTROL_HOOK_OFF],
        default=None, metavar="HOOK",
        help="run the control rollout through this command hook at prediction time "
             "(the inference-only arms); the checkpoint's own hook applies otherwise",
    )
    parser.add_argument(
        "--hook-saturation", choices=HOOK_SATURATIONS, default=None,
        help="with --command-hook: soft (tanh / softplus) or hard (clamp) saturation",
    )
    parser.add_argument(
        "--control-barrier-alpha", type=float, default=None,
        help="with --command-hook barrier: override the checkpoint's class-K gain on the "
             "corridor margins (default: the checkpoint's, normally 0.1)",
    )
    parser.add_argument(
        "--control-barrier-heading-gain", type=float, default=None,
        help="with --command-hook barrier: override the checkpoint's heading-alignment "
             "barrier gain (default: the checkpoint's, normally 0.1)",
    )
    parser.add_argument(
        "--closure-from-labels", default=None, metavar="JSON",
        help="closure output: draw every flight from its LABEL in this file instead of the "
             "model's decision (the family's own ceiling — the oracle arm); records carry "
             "source.closureFromLabels",
    )
    parser.add_argument(
        "--latent-samples", type=int, default=0, metavar="K",
        help="latent control output: besides the top-1 records, decode K prior samples per "
             "flight and write each as a full prediction directory under modes/modeNN/ "
             "(source.modeIndex / modeProbability) for the multimodal readout",
    )
    parser.add_argument("--latent-seed", type=int, default=0,
                           help="seed for --latent-samples and --latent-shuffle")
    parser.add_argument(
        "--cta-offset-s", type=float, default=0.0, metavar="SECONDS",
        help="CTA-conditioned control output: give every flight its truth arrival time plus "
             "this offset (the counterfactual a scheduler asks for); records carry "
             "source.ctaS / ctaOffsetS. 0 = the identity demonstration",
    )
    parser.add_argument(
        "--z-from-posterior", action="store_true",
        help="latent control output: decode every flight from the MEAN of q(z | its own "
             "future) — the z-oracle upper bound (reads the future; records carry "
             "source.zFromPosterior; never a prediction result). The whole output directory "
             "is the oracle arm; cannot be combined with the prior-sample options",
    )
    parser.add_argument(
        "--latent-random", type=int, default=0, metavar="K",
        help="latent control output: decode K latents drawn from N(0, I) instead of the "
             "prior per flight into random/modeNN/ — the same-K control arm minADE_K is "
             "read against (a latent that only adds K chances would match it)",
    )
    parser.add_argument(
        "--latent-shuffle", action="store_true",
        help="latent control output: also decode every flight from ANOTHER flight's top-1 "
             "latent into shuffled/ (source.latentShuffled) — the posterior-collapse reading",
    )
    parser.add_argument(
        "--project-final", choices=CORRIDOR_GATES, default=None, metavar="GATE",
        help="after truncation, clamp each state forecast's established tail (under this "
             "corridor gate) into the LPV corridor and glidepath window — the post-hoc "
             "projection arm; records carry source.projectedOntoFinal",
    )
    parser.add_argument("--split", choices=("test", "val", "train"), default="val",
                           help="which checkpoint split to predict (default: validation)")
    parser.add_argument(
        "--test-release",
        action="store_true",
        help="consume the checkpoint's frozen one-shot test ledger; required for --split test",
    )
    parser.add_argument("--device", default="auto",
                           help='"auto" (default: cuda when available), "cpu", "cuda". '
                                "The device is a runtime property, deliberately NOT taken "
                                "from the checkpoint — a cuda-trained checkpoint must stay "
                                "predictable on a CPU-only machine")


def run_cli(
    args: argparse.Namespace, parser: argparse.ArgumentParser, argv: list[str] | None
) -> int:
    del argv
    if args.split == "test" and not args.test_release:
        parser.error(
            "--split test is sealed; first run freeze-test after all experiment decisions "
            "are final, then pass --test-release"
        )
    if args.split != "test" and args.test_release:
        parser.error("--test-release is valid only with --split test")
    model, config, normalizer, payload = load_checkpoint(args.checkpoint)
    if args.aircraft_filter and args.aircraft_filter != config.aircraft_filter:
        parser.error(
            f"--aircraft-filter {args.aircraft_filter!r} differs from checkpoint policy "
            f"{config.aircraft_filter!r}"
        )
    current_provenance = provenance_from_args(args)
    require_matching_data_provenance(payload, current_provenance, allow_subset=True)
    device = resolve_device(args.device)
    model = model.to(device)

    if args.aircraft_type and args.aircraft_type != config.aircraft_type:
        print(f"  WARNING: predicting with --aircraft-type {args.aircraft_type}, but the "
              f"checkpoint was trained with {config.aircraft_type} — the ENU frames and "
              f"gate targets will differ from the ones the normalizer was fit under")
    # Resolve the exact identities before reading any trajectory rows. For test this claim
    # is deliberately written first: a crash or partial run still counts as exposure.
    split_keys = split_keys_for_current_data(
        payload["split"][args.split], current_provenance
    )
    test_claim: str | None = None
    if args.split == "test":
        try:
            test_claim = begin_test_evaluation(
                args.checkpoint,
                payload,
                current_provenance,
                split_keys,
                output_dir=args.output_dir,
            )
        except TestReleaseError as exc:
            parser.error(str(exc))
    flights = load_flight_dicts(
        args.data,
        include_flight_keys=set(split_keys),
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
    series, _build_report = build_series_or_exit(args, config, parser, flights)
    gains = {
        field: getattr(args, flag[2:].replace("-", "_"))
        for field, flag in PREDICT_CONFIG_FLAGS.items()
        if field.startswith("control_barrier_")
    }
    if args.command_hook is not None:
        if args.hook_saturation is None:
            parser.error("--command-hook needs --hook-saturation")
        config = replace(
            config,
            control_command_hook=args.command_hook,
            control_hook_saturation=args.hook_saturation,
            **{field: value for field, value in gains.items() if value is not None},
        )
        print(f"  command hook at prediction time: {args.command_hook} ({args.hook_saturation}); "
              f"alpha {config.control_barrier_alpha:g}, heading gain "
              f"{config.control_barrier_heading_gain:g}")
    elif args.hook_saturation is not None:
        parser.error("--hook-saturation needs --command-hook")
    elif any(value is not None for value in gains.values()):
        # A gain without a hook would be serialized into nothing and change no trajectory.
        parser.error(
            "--control-barrier-alpha / --control-barrier-heading-gain need --command-hook "
            "barrier; without it the rollout runs the checkpoint's own hook setting"
        )
    closure_labels = None
    if args.closure_from_labels is not None:
        if config.prediction_output != PREDICTION_CLOSURE:
            parser.error("--closure-from-labels requires a closure checkpoint")
        if args.project_final is not None or args.no_truncate:
            parser.error("--closure-from-labels draws the label as it is; --project-final / --no-truncate do not apply")
        closure_labels = load_labels(args.closure_from_labels)
        print(f"  drawing every flight from its label in {args.closure_from_labels} (the oracle arm)")
    if args.cta_offset_s and config.cta_conditioning != CTA_CONDITIONING_GIVEN:
        parser.error("--cta-offset-s needs a checkpoint trained with cta_conditioning=given")
    if config.cta_conditioning == CTA_CONDITIONING_GIVEN:
        print(f"  CTA-conditioned: every flight is given its truth arrival time {args.cta_offset_s:+g} s "
              "(reads the future — a delivery-form demonstration, not a prediction result)")
    if args.z_from_posterior:
        if config.latent_dim < 1:
            parser.error("--z-from-posterior needs a latent control checkpoint")
        if args.latent_samples or args.latent_random or args.latent_shuffle:
            parser.error("--z-from-posterior is an oracle arm of its own; do not combine it with the prior-sample options")
        print("  z-ORACLE: every flight decoded from the posterior mean of its OWN future "
              "(reads the future — an upper bound, never a prediction result)")
    if (args.latent_samples or args.latent_shuffle or args.latent_random) and config.latent_dim < 1:
        parser.error("--latent-samples / --latent-random / --latent-shuffle need a latent control checkpoint")
    if args.latent_samples < 0 or args.latent_random < 0:
        parser.error("--latent-samples and --latent-random must be non-negative")
    print(f"predicting {len(series)} flight(s) from the {args.split!r} split")

    records, flight_metrics = [], []
    # Extra decodes of the same flights: K prior samples (modes) and the shuffled-latent
    # diagnostic, each collected as its own record set and written as a full prediction
    # directory beside the top-1 one, so every readout reads them like any other arm.
    mode_records: list[list] = [[] for _ in range(args.latent_samples)]
    mode_metrics: list[list] = [[] for _ in range(args.latent_samples)]
    random_records: list[list] = [[] for _ in range(args.latent_random)]
    random_metrics: list[list] = [[] for _ in range(args.latent_random)]
    shuffled_records: list = []
    shuffled_metrics: list = []
    rollout_batch_size = max(1, min(config.batch_size, len(series)))
    if args.latent_shuffle:
        if len(series) < 2:
            parser.error("--latent-shuffle needs at least two flights")
        # Shuffling needs another flight in the batch: never a batch of one.
        rollout_batch_size = max(2, rollout_batch_size)
    print(f"  dense rollout batch size: {rollout_batch_size}")
    if args.latent_samples or args.latent_random or args.latent_shuffle:
        # Latents are drawn per batch from seed + batch start, so a flight's latent is
        # reproducible for a fixed (checkpoint, split, batch size) — say so.
        print(f"  latent seed {args.latent_seed} (per batch: seed + batch start; the N(0, I) "
              f"control adds {LATENT_RANDOM_SEED_OFFSET}); batch size {rollout_batch_size}")
    # A trailing batch of ONE flight cannot be shuffled (no other latent to take), so it
    # is folded into the batch before it rather than silently dropped from the diagnostic.
    starts = list(range(0, len(series), rollout_batch_size))
    if args.latent_shuffle and len(starts) > 1 and len(series) - starts[-1] == 1:
        starts.pop()
    for index_start, start in enumerate(starts):
        stop = starts[index_start + 1] if index_start + 1 < len(starts) else len(series)
        batch_series = series[start:stop]
        if args.latent_samples:
            for index, mode_forecasts in enumerate(latent_mode_forecasts(
                model, batch_series, config, normalizer,
                samples=args.latent_samples, seed=args.latent_seed + start, device=device,
                cta_offset_s=args.cta_offset_s,
            )):
                for offset, (s, forecast) in enumerate(zip(batch_series, mode_forecasts, strict=True)):
                    mode_records[index].append(build_prediction_record(
                        s, forecast, index=start + offset, model_name=config.model,
                        horizon_mode=config.horizon_mode, split=args.split,
                    ))
                    mode_metrics[index].append(observed_series_metrics(
                        s, forecast, points=config.validation_common_grid_points,
                    ))
        if args.latent_random:
            for index, random_forecasts in enumerate(random_latent_forecasts(
                model, batch_series, config, normalizer,
                samples=args.latent_random,
                seed=args.latent_seed + LATENT_RANDOM_SEED_OFFSET + start, device=device,
                cta_offset_s=args.cta_offset_s,
            )):
                for offset, (s, forecast) in enumerate(zip(batch_series, random_forecasts, strict=True)):
                    random_records[index].append(build_prediction_record(
                        s, forecast, index=start + offset, model_name=config.model,
                        horizon_mode=config.horizon_mode, split=args.split,
                    ))
                    random_metrics[index].append(observed_series_metrics(
                        s, forecast, points=config.validation_common_grid_points,
                    ))
        if args.latent_shuffle:
            for offset, (s, forecast) in enumerate(zip(batch_series, shuffled_latent_forecasts(
                model, batch_series, config, normalizer, seed=args.latent_seed + start, device=device,
                cta_offset_s=args.cta_offset_s,
            ), strict=True)):
                shuffled_records.append(build_prediction_record(
                    s, forecast, index=start + offset, model_name=config.model,
                    horizon_mode=config.horizon_mode, split=args.split,
                ))
                shuffled_metrics.append(observed_series_metrics(
                    s, forecast, points=config.validation_common_grid_points,
                ))
        if closure_labels is not None:
            forecasts = forecast_closure_from_labels(batch_series, config, closure_labels)
        elif args.z_from_posterior:
            forecasts = posterior_latent_forecasts(
                model, batch_series, config, normalizer, device=device, cta_offset_s=args.cta_offset_s,
            )
        else:
            forecasts = forecast_approaches(
                model,
                batch_series,
                config,
                normalizer,
                device=device,
                truncate=not args.no_truncate,
                project_final=args.project_final,
                cta_offset_s=args.cta_offset_s,
            )
        for offset, (s, forecast) in enumerate(
            zip(batch_series, forecasts, strict=True)
        ):
            records.append(build_prediction_record(
                s,
                forecast,
                index=start + offset,
                model_name=config.model,
                horizon_mode=config.horizon_mode,
                split=args.split,
            ))
            flight_metrics.append(observed_series_metrics(
                s,
                forecast,
                points=config.validation_common_grid_points,
            ))

    paths = write_batch(
        records,
        output_dir=args.output_dir,
        config_dict=config.to_dict(),
        flight_metrics=flight_metrics,
        checkpoint=str(args.checkpoint),
        split=args.split,
    )
    for index, (mode_rows, mode_flight_metrics) in enumerate(zip(mode_records, mode_metrics, strict=True)):
        write_batch(
            mode_rows, output_dir=args.output_dir / "modes" / f"mode{index:02d}",
            config_dict=config.to_dict(), flight_metrics=mode_flight_metrics,
            checkpoint=str(args.checkpoint), split=args.split,
        )
    if args.latent_samples:
        print(f"  wrote {args.latent_samples} prior-sample mode(s) under {args.output_dir / 'modes'}")
    for index, (random_rows, random_flight_metrics) in enumerate(zip(random_records, random_metrics, strict=True)):
        write_batch(
            random_rows, output_dir=args.output_dir / "random" / f"mode{index:02d}",
            config_dict=config.to_dict(), flight_metrics=random_flight_metrics,
            checkpoint=str(args.checkpoint), split=args.split,
        )
    if args.latent_random:
        print(f"  wrote {args.latent_random} N(0, I) control mode(s) under {args.output_dir / 'random'}")
    if shuffled_records:
        write_batch(
            shuffled_records, output_dir=args.output_dir / "shuffled",
            config_dict=config.to_dict(), flight_metrics=shuffled_metrics,
            checkpoint=str(args.checkpoint), split=args.split,
        )
        print(f"  wrote the shuffled-latent diagnostic under {args.output_dir / 'shuffled'}")

    if args.project_final is not None:
        print(f"  projected every state forecast onto the final ({args.project_final} gate)")
    capped = sum(record.source.get("horizonCapped", False) for record in records)
    if capped:
        print(
            f"  WARNING: {capped} forecast(s) reached the fixed H_full cap before a "
            "threshold closest-approach was detected; their final gate states are cap artifacts"
        )

    # Printed from the block that was just persisted, so the terminal and summary.json
    # cannot report different numbers for the same run.
    accuracy = accuracy_block(flight_metrics)
    if accuracy["flights"]:
        ade, fde = accuracy["ade_m"], accuracy["fde_m"]
        print(f"  vs observed track: ADE {ade['mean']:.1f} m (p95 {ade['p95']:.1f})   "
              f"FDE {fde['mean']:.1f} m (p95 {fde['p95']:.1f})   "
              f"over {accuracy['flights']} flight(s)")
    timing = accuracy["final_time_s"]
    print(f"  final time: MAE {timing['mae']:.1f} s   "
          f"p95 {timing['p95_abs']:.1f} s   bias {timing['mean_signed']:+.1f} s")
    raw = accuracy["raw_kinematics"]
    predicted_raw, observed_raw = raw["predicted"], raw["observed_baseline"]
    print(
        "  raw kinematics fleet p95 (prediction vs observed): "
        f"pos/vel RMSE {predicted_raw['position_velocity_rmse_mps']['p95']:.2f}/"
        f"{observed_raw['position_velocity_rmse_mps']['p95']:.2f} m/s   "
        f"turn {predicted_raw['turn_rate_p95_deg_s']['p95']:.2f}/"
        f"{observed_raw['turn_rate_p95_deg_s']['p95']:.2f} deg/s   "
        f"accel {predicted_raw['acceleration_p95_mps2']['p95']:.2f}/"
        f"{observed_raw['acceleration_p95_mps2']['p95']:.2f} m/s²   "
        f"jerk {predicted_raw['jerk_p95_mps3']['p95']:.2f}/"
        f"{observed_raw['jerk_p95_mps3']['p95']:.2f} m/s³"
    )

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
    (args.output_dir / "flyability_report.json").write_text(
        json.dumps(flyability, indent=2), encoding="utf-8")
    predicted, observed = flyability["predicted"], flyability["observed_baseline"]
    print(f"  flyability: {predicted['fully_flyable_rate'] * 100:.1f}% of predictions fully "
          f"flyable vs {observed['fully_flyable_rate'] * 100:.1f}% of the observed tracks "
          f"({flyability['delta']['fully_flyable_rate'] * 100:+.1f} pp)")

    if test_claim is not None:
        complete_test_evaluation(args.checkpoint, test_claim)

    print(f"✓ wrote {len(paths)} evaluation record(s) to {args.output_dir}")
    print(f"  grade them with:  python -m evaluation --input {args.output_dir}")
    return 0
