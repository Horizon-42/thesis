"""``predict``: forecast a checkpoint's split and write the evaluation records.

One dense rollout batch at a time, so the latent arms (K prior samples, the N(0, I)
control, the shuffle) and B3's quantile fan decode the same flights the top-1 record set
does. The outer-test split is sealed: it needs a `freeze-test` ledger and `--test-release`,
and the claim is written BEFORE the first row is read, so a crash still counts as exposure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

from config import (
    DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S,
    DURATION_HEAD_QUANTILE,
    DURATION_MEDIAN_INDEX,
    DURATION_QUANTILES,
    TSConfig,
    CONTROL_HOOKS_AVAILABLE,
    CONTROL_HOOK_OFF,
    CORRIDOR_GATES,
    CTA_CONDITIONING_GIVEN,
    CTA_CONDITIONING_SELF_QUANTILE,
    HOOK_SATURATIONS,
    PREDICTION_CLOSURE,
)
from approach_difficulty import approach_difficulty
from calibration import (
    FAN_INTERVAL_ALPHA,
    QUANTILE_DIR_NAME,
    conformal_intervals,
    interval_directory_name,
    interval_stratum,
    load_conformal_table,
    quantile_directory_name,
)
from data_provenance import require_matching_data_provenance
from closure_output import load_labels
from dataset import dataset_flight_key, load_flight_dicts, truth_duration_s
from evaluation_protocol import (
    TestReleaseError,
    begin_test_evaluation,
    complete_test_evaluation,
)
from export import (
    accuracy_block, build_prediction_record, observed_series_metrics, write_batch,
)
from flyability import report_for_records
from io_utils import file_sha256
from forecast import (
    default_anchor,
    duration_quantile_predictions,
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


@dataclass(frozen=True)
class FanLeaf:
    """One decode of the whole batch at one arrival time the MODEL chose.

    ``cta_s`` is per flight, in batch order. Exactly one of ``quantile`` / ``interval`` is
    set: a level of the duration head, or an endpoint of its calibrated interval.
    """

    directory: str
    cta_s: np.ndarray
    quantile: float | None
    interval: dict[str, object] | None


def fan_leaves(
    quantiles_s: np.ndarray, series, anchor: int, conformal: dict | None
) -> list[FanLeaf]:
    """The fan for one batch: the five levels, plus the calibrated endpoints when there is
    a table. The stratum each flight's interval is read from is decided by the SAME
    `calibration.interval_stratum` the record's own `durationIntervalStratum` comes from."""
    leaves = [
        FanLeaf(quantile_directory_name(tau), quantiles_s[:, column], tau, None)
        for column, tau in enumerate(DURATION_QUANTILES)
    ]
    if conformal is None:
        return leaves
    endpoints = []
    for row, item in zip(quantiles_s, series, strict=True):
        stratum = interval_stratum(conformal, approach_difficulty(item, anchor).to_dict())
        entry = next(
            block for block in conformal_intervals(row, conformal, stratum)
            if block["alpha"] == FAN_INTERVAL_ALPHA
        )
        endpoints.append([entry["lo"], entry["hi"]])
    bounds = np.array(endpoints, dtype=np.float64)
    return leaves + [
        FanLeaf(
            interval_directory_name(FAN_INTERVAL_ALPHA, end),
            bounds[:, column],
            None,
            {"alpha": FAN_INTERVAL_ALPHA, "end": end},
        )
        for column, end in enumerate(("lo", "hi"))
    ]


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
        "--cta-from-quantiles", action="store_true",
        help="CTA-conditioned control output with a quantile duration head: decode every "
             "flight once per duration quantile, using ITS OWN q_tau as the CTA, into "
             f"{QUANTILE_DIR_NAME}/qNN/ (source.ctaQuantile / ctaFromQuantiles). The top-1 "
             "records are the q50 decode. This reads no future — the run is cta=self-q, not "
             "cta=given — and with a conformal table it also decodes the calibrated "
             f"alpha={FAN_INTERVAL_ALPHA:g} interval endpoints",
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


def _fan_forecast(model, series, config, normalizer, device, args, conformal, leaf: FanLeaf):
    """One leaf's decode — the batch flown to the arrival time this leaf names."""
    return forecast_approaches(
        model, series, config, normalizer, device=device,
        truncate=not args.no_truncate, project_final=args.project_final,
        conformal=conformal, cta_s=leaf.cta_s,
        cta_quantile=leaf.quantile, cta_interval=leaf.interval,
    )


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
    if args.cta_from_quantiles:
        # The head trains as a TARGET under `given` even though the given CTA is what drives
        # the rollout in training; at predict the rollout is driven by the model's OWN
        # quantile, which is what makes this the first CTA arm that reads no future.
        if config.cta_conditioning != CTA_CONDITIONING_GIVEN:
            parser.error("--cta-from-quantiles needs a checkpoint trained with "
                         "cta_conditioning=given: without the CTA token the decoder has "
                         "nowhere to put the quantile")
        if config.duration_head != DURATION_HEAD_QUANTILE:
            parser.error("--cta-from-quantiles needs a checkpoint trained with "
                         f"duration_head={DURATION_HEAD_QUANTILE!r}; a point head has one "
                         "duration and there is no fan to decode")
        if args.cta_offset_s:
            parser.error("--cta-from-quantiles and --cta-offset-s are two different arms: "
                         "the first reads the model's own duration, the second shifts the "
                         "truth's")
        if args.z_from_posterior:
            parser.error("--cta-from-quantiles and --z-from-posterior cannot combine: the "
                         "posterior reads the future, which is exactly what the self-quantile "
                         "CTA exists not to do")
        # Everything downstream — the run name, the records, the summary mode — must say
        # `self-q`, never `given`: this directory read no future. The MODEL is unchanged;
        # only the config that names and describes the run is restamped.
        config = replace(config, cta_conditioning=CTA_CONDITIONING_SELF_QUANTILE)
        print("  CTA from the model's OWN duration quantiles (cta=self-q): a fan of "
              f"{len(DURATION_QUANTILES)} decodes per flight; the top-1 records are q50")
    elif config.cta_conditioning == CTA_CONDITIONING_GIVEN:
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
    skipped: dict[str, int] = {}
    if args.cta_offset_s:
        # A counterfactual CTA that leaves less than the package's minimum remaining future
        # is not a plausible arrival time (truth durations start at ~21 s, so −90 s would ask
        # for a landing in the past): those flights are SKIPPED and counted, never clamped —
        # a clamp would silently change the offset the scan is read against.
        anchor = config.seq_len - 1
        kept = [item for item in series
                if truth_duration_s(item, anchor) + args.cta_offset_s >= DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S]
        skipped["cta_below_min_future"] = len(series) - len(kept)
        if not kept:
            parser.error(f"--cta-offset-s {args.cta_offset_s:+g} leaves no flight with at least "
                         f"{DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S:g} s to fly")
        print(f"  CTA offset {args.cta_offset_s:+g} s: {len(kept)} of {len(series)} flights keep >= "
              f"{DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S:g} s of remaining future; "
              f"{skipped['cta_below_min_future']} skipped (stated in summary.json)")
        series = kept
    # B2: the checkpoint's own conformal table, if it has ever been calibrated. Bound to
    # the weights by sha256 — a table left over from another checkpoint RAISES rather than
    # widening this run's intervals by someone else's delta.
    conformal = None
    if config.duration_head == DURATION_HEAD_QUANTILE:
        conformal = load_conformal_table(args.checkpoint, file_sha256(Path(args.checkpoint)))
        if conformal is None:
            print("  quantile duration head, NOT calibrated: records carry the raw "
                  "durationQuantilesS and calibrated=false (run run_ts_eta_calibration.py)")
        else:
            print(f"  quantile duration head, calibrated on {conformal['calibration_flights']} "
                  f"{conformal['split']} flights: records carry durationIntervalS per alpha "
                  f"{list(conformal['alphas'])}")

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
    # B3: one record set per fan leaf, keyed by the directory it will be written to.
    fan_records: dict[str, list] = {}
    fan_metrics: dict[str, list] = {}
    median_directory = quantile_directory_name(DURATION_QUANTILES[DURATION_MEDIAN_INDEX])
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
        elif args.cta_from_quantiles:
            anchor = default_anchor(config)
            quantiles = duration_quantile_predictions(
                model, batch_series, config, normalizer, anchor=anchor, device=device
            )
            leaves = fan_leaves(quantiles, batch_series, anchor, conformal)
            # The rollout's own requirement, and the only one that can bite: the five
            # levels are strictly positive by construction, so a non-positive CTA means a
            # conformal delta wider than the interval it widens. Never clamped and never
            # silently skipped — a fan whose leaves hold different flights is not a fan, and
            # the readout compares them per flight.
            below = sum(int((leaf.cta_s <= 0.0).sum()) for leaf in leaves)
            if below:
                parser.error(
                    f"{below} fan CTA(s) in this batch are not positive, so the rollout "
                    "cannot fly them; a calibrated interval endpoint reached zero, which "
                    "means a conformal delta wider than the interval it widens. Recalibrate "
                    "before decoding the endpoints"
                )
            forecasts = _fan_forecast(
                model, batch_series, config, normalizer, device, args, conformal,
                next(leaf for leaf in leaves if leaf.directory == median_directory),
            )
            for leaf in leaves:
                leaf_forecasts = forecasts if leaf.directory == median_directory else (
                    _fan_forecast(
                        model, batch_series, config, normalizer, device, args, conformal, leaf
                    )
                )
                for offset, (item, forecast) in enumerate(
                    zip(batch_series, leaf_forecasts, strict=True)
                ):
                    fan_records.setdefault(leaf.directory, []).append(build_prediction_record(
                        item, forecast, index=start + offset, model_name=config.model,
                        horizon_mode=config.horizon_mode, split=args.split,
                    ))
                    fan_metrics.setdefault(leaf.directory, []).append(observed_series_metrics(
                        item, forecast, points=config.validation_common_grid_points,
                    ))
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
                conformal=conformal,
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
        skipped=skipped,
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
    for directory, rows in fan_records.items():
        write_batch(
            rows, output_dir=args.output_dir / QUANTILE_DIR_NAME / directory,
            config_dict=config.to_dict(), flight_metrics=fan_metrics[directory],
            checkpoint=str(args.checkpoint), split=args.split,
        )
    if fan_records:
        names = ", ".join(sorted(fan_records))
        print(f"  wrote the quantile fan under {args.output_dir / QUANTILE_DIR_NAME}: {names} "
              f"(the top-1 records above are the {median_directory} decode); read it with "
              f"run_ts_quantile_fan_readout.py --arm {args.output_dir}")
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
