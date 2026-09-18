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

from ts_transformer.config import (
    DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S,
    DURATION_HEADS_WITH_QUANTILES,
    DURATION_MEDIAN_INDEX,
    DURATION_QUANTILES,
    PLAN_CONDITIONING_OFF,
    TSConfig,
    CONTROL_HOOKS_AVAILABLE,
    CONTROL_HOOK_BARRIER,
    CONTROL_HOOK_MEMBERS,
    CONTROL_HOOK_OFF,
    CONTROL_HOOK_TROMBONE,
    CONTROL_SPEED_FLOOR_MARGIN_READERS,
    CORRIDOR_GATE_ON_FINAL,
    CTA_CONDITIONING_GIVEN,
    CTA_CONDITIONING_SELF_QUANTILE,
    HOOK_SATURATIONS,
    TROMBONE_SURPLUS_REFERENCES,
)
from ts_transformer.data.approach_difficulty import approach_difficulty
from ts_transformer.inference.calibration import (
    FAN_INTERVAL_ALPHA,
    QUANTILE_DIR_NAME,
    conformal_intervals,
    interval_directory_name,
    interval_stratum,
    load_conformal_table,
    quantile_directory_name,
)
from ts_transformer.data.data_provenance import require_matching_data_provenance
from ts_transformer.data.dataset import dataset_flight_key, load_flight_dicts, truth_duration_s
from ts_transformer.inference.evaluation_protocol import (
    TestReleaseError,
    begin_test_evaluation,
    complete_test_evaluation,
)
from ts_transformer.inference.export import (
    accuracy_block, build_prediction_record, observed_series_metrics, write_batch,
)
from ts_transformer.geometry.flyability import report_for_records
from ts_transformer.io_utils import file_sha256
from ts_transformer.inference.forecast import cut_at_threshold_crossing, default_anchor, forecast_approaches
from ts_transformer.outputs.control.forecast import (
    duration_quantile_predictions,
    latent_mode_forecasts,
    posterior_latent_forecasts,
    random_latent_forecasts,
    shuffled_latent_forecasts,
)
from ts_transformer.outputs import ForecastOptions
from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.training.train import load_checkpoint

from .common import (
    add_data_args,
    build_series_or_exit,
    provenance_from_args,
    refuse_airport_override_for_pooled_data,
    split_keys_for_current_data,
)

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
    "control_speed_floor_margin": "--control-speed-floor-margin",
    # ...and the third deliberate short name: the arm files spell `--trombone-surplus`, and
    # the field is `trombone_surplus_reference` because "reference" is what it names.
    "trombone_surplus_reference": "--trombone-surplus",
    # The shared data flag (`common.add_data_args`): predicting under another airframe
    # builds the series under IT (target Vref, crossing height), so the config the run
    # writes beside its records must say so too (review C-5: until 2026-09-09 the summary
    # and the run name recorded the checkpoint's type).
    "aircraft_type": "--aircraft-type",
}
_unknown = [name for name in PREDICT_CONFIG_FLAGS if name not in {f.name for f in fields(TSConfig)}]
if _unknown:  # fail at import, like cli.common's list: a renamed field must rename here too
    raise AssertionError(f"predict overrides unknown TSConfig fields: {_unknown}")

#: The hook fields `--command-hook` may override: gains and margins, never the hook itself
#: or its saturation (those two ARE the selection). A value here without `--command-hook`
#: would be serialized into nothing and change no trajectory, so it is refused below.
HOOK_TUNING_FIELDS = frozenset({
    "control_barrier_alpha",
    "control_barrier_heading_gain",
    "control_speed_floor_margin",
    "trombone_surplus_reference",
})
assert HOOK_TUNING_FIELDS < set(PREDICT_CONFIG_FLAGS)

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
        "--truncate-at-threshold",
        action="store_true",
        help="cut every record where it first crosses the landing threshold ON THE FINAL "
             "(past the plane AND inside the on-final gate there) and stamp "
             "source.truncatedAtThreshold — so the evaluation, corridor and "
             "flyability reports see the APPROACH, not the flying a rollout does after it "
             "arrives (L3.d: a floored, late-CTA rollout arrives early and keeps going, "
             "endpoint |xt| p95 43-63 km). Applies to every output kind, the control "
             "rollout included; a forecast that never gets onto the final is left whole — "
             "the plane alone is crossed abeam, on a downwind",
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
        "--control-speed-floor-margin", type=float, default=None,
        help="with a --command-hook containing speed-floor or trombone: override the "
             "checkpoint's stall margin, V_floor = margin x V_stall(n_commanded) — the "
             "speed the floor holds and the speed the trombone sizes its detour against "
             "(default: the checkpoint's, normally 1.1)",
    )
    parser.add_argument(
        "--trombone-surplus", choices=TROMBONE_SURPLUS_REFERENCES, default=None,
        help="with a --command-hook containing trombone: what the stretch is sized "
             "against — 'beeline' (the default and what L3.e ran: the straight-line "
             "distance to the threshold, which reads a vectored flight's intended downwind "
             "and base as surplus time) or 'reference-rollout' (the remaining path of the "
             "hook-free reference rollout, i.e. what the network itself intends to fly)",
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
             "cta=given. The calibrated interval endpoints are a separate opt-in "
             "(--interval-endpoints)",
    )
    parser.add_argument(
        "--interval-endpoints", action="store_true",
        help=f"with --cta-from-quantiles and a conformal table: ALSO decode the calibrated "
             f"alpha={FAN_INTERVAL_ALPHA:g} interval endpoints into "
             f"{QUANTILE_DIR_NAME}/{interval_directory_name(FAN_INTERVAL_ALPHA, 'lo')}/ and "
             f"{interval_directory_name(FAN_INTERVAL_ALPHA, 'hi')}/. Off by default: the fan "
             "readout scores the five quantile leaves, so these two are an extra arm to look "
             "at rather than part of the gate",
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
        "--project-final", choices=(CORRIDOR_GATE_ON_FINAL,), default=None, metavar="GATE",
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


@dataclass(frozen=True)
class PredictOptions:
    """The flag combinations `predict` accepts, checked ONCE by :func:`parse_predict_options`
    (review §4.4): what every forecast is asked (`forecast`), which oracle and diagnostic arms
    are decoded beside the top-1 records, and the cohort the CTA offset skipped."""

    forecast: ForecastOptions
    cta_from_quantiles: bool
    interval_endpoints: bool
    z_from_posterior: bool
    latent_samples: int
    latent_random: int
    latent_shuffle: bool
    latent_seed: int
    skipped: dict[str, int]

    @property
    def conformal(self) -> dict | None:
        return self.forecast.conformal


@dataclass
class PredictionSets:
    """Every record set one `predict` run assembles: the top-1 records and, beside them, the
    latent modes, the N(0, I) controls, the quantile fan and the shuffled-latent diagnostic."""

    records: list
    flight_metrics: list
    mode_records: list[list]
    mode_metrics: list[list]
    random_records: list[list]
    random_metrics: list[list]
    fan_records: dict[str, list]
    fan_metrics: dict[str, list]
    shuffled_records: list
    shuffled_metrics: list
    median_directory: str


def _cut_at_threshold(forecasts, series, truncate_at_threshold: bool):
    """`--truncate-at-threshold` on a list of forecasts the batch produced some other way.

    `forecast_approaches` takes the flag itself; the latent, posterior and label decoders
    are their own entry points, and a flag that cut the main records but left a run's own
    diagnostic arms uncut would make the two incomparable inside one output directory.
    """
    if not truncate_at_threshold:
        return forecasts
    return [
        cut_at_threshold_crossing(forecast, item)
        for forecast, item in zip(forecasts, series, strict=True)
    ]


def _fan_forecast(model, series, config, normalizer, device, options: PredictOptions, leaf: FanLeaf):
    """One leaf's decode — the batch flown to the arrival time this leaf names."""
    return forecast_approaches(
        model, series, config, normalizer, device=device,
        options=replace(
            options.forecast,
            cta_s=leaf.cta_s, cta_quantile=leaf.quantile, cta_interval=leaf.interval,
        ),
    )


def load_predict_checkpoint(args: argparse.Namespace, parser: argparse.ArgumentParser):
    """The checkpoint on the device, its config restamped with the run's own aircraft type,
    and the data provenance the split keys are resolved against."""
    refuse_airport_override_for_pooled_data(args, parser)
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
        # The series are built under this type below; the config written beside the
        # records (and the run name) must carry it, not the checkpoint's (review C-5).
        config = replace(config, aircraft_type=args.aircraft_type)
    return model, config, normalizer, payload, current_provenance, device


def load_predict_series(args, config, parser, payload, current_provenance):
    """The split's flights, in the checkpoint's own order, built into series under the
    run's config; for the test split, the exposure claim is written FIRST."""
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
    return series, test_claim


def parse_predict_options(args, config, parser, series):
    """Every flag-combination rule of `predict`, in one place; returns the options, the
    config restamped with what this run overrides (the hook and its gains, `cta=self-q`),
    and the series the CTA offset keeps."""
    gains = {
        field: getattr(args, flag[2:].replace("-", "_"))
        for field, flag in PREDICT_CONFIG_FLAGS.items()
        if field in HOOK_TUNING_FIELDS
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
        # Only the modules this hook builds have knobs worth printing: a stall margin under
        # `barrier` is a number nothing reads (and TSConfig refuses it away from its default).
        modules = CONTROL_HOOK_MEMBERS[args.command_hook]
        tunings = []
        if CONTROL_HOOK_BARRIER in modules:
            tunings.append(f"alpha {config.control_barrier_alpha:g}")
            tunings.append(f"heading gain {config.control_barrier_heading_gain:g}")
        if any(name in modules for name in CONTROL_SPEED_FLOOR_MARGIN_READERS):
            tunings.append(f"stall margin {config.control_speed_floor_margin:g}")
        if CONTROL_HOOK_TROMBONE in modules:
            tunings.append(f"surplus vs {config.trombone_surplus_reference}")
        print(f"  command hook at prediction time: {args.command_hook} "
              f"({args.hook_saturation}); " + ", ".join(tunings))
    elif args.hook_saturation is not None:
        parser.error("--hook-saturation needs --command-hook")
    elif any(value is not None for value in gains.values()):
        # A gain without a hook would be serialized into nothing and change no trajectory.
        # The list is the table's own, so a flag added there cannot go missing here.
        named = " / ".join(sorted(PREDICT_CONFIG_FLAGS[name] for name in HOOK_TUNING_FIELDS))
        parser.error(
            f"{named} need --command-hook; without it the rollout runs the checkpoint's "
            "own hook setting"
        )
    if args.truncate_at_threshold and args.no_truncate:
        parser.error(
            "--no-truncate keeps a fixed-time STATE forecast past its closest threshold "
            "approach and --truncate-at-threshold cuts every forecast at the crossing: the "
            "two ask for opposite records. (On a control checkpoint --no-truncate is a "
            "no-op — the fixed-time postprocessors never run there — so the pair is refused "
            "on the intent, not on the effect.)"
        )
    if args.cta_offset_s and config.cta_conditioning != CTA_CONDITIONING_GIVEN:
        parser.error("--cta-offset-s needs a checkpoint trained with cta_conditioning=given")
    if (args.cta_offset_s or args.cta_from_quantiles) and config.plan_conditioning != PLAN_CONDITIONING_OFF:
        parser.error(
            f"plan_conditioning={config.plan_conditioning!r} hands the decoder the truth's plan, whose "
            "arrival time is the truth's: a shifted or self-quantile CTA beside it is two arrival times, "
            "and a self-quantile decode would still read the future through the token"
        )
    if args.interval_endpoints and not args.cta_from_quantiles:
        parser.error("--interval-endpoints is part of the quantile fan; it needs "
                     "--cta-from-quantiles")
    if args.cta_from_quantiles:
        # The head trains as a TARGET under `given` even though the given CTA is what drives
        # the rollout in training; at predict the rollout is driven by the model's OWN
        # quantile, which is what makes this the first CTA arm that reads no future.
        if config.cta_conditioning != CTA_CONDITIONING_GIVEN:
            parser.error("--cta-from-quantiles needs a checkpoint trained with "
                         "cta_conditioning=given: without the CTA token the decoder has "
                         "nowhere to put the quantile")
        if config.duration_head not in DURATION_HEADS_WITH_QUANTILES:
            parser.error("--cta-from-quantiles needs a checkpoint whose duration head emits "
                         f"quantiles ({', '.join(DURATION_HEADS_WITH_QUANTILES)}); "
                         f"duration_head={config.duration_head!r} has one duration and "
                         "there is no fan to decode")
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
    if args.latent_shuffle and len(series) < 2:
        parser.error("--latent-shuffle needs at least two flights")
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
        anchor = default_anchor(config)
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
    if config.duration_head in DURATION_HEADS_WITH_QUANTILES:
        conformal = load_conformal_table(args.checkpoint, file_sha256(Path(args.checkpoint)))
        if conformal is None:
            print("  quantile duration head, NOT calibrated: records carry the raw "
                  "durationQuantilesS and calibrated=false (run `run_ts.py eta_calibration`)")
        else:
            cohort = "/".join(conformal["airports"]) or "unstated airports"
            print(f"  quantile duration head, calibrated on {conformal['calibration_flights']} "
                  f"{conformal['split']} flights at {cohort}: records carry durationIntervalS "
                  f"per alpha {list(conformal['alphas'])}"
                  + ("  [SMOKE TABLE: the delta was fitted on a "
                     f"--limit {conformal['limit']} prefix of that split]"
                     if conformal["smoke_test"] else ""))

    return (
        PredictOptions(
            forecast=ForecastOptions(
                truncate=not args.no_truncate,
                project_final=args.project_final,
                cta_offset_s=args.cta_offset_s,
                conformal=conformal,
                truncate_at_threshold=args.truncate_at_threshold,
            ),
            cta_from_quantiles=args.cta_from_quantiles,
            interval_endpoints=args.interval_endpoints,
            z_from_posterior=args.z_from_posterior,
            latent_samples=args.latent_samples,
            latent_random=args.latent_random,
            latent_shuffle=args.latent_shuffle,
            latent_seed=args.latent_seed,
            skipped=skipped,
        ),
        config,
        series,
    )


def predict_sets(model, series, config, normalizer, device, options: PredictOptions, *, split: str) -> PredictionSets:
    """Every decode of the split: the top-1 records and the arms beside them."""
    print(f"predicting {len(series)} flight(s) from the {split!r} split")

    records, flight_metrics = [], []
    # Extra decodes of the same flights: K prior samples (modes) and the shuffled-latent
    # diagnostic, each collected as its own record set and written as a full prediction
    # directory beside the top-1 one, so every readout reads them like any other arm.
    mode_records: list[list] = [[] for _ in range(options.latent_samples)]
    mode_metrics: list[list] = [[] for _ in range(options.latent_samples)]
    random_records: list[list] = [[] for _ in range(options.latent_random)]
    random_metrics: list[list] = [[] for _ in range(options.latent_random)]
    shuffled_records: list = []
    shuffled_metrics: list = []
    # B3: one record set per fan leaf, keyed by the directory it will be written to.
    fan_records: dict[str, list] = {}
    fan_metrics: dict[str, list] = {}
    median_directory = quantile_directory_name(DURATION_QUANTILES[DURATION_MEDIAN_INDEX])
    rollout_batch_size = max(1, min(config.batch_size, len(series)))
    if options.latent_shuffle:
        # Shuffling needs another flight in the batch: never a batch of one.
        rollout_batch_size = max(2, rollout_batch_size)
    print(f"  dense rollout batch size: {rollout_batch_size}")
    if options.latent_samples or options.latent_random or options.latent_shuffle:
        # Latents are drawn per batch from seed + batch start, so a flight's latent is
        # reproducible for a fixed (checkpoint, split, batch size) — say so.
        print(f"  latent seed {options.latent_seed} (per batch: seed + batch start; the N(0, I) "
              f"control adds {LATENT_RANDOM_SEED_OFFSET}); batch size {rollout_batch_size}")
    # A trailing batch of ONE flight cannot be shuffled (no other latent to take), so it
    # is folded into the batch before it rather than silently dropped from the diagnostic.
    starts = list(range(0, len(series), rollout_batch_size))
    if options.latent_shuffle and len(starts) > 1 and len(series) - starts[-1] == 1:
        starts.pop()
    for index_start, start in enumerate(starts):
        stop = starts[index_start + 1] if index_start + 1 < len(starts) else len(series)
        batch_series = series[start:stop]
        if options.latent_samples:
            for index, mode_forecasts in enumerate(latent_mode_forecasts(
                model, batch_series, config, normalizer,
                samples=options.latent_samples, seed=options.latent_seed + start, device=device,
                cta_offset_s=options.forecast.cta_offset_s,
            )):
                for offset, (s, forecast) in enumerate(zip(
                    batch_series, _cut_at_threshold(mode_forecasts, batch_series, options.forecast.truncate_at_threshold), strict=True
                )):
                    mode_records[index].append(build_prediction_record(
                        s, forecast, index=start + offset, model_name=config.model,
                        horizon_mode=config.horizon_mode, split=split,
                    ))
                    mode_metrics[index].append(observed_series_metrics(
                        s, forecast, points=config.validation_common_grid_points,
                    ))
        if options.latent_random:
            for index, random_forecasts in enumerate(random_latent_forecasts(
                model, batch_series, config, normalizer,
                samples=options.latent_random,
                seed=options.latent_seed + LATENT_RANDOM_SEED_OFFSET + start, device=device,
                cta_offset_s=options.forecast.cta_offset_s,
            )):
                for offset, (s, forecast) in enumerate(zip(
                    batch_series, _cut_at_threshold(random_forecasts, batch_series, options.forecast.truncate_at_threshold), strict=True
                )):
                    random_records[index].append(build_prediction_record(
                        s, forecast, index=start + offset, model_name=config.model,
                        horizon_mode=config.horizon_mode, split=split,
                    ))
                    random_metrics[index].append(observed_series_metrics(
                        s, forecast, points=config.validation_common_grid_points,
                    ))
        if options.latent_shuffle:
            for offset, (s, forecast) in enumerate(zip(batch_series, _cut_at_threshold(
                shuffled_latent_forecasts(
                    model, batch_series, config, normalizer,
                    seed=options.latent_seed + start, device=device,
                    cta_offset_s=options.forecast.cta_offset_s,
                ), batch_series, options.forecast.truncate_at_threshold,
            ), strict=True)):
                shuffled_records.append(build_prediction_record(
                    s, forecast, index=start + offset, model_name=config.model,
                    horizon_mode=config.horizon_mode, split=split,
                ))
                shuffled_metrics.append(observed_series_metrics(
                    s, forecast, points=config.validation_common_grid_points,
                ))
        if options.z_from_posterior:
            forecasts = _cut_at_threshold(posterior_latent_forecasts(
                model, batch_series, config, normalizer, device=device, cta_offset_s=options.forecast.cta_offset_s,
            ), batch_series, options.forecast.truncate_at_threshold)
        elif options.cta_from_quantiles:
            anchor = default_anchor(config)
            quantiles = duration_quantile_predictions(
                model, batch_series, config, normalizer, anchor=anchor, device=device
            )
            # The endpoints are opt-in: without the flag the fan is exactly the five
            # levels the readout's gate scores.
            leaves = fan_leaves(
                quantiles, batch_series, anchor,
                options.conformal if options.interval_endpoints else None,
            )
            # The rollout's own requirement, and the only one that can bite: the five
            # levels are strictly positive by construction, so a non-positive CTA means a
            # conformal delta wider than the interval it widens. Never clamped and never
            # silently skipped — a fan whose leaves hold different flights is not a fan, and
            # the readout compares them per flight.
            below = sum(int((leaf.cta_s <= 0.0).sum()) for leaf in leaves)
            if below:
                raise ValueError(
                    f"{below} fan CTA(s) in this batch are not positive, so the rollout "
                    "cannot fly them; a calibrated interval endpoint reached zero, which "
                    "means a conformal delta wider than the interval it widens. Recalibrate "
                    "before decoding the endpoints"
                )
            forecasts = _fan_forecast(
                model, batch_series, config, normalizer, device, options,
                next(leaf for leaf in leaves if leaf.directory == median_directory),
            )
            for leaf in leaves:
                leaf_forecasts = forecasts if leaf.directory == median_directory else (
                    _fan_forecast(
                        model, batch_series, config, normalizer, device, options, leaf
                    )
                )
                for offset, (item, forecast) in enumerate(
                    zip(batch_series, leaf_forecasts, strict=True)
                ):
                    fan_records.setdefault(leaf.directory, []).append(build_prediction_record(
                        item, forecast, index=start + offset, model_name=config.model,
                        horizon_mode=config.horizon_mode, split=split,
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
                options=options.forecast,
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
                split=split,
            ))
            flight_metrics.append(observed_series_metrics(
                s,
                forecast,
                points=config.validation_common_grid_points,
            ))

    return PredictionSets(
        records=records,
        flight_metrics=flight_metrics,
        mode_records=mode_records,
        mode_metrics=mode_metrics,
        random_records=random_records,
        random_metrics=random_metrics,
        fan_records=fan_records,
        fan_metrics=fan_metrics,
        shuffled_records=shuffled_records,
        shuffled_metrics=shuffled_metrics,
        median_directory=median_directory,
    )


def write_prediction_sets(sets: PredictionSets, options: PredictOptions, config, *, output_dir: Path, checkpoint, split: str):
    """One emitter for the main directory and every arm beside it."""
    records, flight_metrics = sets.records, sets.flight_metrics
    mode_records, mode_metrics = sets.mode_records, sets.mode_metrics
    random_records, random_metrics = sets.random_records, sets.random_metrics
    fan_records, fan_metrics = sets.fan_records, sets.fan_metrics
    shuffled_records, shuffled_metrics = sets.shuffled_records, sets.shuffled_metrics
    median_directory = sets.median_directory
    def emit(rows, directory, metrics):
        # ONE emitter for the main directory and every arm beside it (modes, random, the
        # quantile fan, shuffled): each states the same `skipped`, so no directory can
        # publish a subset as the split (review C-6 — the four arm calls omitted it).
        return write_batch(
            rows,
            output_dir=directory,
            config_dict=config.to_dict(),
            flight_metrics=metrics,
            checkpoint=str(checkpoint),
            split=split,
            skipped=options.skipped,
        )

    paths = emit(records, output_dir, flight_metrics)
    for index, (mode_rows, mode_flight_metrics) in enumerate(zip(mode_records, mode_metrics, strict=True)):
        emit(mode_rows, output_dir / "modes" / f"mode{index:02d}", mode_flight_metrics)
    if options.latent_samples:
        print(f"  wrote {options.latent_samples} prior-sample mode(s) under {output_dir / 'modes'}")
    for index, (random_rows, random_flight_metrics) in enumerate(zip(random_records, random_metrics, strict=True)):
        emit(random_rows, output_dir / "random" / f"mode{index:02d}", random_flight_metrics)
    if options.latent_random:
        print(f"  wrote {options.latent_random} N(0, I) control mode(s) under {output_dir / 'random'}")
    for directory, rows in fan_records.items():
        emit(rows, output_dir / QUANTILE_DIR_NAME / directory, fan_metrics[directory])
    if fan_records:
        names = ", ".join(sorted(fan_records))
        print(f"  wrote the quantile fan under {output_dir / QUANTILE_DIR_NAME}: {names} "
              f"(the top-1 records above are the {median_directory} decode); read it with "
              f"run_ts.py quantile_fan_readout --arm {output_dir}")
    if shuffled_records:
        emit(shuffled_records, output_dir / "shuffled", shuffled_metrics)
        print(f"  wrote the shuffled-latent diagnostic under {output_dir / 'shuffled'}")

    return paths


def report_predictions(sets: PredictionSets, series, options: PredictOptions, *, output_dir: Path) -> None:
    """The run's readout: the cuts and caps, the accuracy block just persisted, and the
    flyability delta against the observed tracks."""
    records, flight_metrics = sets.records, sets.flight_metrics
    if options.forecast.project_final is not None:
        print(f"  projected every state forecast onto the final ({options.forecast.project_final} gate)")
    if options.forecast.truncate_at_threshold:
        cut = sum(record.source.get("truncatedAtThreshold", False) for record in records)
        print(
            f"  {cut} of {len(records)} main record(s) end at the threshold crossing; the "
            f"remaining {len(records) - cut} never cross it on the final and are whole. "
            "The run's own latent / fan / posterior arms were cut on the same rule and are "
            "not in this count"
        )
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
    (output_dir / "flyability_report.json").write_text(
        json.dumps(flyability, indent=2), encoding="utf-8")
    predicted, observed = flyability["predicted"], flyability["observed_baseline"]
    print(f"  flyability: {predicted['fully_flyable_rate'] * 100:.1f}% of predictions fully "
          f"flyable vs {observed['fully_flyable_rate'] * 100:.1f}% of the observed tracks "
          f"({flyability['delta']['fully_flyable_rate'] * 100:+.1f} pp)")



def run_cli(
    args: argparse.Namespace, parser: argparse.ArgumentParser, argv: list[str] | None
) -> int:
    del argv
    model, config, normalizer, payload, current_provenance, device = load_predict_checkpoint(args, parser)
    series, test_claim = load_predict_series(args, config, parser, payload, current_provenance)
    options, config, series = parse_predict_options(args, config, parser, series)
    try:
        sets = predict_sets(model, series, config, normalizer, device, options, split=args.split)
    except ValueError as exc:
        parser.error(str(exc))
    paths = write_prediction_sets(
        sets, options, config, output_dir=args.output_dir, checkpoint=args.checkpoint,
        split=args.split,
    )
    report_predictions(sets, series, options, output_dir=args.output_dir)
    if test_claim is not None:
        complete_test_evaluation(args.checkpoint, test_claim)

    print(f"✓ wrote {len(paths)} evaluation record(s) to {args.output_dir}")
    print(f"  grade them with:  python -m evaluation --input {args.output_dir}")
    return 0
