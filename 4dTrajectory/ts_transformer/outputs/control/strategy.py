"""The control path: bounded controls rolled through the point-mass twin.

The path's own code lives beside this module — `heads`, `latent`, `dynamics/`,
`constraints/`, `loss/`, `supervision`, `forecast`, `basis_fit`, `training/` — and this
class is the one door the spine reaches it through.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.data.batch_contract import LossComponents, anchor_state
from ts_transformer.data.channels import IDX
from ts_transformer.config import (
    CONTROL_THRUST_FRACTION,
    CONTROL_DURATION_UNIFORM,
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_HOOK_OFF,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    CTA_CONDITIONING_GIVEN,
    CTA_CONDITIONING_OFF,
    DURATION_HEAD_TWO_HEAD,
    PLAN_CONDITIONING_OFF,
    PLAN_CONDITIONING_TRUTH_NEXT,
    HOOK_SATURATION_HARD,
    PREDICTION_CONTROL,
    ControlOutput,
    TSConfig,
    control_recipe,
)
from ts_transformer.data.dataset import truth_duration_s
from ts_transformer.outputs.control.plan_token import PLAN_TOKEN_KEY, truth_plan_token
from ts_transformer.outputs.plan.skeleton import SkeletonCache
from ts_transformer.data.fixed_dt_supervision import (
    FixedDTControlSupervision,
    FixedDTSupervisionRow,
    build_fixed_dt_supervision,
    cache_fixed_dt_supervision_rows,
    pack_fixed_dt_supervision_rows,
)
from ts_transformer.geometry.flyability import G as GRAVITY_MPS2
from ts_transformer.backbone.adapters import build_state_forecaster
from ts_transformer.outputs.base import ForecastOptions, OutputStrategy, Replay, WindowContext
from ts_transformer.outputs.control.basis_fit import FittedTeacherTable, load_fitted_teacher
from ts_transformer.outputs.constraints import build_command_hook
from ts_transformer.outputs.dynamics import rollout as control_rollout
from ts_transformer.outputs.dynamics.hooks import HOOK_DIAGNOSTIC_PREFIX, HOOK_STEPS_KEY
from ts_transformer.outputs.control.forecast import forecast_control_batch
from ts_transformer.outputs.control.heads import ControlOutputModel, ControlPrediction
from ts_transformer.outputs.control.latent import (
    LATENT_AUX_COMPONENT,
    LATENT_KL_COMPONENT,
    LatentControlModel,
    LatentControlPrediction,
    effective_latent_beta,
    latent_epoch_record,
    with_latent_aux_duration,
    with_latent_kl,
)
from ts_transformer.outputs.control.loss.objective import (
    CONTROL_LOSS_COMPONENT_NAMES,
    CONTROL_TARGET_CONTRACTS,
    DURATION_QUANTILE_COMPONENT,
    align_control_targets_to_query_clock,
    control_prediction_loss_components,
)
from ts_transformer.outputs.dynamics.context import dynamics_arrays
from ts_transformer.outputs.envelope import control_contract
from ts_transformer.outputs.control.supervision import (
    probe_dynamics,
    reference_control_supervision,
    reference_heading_rate_supervision,
)
from ts_transformer.outputs.control.training.diagnostics import (
    ControlTrainingDiagnosticsAccumulator,
    saturation_labels,
)

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries, Normalizer, TrajectoryWindows
    from ts_transformer.inference.forecast import Forecast
    from ts_transformer.training.objective import ProcedureMultipliers


# ── the anchor domain ────────────────────────────────────────────────────────

SEA_LEVEL_DENSITY_KG_M3 = 1.225
#: The airborne floor a train anchor must clear; the same 1.10 × stall margin the
#: optimisation counterpart and the envelope's speed floor use.
CONTROL_ANCHOR_STALL_MARGIN = 1.10


def airborne_control_candidates(series: FlightSeries, anchors: Sequence[int]) -> list[int]:
    """Retain anchors where the observed state is inside the airborne model domain.

    The control ODE divides by airspeed and is not defined for stopped ground reports.
    Its existing optimization counterpart uses the same 1.10 x sea-level stall-speed
    floor. Channel horizontal derivatives differ from physical ENU velocity only by the
    local chart transport factors (well below this 10% margin in terminal airspace).
    """
    candidate_indices = np.fromiter(anchors, dtype=np.int64)
    if not len(candidate_indices):
        return []
    velocity_indices = [IDX["edot"], IDX["ndot"], IDX["udot"]]
    observed_speed = np.linalg.norm(
        series.values[candidate_indices][:, velocity_indices], axis=1
    )
    scenario = series.scenario
    stall_speed = math.sqrt(
        2.0 * float(scenario.initial.m) * GRAVITY_MPS2
        / (
            SEA_LEVEL_DENSITY_KG_M3
            * float(scenario.aero.S)
            * float(scenario.aero.Cl_max)
        )
    )
    return candidate_indices[
        observed_speed >= CONTROL_ANCHOR_STALL_MARGIN * stall_speed
    ].tolist()


# ── the batch context ────────────────────────────────────────────────────────

class ControlContext(WindowContext):
    """The per-sample physical tensors a control model and its rollout need (`dynamics_arrays`),
    the CTA under ``cta_conditioning``, and — on a supervised window set — the imitation
    schedule and the heading-rate reference; under the fixed-dt grid, the dense supervision.

    A fixed-anchor window set (``cache_context_rows``) is reused on every validation epoch
    and on final replay, so its rows are built once here; a random-anchor set builds each
    row for the anchor it drew.
    """

    def __init__(self, windows: TrajectoryWindows, fitted_teacher: FittedTeacherTable | None):
        self.windows = windows
        self.config = windows.config
        # The imitation term's per-flight teacher table (control_imitation_target=
        # "fitted"). It is a TRAINING-TIME INPUT, handed in by `train.fit_model` for the
        # train and validation window sets it supervises — never opened here. Every other
        # consumer replays a checkpoint and needs no teacher at all.
        #
        # What IS checked here is coverage, because this is where the flights are known: a
        # fitted schedule reproduces its truth only at the width, the anchor and the total
        # duration it was fitted under, and a flight the table does not carry has no
        # teacher at all. A table that does not cover this cohort refuses the build — there
        # is no partial mode, which would train part of every batch on nothing while the
        # loss still reported an imitation number.
        self.fitted_teacher = fitted_teacher
        if fitted_teacher is not None:
            covered = [
                (windows.series[int(index)], windows.index[int(windows.range_starts[int(index)])][1])
                for index in windows.eligible_series
            ]
            fitted_teacher.require_cover(
                [(item.flight_id, truth_duration_s(item, anchor)) for item, anchor in covered],
                airports={item.airport for item, _anchor in covered},
                anchor_indices={int(anchor) for _item, anchor in covered},
                n_segments=int(self.config.n_segments),
            )
        # the plan token's skeletons (two-tier T1), one per runway, read on first use
        self.skeletons = SkeletonCache()
        self._rows: list[dict[str, np.ndarray]] | None = None
        self._fixed_dt: tuple[FixedDTSupervisionRow, ...] | None = None
        if windows.cache_context_rows:
            self._rows = [self._build_row(index) for index in range(len(windows.index))]
            if self.config.control_state_loss_grid == CONTROL_STATE_LOSS_GRID_FIXED_DT:
                self._fixed_dt = cache_fixed_dt_supervision_rows(
                    windows.series, windows.encoded, windows.index, dt_s=self.config.dt_s
                )

    def row(self, i: int) -> dict[str, np.ndarray]:
        return self._rows[i] if self._rows is not None else self._build_row(i)

    def _build_row(self, i: int) -> dict[str, np.ndarray]:
        windows, config = self.windows, self.config
        s_idx, anchor = windows.index[i]
        series = windows.series[s_idx]
        arrays = dynamics_arrays(
            series, anchor, parameterization=config.control_thrust_parameterization,
            condition_features=config.control_condition_features,
        )
        if config.cta_conditioning == CTA_CONDITIONING_GIVEN:
            # Training feeds the truth as the controlled time of arrival.
            arrays["cta_s"] = np.array(truth_duration_s(series, anchor), dtype=np.float64)
        elif config.cta_conditioning != CTA_CONDITIONING_OFF:
            # The head builds its CTA token for every mode but `off` (heads.py), so a mode
            # this branch does not fill would reach the decoder with no `cta_s`. `self-q`
            # is a predict-time label (`CTA_CONDITIONINGS_AVAILABLE` keeps it out of a new
            # run); if it ever reaches a window set, say so here, not in the head.
            raise ValueError(
                f"cta_conditioning={config.cta_conditioning!r} names no training-time "
                "source for the CTA token; only 'given' (the truth duration) is defined"
            )
        if config.plan_conditioning == PLAN_CONDITIONING_TRUTH_NEXT:
            # the truth's plan at this anchor: the plan head's label there (reads the future)
            arrays[PLAN_TOKEN_KEY] = truth_plan_token(series, anchor, self.skeletons)
        elif config.plan_conditioning != PLAN_CONDITIONING_OFF:
            raise ValueError(
                f"plan_conditioning={config.plan_conditioning!r} names no training-time source "
                "for the plan token; only 'truth-next' is defined"
            )
        if not windows.control_supervision:
            return arrays
        anchor_time = float(series.times[anchor])
        total_duration_s = float(series.supervision_times[-1] - anchor_time)
        last_measured_time_s = float(
            windows.last_supervised_times[s_idx][windows.kinematic_channels].min() - anchor_time
        )
        if config.control_imitation_loss_weight:
            # The fitted teacher replaces the inversion outright — its schedule was fitted
            # over the WHOLE supervised horizon through the rollout, so every segment
            # carries weight one, where the inversion has to mask the fitted tail it has
            # no measured velocity to differentiate.
            arrays.update(
                self.fitted_teacher.supervision(series.flight_id)
                if self.fitted_teacher is not None
                else reference_control_supervision(
                    series, anchor, config,
                    total_duration_s=total_duration_s,
                    last_measured_time_s=last_measured_time_s,
                )
            )
        if config.control_heading_rate_loss_weight:
            # Same supervised horizon and same last-measured instant as the imitation
            # target above; the heading-rate target only masks at the endpoints instead of
            # the midpoints, and costs no inverse-dynamics solve.
            arrays.update(
                reference_heading_rate_supervision(
                    series, anchor, config,
                    total_duration_s=total_duration_s,
                    last_measured_time_s=last_measured_time_s,
                )
            )
        return arrays

    def dense(
        self, indices: Sequence[int] | np.ndarray
    ) -> FixedDTControlSupervision | None:
        if self.config.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_FIXED_DT:
            return None
        if self._fixed_dt is not None:
            return pack_fixed_dt_supervision_rows(
                [self._fixed_dt[int(index)] for index in indices],
                channels=len(self.config.channels),
            )
        windows = self.windows
        return build_fixed_dt_supervision(
            windows.series, windows.encoded,
            [windows.index[int(index)] for index in indices],
            dt_s=self.config.dt_s,
        )


# ── the batch-size probe ─────────────────────────────────────────────────────

def heterogeneous_control_probe_prediction(prediction: ControlPrediction) -> ControlPrediction:
    """Replace identical probe partitions with deterministic heterogeneous schedules.

    Batched rollout graph length is governed by the maximum duration in each segment, not
    by one row's total duration. Zero histories otherwise make every probe row identical and
    systematically understate that graph. A phase-shifted circular profile exercises about
    three times the uniform graph depth while remaining smooth, positive, and connected to
    the model's duration output for backward-memory fidelity.
    """
    batch, segments = prediction.segment_durations.shape
    dtype, device = prediction.segment_durations.dtype, prediction.segment_durations.device
    learned_fractions = prediction.segment_durations / prediction.final_time_s.unsqueeze(1)
    segment_phase = (
        torch.arange(segments, dtype=dtype, device=device) * (2.0 * torch.pi / segments)
    ).unsqueeze(0)
    row_phase = (
        torch.arange(batch, dtype=dtype, device=device) * (2.0 * torch.pi / batch)
    ).unsqueeze(1)
    probe_profile = torch.softmax(2.0 * torch.cos(segment_phase - row_phase), dim=-1)
    fractions = learned_fractions * probe_profile
    fractions = fractions / fractions.sum(dim=-1, keepdim=True)
    durations = fractions * prediction.final_time_s.unsqueeze(1)
    # `replace`, not a rebuilt ControlPrediction: the quantile head's `duration_quantiles_s`
    # (and a latent prediction's extra fields) must survive the probe, or the objective
    # reads None where the real epoch reads a tensor (review B-2).
    return replace(prediction, segment_durations=durations)


# ── the strategy ─────────────────────────────────────────────────────────────

class ControlStrategy(OutputStrategy):
    name = PREDICTION_CONTROL
    view = ControlOutput
    anchor_policy = "airborne-1.10-stall-margin-v1"
    # Control rollout depth can grow further as learned duration logits sharpen: the probe
    # keeps one tested power of two as a memory margin after the heterogeneous probe.
    keeps_batch_margin = True

    def eligible_anchors(self, series: FlightSeries, anchors: Sequence[int]) -> list[int]:
        return airborne_control_candidates(series, anchors)

    def bind_windows(
        self, windows: TrajectoryWindows, *, training_input: FittedTeacherTable | None = None
    ) -> WindowContext:
        return ControlContext(windows, training_input)

    def build_model(self, config: TSConfig, normalizer: Normalizer | None) -> nn.Module:
        del normalizer  # controls are rolled out in physical units already
        if config.latent_dim > 0:
            return LatentControlModel(config, build_state_forecaster(config))
        # One model class: the duration parameterization is decided inside the head
        # (`heads.control_head_for`), not by a second model type.
        return ControlOutputModel(config, build_state_forecaster(config))

    def target_contract(self, config: TSConfig) -> str:
        # A contract whose constants are module constants rather than config fields spells them
        # (`ControlContract.identity_suffix`), so a checkpoint trained under other values is
        # refused at load instead of flying under these.
        return (
            self._objective_target_contract(config)
            + control_contract(config.control_thrust_parameterization).identity_suffix
        )

    def _objective_target_contract(self, config: TSConfig) -> str:
        base = CONTROL_TARGET_CONTRACTS[
            (
                config.control_duration_parameterization,
                config.control_state_supervision_clock,
                config.control_state_loss_grid,
            )
        ]
        if config.control_duration_parameterization != CONTROL_DURATION_UNIFORM:
            base += f"+duration-uniform-floor={config.control_duration_uniform_floor:g}-v1"
        if config.control_dynamics_backend != CONTROL_DYNAMICS_REANCHORED_RK4:
            base += f"+dynamics={config.control_dynamics_backend}-v1"
        if config.control_duration_parameterization == CONTROL_DURATION_UNIFORM:
            return (
                base
                if config.control_state_objective == CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE
                else f"{base}+{config.control_state_objective}"
            )
        if (
            config.control_state_objective == CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE
            and config.control_state_duration_gradient
        ):
            return base
        gradient_contract = (
            "joint-duration-gradient"
            if config.control_state_duration_gradient
            else "detached-duration-gradient"
        )
        return f"{base}+{config.control_state_objective}+{gradient_contract}"

    def loss_component_names(self, config: TSConfig) -> tuple[str, ...]:
        extensions = {
            CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION: (
                *(("velocity",) if config.control_velocity_loss_weight else ()),
                *(("imitation",) if config.control_imitation_loss_weight else ()),
                *(("heading_rate",) if config.control_heading_rate_loss_weight else ()),
                *(("bank_tv",) if config.control_bank_tv_loss_weight else ()),
            ),
        }
        return (
            *CONTROL_LOSS_COMPONENT_NAMES,
            *extensions.get(config.control_state_objective, ()),
            *(("procedure",) if config.procedure_loss_active else ()),
            # The latent's KL is charged on every latent run; the auxiliary intent target
            # only when weighted. Both are registered under exactly the objective's own
            # condition, and the two must be read together.
            *((LATENT_KL_COMPONENT,) if config.latent_dim > 0 else ()),
            *((LATENT_AUX_COMPONENT,) if config.latent_aux_duration_weight else ()),
            # B1.b: the pinball sum beside the point term, never instead of it.
            *((DURATION_QUANTILE_COMPONENT,)
              if config.duration_head == DURATION_HEAD_TWO_HEAD else ()),
        )

    def loss(
        self,
        prediction: Any,
        normalized_anchor_state: torch.Tensor,
        target_states: torch.Tensor,
        state_weights: torch.Tensor,
        target_final_time_s: torch.Tensor,
        flight_weights: torch.Tensor,
        config: TSConfig,
        normalizer: Normalizer,
        context: dict[str, torch.Tensor] | None,
        dense_supervision: FixedDTControlSupervision | None,
        *,
        multipliers: ProcedureMultipliers | None = None,
    ) -> LossComponents:
        if context is None:
            raise ValueError("control prediction loss requires per-flight dynamics")
        components = control_prediction_loss_components(
            prediction, normalized_anchor_state, target_states, state_weights,
            target_final_time_s, flight_weights, config, normalizer, context,
            dense_supervision, multipliers=multipliers,
        )
        if isinstance(prediction, LatentControlPrediction):
            # The control objective on the decoded schedule, plus the latent's own terms —
            # registered under the same conditions as `loss_component_names`.
            components = with_latent_kl(components, prediction, config, flight_weights)
            if config.latent_aux_duration_weight:
                components = with_latent_aux_duration(
                    components, prediction, config, target_final_time_s, flight_weights
                )
        return components

    # ── the batch-size probe ─────────────────────────────────────────────────

    def probe_context(
        self, batch_size: int, device: torch.device, config: TSConfig
    ) -> dict[str, torch.Tensor] | None:
        return probe_dynamics(batch_size, device, config)

    def probe_dense_supervision(
        self, batch_size: int, device: torch.device, config: TSConfig
    ) -> FixedDTControlSupervision | None:
        if config.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_FIXED_DT:
            return None
        points = int(config.final_time_scale_s // config.dt_s)
        offsets = (
            torch.arange(1, points + 1, dtype=torch.float64, device=device) * config.dt_s
        ).unsqueeze(0).expand(batch_size, -1)
        dense_states = torch.zeros(
            (batch_size, points, len(config.channels)), dtype=torch.float32, device=device
        )
        return FixedDTControlSupervision(
            query_offsets_s=offsets,
            states=dense_states,
            weights=torch.ones_like(dense_states),
            valid=torch.ones((batch_size, points), dtype=torch.bool, device=device),
        )

    def probe_prediction(self, prediction: Any) -> Any:
        return heterogeneous_control_probe_prediction(prediction)

    # ── the training loop ────────────────────────────────────────────────────

    def check_trainable(self, config: TSConfig) -> None:
        if (
            config.control_command_hook != CONTROL_HOOK_OFF
            and config.control_hook_saturation == HOOK_SATURATION_HARD
        ):
            raise ValueError(
                "hard hook saturation is for inference-only arms: a clamped command has no "
                "gradient, so training would learn nothing on the clamped steps"
            )

    def training_input(self, config: TSConfig) -> FittedTeacherTable | None:
        # The imitation term's teacher table is a TRAINING input and is opened exactly
        # here — the one place that builds supervised window sets. Every replay path
        # builds its own window set without one and must keep working when the table is gone.
        if config.uses_fitted_teacher:
            return load_fitted_teacher(config.control_fitted_teacher_path)
        return None

    def epoch_config(self, config: TSConfig, epoch: int) -> TSConfig:
        # The objective THIS epoch optimizes: the run's config carrying the annealed KL
        # weight (identical to `config` itself once the warm-up is over, and always when
        # there is none). The training batches and the validation pass are both scored
        # under it, so an epoch's train and val `latent_kl` mean the same thing — the rule
        # the procedure penalty's λ already follows.
        return replace(config, latent_beta=effective_latent_beta(config, epoch))

    def training_diagnostics(self, config: TSConfig) -> ControlTrainingDiagnosticsAccumulator | None:
        if config.control_gradient_clip_norm > 0.0:
            return ControlTrainingDiagnosticsAccumulator(
                config.control_gradient_clip_norm,
                saturation_labels(config.control_thrust_parameterization),
            )
        return None

    def epoch_record(
        self,
        config: TSConfig,
        epoch: int,
        diagnostic_totals: dict[str, float],
        diagnostics: ControlTrainingDiagnosticsAccumulator | None,
    ) -> dict[str, Any]:
        # The command hook's epoch record: how often it was gated on and how hard it acted
        # (per-step shares over the epoch's rollouts; the step count beside them).
        hook_steps = diagnostic_totals.get(HOOK_STEPS_KEY, 0.0)
        hook_epoch: dict[str, float] = {}
        if hook_steps > 0.0:
            hook_epoch = {
                name.removeprefix(HOOK_DIAGNOSTIC_PREFIX): value / hook_steps
                for name, value in diagnostic_totals.items()
                if name.startswith(HOOK_DIAGNOSTIC_PREFIX) and name != HOOK_STEPS_KEY
            }
            hook_epoch["steps"] = hook_steps
        latent_epoch: dict[str, Any] = {}
        if config.latent_dim > 0:
            latent_epoch = latent_epoch_record(
                diagnostic_totals, config, beta_effective=effective_latent_beta(config, epoch)
            )
        return {
            "command_hook": hook_epoch,
            "latent": latent_epoch,
            "control_training_diagnostics": (
                diagnostics.summary() if diagnostics is not None else {}
            ),
        }

    def checkpoint_metadata(self, config: TSConfig) -> dict[str, Any]:
        return {"control_recipe": control_recipe(config)}

    # ── inference ────────────────────────────────────────────────────────────

    def forecast(
        self,
        model: nn.Module,
        series: Sequence[FlightSeries],
        config: TSConfig,
        normalizer: Normalizer,
        anchor: int,
        device: torch.device,
        options: ForecastOptions,
    ) -> list[Forecast]:
        if options.project_final is not None:
            raise ValueError("the final-approach projection applies to state forecasts only")
        return forecast_control_batch(
            model, series, config, normalizer, anchor, device,
            cta_offset_s=options.cta_offset_s, conformal=options.conformal,
            cta_s=options.cta_s, cta_quantile=options.cta_quantile,
            cta_interval=options.cta_interval,
        )

    def replay(
        self,
        output: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        final_time_s: torch.Tensor,
        context: dict[str, torch.Tensor] | None,
        dataset: TrajectoryWindows,
    ) -> Replay:
        if context is None:
            raise ValueError("control replay requires per-flight dynamics")
        config = dataset.config
        points = config.validation_common_grid_points
        progress = torch.arange(
            1, points + 1, dtype=torch.float64, device=output.segment_durations.device
        ) / points
        predicted_total_s = output.segment_durations.to(torch.float64).sum(dim=1)
        query_offsets_s = predicted_total_s.unsqueeze(1) * progress.unsqueeze(0)
        query_valid = torch.ones_like(query_offsets_s, dtype=torch.bool)
        rollout = control_rollout.rollout_control_dense(
            output.controls,
            output.segment_durations,
            context,
            query_offsets_s,
            query_valid,
            config,
            command_hook=build_command_hook(config, context),
        )
        predicted = rollout.query_channels.detach().cpu().numpy().astype(np.float32)
        metric_targets, metric_weights = align_control_targets_to_query_clock(
            anchor_state(x, len(config.channels)), y, mask, query_offsets_s, final_time_s,
        )
        segment_durations_s = np.broadcast_to(
            (predicted_total_s.detach().cpu().numpy().astype(np.float64) / points)[:, None],
            (len(x), points),
        ).copy()
        return Replay(
            predicted_physical=predicted,
            segment_durations_s=segment_durations_s,
            predicted_time_s=output.final_time_s.detach().cpu().numpy(),
            metric_targets=metric_targets,
            metric_weights=metric_weights,
        )

    def record_fields(self, forecast: Forecast) -> dict[str, Any]:
        return {
            # The contract the schedule was predicted in, off the default; absent means
            # thrust-fraction, so every such record reproduces to the bit.
            **({"controlThrustParameterization": forecast.control_parameterization}
               if forecast.control_parameterization != CONTROL_THRUST_FRACTION else {}),
            # Latent control output: which prior sample this is (None = the top-1 the
            # contract carries) and its probability; whether it was decoded from another
            # flight's latent (the collapse diagnostic). z itself is never written.
            **({"modeIndex": forecast.mode_index, "modeProbability": forecast.mode_probability}
               if forecast.mode_index is not None else {}),
            **({"latentShuffled": True} if forecast.latent_shuffled else {}),
            **({"zFromPosterior": True} if forecast.z_from_posterior else {}),
            # CTA-conditioned control output: the arrival time the decoder was GIVEN (truth
            # + offset) — a record that reads the future says so. Under B3 the arrival time
            # is the model's OWN duration quantile, `ctaOffsetS` is null (there is no truth
            # to offset) and `ctaFromQuantiles` is what separates the two arms in any later
            # reading.
            **({"ctaS": forecast.cta_s, "ctaOffsetS": forecast.cta_offset_s,
                **({"ctaFromQuantiles": True, "ctaQuantile": forecast.cta_quantile,
                    **({"ctaInterval": forecast.cta_interval}
                       if forecast.cta_interval is not None else {})}
                   if forecast.cta_from_quantiles else {})}
               if forecast.cta_s is not None else {}),
            # B1, quantile duration head: all five DURATION_QUANTILES in seconds, in level
            # order — the median included, so the record is a complete interval and a
            # reader never has to splice `durationHeadFinalTimeS` back into position 2.
            # Under B1.b's `two-head` those are two DIFFERENT numbers on purpose:
            # `durationHeadFinalTimeS` is the POINT head's duration (the one the states
            # were rolled over) and this is the quantile head's distribution (the one B2
            # calibrates and B3 decodes).
            **({"durationQuantilesS": [float(value) for value in forecast.duration_quantiles_s],
                # B2: `calibrated` is written for EVERY quantile record — false is the
                # claim that this checkpoint has no conformal table, and a reader must not
                # have to infer it from a missing key (design §六 4).
                "calibrated": forecast.duration_interval_s is not None,
                **({"durationIntervalS": forecast.duration_interval_s,
                    "durationIntervalStratum": forecast.duration_interval_stratum,
                    "durationIntervalCohort": forecast.duration_interval_cohort}
                   if forecast.duration_interval_s is not None else {})}
               if forecast.duration_quantiles_s is not None else {}),
        }


STRATEGY = ControlStrategy()
