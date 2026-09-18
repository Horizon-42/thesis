"""The strategy interface: what the spine asks of a prediction path.

Collected from the branch sites it replaces, one method per site:

======================  ============================================================
spine call site         method
======================  ============================================================
`models.build_model`    :meth:`OutputStrategy.build_model`
`dataset` (anchors)     :attr:`OutputStrategy.anchor_policy`, :meth:`eligible_anchors`
`dataset` (batch)       :meth:`bind_windows` → :class:`WindowContext` (context rows,
                        dense supervision)
`objective`             :meth:`target_contract`, :meth:`loss_component_names`,
                        :meth:`loss`
`batching` (probe)      :meth:`probe_context`, :meth:`probe_dense_supervision`,
                        :meth:`probe_prediction`, :attr:`keeps_batch_margin`
`train.fit_model`       :meth:`check_trainable`, :meth:`training_input`,
                        :meth:`epoch_config`, :meth:`training_diagnostics`,
                        :meth:`epoch_record`, :meth:`validation_extras`,
                        :meth:`checkpoint_metadata`
`train.load_checkpoint` :meth:`verify_checkpoint_payload`
`forecast`              :meth:`forecast` with :class:`ForecastOptions`
`validation` (replay)   :meth:`replay` → :class:`Replay`
`export` (record)       :meth:`record_fields`
======================  ============================================================

A method a path has no use for keeps the base's default (nothing), so a new path
implements the abstract five and overrides what it needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.data.batch_contract import LossComponents
from ts_transformer.config import OutputSpec, TSConfig

if TYPE_CHECKING:  # value types of the spine; importing them at runtime would be a cycle
    from ts_transformer.data.dataset import FlightSeries, Normalizer, TrajectoryWindows
    from ts_transformer.data.fixed_dt_supervision import FixedDTControlSupervision
    from ts_transformer.inference.forecast import Forecast
    from ts_transformer.training.objective import ProcedureMultipliers


@dataclass(frozen=True)
class ForecastOptions:
    """What a predict-time caller may ask of a forecast beyond the checkpoint's contract.

    One value threaded from the CLI to the strategy, instead of seven keyword arguments
    through `forecast_approaches` (review §4.4). Every field is the CLI flag it came from;
    a strategy refuses the ones that do not apply to its path rather than ignoring them.
    """

    #: The fixed-time state postprocessor: cut at the closest threshold approach
    #: (``--no-truncate`` turns it off). The normalized horizon never truncates.
    truncate: bool = True
    #: ``--project-final``: clamp a state forecast into the corridor by this gate.
    project_final: str | None = None
    #: ``--truncate-at-threshold``: cut ANY forecast where it first crosses the threshold
    #: on the final (`forecast.cut_at_threshold_crossing`).
    truncate_at_threshold: bool = False
    #: ``--cta-offset-s``: the counterfactual offset added to the truth duration a
    #: ``cta_conditioning=given`` decoder is handed.
    cta_offset_s: float = 0.0
    #: B3 (``--cta-from-quantiles``): the model's own duration quantile as the CTA, per
    #: flight, with the quantile level and the calibrated interval it was read from.
    cta_s: np.ndarray | None = None
    cta_quantile: float | None = None
    cta_interval: dict[str, object] | None = None
    #: B2: the conformal table a quantile duration head's interval is widened by.
    conformal: dict | None = None


@dataclass(frozen=True)
class Replay:
    """A batch's deployable prediction in physical units, on the path's own clock, with
    the truth aligned to that clock (`validation._prediction_batch_replay` decodes the
    truth and anchors, and stores everything float32)."""

    #: ``[B, M, C]`` physical channels, velocities included.
    predicted_physical: np.ndarray
    #: ``[B, M]`` seconds between consecutive rows.
    segment_durations_s: np.ndarray
    #: ``[B]`` the predicted total duration.
    predicted_time_s: np.ndarray
    #: The NORMALIZED truth and its weights the metrics compare against — the batch's own
    #: when the prediction sits on the truth's grid, or interpolated onto the prediction's
    #: clock when it does not (the control rollout).
    metric_targets: torch.Tensor
    metric_weights: torch.Tensor
    #: ``[B, M]`` which rows are the path's OWN prediction, or None for every row: a path
    #: that pads a fixed-width replay (the segment plan holds the threshold after the
    #: arrival) says so, and the raw kinematics are scored on the prediction alone.
    row_valid: np.ndarray | None = None


class WindowContext:
    """What a path adds to every batch of one window set: the per-sample context rows
    (`TrajectoryWindows.batch`'s context slot) and, under the fixed-dt grid, the dense
    supervision beside them. The base carries nothing — the plain state path."""

    #: One line for the training log (the closure's label coverage), or None.
    summary: str | None = None

    def row(self, i: int) -> dict[str, np.ndarray] | None:
        return None

    def override(self, i: int, epoch_seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
        """A TRAINING draw's substitute for window ``i`` this epoch, or None: the NORMALIZED
        ``[L, C]`` window that replaces the observed one (the set's conditioning is added
        by `batch`) and the context row that goes with it. Only the training iterator asks
        (`iter_batches(shuffle=True)` passes the epoch's seed); the plan path answers with
        one of the flight's rolled windows at its share (design v5.2), every other path
        with nothing."""
        return None

    def dense(
        self, indices: Sequence[int] | np.ndarray
    ) -> FixedDTControlSupervision | None:
        return None


class OutputStrategy:
    """One prediction path's answers to the spine. Stateless: one instance per path."""

    #: ``prediction_output``.
    name: str
    #: The typed view of `TSConfig` this path reads (``config.output``, review §4.3).
    view: type[OutputSpec]
    #: The random-train-anchor eligibility policy, named into the run record.
    anchor_policy: str = "temporal-only-v1"
    #: Whether the auto batch-size probe keeps one tested power of two in reserve.
    keeps_batch_margin: bool = False

    # ── data ─────────────────────────────────────────────────────────────────

    def eligible_anchors(self, series: FlightSeries, anchors: Sequence[int]) -> list[int]:
        """The temporal candidates this path's model domain admits (all, by default)."""
        return list(anchors)

    def bind_windows(
        self, windows: TrajectoryWindows, *, training_input: Any | None = None
    ) -> WindowContext:
        return WindowContext()

    # ── model and objective ──────────────────────────────────────────────────

    def build_model(self, config: TSConfig, normalizer: Normalizer | None) -> nn.Module:
        raise NotImplementedError

    def target_contract(self, config: TSConfig) -> str:
        raise NotImplementedError

    def loss_component_names(self, config: TSConfig) -> tuple[str, ...]:
        raise NotImplementedError

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
        raise NotImplementedError

    # ── the batch-size probe ─────────────────────────────────────────────────

    def probe_context(
        self, batch_size: int, device: torch.device, config: TSConfig
    ) -> dict[str, torch.Tensor] | None:
        return None

    def probe_dense_supervision(
        self, batch_size: int, device: torch.device, config: TSConfig
    ) -> FixedDTControlSupervision | None:
        return None

    def probe_prediction(self, prediction: Any) -> Any:
        """The prediction the probe scores — a path whose graph depth depends on the
        prediction itself (the control rollout) substitutes a worst case here."""
        return prediction

    # ── the training loop ────────────────────────────────────────────────────

    def check_trainable(self, config: TSConfig) -> None:
        """Refuse a config that loads and predicts but cannot be TRAINED."""

    def training_input(self, config: TSConfig) -> Any | None:
        """The path's training-time input, opened by `fit_model` once and handed to every
        window set it builds (the control path's fitted teacher table, the plan path's
        rolled-window table): an object with ``provenance`` (what the checkpoint records)
        and ``metadata_key`` (under which). Every replay path builds its window sets
        without one and must keep working when the file is gone."""
        return None

    def epoch_config(self, config: TSConfig, epoch: int) -> TSConfig:
        """The objective THIS epoch (1-based) optimizes — the config itself unless a
        schedule (the latent's KL warm-up) moves a weight epoch by epoch."""
        return config

    def training_diagnostics(self, config: TSConfig) -> Any | None:
        """A per-epoch accumulator with ``record_prediction(prediction, context)``,
        ``record_gradients_and_clip(model)`` and ``summary()``, or None."""
        return None

    def epoch_record(
        self,
        config: TSConfig,
        epoch: int,
        diagnostic_totals: dict[str, float],
        diagnostics: Any | None,
    ) -> dict[str, Any]:
        """This path's blocks of the epoch record (`train.EpochResult` fields)."""
        return {}

    def validation_extras(
        self,
        model: nn.Module,
        val_sets: dict[str, TrajectoryWindows],
        device: torch.device,
        config: TSConfig,
    ) -> dict[str, Any]:
        """This path's own READOUTS of the validation pass, one block per key of the epoch
        record (`train.EpochResult` fields) — measured, never selected on: the plan path's
        loss over the val split's rolled windows (design v5.2). The base has none."""
        return {}

    def checkpoint_metadata(self, config: TSConfig) -> dict[str, Any]:
        return {}

    def verify_checkpoint_payload(self, config: TSConfig, payload: dict[str, Any]) -> None:
        """Refuse a stored payload this path cannot honour today (the control path: an
        executor whose frozen codebook no longer holds the weights it trained against).
        Nothing to check, by default."""
        return None

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
        raise NotImplementedError

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
        raise NotImplementedError

    def record_fields(self, forecast: Forecast) -> dict[str, Any]:
        """This path's own entries of a prediction record's ``source`` block."""
        return {}
