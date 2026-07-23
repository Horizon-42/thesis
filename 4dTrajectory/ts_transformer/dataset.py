"""Observed arrival tracks -> uniformly-sampled channel series -> (x, y, mask) windows.

Pipeline per flight::

    flight dict  ({id, runway, waypoints: [[t, lon, lat, alt], ...]})
      -> flight_scenarios.build_scenario(..., target_from_threshold=True)   # target, aircraft, mass
      -> flight_scenarios.state_samples_from_track(...)                     # V/psi/gamma per sample
      -> channels.channels_from_states(...)                                 # ENU metres, threshold origin
      -> channels.resample_uniform(...)                                     # regular dt grid
      -> measured FlightSeries + position-only fitted-tail supervision

Every one of those steps is an existing, tested seam except the last two. That is on
purpose: the reference records the predictions get judged against are built by the same
functions, so a divergence here would read as model error rather than as a bug.

**The split(train/validation/test) is BY FLIGHT, never by window.** Consecutive windows of one approach overlap by
``seq_len - 1`` samples, so splitting windows at random puts near-duplicates of a validation
window in the training set and the val loss becomes a memorisation score. Splitting whole
flights is the only honest option, and it is done here rather than left to the caller.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# flight_key is the identity ``id_runway_icao24_landingTime`` — single-sourced in
# flight_scenarios.identity because the optimizer batch derives its record filenames from
# the SAME function; the split key here and both writers' filename stems cannot drift.
from flight_scenarios import (
    FlightScenario,
    build_scenario,
    fit_flight_final_approach,
    flight_key,
    state_samples_from_track,
)
from flight_scenarios.datum import flight_to_msl
from trajectory_data_process.harvest.arrivals import load_arrival_flights

from channels import (
    CHANNELS,
    POSITION_IDX,
    Frame,
    channels_from_states,
    frame_for_state,
    resample_uniform,
)
from config import DEFAULT_AIRCRAFT_TYPE, HORIZON_FULL, TSConfig


@dataclass
class FlightSeries:
    """One observed arrival plus its training-only fitted position supervision."""

    flight_id: str
    scenario: FlightScenario
    frame: Frame
    times: np.ndarray        # [N] seconds, uniform dt, rebased to 0 at the first sample
    values: np.ndarray       # [N, C] channel space (see channels.CHANNELS)
    # The observed arrays above remain the only model INPUT and the only arrays exposed to
    # forecast/export.  These arrays extend them with a fitted tail for training TARGETS.
    supervision_times: np.ndarray | None = None    # [M], M >= N
    supervision_values: np.ndarray | None = None   # [M, C]
    supervision_weights: np.ndarray | None = None  # [M, C], fitted velocities are zero

    def __post_init__(self) -> None:
        supplied = (
            self.supervision_times is not None,
            self.supervision_values is not None,
            self.supervision_weights is not None,
        )
        if not any(supplied):
            # Backward-compatible measured-only construction for small fixtures and generic
            # consumers that do not need fitted labels.
            self.supervision_times = self.times
            self.supervision_values = self.values
            self.supervision_weights = np.full(
                self.values.shape, 1.0 / self.values.shape[1], dtype=np.float64
            )
        elif not all(supplied):
            raise ValueError("supervision_times/values/weights must be supplied together")
        if not (
            len(self.supervision_times) == len(self.supervision_values)
            == len(self.supervision_weights)
        ):
            raise ValueError("supervision times, values, and weights must align")

    @property
    def n_samples(self) -> int:
        return len(self.times)

    @property
    def n_supervision_samples(self) -> int:
        return len(self.supervision_times)


@dataclass(frozen=True)
class Normalizer:
    """Per-channel standardisation, fit on the TRAINING split only.

    Both architectures already normalise each window internally (iTransformer's ``use_norm``,
    PatchTST's RevIN), so this is not about conditioning the attention — it is about the
    LOSS. Predictions come back in physical units, where ``e``/``n`` span ~2.5e4 m and
    ``udot`` spans ~1e1 m/s; an unweighted MSE over raw channels is ~99% a
    horizontal-position loss and the vertical channel never trains. Standardising first
    makes the loss weight channels comparably.
    """

    mean: np.ndarray   # [C]
    std: np.ndarray    # [C]

    @classmethod
    def fit(cls, series: Sequence[FlightSeries]) -> Normalizer:
        """Fit a normaliser to the training split."""
        # series is a list of FlightSeries, each with values [N, C]; stack them to [N_total, C]
        stacked = np.concatenate([s.values for s in series], axis=0) # [N, C]
        mean = stacked.mean(axis=0) # [C]
        std = stacked.std(axis=0) # [C]
        # A channel that never varies across the training set carries no signal; leaving its
        # std at 0 would produce inf on the first divide. Scale it by 1 and let it ride as a
        # constant (the model can still use it as a bias).
        std = np.where(std > 1e-9, std, 1.0) # normal Z-score standardization: z=(x-mean)/std;
        # For channel with std=0, there is no need to standardize, so we set std=1.0 to avoid division by zero.
        return cls(mean=mean, std=std)

    def encode(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def decode(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> Normalizer:
        return cls(mean=np.asarray(data["mean"], dtype=np.float64),
                   std=np.asarray(data["std"], dtype=np.float64))


# ── Loading ──────────────────────────────────────────────────────────────────

def load_flight_dicts(path: str | Path, *, verbose: bool = True) -> list[dict[str, Any]]:
    """Model-ready flights from the harvest's authoritative arrival manifest."""
    flights = load_arrival_flights(path)
    if verbose:
        print(f"  {path}: {len(flights)} manifest-rostered arrival(s)")
    return flights


@dataclass
class BuildReport:
    """What survived the build, and why the rest did not.

    Skips are counted and reported rather than silently dropped: a run that quietly trains
    on 20% of the data because most thresholds were missing from the config should be
    obvious from the console, not from a confusing loss curve.
    """

    built: int = 0
    skipped: dict[str, int] = field(default_factory=dict)  # reason -> count

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    @property
    def total(self) -> int:
        return self.built + sum(self.skipped.values())

    def format(self) -> str:
        lines = [f"built {self.built}/{self.total} series"]
        for reason, count in sorted(self.skipped.items(), key=lambda kv: -kv[1]):
            lines.append(f"  skipped {count:5d}  {reason}")
        return "\n".join(lines)


def build_series(
    flights: Sequence[dict[str, Any]],
    config: TSConfig,
    *,
    airport: str | None = None,
    aircraft_type: str = DEFAULT_AIRCRAFT_TYPE,
) -> tuple[list[FlightSeries], BuildReport]:
    """Flight dicts -> :class:`FlightSeries`, skipping what cannot be built.

    A flight is skipped when it has no published runway threshold (no ENU frame and no
    target to judge against), when the track is too short to resample onto the grid, or
    when it cannot furnish one full window (``seq_len + 1`` samples minimum).

    ``aircraft_type`` is the fallback when the flight dict does not name a resolvable type.
    Every harvested arrival currently carries ``"type": "UNK"`` (``czml_export`` hardcodes
    it), and ``flight_scenarios._resolve_aircraft`` RAISES rather than guessing — so
    without this the whole batch dies on the first flight. The choice is not cosmetic: it
    sets the target state's Vref and threshold-crossing height, which is what the
    evaluation gates measure the final state against.
    """
    minimum_samples = config.seq_len + 1
    series: list[FlightSeries] = []
    report = BuildReport()

    for index, flight in enumerate(flights):
        # Real inputs come through the harvest arrival manifest and must carry the
        # published CIFP target that made them model-ready. Synthetic fixtures are built
        # from the static threshold configuration and intentionally have no manifest
        # metadata, so they retain that explicit alternate path.
        if flight.get("altitude_source") != "synthetic":
            target_meta = flight.get("runway_target") or {}
            if target_meta.get("threshold_crossing_height_m") is None:
                report.skip("no published runway TCH")
                continue
            if target_meta.get("published_glidepath_deg") is None:
                report.skip("no published runway glidepath")
                continue
        waypoints = flight.get("waypoints") or []
        if len(waypoints) < 2:
            report.skip("track has fewer than 2 waypoints")
            continue

        # Waypoints are [t, lon, lat, alt] and state_samples_from_track keeps every
        # waypoint at time t - t0, so the span is known from the raw dict — check it
        # BEFORE the expensive build (aircraft resolution + per-sample least-squares
        # velocity fits), which too-short flights would otherwise pay for in full.
        span = float(waypoints[-1][0]) - float(waypoints[0][0])
        if span < config.dt_s * (minimum_samples - 1):
            report.skip(f"track shorter than one window ({config.lookback_s:.0f}s)")
            continue

        # Into the modeling plane: harvested altitude is ellipsoidal (HAE) while the
        # threshold-anchored channels and the evaluation gates are MSL. Converted HERE, not
        # inside build_scenario alone, because state_samples_from_track below takes the bare
        # waypoint list and so cannot convert itself. Idempotent (see flight_scenarios/datum).
        # Placed after the cheap skips (which read no altitude) so rejected flights don't
        # pay the pyproj transform; ``waypoints`` must be rebound to the converted rows.
        flight = flight_to_msl(flight)
        waypoints = flight["waypoints"]

        scenario = build_scenario(flight, aircraft_type, airport=airport,
                                  target_from_threshold=True)
        if scenario.target is None:
            runway = flight.get("runway") or "?"
            report.skip(f"no published threshold for runway {runway}")
            continue

        samples = state_samples_from_track(waypoints, mass_kg=scenario.initial.m)
        frame = frame_for_state(scenario.target)
        times, values = channels_from_states(samples, frame)
        grid, resampled = resample_uniform(times, values, config.dt_s)
        # Not redundant with the span pre-check: for a non-dyadic dt the multiply and
        # resample_uniform's floor-divide can round differently at the boundary.
        if len(grid) < minimum_samples:
            report.skip(f"track shorter than one window ({config.lookback_s:.0f}s)")
            continue

        supervision_times, supervision_values, supervision_weights = _build_supervision(
            flight, samples, frame, grid, resampled, config
        )
        series.append(FlightSeries(
            flight_id=flight_key(scenario.source, index), scenario=scenario, frame=frame,
            times=grid, values=resampled,
            supervision_times=supervision_times,
            supervision_values=supervision_values,
            supervision_weights=supervision_weights,
        ))
        report.built += 1

    return series, report


def _build_supervision(
    flight: dict[str, Any],
    measured_samples,
    frame: Frame,
    grid: np.ndarray,
    measured_values: np.ndarray,
    config: TSConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed six-channel labels plus low-weight, position-only fitted tail labels."""
    channel_count = len(CHANNELS)
    # Row weights sum to one, preserving the previous all-channel mean-MSE scale.
    measured_weights = np.full(
        measured_values.shape, 1.0 / channel_count, dtype=np.float64
    )
    if (
        config.fitted_tail_position_weight == 0.0
        and config.fitted_terminal_position_weight == 0.0
    ):
        return grid, measured_values, measured_weights

    fitted = fit_flight_final_approach(flight)
    if fitted is None:
        return grid, measured_values, measured_weights
    tail = fitted.uniform_tail(after_time_s=float(grid[-1]), dt_s=config.dt_s)
    if not tail:
        return grid, measured_values, measured_weights

    # Kinematics are placeholders required by the fixed six-channel tensor shape.  Their
    # supervision weights below are zero, so no extrapolated velocity enters the loss.
    terminal_state = measured_samples[-1][1]
    tail_states = [
        (
            row.time_s,
            type(terminal_state)(
                latitude=row.point.lat,
                longitude=row.point.lon,
                altitude=row.point.alt_m,
                V=terminal_state.V,
                psi=terminal_state.psi,
                gamma=terminal_state.gamma,
                m=terminal_state.m,
            ),
        )
        for row in tail
    ]
    tail_times, tail_values = channels_from_states(tail_states, frame)
    tail_weights = np.zeros_like(tail_values)
    for index in POSITION_IDX:
        tail_weights[:, index] = config.fitted_tail_position_weight / len(POSITION_IDX)
        tail_weights[-1, index] += (
            config.fitted_terminal_position_weight / len(POSITION_IDX)
        )

    return (
        np.concatenate([grid, tail_times]),
        np.concatenate([measured_values, tail_values], axis=0),
        np.concatenate([measured_weights, tail_weights], axis=0),
    )


# ── Windowing ────────────────────────────────────────────────────────────────

def window_anchors(series: FlightSeries, config: TSConfig) -> range:
    """Valid anchor indices for ``series``.

    An anchor ``i`` is the index of the LAST observed sample: the model is shown
    ``values[i - seq_len + 1 : i + 1]`` and must predict what follows.

    - window mode requires a full, unpadded horizon after the anchor.
    - full mode requires only that something follows; a short remainder is padded and
      masked. Where the remainder EXCEEDS ``pred_len``, the target is truncated to
      ``pred_len`` — the model predicts as far ahead as the architecture allows.
    """
    first = config.seq_len - 1
    if config.horizon_mode == HORIZON_FULL:
        # An anchor is always observed; fitted rows can be TARGETS but never model inputs.
        last = min(series.n_samples - 1, series.n_supervision_samples - 2)
    else:
        last = min(
            series.n_samples - 1,
            series.n_supervision_samples - config.pred_len - 1,
        )
    return range(first, last + 1) if last >= first else range(0)


class TrajectoryWindows(Dataset):
    """``(x[L,C], y[H,C], weights[H,C])`` windows over :class:`FlightSeries`.

    Measured rows supervise all six channels. Fitted rows supervise only ``e/n/u`` at lower
    weight, and padding is zero everywhere. Thus extrapolated velocity never becomes a
    label, while the fitted crossing still contributes terminal position error.
    """

    def __init__(self, series: Sequence[FlightSeries], config: TSConfig, normalizer: Normalizer):
        self.series = list(series)
        self.config = config
        self.normalizer = normalizer
        self.index: list[tuple[int, int]] = [
            (s_idx, anchor)
            for s_idx, s in enumerate(self.series)
            for anchor in window_anchors(s, config)
        ]
        # Standardise once up front rather than per __getitem__ — the series are small
        # (a few hundred rows each) and this is read on every epoch.
        self.encoded = [
            normalizer.encode(s.supervision_values).astype(np.float32) for s in self.series
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_idx, anchor = self.index[i]
        values = self.encoded[s_idx]
        L, H, C = self.config.seq_len, self.config.pred_len, len(self.config.channels)

        x = values[anchor - L + 1 : anchor + 1]
        future = values[anchor + 1 : anchor + 1 + H]
        future_weights = self.series[s_idx].supervision_weights[
            anchor + 1 : anchor + 1 + H
        ]
        y = np.zeros((H, C), dtype=np.float32)
        weights = np.zeros((H, C), dtype=np.float32)
        y[: len(future)] = future
        weights[: len(future_weights)] = future_weights

        return torch.from_numpy(x.copy()), torch.from_numpy(y), torch.from_numpy(weights)


def _split_fraction(flight_id: str, seed: int) -> float:
    """One flight's deterministic position in [0, 1), independent of every other flight.

    hashlib, not the builtin ``hash()`` — that one is salted per process (PYTHONHASHSEED),
    so it would deal every run a different split.
    """
    digest = hashlib.sha256(f"{seed}:{flight_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_by_flight(
    series: Sequence[FlightSeries], config: TSConfig
) -> tuple[list[FlightSeries], list[FlightSeries], list[FlightSeries]]:
    """Deterministic train / val / test split at FLIGHT granularity.

    Each flight's split is a pure function of ``(config.seed, flight_id)`` — never of its
    POSITION in the list. A positional shuffle looks deterministic but reshuffles the whole
    assignment the moment one flight is added to or dropped from the harvest, silently
    promoting old test flights into training on the next retrain. The cost of per-flight
    hashing is that the realised fractions only approximate ``val_fraction`` /
    ``test_fraction`` (exact in expectation); the win is that a flight, once in the test
    set, stays there for every future harvest with the same seed.
    """
    train, val, test = [], [], []
    for s in series:
        fraction = _split_fraction(s.flight_id, config.seed)
        if fraction < config.test_fraction:
            test.append(s)
        elif fraction < config.test_fraction + config.val_fraction:
            val.append(s)
        else:
            train.append(s)
    if not train or not val:
        raise ValueError(
            f"split of {len(series)} flight(s) left train={len(train)}, val={len(val)}, "
            f"test={len(test)} at val={config.val_fraction}, test={config.test_fraction} — "
            f"too few flights for these fractions (training needs non-empty train AND val)"
        )
    return train, val, test


def iter_batches(dataset: TrajectoryWindows, batch_size: int, *, shuffle: bool, seed: int) -> Iterator:
    """A DataLoader with this project's defaults (no workers — the data is already in RAM)."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=0, drop_last=False)
