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
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler

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
from trajectory_data_process.harvest.arrivals import (
    load_arrival_flights,
    resolve_arrival_manifest,
)

from channels import (
    CHANNELS,
    POSITION_IDX,
    Frame,
    channels_from_states,
    frame_for_state,
    resample_uniform,
)
from config import DEFAULT_AIRCRAFT_TYPE, HORIZON_FULL, TSConfig

ARRIVAL_DATA_PROVENANCE_SCHEMA = "ts-arrival-data-v2-multi-airport"


def dataset_flight_key(source: dict[str, Any], index: int) -> str:
    """Airport-qualified identity used by splits and checkpoint membership."""
    airport = str(source.get("arr_airport") or "").strip().upper()
    key = flight_key(source, index)
    return f"{airport}:{key}" if airport else key


def _manifest_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve one or more manifests, rejecting duplicate airport inputs."""
    raw_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not raw_paths:
        raise ValueError("at least one arrival manifest is required")

    resolved: list[tuple[str, Path]] = []
    seen_airports: set[str] = set()
    for raw_path in raw_paths:
        manifest_path = resolve_arrival_manifest(raw_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(
                f"{manifest_path} is not an arrival manifest object; legacy flight-array "
                "inputs are no longer supported"
            )
        airport = str(manifest.get("airport") or "").strip().upper()
        if not airport:
            raise ValueError(f"{manifest_path} does not declare an airport")
        if airport in seen_airports:
            raise ValueError(f"multiple arrival manifests supplied for airport {airport}")
        seen_airports.add(airport)
        resolved.append((airport, manifest_path))
    return [path for _airport, path in sorted(resolved)]


def arrival_data_provenance(
    paths: str | Path | Sequence[str | Path],
) -> dict[str, Any]:
    """Fingerprint the exact canonical arrival rosters used by a training run.

    The manifest digest catches any roster, slice, target, or metadata change.  Keeping
    each flight's canonical source digest as well makes the checkpoint independently
    auditable without reopening every source track.
    """
    manifest_entries: list[dict[str, Any]] = []
    for manifest_path in _manifest_paths(paths):
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        records = manifest.get("records") if isinstance(manifest, dict) else None
        if not isinstance(records, list):
            raise ValueError(f"{manifest_path} lacks an arrival records roster")
        airport = str(manifest.get("airport") or "").strip().upper()

        source_records: list[dict[str, str]] = []
        seen: set[str] = set()
        for index, row in enumerate(records):
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}: arrival record {index} is not an object")
            key = row.get("flight_key")
            source_sha256 = row.get("source_sha256")
            if not isinstance(key, str) or not key:
                raise ValueError(f"{manifest_path}: arrival record {index} lacks flight_key")
            if key in seen:
                raise ValueError(f"{manifest_path} lists duplicate flight_key {key!r}")
            if (
                not isinstance(source_sha256, str)
                or len(source_sha256) != 64
                or any(char not in "0123456789abcdef" for char in source_sha256.lower())
            ):
                raise ValueError(
                    f"{manifest_path}: arrival record {index} has invalid source_sha256"
                )
            seen.add(key)
            source_records.append(
                {"flight_key": key, "source_sha256": source_sha256.lower()}
            )

        source_records.sort(key=lambda item: item["flight_key"])
        manifest_entries.append({
            "airport": airport,
            "arrival_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_records": source_records,
        })

    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": manifest_entries,
    }


def provenance_manifest_digests(provenance: dict[str, Any]) -> dict[str, str]:
    """Compact airport -> manifest digest view used by import-light runners."""
    if provenance.get("schema_version") != ARRIVAL_DATA_PROVENANCE_SCHEMA:
        raise ValueError("data_provenance is not a multi-airport TS fingerprint")
    manifests = provenance.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("data_provenance has no arrival manifests")
    result: dict[str, str] = {}
    for entry in manifests:
        if not isinstance(entry, dict):
            raise ValueError("data_provenance manifest entry is not an object")
        airport = entry.get("airport")
        digest = entry.get("arrival_manifest_sha256")
        if not isinstance(airport, str) or not isinstance(digest, str):
            raise ValueError("data_provenance manifest entry lacks airport or digest")
        if airport in result:
            raise ValueError(f"data_provenance repeats airport {airport}")
        result[airport] = digest
    return result


def require_matching_data_provenance(
    checkpoint_payload: dict[str, Any],
    current: dict[str, Any],
    *,
    allow_subset: bool = False,
) -> None:
    """Reject stale data; prediction may verify an exact airport subset of training data."""
    stored = checkpoint_payload.get("data_provenance")
    if not isinstance(stored, dict):
        raise ValueError(
            "checkpoint has no arrival-data provenance; retrain it against the current "
            "arrivals/manifest.json"
        )
    if not allow_subset and stored != current:
        raise ValueError(
            "checkpoint training data does not match the current arrival manifests; "
            "retrain instead of reusing this checkpoint"
        )
    if allow_subset:
        stored_entries = {
            entry["airport"]: entry for entry in stored.get("manifests", [])
            if isinstance(entry, dict) and isinstance(entry.get("airport"), str)
        }
        current_entries = current.get("manifests")
        if (
            current.get("schema_version") != ARRIVAL_DATA_PROVENANCE_SCHEMA
            or not isinstance(current_entries, list)
            or not current_entries
            or any(
                not isinstance(entry, dict)
                or stored_entries.get(entry.get("airport")) != entry
                for entry in current_entries
            )
        ):
            raise ValueError(
                "prediction data is not an exact airport subset of the checkpoint training "
                "data; retrain or use the matching manifests"
            )


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

    @property
    def airport(self) -> str:
        """Arrival airport carried by the canonical manifest record."""
        return str(self.scenario.source.get("arr_airport") or "").strip().upper()

    @property
    def dataset_id(self) -> str:
        """Cross-airport split/checkpoint identity; export stems remain ``flight_id``."""
        return f"{self.airport}:{self.flight_id}" if self.airport else self.flight_id


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
    def fit(
        cls,
        series: Sequence[FlightSeries],
        *,
        balance_airports_and_flights: bool = False,
    ) -> Normalizer:
        """Fit on training only, optionally matching the hierarchical sampler's measure."""
        if balance_airports_and_flights:
            by_airport: dict[str, list[FlightSeries]] = {}
            for item in series:
                by_airport.setdefault(item.airport or "<unknown>", []).append(item)
            airport_means, airport_second_moments = [], []
            for group in by_airport.values():
                airport_means.append(np.mean([item.values.mean(axis=0) for item in group], axis=0))
                airport_second_moments.append(np.mean([
                    np.square(item.values).mean(axis=0) for item in group
                ], axis=0))
            mean = np.mean(airport_means, axis=0)
            variance = np.maximum(np.mean(airport_second_moments, axis=0) - mean**2, 0.0)
            std = np.sqrt(variance)
            std = np.where(std > 1e-9, std, 1.0)
            return cls(mean=mean, std=std)
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

def load_flight_dicts(
    paths: str | Path | Sequence[str | Path], *, verbose: bool = True
) -> list[dict[str, Any]]:
    """Model-ready flights from one or more authoritative arrival manifests."""
    flights: list[dict[str, Any]] = []
    for manifest_path in _manifest_paths(paths):
        manifest_flights = load_arrival_flights(manifest_path)
        flights.extend(manifest_flights)
        if verbose:
            print(f"  {manifest_path}: {len(manifest_flights)} manifest-rostered arrival(s)")
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
        # pay the conversion; ``waypoints`` must be rebound to the converted rows.
        flight = flight_to_msl(flight)
        waypoints = flight["waypoints"]

        scenario = build_scenario(flight, aircraft_type, airport=airport,
                                  target_from_threshold=True)
        if scenario.target is None:
            runway = flight.get("runway") or "?"
            report.skip(f"no published threshold for runway {runway}")
            continue

        samples = state_samples_from_track(waypoints, mass_kg=scenario.initial.m)
        frame = frame_for_state(scenario.target, config.coordinate_frame)
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

    def __init__(
        self,
        series: Sequence[FlightSeries],
        config: TSConfig,
        normalizer: Normalizer,
        *,
        anchor_policy: str = "all",
    ):
        if anchor_policy not in ("all", "first"):
            raise ValueError(f"unknown anchor policy {anchor_policy!r}")
        self.series = list(series)
        self.config = config
        self.normalizer = normalizer
        self.anchor_policy = anchor_policy
        self.index: list[tuple[int, int]] = []
        self.series_ranges: dict[int, tuple[int, int]] = {}
        for s_idx, item in enumerate(self.series):
            anchors = window_anchors(item, config)
            chosen = [anchors.start] if anchor_policy == "first" and len(anchors) else anchors
            start = len(self.index)
            self.index.extend((s_idx, anchor) for anchor in chosen)
            self.series_ranges[s_idx] = (start, len(self.index) - start)
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


def split_name_for_dataset_id(dataset_id: str, config: TSConfig) -> str:
    """Return the locked outer split without reading any trajectory values."""
    fraction = _split_fraction(dataset_id, config.seed)
    if fraction < config.test_fraction:
        return "test"
    if fraction < config.test_fraction + config.val_fraction:
        return "val"
    return "train"


def split_by_flight(
    series: Sequence[FlightSeries], config: TSConfig
) -> tuple[list[FlightSeries], list[FlightSeries], list[FlightSeries]]:
    """Deterministic train / val / test split at FLIGHT granularity.

    Each flight's split is a pure function of ``(config.seed, airport:flight_id)`` — never of its
    POSITION in the list. A positional shuffle looks deterministic but reshuffles the whole
    assignment the moment one flight is added to or dropped from the harvest, silently
    promoting old test flights into training on the next retrain. The cost of per-flight
    hashing is that the realised fractions only approximate ``val_fraction`` /
    ``test_fraction`` (exact in expectation); the win is that a flight, once in the test
    set, stays there for every future harvest with the same seed.
    """
    train, val, test = [], [], []
    for s in series:
        split = split_name_for_dataset_id(s.dataset_id, config)
        if split == "test":
            test.append(s)
        elif split == "val":
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


def cross_validation_folds(
    series: Sequence[FlightSeries], n_splits: int, *, seed: int
) -> list[list[FlightSeries]]:
    """Deterministic airport-stratified folds over an already locked outer-train set."""
    if n_splits < 2:
        raise ValueError(f"cross validation needs at least 2 folds, got {n_splits}")
    if len(series) < n_splits:
        raise ValueError(f"cannot split {len(series)} flight(s) into {n_splits} folds")

    by_airport: dict[str, list[FlightSeries]] = {}
    for item in series:
        by_airport.setdefault(item.airport or "<unknown>", []).append(item)

    folds: list[list[FlightSeries]] = [[] for _ in range(n_splits)]
    for airport, group in sorted(by_airport.items()):
        if len(group) < n_splits:
            raise ValueError(
                f"airport {airport} has only {len(group)} outer-train flight(s), fewer than "
                f"the requested {n_splits} folds"
            )
        ordered = sorted(
            group,
            key=lambda item: hashlib.sha256(
                f"cv:{seed}:{airport}:{item.dataset_id}".encode()
            ).digest(),
        )
        for index, item in enumerate(ordered):
            folds[index % n_splits].append(item)
    if any(not fold for fold in folds):
        raise ValueError("airport-stratified cross validation produced an empty fold")
    return folds


class AirportFlightWindowSampler(Sampler[int]):
    """Sample airport, then flight, then one of that flight's valid windows uniformly."""

    def __init__(self, dataset: TrajectoryWindows, *, num_samples: int, seed: int):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.dataset = dataset
        self.num_samples = num_samples
        self.seed = seed
        self.by_airport: dict[str, list[int]] = {}
        for s_idx, item in enumerate(dataset.series):
            _start, count = dataset.series_ranges[s_idx]
            if count:
                self.by_airport.setdefault(item.airport or "<unknown>", []).append(s_idx)
        if not self.by_airport:
            raise ValueError("balanced sampler received a dataset with no windows")
        self.airports = sorted(self.by_airport)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.num_samples):
            airport_idx = int(torch.randint(len(self.airports), (1,), generator=generator))
            flights = self.by_airport[self.airports[airport_idx]]
            flight_idx = int(torch.randint(len(flights), (1,), generator=generator))
            start, count = self.dataset.series_ranges[flights[flight_idx]]
            offset = int(torch.randint(count, (1,), generator=generator))
            yield start + offset


def iter_batches(
    dataset: TrajectoryWindows,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    balanced: bool = False,
    num_samples: int | None = None,
) -> Iterator:
    """A DataLoader with this project's defaults (no workers — the data is already in RAM)."""
    if balanced:
        sampler = AirportFlightWindowSampler(
            dataset, num_samples=num_samples or len(dataset), seed=seed
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
                          drop_last=False)
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    if shuffle and num_samples is not None and num_samples < len(dataset):
        sampler = RandomSampler(
            dataset, replacement=False, num_samples=num_samples, generator=generator
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
                          drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=0, drop_last=False)
