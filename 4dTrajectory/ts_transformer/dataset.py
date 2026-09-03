"""Observed arrivals -> uniform histories -> configured trajectory-time targets.

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

import functools
import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from aircraft.identity import get_default_identity_resolver
from aircraft.query_aircraft_parameters import (
    openap_direct_typecodes,
    openap_source_label,
    openap_support_kind,
)

# flight_key is the identity ``id_runway_icao24_landingTime`` — single-sourced in
# flight_scenarios.identity because the optimizer batch derives its record filenames from
# the SAME function; the split key here and both writers' filename stems cannot drift.
from final_approach import bracket_fraction
from flight_scenarios import (
    FittedApproach,
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
    VELOCITY_IDX,
    channels_from_states,
    resample_uniform,
    states_from_channels,
    target_chart_position,
)
from anchor_eligibility import (
    eligible_random_train_anchors,
    random_train_anchor_eligibility_policy,
)
from config import (
    AIRCRAFT_FILTER_OPENAP_DIRECT,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    DEFAULT_AIRCRAFT_TYPE,
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    TSConfig,
    uses_control_dynamics,
)
from control.conditioning import DYNAMICS_CONDITION_NAMES, condition_vector
from control.envelope import CONTROL_LOWER, CONTROL_UPPER
from control.dynamics.inverse import actual_controls, segment_controls
from coordinate_frames import (
    COORDINATE_FRAME_AIRPORT_ENU,
    AirportReference,
    CoordinateFrame,
    frame_for_state,
)
from flight_scenarios.runway_target import airport_reference_point
from fixed_dt_supervision import (
    FixedDTControlSupervision,
    FixedDTSupervisionRow,
    build_fixed_dt_supervision,
    cache_fixed_dt_supervision_rows,
    pack_fixed_dt_supervision_rows,
)
from time_grids import output_time_grid
from reference_velocity import rebuild_reference_velocities

ARRIVAL_DATA_PROVENANCE_SCHEMA = "ts-arrival-data-v3-eligibility-bound"
DATA_SELECTION_SCHEMA = "ts-data-selection-v2-pre-split-eligibility"


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
    *,
    eligibility_rosters: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Fingerprint the exact canonical arrival rosters used by a training run.

    The manifest digest catches any roster, slice, target, or metadata change.  Keeping
    each flight's canonical source digest as well makes the checkpoint independently
    auditable without reopening every source track.
    """
    roster_by_airport: dict[str, tuple[Path, bytes, dict[str, Any]]] = {}
    for raw_roster in eligibility_rosters or ():
        roster_path = Path(raw_roster).resolve()
        roster_bytes = roster_path.read_bytes()
        roster = json.loads(roster_bytes)
        airport = str(roster.get("airport") or "").strip().upper()
        if not airport:
            raise ValueError(f"{roster_path} does not declare an airport")
        if airport in roster_by_airport:
            raise ValueError(f"multiple eligibility rosters supplied for airport {airport}")
        roster_by_airport[airport] = (roster_path, roster_bytes, roster)

    manifest_entries: list[dict[str, Any]] = []
    manifest_airports: set[str] = set()
    for manifest_path in _manifest_paths(paths):
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        records = manifest.get("records") if isinstance(manifest, dict) else None
        if not isinstance(records, list):
            raise ValueError(f"{manifest_path} lacks an arrival records roster")
        airport = str(manifest.get("airport") or "").strip().upper()
        manifest_airports.add(airport)

        eligible_keys: set[str] | None = None
        eligibility: dict[str, Any] | None = None
        if eligibility_rosters is not None:
            if airport not in roster_by_airport:
                raise ValueError(f"no eligibility roster supplied for airport {airport}")
            roster_path, roster_bytes, roster = roster_by_airport[airport]
            from lateral_eligibility import (  # local import keeps the loader policy-agnostic
                LATERAL_PASS_POLICY,
                LATERAL_PASS_ROSTER_SCHEMA,
            )
            if roster.get("schema_version") != LATERAL_PASS_ROSTER_SCHEMA:
                raise ValueError(f"{roster_path} has the wrong eligibility schema")
            sources = roster.get("sources")
            manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
            if (
                not isinstance(sources, dict)
                or sources.get("arrival_manifest_sha256") != manifest_digest
            ):
                raise ValueError(f"{roster_path} was built for a different arrival manifest")
            if roster.get("policy") != LATERAL_PASS_POLICY:
                raise ValueError(f"{roster_path} has the wrong eligibility policy")
            keys = roster.get("eligible_flight_keys")
            if (
                not isinstance(keys, list)
                or any(not isinstance(key, str) or not key for key in keys)
                or len(set(keys)) != len(keys)
            ):
                raise ValueError(f"{roster_path} has invalid eligible flight identities")
            eligible_keys = set(keys)
            eligibility = {
                "schema_version": roster["schema_version"],
                "policy": roster["policy"],
                "roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
                "evaluation_report_sha256": sources.get("evaluation_report_sha256"),
                "counts": roster.get("counts"),
            }

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
            if eligible_keys is not None and key not in eligible_keys:
                continue
            source_records.append(
                {"flight_key": key, "source_sha256": source_sha256.lower()}
            )

        source_records.sort(key=lambda item: item["flight_key"])
        if eligible_keys is not None:
            missing_eligible = eligible_keys - seen
            if missing_eligible:
                raise ValueError(
                    f"{roster_path} names flight absent from arrival manifest: "
                    f"{min(missing_eligible)!r}"
                )
        manifest_entries.append({
            "airport": airport,
            "arrival_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "arrival_candidate_count": len(records),
            "eligibility": eligibility,
            "source_records": source_records,
        })

    unused_rosters = roster_by_airport.keys() - manifest_airports
    if unused_rosters:
        raise ValueError(
            f"eligibility roster supplied without arrival manifest for {min(unused_rosters)}"
        )

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


def provenance_eligibility_digests(
    provenance: dict[str, Any],
) -> dict[str, str]:
    """Compact airport -> pre-split eligibility artifact digest."""
    if provenance.get("schema_version") != ARRIVAL_DATA_PROVENANCE_SCHEMA:
        raise ValueError("data_provenance is not a multi-airport TS fingerprint")
    result: dict[str, str] = {}
    for entry in provenance.get("manifests", []):
        if not isinstance(entry, dict):
            raise ValueError("data_provenance manifest entry is not an object")
        airport = entry.get("airport")
        eligibility = entry.get("eligibility")
        digest = (
            eligibility.get("roster_sha256")
            if isinstance(eligibility, dict)
            else None
        )
        if not isinstance(airport, str) or (
            digest is not None and not isinstance(digest, str)
        ):
            raise ValueError("data_provenance has invalid eligibility identity")
        if digest is not None:
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
    frame: CoordinateFrame
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

    @property
    def target_chart(self) -> np.ndarray:
        """The runway target's chart position ``(e, n, u)`` — where "distance to go" is
        measured from. ``(0, 0, 0)`` under the threshold-anchored frames; the frame's
        anchor and the target are different points under ``airport-enu``."""
        return target_chart_position(self.scenario.target, self.frame)




# How much observed lookback the anchor-state control inversion differentiates. It needs
# at least three samples for a second-order gradient; a few more absorb ADS-B jitter
# without reaching back into a different phase of flight (11 x 2 s = 20 s).
ANCHOR_CONTROL_SAMPLES = 11


def anchor_controls(series: FlightSeries, anchor: int, mass_kg: float) -> np.ndarray:
    """Return the controls the observed lookback implies are in effect at ``anchor``.

    This is the lagged model's actuator initial condition. It reads only samples at or
    before the anchor, so it is as deployable as the history window itself, and it is the
    ACTUAL control (never a command) for every flight model — the commands that produced
    it are a separate inversion in ``control_inverse_dynamics``.
    """
    start = max(0, anchor + 1 - ANCHOR_CONTROL_SAMPLES)
    window = slice(start, anchor + 1)
    times = series.times[window]
    samples = states_from_channels(
        times, series.values[window], series.frame, mass_kg=mass_kg
    )
    states = np.asarray(
        [
            [s.latitude, s.longitude, s.altitude, s.V, s.psi, s.gamma, s.m]
            for _t, s in samples
        ],
        dtype=np.float64,
    )
    aero = series.scenario.aero
    return actual_controls(
        states,
        times,
        aero_params=np.array(
            [aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall],
            dtype=np.float64,
        ),
        max_thrust_n=float(series.scenario.aircraft.engine.max_thrust_total_n),
    )[-1]


def reference_control_supervision(
    series: FlightSeries,
    anchor: int,
    config: TSConfig,
    total_duration_s: float,
    last_measured_time_s: float,
) -> dict[str, np.ndarray]:
    """Return the control schedule the flown track implies, as a TRAINING TARGET.

    This is supervision, not dynamics: it reads the future, so it is deliberately built
    here and not in :func:`dynamics_arrays`, which forecast and predict also call and which
    must stay deployable from the lookback alone.

    The schedule comes from the SAME inverse registry the forward rollout dispatches
    through, keyed on ``config.control_dynamics_model``, so the lagged model is supervised
    on COMMANDS and the point-mass model on actual controls -- a target can never be the
    solution of equations the training rollout does not integrate.

    ``total_duration_s`` is the full supervised horizon (fitted tail included) because the
    model's segments span it, but the fitted tail has no measured velocity to differentiate.
    Segments whose midpoint falls past ``last_measured_time_s`` therefore get weight zero,
    the same masking the velocity term already relies on.
    """
    mass_kg = float(series.scenario.initial.m)
    anchor_time = float(series.times[anchor])
    times = series.times[anchor:] - anchor_time
    samples = states_from_channels(
        times, series.values[anchor:], series.frame, mass_kg=mass_kg
    )
    states = np.asarray(
        [[s.latitude, s.longitude, s.altitude, s.V, s.psi, s.gamma, s.m]
         for _t, s in samples],
        dtype=np.float64,
    )
    aero = series.scenario.aero
    n_segments = int(config.n_segments)
    inverted = segment_controls(
        states,
        times,
        config=config,
        aero_params=np.array(
            [aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall],
            dtype=np.float64,
        ),
        max_thrust_n=float(series.scenario.aircraft.engine.max_thrust_total_n),
        control_lower=CONTROL_LOWER,
        control_upper=CONTROL_UPPER,
        n_segments=n_segments,
        total_duration_s=total_duration_s,
    )
    midpoints = (np.arange(n_segments, dtype=np.float64) + 0.5) * (
        total_duration_s / n_segments
    )
    return {
        "reference_controls": inverted.controls.astype(np.float64),
        "reference_control_weight": (
            midpoints <= last_measured_time_s
        ).astype(np.float64),
    }


def dynamics_arrays(series: FlightSeries, anchor: int) -> dict[str, np.ndarray]:
    """Physical per-flight tensors required by a control model and its rollout."""
    scenario = series.scenario
    mass_kg = float(scenario.initial.m)
    initial = states_from_channels(
        np.array([0.0], dtype=np.float64),
        series.values[anchor : anchor + 1],
        series.frame,
        mass_kg=mass_kg,
    )[0][1]
    aero = scenario.aero
    max_thrust = float(scenario.aircraft.engine.max_thrust_total_n)
    condition = condition_vector(mass_kg, max_thrust, aero)
    heading = float(getattr(series.frame, "heading_rad", 0.0))
    return {
        "condition": condition,
        "initial_state": np.array(
            [
                initial.latitude,
                initial.longitude,
                initial.altitude,
                initial.V,
                initial.psi,
                initial.gamma,
                initial.m,
            ],
            dtype=np.float64,
        ),
        "aero_params": np.array(
            [aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall],
            dtype=np.float64,
        ),
        "max_thrust_n": np.array(max_thrust, dtype=np.float64),
        # The dimensionless envelope is the same box on every airframe (see
        # control_envelope); the flight's installed thrust enters through max_thrust_n.
        "control_lower": CONTROL_LOWER.astype(np.float32),
        "control_upper": CONTROL_UPPER.astype(np.float32),
        # The lagged model's actuator initial condition. Emitted unconditionally so the
        # batch contract does not depend on the flight model, and clipped to the same box
        # the head predicts in — an anchor whose implied thrust is outside the envelope is
        # a starting point the model could not have commanded.
        "initial_controls": np.clip(
            anchor_controls(series, anchor, mass_kg), CONTROL_LOWER, CONTROL_UPPER
        ).astype(np.float64),
        "frame_params": np.array(
            [series.frame.lat0, series.frame.lon0, series.frame.alt0, heading],
            dtype=np.float64,
        ),
        # The rollout frame rotation and the runway heading coincide only for the
        # runway-aligned coordinate frame.  Keep the terminal-loss reference separate
        # so ENU rollouts are decomposed along/across the actual runway, not east/north.
        "runway_heading_rad": np.array(float(scenario.target.psi), dtype=np.float64),
    }


def probe_dynamics(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """One representative dynamics batch for shape/throughput probes.

    Kept beside :func:`dynamics_arrays` so the batch-size probe and the gradient
    diagnostics cannot carry a stale copy of the contract: a key added to the real batch
    appears here in the same commit or the probe fails immediately.
    """
    rows = {
        "condition": [0.66, 0.24, 0.2452, 0.9, 0.2, 0.4, 0.9, 0.5],
        "initial_state": [35.9, -78.8, 1000.0, 80.0, 2.0, -0.05, 66_000.0],
        "aero_params": [122.6, 2.7, 0.02, 0.04, 0.9, 0.1],
        "control_lower": CONTROL_LOWER.tolist(),
        "control_upper": CONTROL_UPPER.tolist(),
        "initial_controls": [0.2, 0.0, 1.0],
        "frame_params": [35.9, -78.8, 100.0, 0.0],
    }
    dynamics = {
        name: torch.tensor([value], dtype=torch.float32, device=device).expand(
            batch_size, -1
        )
        for name, value in rows.items()
    }
    dynamics["max_thrust_n"] = torch.full(
        (batch_size,), 240_000.0, dtype=torch.float32, device=device
    )
    dynamics["runway_heading_rad"] = torch.zeros(
        batch_size, dtype=torch.float32, device=device
    )
    return dynamics


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
    paths: str | Path | Sequence[str | Path],
    *,
    include_flight_keys: set[str] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Model-ready flights from authoritative manifests, optionally split-filtered.

    ``include_flight_keys`` uses the airport-qualified checkpoint identity
    ``AIRPORT:flight_key``. The set is reduced to each manifest's local keys before
    :func:`load_arrival_flights` opens source tracks, so an excluded split's trajectory
    values are never read merely to discard them later.
    """
    requested = None if include_flight_keys is None else set(include_flight_keys)
    flights: list[dict[str, Any]] = []
    loaded_keys: set[str] = set()
    for manifest_path in _manifest_paths(paths):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        airport = str(manifest.get("airport") or "").strip().upper()
        local_keys = None
        if requested is not None:
            prefix = f"{airport}:"
            local_keys = {
                key[len(prefix):] for key in requested if key.startswith(prefix)
            }
            if not local_keys:
                if verbose:
                    print(f"  {manifest_path}: 0 selected manifest-rostered arrival(s)")
                continue
        manifest_flights = load_arrival_flights(
            manifest_path,
            include_flight_keys=local_keys,
        )
        flights.extend(manifest_flights)
        loaded_keys.update(
            dataset_flight_key(flight, index)
            for index, flight in enumerate(manifest_flights)
        )
        if verbose:
            qualifier = "selected " if requested is not None else ""
            print(
                f"  {manifest_path}: {len(manifest_flights)} "
                f"{qualifier}manifest-rostered arrival(s)"
            )
    if requested is not None:
        missing = requested - loaded_keys
        if missing:
            raise ValueError(
                f"arrival manifests do not roster requested flight {min(missing)!r}"
            )
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
    selected_typecodes: dict[str, int] = field(default_factory=dict)
    rejected_aircraft: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def select_typecode(self, typecode: str) -> None:
        self.selected_typecodes[typecode] = self.selected_typecodes.get(typecode, 0) + 1

    def reject_aircraft(self, reason: str) -> None:
        self.rejected_aircraft[reason] = self.rejected_aircraft.get(reason, 0) + 1
        self.skip("aircraft filter rejected")

    @property
    def total(self) -> int:
        return self.built + sum(self.skipped.values())

    def format(self) -> str:
        lines = [f"built {self.built}/{self.total} series"]
        for reason, count in sorted(self.skipped.items(), key=lambda kv: -kv[1]):
            lines.append(f"  skipped {count:5d}  {reason}")
        if self.selected_typecodes:
            selected = ", ".join(
                f"{code}:{count}"
                for code, count in sorted(
                    self.selected_typecodes.items(), key=lambda item: (-item[1], item[0])
                )
            )
            lines.append(f"  selected aircraft  {selected}")
        rejected = sorted(
            self.rejected_aircraft.items(), key=lambda item: (-item[1], item[0])
        )
        for reason, count in rejected[:20]:
            lines.append(f"  rejected {count:5d}  {reason}")
        if len(rejected) > 20:
            omitted = sum(count for _reason, count in rejected[20:])
            lines.append(
                f"  rejected {omitted:5d}  {len(rejected) - 20} additional type reasons "
                "(see data_selection.json)"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "built": self.built,
            "total": self.total,
            "skipped": dict(sorted(self.skipped.items())),
            "selected_typecodes": dict(sorted(self.selected_typecodes.items())),
            "rejected_aircraft": dict(sorted(self.rejected_aircraft.items())),
        }


@functools.lru_cache(maxsize=None)
def _airport_reference(airport: str) -> AirportReference:
    """The airport-fixed frame's anchor, from the harvest's own airport roster."""
    point = airport_reference_point(airport)
    return AirportReference(
        code=airport,
        lat=point["lat"],
        lon=point["lon"],
        elevation_msl_m=point["elevation_m"],
    )


def _frame_for_scenario(scenario: FlightScenario, config: TSConfig) -> CoordinateFrame:
    """Resolve the chart once per flight; ``airport-enu`` needs the arrival airport."""
    airport_ref = None
    if config.coordinate_frame == COORDINATE_FRAME_AIRPORT_ENU:
        code = str(scenario.source.get("arr_airport") or "").strip().upper()
        if not code:
            raise ValueError(
                f"flight {scenario.source.get('id')!r}: coordinate frame "
                f"{config.coordinate_frame!r} needs the arrival airport, but the flight "
                "carries no arr_airport and build_series was given none"
            )
        airport_ref = _airport_reference(code)
    return frame_for_state(scenario.target, config.coordinate_frame, airport_ref=airport_ref)


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

        # Apply the strict fleet contract before geoid conversion, scenario construction,
        # velocity fitting or resampling. Split identity is a per-flight hash assigned from
        # manifest metadata, so rejecting a row here removes it *within* its original split;
        # it cannot reshuffle any other flight between train/validation/test.
        if config.aircraft_filter == AIRCRAFT_FILTER_OPENAP_DIRECT:
            identity = get_default_identity_resolver().resolve(
                declared_type=flight.get("type"),
                icao24=flight.get("icao24"),
            )
            support_kind = openap_support_kind(identity.typecode)
            if support_kind != "direct":
                if identity.typecode is None:
                    reason = "unresolved ICAO Doc 8643 typecode"
                elif support_kind == "synonym":
                    reason = f"OpenAP synonym model ({identity.typecode})"
                else:
                    reason = f"no native OpenAP model ({identity.typecode})"
                report.reject_aircraft(reason)
                continue
            report.select_typecode(identity.typecode)

        # Into the modeling plane: harvested altitude is ellipsoidal (HAE) while the
        # threshold-anchored channels and the evaluation gates are MSL. Converted HERE, not
        # inside build_scenario alone, because state_samples_from_track below takes the bare
        # waypoint list and so cannot convert itself. Idempotent (see flight_scenarios/datum).
        # Placed after the cheap skips (which read no altitude) so rejected flights don't
        # pay the conversion; ``waypoints`` must be rebound to the converted rows.
        flight = flight_to_msl(flight)
        waypoints = flight["waypoints"]

        scenario = build_scenario(flight,
            None if config.aircraft_filter == AIRCRAFT_FILTER_OPENAP_DIRECT
            else aircraft_type,
            airport=airport,
            target_from_threshold=True,
            aircraft_provider=(
                "openap"
                if config.aircraft_filter == AIRCRAFT_FILTER_OPENAP_DIRECT
                else "auto"
            ),
        )
        if scenario.target is None:
            runway = flight.get("runway") or "?"
            report.skip(f"no published threshold for runway {runway}")
            continue

        samples = state_samples_from_track(waypoints, mass_kg=scenario.initial.m)
        frame = _frame_for_scenario(scenario, config)
        target_chart = target_chart_position(scenario.target, frame)
        times, values = channels_from_states(samples, frame)
        grid, resampled = resample_uniform(times, values, config.dt_s)
        # Not redundant with the span pre-check: for a non-dyadic dt the multiply and
        # resample_uniform's floor-divide can round differently at the boundary.
        if len(grid) < minimum_samples:
            report.skip(f"track shorter than one window ({config.lookback_s:.0f}s)")
            continue

        fitted = fit_flight_final_approach(flight)
        observed_crossing = _observed_threshold_crossing(
            times,
            values,
            frame,
            target_chart=target_chart,
            runway_heading_rad=scenario.target.psi,
            fitted=fitted,
        )
        if observed_crossing is not None:
            crossing_time, _crossing_values = observed_crossing
            before_crossing = grid < crossing_time
            grid = grid[before_crossing]
            resampled = resampled[before_crossing]
            if len(grid) < config.seq_len:
                report.skip(f"track shorter than one window ({config.lookback_s:.0f}s)")
                continue

        supervision_times, supervision_values, supervision_weights = _build_supervision(
            samples,
            frame,
            grid,
            resampled,
            config,
            fitted=fitted,
            observed_crossing=observed_crossing,
        )
        measured_velocity_rows = (
            np.arange(len(supervision_times)) < len(grid)
        ) & np.all(
            supervision_weights[:, list(VELOCITY_IDX)] > 0.0, axis=1
        )
        supervision_values = rebuild_reference_velocities(
            supervision_times,
            supervision_values,
            source=config.reference_velocity_source,
            valid_rows=measured_velocity_rows,
        )
        resampled = supervision_values[: len(grid)].copy()
        series.append(FlightSeries(
            flight_id=flight_key(scenario.source, index), scenario=scenario, frame=frame,
            times=grid, values=resampled,
            supervision_times=supervision_times,
            supervision_values=supervision_values,
            supervision_weights=supervision_weights,
        ))
        report.built += 1

    return series, report


def _observed_threshold_crossing(
    measured_times: np.ndarray,
    measured_values: np.ndarray,
    frame: CoordinateFrame,
    *,
    target_chart: np.ndarray,
    runway_heading_rad: float,
    fitted: FittedApproach | None,
) -> tuple[float, np.ndarray] | None:
    """Interpolate the measured final's first crossing of the runway threshold plane.

    The plane passes through the TARGET (``target_chart``), normal to the runway course —
    not through the frame origin, which only coincides with the threshold under the
    threshold-anchored frames.
    """
    cosine = np.cos(runway_heading_rad)
    sine = np.sin(runway_heading_rad)
    along = np.asarray([
        east * cosine + north * sine
        for east, north in (
            frame.to_world_horizontal(
                float(row[0]) - float(target_chart[0]),
                float(row[1]) - float(target_chart[1]),
            )
            for row in measured_values
        )
    ])
    first_right_index = fitted.fit.last_sample_index + 1 if fitted is not None else 1
    for right_index in range(first_right_index, len(along)):
        left_index = right_index - 1
        left_along = along[left_index]
        right_along = along[right_index]
        if left_along <= 0.0 <= right_along and right_along > left_along:
            # The project's single crossing-fraction definition (final_approach) —
            # the same two-point operation the harvest bracket and the evaluator use.
            fraction = bracket_fraction(float(left_along), float(right_along))
            crossing_time = measured_times[left_index] + fraction * (
                measured_times[right_index] - measured_times[left_index]
            )
            crossing_values = measured_values[left_index] + fraction * (
                measured_values[right_index] - measured_values[left_index]
            )
            return float(crossing_time), crossing_values
    return None


def _build_supervision(
    measured_samples,
    frame: CoordinateFrame,
    grid: np.ndarray,
    measured_values: np.ndarray,
    config: TSConfig,
    *,
    fitted: FittedApproach | None,
    observed_crossing: tuple[float, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed six-channel labels plus low-weight, position-only fitted tail labels."""
    channel_count = len(CHANNELS)
    # Row weights sum to one, preserving the previous all-channel mean-MSE scale.
    measured_weights = np.full(
        measured_values.shape, 1.0 / channel_count, dtype=np.float64
    )
    if observed_crossing is not None:
        crossing_time, crossing_values = observed_crossing
        return (
            np.concatenate([grid, np.asarray([crossing_time])]),
            np.concatenate([measured_values, crossing_values[None, :]], axis=0),
            np.concatenate([
                measured_weights,
                np.full((1, channel_count), 1.0 / channel_count, dtype=np.float64),
            ], axis=0),
        )
    if (
        config.fitted_tail_position_weight == 0.0
        and config.fitted_terminal_position_weight == 0.0
    ):
        return grid, measured_values, measured_weights

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

def window_anchors(
    series: FlightSeries,
    config: TSConfig,
    *,
    minimum_anchor_index: int | None = None,
    minimum_future_s: float = 0.0,
) -> range:
    """Valid anchor indices for ``series``.

    An anchor ``i`` is the index of the LAST observed sample: the model is shown
    ``values[i - seq_len + 1 : i + 1]`` and must predict what follows.

    Normalized and full modes accept any non-empty remainder; normalized resamples the
    complete remainder and full masks a short padded suffix. Window mode requires a complete
    fixed-dt short horizon because those targets train one recursive forecasting pass.
    """
    if minimum_anchor_index is not None and minimum_anchor_index < 0:
        raise ValueError("minimum_anchor_index must be non-negative")
    if minimum_future_s < 0.0:
        raise ValueError("minimum_future_s must be non-negative")
    first = max(config.seq_len - 1, minimum_anchor_index or 0)
    # An anchor is always observed; fitted rows can be targets but never model inputs.
    last_with_remainder = series.n_supervision_samples - 2
    required_window_s = config.pred_len * config.dt_s
    complete_window_indices = np.flatnonzero(
        series.supervision_times[-1] - series.times >= required_window_s - 1e-9
    )
    last_with_complete_window = (
        int(complete_window_indices[-1]) if len(complete_window_indices) else -1
    )
    last_by_mode = {
        HORIZON_NORMALIZED: last_with_remainder,
        HORIZON_FULL: last_with_remainder,
        HORIZON_WINDOW: last_with_complete_window,
    }
    future_eligible = np.flatnonzero(
        series.supervision_times[-1] - series.times >= minimum_future_s - 1e-9
    )
    last_with_minimum_future = (
        int(future_eligible[-1]) if len(future_eligible) else -1
    )
    last = min(
        series.n_samples - 1,
        last_by_mode[config.horizon_mode],
        last_with_minimum_future,
    )
    return range(first, last + 1) if last >= first else range(0)


class TrajectoryWindows(Dataset, ABC):
    """Shared target-grid and batch mechanics for an explicit anchor population.

    Measured rows supervise all six channels. Fitted rows supervise only ``e/n/u`` at lower
    weight. Linear interpolation maps the remainder either to a complete normalized grid or
    to a physical full/window grid with a masked suffix. Concrete subclasses own anchor
    population and per-epoch index selection; this class contains no fixed-versus-random
    anchor-policy branch.
    """

    anchor_description: str
    anchor_policy: str
    sampling_version: str

    def __init__(
        self,
        series: Sequence[FlightSeries],
        config: TSConfig,
        normalizer: Normalizer,
        *,
        minimum_anchor_index: int | None = None,
        minimum_future_s: float = 0.0,
    ):
        self.series = list(series)
        self.config = config
        self.normalizer = normalizer
        self.minimum_anchor_index = minimum_anchor_index
        self.minimum_future_s = float(minimum_future_s)
        self.index: list[tuple[int, int]] = []
        self.series_ranges: dict[int, tuple[int, int]] = {}
        self.temporal_candidate_anchors = 0
        self.eligible_candidate_anchors = 0
        for s_idx, item in enumerate(self.series):
            anchors = window_anchors(
                item,
                config,
                minimum_anchor_index=minimum_anchor_index,
                minimum_future_s=self.minimum_future_s,
            )
            self.temporal_candidate_anchors += len(anchors)
            anchors = self._eligible_anchors(item, anchors)
            self.eligible_candidate_anchors += len(anchors)
            chosen = self._select_anchors(anchors)
            start = len(self.index)
            self.index.extend((s_idx, anchor) for anchor in chosen)
            self.series_ranges[s_idx] = (start, len(self.index) - start)
        self.range_starts = np.array(
            [self.series_ranges[index][0] for index in range(len(self.series))],
            dtype=np.int64,
        )
        self.range_counts = np.array(
            [self.series_ranges[index][1] for index in range(len(self.series))],
            dtype=np.int64,
        )
        self.eligible_series = np.flatnonzero(self.range_counts)
        # Standardise once up front rather than per __getitem__ — the series are small
        # (a few hundred rows each) and this is read on every epoch.
        self.encoded = [
            normalizer.encode(s.supervision_values).astype(np.float32) for s in self.series
        ]
        # Public diagnostic for normalized-time experiments. Actual query times come from
        # the shared clock below, which also defines fixed-time loss and inference timing.
        self.progress = (
            np.arange(1, config.pred_len + 1, dtype=np.float64) / config.pred_len
        )
        self.kinematic_channels = np.array(
            [channel for channel in range(len(config.channels)) if channel not in POSITION_IDX]
        )
        # Fitted rows retain placeholder velocity values for tensor shape only. Cache the
        # final valid supervision time per flight/channel once; recomputing flatnonzero for
        # every sampled anchor was a substantial part of host-side batch preparation.
        self.last_supervised_times = [
            np.array([
                item.supervision_times[
                    np.flatnonzero(item.supervision_weights[:, channel] > 0.0)[-1]
                ]
                for channel in range(len(config.channels))
            ])
            for item in self.series
        ]
        eligible_airports = [
            item.airport or "<unknown>"
            for series_index, item in enumerate(self.series)
            if self.series_ranges[series_index][1]
        ]
        airport_flights = Counter(eligible_airports)
        eligible_count = len(eligible_airports)
        airport_count = len(airport_flights)
        # The mean weight is one, and summing a complete epoch gives the same
        # eligible_count / airport_count weight to every airport. This lets a normal batch
        # mean estimate the airport-macro objective without repeating smaller airports.
        self.flight_weights = np.array([
            eligible_count / (
                airport_count * airport_flights[item.airport or "<unknown>"]
            )
            if self.series_ranges[series_index][1]
            else 0.0
            for series_index, item in enumerate(self.series)
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.index)

    @abstractmethod
    def _select_anchors(self, anchors: Sequence[int]) -> Sequence[int]:
        """Return the concrete mode's stored anchor population for one flight."""

    def _eligible_anchors(
        self, _series: FlightSeries, anchors: Sequence[int]
    ) -> Sequence[int]:
        """Apply mode-specific non-temporal eligibility before anchor sampling."""
        return anchors

    @abstractmethod
    def epoch_indices(self, seed: int) -> np.ndarray:
        """Return the concrete mode's one-flight-per-epoch sample indices."""

    def anchor_statistics(self, seed: int) -> dict[str, Any]:
        """Audit the exact one-window-per-flight sample selected for an epoch."""
        indices = self.epoch_indices(seed)
        anchors = np.array([self.index[int(index)][1] for index in indices], dtype=np.int64)
        series_indices = np.array(
            [self.index[int(index)][0] for index in indices], dtype=np.int64
        )
        if not len(indices):
            return {
                "policy": self.anchor_policy,
                "sampling_version": self.sampling_version,
                "minimum_future_s": self.minimum_future_s,
                "eligibility_policy": getattr(
                    self, "anchor_eligibility_policy", "temporal-only-v1"
                ),
                "temporal_candidate_anchors": self.temporal_candidate_anchors,
                "eligible_candidate_anchors": self.eligible_candidate_anchors,
                "excluded_candidate_anchors": (
                    self.temporal_candidate_anchors - self.eligible_candidate_anchors
                ),
                "samples": 0,
            }
        anchor_times = np.array([
            self.series[int(series_index)].times[int(anchor)]
            for series_index, anchor in zip(series_indices, anchors)
        ])
        remaining_times = np.array([
            self.series[int(series_index)].supervision_times[-1]
            - self.series[int(series_index)].times[int(anchor)]
            for series_index, anchor in zip(series_indices, anchors)
        ])
        digest_rows = sorted(
            f"{self.series[int(series_index)].dataset_id}:{int(anchor)}"
            for series_index, anchor in zip(series_indices, anchors)
        )
        digest = hashlib.sha256("\n".join(digest_rows).encode()).hexdigest()

        def distribution(values: np.ndarray) -> dict[str, float]:
            return {
                "min": float(np.min(values)),
                "p50": float(np.median(values)),
                "max": float(np.max(values)),
            }

        return {
            "policy": self.anchor_policy,
            "sampling_version": self.sampling_version,
            "minimum_future_s": self.minimum_future_s,
            "eligibility_policy": getattr(
                self, "anchor_eligibility_policy", "temporal-only-v1"
            ),
            "temporal_candidate_anchors": self.temporal_candidate_anchors,
            "eligible_candidate_anchors": self.eligible_candidate_anchors,
            "excluded_candidate_anchors": (
                self.temporal_candidate_anchors - self.eligible_candidate_anchors
            ),
            "samples": len(indices),
            "sample_sha256": digest,
            "anchor_index": distribution(anchors),
            "anchor_time_s": distribution(anchor_times),
            "remaining_time_s": distribution(remaining_times),
            "fixed_anchor_fraction": float(np.mean(anchors == self.config.seq_len - 1)),
        }

    def _sample_arrays(
        self, i: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32, np.float32]:
        s_idx, anchor = self.index[i]
        values = self.encoded[s_idx]
        series = self.series[s_idx]
        L = self.config.seq_len

        x = values[anchor - L + 1 : anchor + 1]
        anchor_time = float(series.times[anchor])
        final_time_s = float(series.supervision_times[-1] - anchor_time)
        time_grid = output_time_grid(final_time_s, self.config)
        query_times = anchor_time + time_grid.offsets_s

        # One search locates interpolation neighbours for every output time. The same
        # [N] neighbour arrays then interpolate all C state and weight channels together;
        # this replaces 2*C separate np.interp calls without hiding the interpolation math.
        right = np.searchsorted(series.supervision_times, query_times, side="left")
        right = np.clip(right, 1, len(series.supervision_times) - 1)
        left = right - 1
        interval = series.supervision_times[right] - series.supervision_times[left]
        fraction = ((query_times - series.supervision_times[left]) / interval)[:, None]
        y = values[left] + fraction * (values[right] - values[left])
        source_weights = series.supervision_weights
        weights = source_weights[left] + fraction * (
            source_weights[right] - source_weights[left]
        )
        weights[~time_grid.active] = 0.0

        # An observed crossing remains supervised even when it lies off the regular input
        # grid, but fitted-tail kinematics beyond the last measured velocity stay masked.
        weights[:, self.kinematic_channels] = np.where(
            query_times[:, None]
            > self.last_supervised_times[s_idx][self.kinematic_channels][None, :],
            0.0,
            weights[:, self.kinematic_channels],
        )

        return (
            x,
            y.astype(np.float32),
            weights.astype(np.float32),
            np.float32(final_time_s),
            self.flight_weights[s_idx],
        )

    def _dynamics_arrays(self, i: int) -> dict[str, np.ndarray]:
        s_idx, anchor = self.index[i]
        series = self.series[s_idx]
        arrays = dynamics_arrays(series, anchor)
        if self.config.control_imitation_loss_weight:
            anchor_time = float(series.times[anchor])
            arrays.update(
                reference_control_supervision(
                    series,
                    anchor,
                    self.config,
                    total_duration_s=float(
                        series.supervision_times[-1] - anchor_time
                    ),
                    last_measured_time_s=float(
                        self.last_supervised_times[s_idx][
                            self.kinematic_channels
                        ].min()
                        - anchor_time
                    ),
                )
            )
        return arrays

    def _fixed_dt_supervision(
        self, indices: Sequence[int] | np.ndarray
    ) -> FixedDTControlSupervision:
        return build_fixed_dt_supervision(
            self.series,
            self.encoded,
            [self.index[int(index)] for index in indices],
            dt_s=self.config.dt_s,
        )

    def __getitem__(
        self, i: int
    ) -> tuple:
        x, y, weights, final_time_s, flight_weight = self._sample_arrays(i)
        result = (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y),
            torch.from_numpy(weights),
            torch.from_numpy(np.asarray(final_time_s)),
            torch.from_numpy(np.asarray(flight_weight)),
        )
        if not uses_control_dynamics(self.config.prediction_output):
            return result
        dynamics = {
            key: torch.from_numpy(value)
            for key, value in self._dynamics_arrays(i).items()
        }
        if self.config.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_FIXED_DT:
            return (*result, dynamics)
        dense = self._fixed_dt_supervision([i])
        return (*result, dynamics, dense)

    def batch(
        self, indices: Sequence[int] | np.ndarray
    ) -> tuple:
        """Build one contiguous batch without per-sample Tensor creation and collation."""
        batch_size = len(indices)
        L, N, C = self.config.seq_len, self.config.pred_len, len(self.config.channels)
        x = np.empty((batch_size, L, C), dtype=np.float32)
        y = np.empty((batch_size, N, C), dtype=np.float32)
        weights = np.empty_like(y)
        final_time_s = np.empty(batch_size, dtype=np.float32)
        flight_weights = np.empty(batch_size, dtype=np.float32)

        # Flights are ragged, so locating each source array remains a short explicit loop.
        # All expensive work inside a sample is vectorized over N progress points and C
        # channels, and conversion to Torch happens once per complete batch below.
        for row, index in enumerate(indices):
            sample_x, sample_y, sample_weights, sample_time, flight_weight = (
                self._sample_arrays(int(index))
            )
            x[row] = sample_x
            y[row] = sample_y
            weights[row] = sample_weights
            final_time_s[row] = sample_time
            flight_weights[row] = flight_weight

        result = tuple(
            torch.from_numpy(array)
            for array in (x, y, weights, final_time_s, flight_weights)
        )
        if not uses_control_dynamics(self.config.prediction_output):
            return result
        dynamics_rows = [self._dynamics_arrays(int(index)) for index in indices]
        dynamics = {
            key: torch.from_numpy(np.stack([row[key] for row in dynamics_rows]))
            for key in dynamics_rows[0]
        }
        if self.config.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_FIXED_DT:
            return (*result, dynamics)
        dense = self._fixed_dt_supervision(indices)
        return (*result, dynamics, dense)


class FixedAnchorTrajectoryWindows(TrajectoryWindows):
    """One deterministic `L-1` (or experiment-supplied common) anchor per flight."""

    anchor_description = "fixed train anchor L-1"
    anchor_policy = "fixed"
    sampling_version = "fixed-anchor-v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Every fixed-anchor row is reused on every validation epoch and on final replay.
        # Cache the ragged source rows, not padded batches, so duration bucketing remains a
        # pure batching decision and cannot change any target or dynamics value.
        self._sample_cache = [
            TrajectoryWindows._sample_arrays(self, index)
            for index in range(len(self.index))
        ]
        self._dynamics_cache: list[dict[str, np.ndarray]] | None = None
        self._fixed_dt_cache: tuple[FixedDTSupervisionRow, ...] | None = None
        if uses_control_dynamics(self.config.prediction_output):
            self._dynamics_cache = [
                TrajectoryWindows._dynamics_arrays(self, index)
                for index in range(len(self.index))
            ]
            if self.config.control_state_loss_grid == CONTROL_STATE_LOSS_GRID_FIXED_DT:
                self._fixed_dt_cache = cache_fixed_dt_supervision_rows(
                    self.series,
                    self.encoded,
                    self.index,
                    dt_s=self.config.dt_s,
                )

    def _sample_arrays(self, i: int):
        return self._sample_cache[i]

    def _dynamics_arrays(self, i: int) -> dict[str, np.ndarray]:
        if self._dynamics_cache is None:
            return super()._dynamics_arrays(i)
        return self._dynamics_cache[i]

    def _fixed_dt_supervision(
        self, indices: Sequence[int] | np.ndarray
    ) -> FixedDTControlSupervision:
        if self._fixed_dt_cache is None:
            return super()._fixed_dt_supervision(indices)
        return pack_fixed_dt_supervision_rows(
            [self._fixed_dt_cache[int(index)] for index in indices],
            channels=len(self.config.channels),
        )

    def _select_anchors(self, anchors: Sequence[int]) -> Sequence[int]:
        return [anchors[0]] if len(anchors) else []

    def epoch_indices(self, seed: int) -> np.ndarray:
        indices = self.range_starts[self.eligible_series].copy()
        np.random.default_rng(seed).shuffle(indices)
        return indices


class RandomAnchorTrajectoryWindows(TrajectoryWindows):
    """All valid anchors available; each epoch selects one uniformly per flight."""

    anchor_description = "one random valid train anchor per flight and epoch"
    anchor_policy = "uniform-random"
    sampling_version = "per-flight-hash-v2-output-eligibility"

    def __init__(
        self,
        series: Sequence[FlightSeries],
        config: TSConfig,
        normalizer: Normalizer,
        *,
        minimum_anchor_index: int | None = None,
    ):
        self.anchor_eligibility_policy = random_train_anchor_eligibility_policy(config)
        super().__init__(
            series,
            config,
            normalizer,
            minimum_anchor_index=minimum_anchor_index,
            minimum_future_s=config.random_train_anchor_min_future_s,
        )

    def _select_anchors(self, anchors: Sequence[int]) -> Sequence[int]:
        return anchors

    def _eligible_anchors(
        self, series: FlightSeries, anchors: Sequence[int]
    ) -> Sequence[int]:
        return eligible_random_train_anchors(series, anchors, self.config)

    def epoch_indices(self, seed: int) -> np.ndarray:
        starts = self.range_starts[self.eligible_series]
        counts = self.range_counts[self.eligible_series]
        offsets = np.array([
            int.from_bytes(
                hashlib.sha256(
                    f"{self.sampling_version}:{seed}:"
                    f"{self.series[int(series_index)].dataset_id}".encode()
                ).digest()[:8],
                "big",
            ) % int(count)
            for series_index, count in zip(self.eligible_series, counts)
        ], dtype=np.int64)
        indices = starts + offsets
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        return indices


def _split_fraction(flight_id: str, seed: int) -> float:
    """One flight's deterministic position in [0, 1), independent of every other flight.

    hashlib, not the builtin ``hash()`` — that one is salted per process (PYTHONHASHSEED),
    so it would deal every run a different split.
    """
    digest = hashlib.sha256(f"{seed}:{flight_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_name_for_dataset_id(dataset_id: str, config: TSConfig) -> str:
    """Return the locked outer split without reading any trajectory values."""
    fraction = _split_fraction(dataset_id, config.resolved_split_seed)
    if fraction < config.test_fraction:
        return "test"
    if fraction < config.test_fraction + config.val_fraction:
        return "val"
    return "train"


def flight_keys_by_split(
    data_provenance: dict[str, Any], config: TSConfig
) -> dict[str, list[str]]:
    """Resolve airport-qualified split identities from manifest metadata only.

    Arrival provenance already carries the authoritative roster and source digests. This
    helper deliberately does not open any source trajectory file; callers can pass the
    returned keys to :func:`load_flight_dicts` before loading model inputs.
    """
    if data_provenance.get("schema_version") != ARRIVAL_DATA_PROVENANCE_SCHEMA:
        raise ValueError("data_provenance is not a TS arrival-data fingerprint")
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for manifest in data_provenance.get("manifests", []):
        airport = str(manifest.get("airport") or "").strip().upper()
        for record in manifest.get("source_records", []):
            dataset_id = f"{airport}:{record['flight_key']}"
            result[split_name_for_dataset_id(dataset_id, config)].append(dataset_id)
    return result


def data_selection_audit(
    series: Sequence[FlightSeries],
    report: BuildReport,
    config: TSConfig,
    outer_split_keys: dict[str, list[str]],
) -> dict[str, Any]:
    """Describe the post-split fleet selection without opening outer-test tracks.

    ``outer_split_keys`` comes from manifest metadata only. ``series`` must contain only
    development rows loaded by the caller; the sealed test population is recorded by
    identity/hash and its aircraft eligibility is intentionally deferred until release.
    """
    selected: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for item in series:
        selected[split_name_for_dataset_id(item.dataset_id, config)].append(item.dataset_id)
    if selected["test"]:
        raise ValueError(
            "data-selection audit received outer-test trajectory series during development"
        )

    def digest(keys: Sequence[str]) -> str:
        payload = "\n".join(sorted(keys)).encode()
        return hashlib.sha256(payload).hexdigest()

    direct_typecodes = openap_direct_typecodes()
    direct_digest = hashlib.sha256("\n".join(direct_typecodes).encode()).hexdigest()
    return {
        "schema_version": DATA_SELECTION_SCHEMA,
        "aircraft_filter": config.aircraft_filter,
        "identity_standard": "ICAO Doc 8643",
        "performance_provider": openap_source_label(),
        "openap_direct_typecodes": {
            "count": len(direct_typecodes),
            "sha256": direct_digest,
            "values": list(direct_typecodes),
        },
        "split_policy": {
            "method": "sha256(seed:airport-qualified-flight-id)",
            "split_seed": config.resolved_split_seed,
            "eligibility_applied_before_split_assignment": True,
            "aircraft_filter_applied_after_split_assignment": True,
            "outer_test_tracks_loaded": False,
            "outer_test_aircraft_filter_status": "deferred_until_test_release",
        },
        "splits": {
            name: {
                "eligible_roster_flights": len(outer_split_keys[name]),
                "eligible_identity_sha256": digest(outer_split_keys[name]),
                "selected_flights": (
                    len(selected[name]) if name != "test" else None
                ),
                "selected_identity_sha256": (
                    digest(selected[name]) if name != "test" else None
                ),
            }
            for name in ("train", "val", "test")
        },
        "development_build": report.to_dict(),
    }


def split_by_flight(
    series: Sequence[FlightSeries], config: TSConfig
) -> tuple[list[FlightSeries], list[FlightSeries], list[FlightSeries]]:
    """Deterministic train / val / test split at FLIGHT granularity.

    Each flight's split is a pure function of ``(config.resolved_split_seed,
    airport:flight_id)`` — never of its POSITION in the list or of the model-training seed.
    A positional shuffle looks deterministic but reshuffles the whole assignment the moment
    one flight is added to or dropped from the harvest, silently promoting old test flights
    into training on the next retrain. The cost of per-flight hashing is that the realised
    fractions only approximate ``val_fraction`` / ``test_fraction`` (exact in expectation);
    the win is that a flight, once in the test set, stays there for every future harvest with
    the same split seed.
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


class FlightEpochSampler(Sampler[int]):
    """Select one example per flight, then reshuffle the complete epoch."""

    def __init__(self, dataset: TrajectoryWindows, *, seed: int):
        self.dataset = dataset
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset.eligible_series)

    def __iter__(self):
        return iter(self.dataset.epoch_indices(self.seed).tolist())


def iter_batches(
    dataset: TrajectoryWindows,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterator:
    """Yield contiguous in-RAM batches with deterministic index selection."""
    if shuffle:
        sampler = FlightEpochSampler(dataset, seed=seed)
        indices = np.fromiter(sampler, dtype=np.int64, count=len(sampler))
    else:
        indices = np.arange(len(dataset), dtype=np.int64)

    for start in range(0, len(indices), batch_size):
        yield dataset.batch(indices[start : start + batch_size])
