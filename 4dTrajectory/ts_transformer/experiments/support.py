"""What every runner shares: where the repository is, and the helpers five of them carried as
byte-identical copies (review §4.5). The paths are `repo_layout`'s and `parse_airports` is
`cli.common`'s (the benchmark CLI reads it too), both re-exported here so a runner has one
import; `series_digest` lives here; `write_json_atomic` / `file_sha256` are `io_utils`'s —
the private copies are gone; `forecast_geometry` is the time-free reading of one forecast
against its truth that the anytime curve and the plan oracle both take."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Sequence

import numpy as np

from pathlib import Path
from typing import Any

from ts_transformer.cli.common import parse_airports
from ts_transformer.config import TSConfig, recipe_settings
from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.data.data_provenance import checkpoint_data_provenance, require_matching_data_provenance
from ts_transformer.data.dataset import build_series, load_flight_dicts
import ts_transformer.geometry.geometric_metrics as gm
from ts_transformer.repo_layout import REPO_ROOT, TS_DIR, TS_SCRIPT, arrival_manifest_path
from ts_transformer.training.train import usable_series

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries
    from ts_transformer.inference.forecast import Forecast

__all__ = [
    "EXPERIMENTS_MAIN", "REPO_ROOT", "RUN_TS", "TS_DIR", "TS_SCRIPT",
    "arm_config", "checkpoint_manifests", "cohort_from_manifests", "declaration_base", "forecast_geometry",
    "parse_airports", "rebuild_cohort", "series_digest",
]

#: The runners' own entry point, for a runner that spawns another runner.
EXPERIMENTS_MAIN = TS_DIR / "experiments" / "__main__.py"
RUN_TS = REPO_ROOT / "run_ts.py"


def declaration_base(declaration: dict[str, Any]) -> dict[str, Any]:
    """The settings every arm of an arm declaration starts from: ``"base"``, laid over a named
    ``"base_recipe"``'s content when one is named (the recipe keeps its name, so an arm may only
    touch the fields it leaves open — TSConfig refuses the rest)."""
    base = dict(declaration.get("base", {}))
    base_recipe = declaration.get("base_recipe")
    if base_recipe:
        base = {**recipe_settings(base_recipe, keep_name=True), **base}
    return base


def arm_config(base: dict[str, Any], overrides: dict[str, Any]) -> tuple[TSConfig, dict[str, Any]]:
    """One arm's complete settings and the config they construct: ``(config, settings)``.

    The config is CONSTRUCTED on every read, dry or not — an unrunnable arm must fail here,
    before anything is trained or written, and finding that out is most of what a dry run is for.
    """
    settings = dict(base)
    settings.update(overrides)
    config = TSConfig(**settings)  # validates: an unrunnable arm fails here, before training
    return config, settings


def checkpoint_manifests(payload: dict[str, Any]) -> list[Path]:
    """The arrival manifests a checkpoint's own provenance names, in its order."""
    return [arrival_manifest_path(entry["airport"]) for entry in payload["data_provenance"]["manifests"]]


def cohort_splits(payload: dict[str, Any], cohort, limit: int = 0) -> dict[str, list[str]]:
    """The development cohort's train and val rosters (`--limit`: a PREFIX of each, a smoke
    test), each refused unless the executor checkpoint's SAME split holds every flight of it —
    the cohort is a subset of the executor's own population, never another one."""
    splits = {"train": list(cohort.train_flight_ids), "val": list(cohort.val_flight_ids)}
    if limit:
        splits = {name: keys[:limit] for name, keys in splits.items()}
    for name, keys in splits.items():
        held = set(payload["split"][name])
        missing = [key for key in keys if key not in held]
        if missing:
            raise ValueError(f"{len(missing)} {name} flight(s) of the cohort are not in the executor's {name} split (first {missing[0]!r})")
    return splits


def rebuild_cohort(payload: dict[str, Any], config: TSConfig, keys: Sequence[str]) -> list[FlightSeries]:
    """The checkpoint's flights ``keys`` rebuilt under ``config`` in the given order, the
    provenance verified against today's manifests first (C25: through
    `checkpoint_data_provenance`); a flight that cannot be rebuilt refuses the run — a readout
    over a silent subset is a different cohort. The one cohort door of the manoeuvre runners."""
    manifests = checkpoint_manifests(payload)
    require_matching_data_provenance(payload, checkpoint_data_provenance(payload, manifests))
    built, report = build_series(
        load_flight_dicts(manifests, include_flight_keys=set(keys), verbose=False), config, aircraft_type=config.aircraft_type,
    )
    print(f"  {report.format()}", flush=True)
    by_id = {item.dataset_id: item for item in usable_series(built, config, verbose=False)}
    missing = [key for key in keys if key not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(keys)} checkpoint flights could not be rebuilt (first: {missing[0]!r})")
    return [by_id[key] for key in keys]


def cohort_from_manifests(airports: Sequence[str], cohort, config: TSConfig, limit: int = 0) -> tuple[list[FlightSeries], dict[str, list[str]], dict[str, Any]]:
    """The cohort rebuilt STRAIGHT FROM the arrival manifests, plus its splits and its provenance.

    The other door (`rebuild_cohort`) goes through an executor checkpoint, which carries both the
    data provenance and the config. A POOLED cohort has no executor — none was ever trained on it —
    so this door reads the manifests themselves and records their digests in the checkpoint's place.
    The caller passes the ``config`` the series are built under and says so in its own output, and
    ``limit`` takes a PREFIX of each split (a smoke test) before the build rather than after it.

    NO `usable_series` here, deliberately: that filter drops flights too short to yield one TRAINING
    WINDOW, which is the executor's requirement. Reading a sentence, or training a prior over one,
    needs neither a lookback nor truth after an anchor, so importing the executor's requirement would
    shrink the population for a reason that has nothing to do with the reading.

    `build_series` itself skips a track shorter than one window — the data layer's own contract, not
    a filter applied here. Those flights are NAMED in the returned provenance rather than silently
    missing, and more than a percent of them means the cohort file and this config disagree about the
    population, which is a different cohort, not an exclusion.
    """
    manifests = [arrival_manifest_path(code.upper()) for code in airports]
    provenance: dict[str, Any] = {
        "read_from": "arrival manifests", "airports": [code.upper() for code in airports],
        "manifests": [str(path) for path in manifests],
        "manifest_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in manifests],
    }
    splits = {"train": list(cohort.train_flight_ids), "val": list(cohort.val_flight_ids)}
    if limit:                                   # a PREFIX of each split, as `cohort_splits` means it —
        splits = {name: keys[:limit] for name, keys in splits.items()}   # applied BEFORE the build, which is the slow part
    keys = [*splits["train"], *splits["val"]]
    built, report = build_series(load_flight_dicts(manifests, include_flight_keys=set(keys), verbose=False),
                                 config, aircraft_type=config.aircraft_type)
    print(f"  {report.format()}", flush=True)
    by_id = {item.dataset_id: item for item in built}
    missing = [key for key in keys if key not in by_id]
    if missing:
        share = len(missing) / len(keys)
        print(f"  {len(missing)} of {len(keys)} cohort flights ({share:.2%}) were skipped by the series "
              f"builder (track shorter than one window); first {missing[0]!r}", flush=True)
        if share > 0.01:
            raise SystemExit(f"{share:.2%} of the cohort cannot be built: that is a different cohort")
        provenance["excluded_short_track"] = missing
        splits = {name: [key for key in ids if key in by_id] for name, ids in splits.items()}
        keys = [*splits["train"], *splits["val"]]
    return [by_id[key] for key in keys], splits, provenance


def series_digest(series: Sequence[FlightSeries]) -> str:
    """The cohort's identity for a runner's resume check: sha256 over the sorted dataset ids."""
    payload = "\n".join(sorted(item.dataset_id for item in series)).encode()
    return hashlib.sha256(payload).hexdigest()


def forecast_geometry(series: FlightSeries, forecast: Forecast) -> dict[str, float]:
    """The time-free metrics for one forecast against the truth AFTER its anchor.

    Both paths are ``[N, 4]`` ``(e, n, u, t)`` in the flight's own chart, which is all
    `geometric_metrics` needs — chamfer and Fréchet are relative, so the chart origin
    (threshold or airport) does not enter.
    """
    anchor_time = float(series.times[forecast.anchor])
    future = series.supervision_times > anchor_time
    truth = np.column_stack([
        np.asarray(series.supervision_values, dtype=np.float64)[future][:, list(POSITION_IDX)],
        np.asarray(series.supervision_times, dtype=np.float64)[future] - anchor_time,
    ])
    predicted = np.column_stack([
        np.asarray(forecast.values, dtype=np.float64)[:, list(POSITION_IDX)],
        np.cumsum(forecast.sample_durations_s),
    ])
    return gm.path_metrics(predicted, truth)
