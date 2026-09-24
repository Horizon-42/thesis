"""A labelled flight's inputs, rebuilt from the data plane (executor design §2.4).

The artefact stores each flight's signals, not its dynamics. The aircraft's aero row, installed
thrust, mass and the chart the rollout integrates in come from the data plane's own `FlightSeries`,
rebuilt here with the call that built the signals (`data.dataset.build_series` under the default
`TSConfig`, from the arrival manifest the artefact recorded — refused if the manifest moved since).
The rebuilt flight must reproduce the stored signals row for row and name the same runway and
aircraft (`instructions.signals.signals_from_series`), and the build configuration must be the one the
signals recorded, or it is refused: an executor flown from another flight — or another airframe — than
the one the sentence was read off answers nothing.

The executor starts at the flight's first row: the state is the signals' row 0, and the flight's
physical context is `outputs.dynamics.context.rollout_context` at anchor 0 — the reading of a series
every control rollout makes (`dynamics_arrays` is it plus what a control model reads).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ts_transformer.config import TSConfig
from ts_transformer.data.dataset import FlightSeries, build_series, load_flight_dicts
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import load_candidates
from ts_transformer.instructions.signals import ROW_FIELDS, FlightSignals, signals_from_series
from ts_transformer.io_utils import file_sha256
from ts_transformer.outputs.dynamics.context import rollout_context
from ts_transformer.repo_layout import arrival_manifest_path

#: How closely a rebuilt flight must reproduce the stored signals (the same code on the same
#: manifest reproduces them exactly; the tolerance absorbs only floating-point reassociation).
SIGNAL_TOLERANCE = 1e-6


def rebuild_series(directory: Path, signals: Sequence[FlightSignals]) -> list[FlightSeries]:
    """The `FlightSeries` behind each of ``signals``, in the same order, each checked to be the flight the
    signals were read from (`require_same_flight`)."""
    record = json.loads((directory / "signals.json").read_text(encoding="utf-8"))
    recorded = {source["airport"]: source["arrival_manifest_sha256"] for source in record["sources"]}
    config = TSConfig()
    changed = {name: (value, getattr(config, name)) for name, value in record["config"].items()
               if getattr(config, name) != value}
    if changed:
        raise ValueError(f"the build configuration moved since the signals were read: {changed}")
    geometries = load_candidates(directory)
    built: dict[str, FlightSeries] = {}
    for airport in sorted({item.airport for item in signals}):
        manifest = arrival_manifest_path(airport)
        if file_sha256(manifest) != recorded[airport]:
            raise ValueError(f"{manifest} is not the manifest the artefact's signals were read from "
                             f"(sha256 {recorded[airport][:12]} recorded)")
        keys = {item.dataset_id for item in signals if item.airport == airport}
        flights = load_flight_dicts([manifest], include_flight_keys=keys, verbose=False)
        series, _report = build_series(flights, config, aircraft_type=config.aircraft_type)
        built.update({item.dataset_id: item for item in series})
    missing = [item.dataset_id for item in signals if item.dataset_id not in built]
    if missing:
        raise ValueError(f"{len(missing)} flight(s) did not rebuild, e.g. {missing[:3]}")
    for item in signals:
        require_same_flight(built[item.dataset_id], item, geometries[item.airport])
    return [built[item.dataset_id] for item in signals]


def require_same_flight(series: FlightSeries, signals: FlightSignals, geometry: AirportGeometry) -> None:
    """The rebuilt flight reproduces the stored signals: the same runway and aircraft, every row of
    every field."""
    again = signals_from_series(series, geometry)
    for name in ("runway", "typecode"):
        if getattr(again, name) != getattr(signals, name):
            raise ValueError(f"{signals.dataset_id}: the rebuilt flight's {name} is {getattr(again, name)!r}, "
                             f"the stored signals' {getattr(signals, name)!r}")
    for name in ROW_FIELDS:
        stored, rebuilt = getattr(signals, name), getattr(again, name)
        if stored.shape != rebuilt.shape or not np.allclose(stored, rebuilt, rtol=0.0, atol=SIGNAL_TOLERANCE):
            raise ValueError(f"{signals.dataset_id}: the rebuilt flight's {name} differs from the stored signals")


@dataclass(frozen=True)
class FlightInputs:
    """The batch's physical context, float64, one row per flight."""

    initial_state: torch.Tensor     # [B,7] geodetic, the flight's row 0
    aero_params: torch.Tensor       # [B,6]
    frame_params: torch.Tensor      # [B,4] the chart the rollout integrates in
    max_thrust_n: torch.Tensor      # [B]


def flight_inputs(series: Sequence[FlightSeries], *, device: torch.device) -> FlightInputs:
    """`rollout_context` at each flight's first row."""
    rows = [rollout_context(item, 0) for item in series]

    def stack(key: str) -> torch.Tensor:
        return torch.as_tensor(np.stack([row[key] for row in rows]), dtype=torch.float64, device=device)

    return FlightInputs(initial_state=stack("initial_state"), aero_params=stack("aero_params"),
                        frame_params=stack("frame_params"), max_thrust_n=stack("max_thrust_n"))
