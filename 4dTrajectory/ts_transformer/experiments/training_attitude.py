"""The attitude an aircraft is DRAWN in by the Training views (outline §6; brought back from
`archive/two_tier_v3_2026_10/experiments/training_attitude.py` for A23, the reading unchanged): its heading, path angle
and bank, and a reading of its angle of attack — computed once here and written beside each exported track, so the
viewer computes no attitude. A shared module of the Training export and the backend's live executor, not a runner.

One reading, :func:`attitude`, of a state row ``(lat, lon, alt, V, ψ, γ, m)`` with the bank and load factor in force
(the dynamics' sign: a positive bank turns LEFT), fed from two sources:

- a flight the EXECUTOR flew (`executor_attitude`): its state rows and the commands of the cycle that starts at each
  row — what it flies from there; the track's last row, the cycle that ended there (a later one is another flight's
  of the batch, stepped on after this one was done, or none) — with its own aero row (`FlightInputs.aero_params`);
- an OBSERVED flight (`observed_attitude`): the data plane's own reading of it — `states_from_channels` of its
  `FlightSeries` at the scenario's mass, as the executor starts from and `signals_from_series` reads, and the controls
  `outputs.dynamics.inverse.actual_controls` recovers under the thrust-fraction contract (the teacher's inversion,
  whose bank and load factor are the executor's command columns). A flight the dynamics has no airframe for (C31:
  kept, mass NaN) has its heading and path angle — its states say them — and NO bank or attack: the inversion reads an
  airframe (its drag), and none is guessed; those two fields are null for it.

The angle of attack is a READING: the point-mass dynamics carries a load factor, not an angle. The lift coefficient the
dynamics asks for at that load factor (`torch_dynamics.aerodynamic_coefficients`, capped at ``Cl_max`` as it flies) is
read back through the lift curve of `aerodynamic_model.simulator.Simulator` (``CL0``, ``CL_alpha``) — a clean wing, so on
a flapped final it reads high (≈ 15° at 70 m/s for an A320 at 64 t). The views write it in the label only, never into
the drawn pitch (the user, 2026-09-30).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from aerodynamic_model.simulator import Simulator
from aerodynamic_model.torch_dynamics import aerodynamic_coefficients, isa_density
from ts_transformer.autopilot.flights import rebuild_series
from ts_transformer.config import CONTROL_THRUST_FRACTION
from ts_transformer.data.channels import states_from_channels
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.training_files import rounded
from ts_transformer.instructions.words import compass_from_math_rad
from ts_transformer.outputs.dynamics.context import rollout_context
from ts_transformer.outputs.dynamics.inverse import actual_controls
from ts_transformer.outputs.envelope import CONTROL_NAMES

#: The drawn attitude's fields, one value per track point, in the order :func:`attitude_payload` writes them.
ATTITUDE_FIELDS = ("headingDeg", "pathAngleDeg", "bankRightDeg", "attackDeg")
#: The state row's columns this module reads (`torch_dynamics.STATE_NAMES`).
_ALT, _SPEED, _PSI, _GAMMA, _MASS = 2, 3, 4, 5, 6


def attitude(states: np.ndarray, bank_rad: np.ndarray, load_factor: np.ndarray, aero_params: np.ndarray) -> dict[str, np.ndarray]:
    """``states [M, 7]`` geodetic rows with the ``bank_rad`` (the dynamics' sign) and ``load_factor [M]`` in force, and the
    airframe's ``aero_params [6]`` → the heading (compass degrees), path angle and RIGHT bank (degrees), and the angle of
    attack's reading (degrees, the module docstring), each ``[M]``, keyed by `ATTITUDE_FIELDS`."""
    states = np.asarray(states, dtype=np.float64)
    bank_rad, load_factor = np.asarray(bank_rad, dtype=np.float64), np.asarray(load_factor, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 7 or bank_rad.shape != (len(states),) or load_factor.shape != (len(states),):
        raise ValueError(f"states [M, 7] with a bank and a load factor each: got {states.shape}, {bank_rad.shape}, "
                         f"{load_factor.shape}")
    lift, _drag, _stalled = aerodynamic_coefficients(
        *(torch.as_tensor(values, dtype=torch.float64) for values in (load_factor, states[:, _SPEED], states[:, _MASS])),
        isa_density(torch.as_tensor(states[:, _ALT], dtype=torch.float64)), torch.as_tensor(aero_params, dtype=torch.float64))
    attack_rad = (lift.numpy() - Simulator.CL0) / Simulator.CL_alpha
    return {"headingDeg": compass_from_math_rad(states[:, _PSI]), "pathAngleDeg": np.degrees(states[:, _GAMMA]),
            "bankRightDeg": 0.0 - np.degrees(bank_rad), "attackDeg": np.degrees(attack_rad)}


def attitude_payload(values: dict[str, np.ndarray | None], rows: slice = slice(None)) -> dict[str, list[float] | None]:
    """:func:`attitude` (or `observed_attitude`) at ``rows`` as a Training file writes it, beside its track's points
    (0.01°); a field the flight has none of (no airframe) is null."""
    return {name: None if values[name] is None else rounded(values[name][rows], 2) for name in ATTITUDE_FIELDS}


def executor_attitude(flown: Any, index: int, rows: Sequence[int], aero_params: np.ndarray) -> dict[str, np.ndarray]:
    """Flight ``index`` of an executor's ``flown`` (`autopilot.executor.Flown`) at its state ``rows`` (ascending, the track
    written): each row's state and the commands of the cycle that starts there; the last row — the track's end — the
    cycle that ended there, never one after it (in a batch the executor steps a flight on after it is done)."""
    rows = np.asarray(rows, dtype=np.int64)
    commands = flown.commands[index].cpu().numpy()
    cycles = np.minimum(rows, max(int(rows[-1]) - 1, 0))
    return attitude(flown.states[index].cpu().numpy()[rows], commands[cycles, 1], commands[cycles, 2], aero_params)


def observed_attitude(series: Any) -> dict[str, np.ndarray | None]:
    """An observed flight (a `data.dataset.FlightSeries`) at every one of its rows — the rows `signals_from_series` reads;
    one without an airframe: its heading and path angle only (the module docstring)."""
    states = np.array([[s.latitude, s.longitude, s.altitude, s.V, s.psi, s.gamma, s.m]
                       for _t, s in states_from_channels(series.times, series.values, series.frame,
                                                         mass_kg=float(series.scenario.initial.m))], dtype=np.float64)
    if not series.scenario.has_dynamics:
        return {"headingDeg": compass_from_math_rad(states[:, _PSI]), "pathAngleDeg": np.degrees(states[:, _GAMMA]),
                "bankRightDeg": None, "attackDeg": None}
    context = rollout_context(series, 0)
    controls = actual_controls(states, np.asarray(series.times, dtype=np.float64), aero_params=context["aero_params"],
                               max_thrust_n=float(context["max_thrust_n"]), parameterization=CONTROL_THRUST_FRACTION)
    return attitude(states, controls[:, CONTROL_NAMES.index("bank_rad")], controls[:, CONTROL_NAMES.index("load_factor")],
                    context["aero_params"])


def observed_attitudes(directory: Path, signals: Sequence[FlightSignals]) -> dict[str, dict[str, np.ndarray | None]]:
    """Every flight of ``signals`` (the instruction artefact ``directory``'s) by dataset id: `observed_attitude` of its
    series, rebuilt as the executor's exports rebuild it (`autopilot.flights.rebuild_series`: the same flight, row for
    row, or refused) — its rows are the signals' rows."""
    out = {}
    for flight, series in zip(signals, rebuild_series(directory, signals), strict=True):
        got = observed_attitude(series)
        if len(got["headingDeg"]) != flight.n_rows:
            raise ValueError(f"{flight.dataset_id}: {len(got['headingDeg'])} attitude rows for {flight.n_rows} signal rows")
        out[flight.dataset_id] = got
    return out
