"""One control cycle of the dynamics the executor flies (executor design §2.1–§2.3).

The dynamics are the control path's own point-mass model on the scaled transport chart with its three first-order
actuators (`control_dynamics_model` ``first-order-lag``: thrust fraction, bank, load factor, ``ȧ = (u − a) / τ``), commands
in the thrust-fraction contract (`outputs/envelope.py`), rolled through `outputs.dynamics.rollout.rollout_control_endpoints`
— the lag model wraps the same chart RHS, so this is the point-mass plant with the actuators the design assumes (§2.2,
§9.2: the time constants are the executor's parameters, not identified).

The plant holds no state: each cycle is rolled from the cycle's start state and the ACTUAL actuator state the executor
hands it (what the actuators had reached), and returns both at the cycle's end (the lag by RK4 at the control path's
0.5 s step, not the closed form the observer uses: the two agree to about 1e-4 of the change, T1); the executor never
reads the actual actuators for its laws — it keeps its own observer (§4.2). The end state comes back as a geodetic row and goes in as one;
the two chart conversions between them are exact inverses (`aerodynamic_model.torch_transport_chart_dynamics`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.config import (
    CONTROL_DYNAMICS_FIRST_ORDER_LAG, CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY, CONTROL_THRUST_FRACTION,
    PREDICTION_CONTROL, TSConfig, default_aircraft_filter,
)
from ts_transformer.outputs.dynamics.rollout import rollout_control_endpoints

if TYPE_CHECKING:
    from ts_transformer.autopilot.params import ExecutorParams

#: What every executor cycle shares whatever its time constants: the control path, the thrust contract, the integrator.
EXECUTOR_DYNAMICS = replace(
    TSConfig(),
    prediction_output=PREDICTION_CONTROL,
    # A control config keeps only flights with dynamics (C31); the rollout never reads the filter.
    aircraft_filter=default_aircraft_filter(PREDICTION_CONTROL),
    control_dynamics_model=CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    control_dynamics_backend=CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    control_thrust_parameterization=CONTROL_THRUST_FRACTION,
)


def executor_dynamics(params: ExecutorParams) -> TSConfig:
    """`EXECUTOR_DYNAMICS` with the executor's actuator time constants (refused by the config when its integrator cannot
    hold one, C16)."""
    thrust, bank, load = params.time_constants_s
    return replace(EXECUTOR_DYNAMICS, control_thrust_time_constant_s=thrust, control_bank_time_constant_s=bank,
                   control_load_time_constant_s=load)


class Plant:
    """The batch's airframes and the dynamics; :meth:`step` integrates one cycle."""

    def __init__(self, inputs: FlightInputs, params: ExecutorParams) -> None:
        self.inputs = inputs
        self.dynamics = executor_dynamics(params)

    def step(self, state: torch.Tensor, actuators: torch.Tensor, command: torch.Tensor,
             duration_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        """``state`` ``[B,7]`` geodetic and ``actuators`` ``[B,3]`` (what the thrust fraction, the bank and the load
        factor had reached) under ``command`` ``[B,3]`` held for ``duration_s`` → both at the cycle's end."""
        inputs = self.inputs
        rollout = rollout_control_endpoints(
            command[:, None, :],
            torch.full((len(state), 1), duration_s, dtype=state.dtype, device=state.device),
            {"initial_state": state, "initial_controls": actuators, "aero_params": inputs.aero_params,
             "frame_params": inputs.frame_params, "max_thrust_n": inputs.max_thrust_n},
            self.dynamics,
        )
        return rollout.geodetic_states[:, -1], rollout.actual_controls[:, -1]
