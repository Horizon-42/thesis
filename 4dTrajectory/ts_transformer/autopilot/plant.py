"""One control cycle of the dynamics the executor flies (executor design §2.1–§2.3).

The dynamics are the control path's own: the point-mass model on the scaled transport chart, commands
in the thrust-fraction contract (`outputs/envelope.py`: thrust as a fraction of the installed thrust,
bank in radians, load factor) — the pair the `simple-v3` recipe names, rolled through
`outputs.dynamics.rollout.rollout_control_endpoints`.

That backend runs no command hooks (only the first-order-lag rows do: `backends.ControlDynamicsBackend.
runs_hooks`), so the executor does not sit INSIDE a rollout as a hook; it rolls one cycle at a time
and hands each cycle's end state to the next. That is the same computation: a hook is also called
once per segment with the segment's start state, and the integrator restarts at every segment
boundary either way. The end state comes back as a geodetic row and goes in as one; the two chart
conversions between them are exact inverses (`aerodynamic_model.torch_transport_chart_dynamics`).
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.config import (
    CONTROL_DYNAMICS_POINT_MASS, CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY, CONTROL_THRUST_FRACTION,
    PREDICTION_CONTROL, TSConfig,
)
from ts_transformer.outputs.dynamics.rollout import rollout_control_endpoints

#: The dynamics every executor cycle is integrated with.
EXECUTOR_DYNAMICS = replace(
    TSConfig(),
    prediction_output=PREDICTION_CONTROL,
    control_dynamics_model=CONTROL_DYNAMICS_POINT_MASS,
    control_dynamics_backend=CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    control_thrust_parameterization=CONTROL_THRUST_FRACTION,
)


class Plant:
    """The batch's airframes and charts; :meth:`step` integrates one cycle."""

    def __init__(self, inputs: FlightInputs) -> None:
        self.inputs = inputs

    def step(self, state: torch.Tensor, command: torch.Tensor, duration_s: float) -> torch.Tensor:
        """``state`` ``[B,7]`` geodetic, ``command`` ``[B,3]`` (thrust fraction, bank, load factor)
        held for ``duration_s`` → the geodetic state at the cycle's end."""
        inputs = self.inputs
        # ``initial_controls`` is a lagged model's actuator state; the point-mass rows never read it
        assert EXECUTOR_DYNAMICS.control_dynamics_model == CONTROL_DYNAMICS_POINT_MASS
        rollout = rollout_control_endpoints(
            command[:, None, :],
            torch.full((len(state), 1), duration_s, dtype=state.dtype, device=state.device),
            {"initial_state": state, "initial_controls": command, "aero_params": inputs.aero_params,
             "frame_params": inputs.frame_params, "max_thrust_n": inputs.max_thrust_n},
            EXECUTOR_DYNAMICS,
        )
        return rollout.geodetic_states[:, -1]
