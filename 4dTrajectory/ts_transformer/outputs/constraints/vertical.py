"""The contract's third column, as a constraint module needs to read and write it.

A module that prices a turn needs the vertical lift factor being flown (``n cos μ``); a floor on the
stall speed needs the load factor a held command will fly; a module that moves the bank must keep the
vertical lift the network paired with its command. Under the thrust-fraction, specific-force and
speed-command contracts the third column IS the load factor, so all three are that column. Under
``specific-force+path-angle`` it is a path-angle target: the load is what the path loop resolves at the
state (`resolve_load`, and `held_load` for a command held over a segment), and the loop keeps the
vertical lift at any bank by itself (``RESOLVES_LOAD_AT_FLOWN_BANK``), so no module reads column 2 blind
(two-tier feasibility design §10.8).
"""

from __future__ import annotations

import torch

from aerodynamic_model.torch_lag_dynamics import VERTICAL_ACTUATOR
from aerodynamic_model.torch_transport_chart_dynamics import transport_chart_kinematics
from ts_transformer.config import TSConfig
from ts_transformer.outputs.dynamics.hooks import RolloutStateView
from ts_transformer.outputs.envelope import control_contract


class VerticalChannel:
    """The load factor behind the third column, and the third column that keeps the lift."""

    def __init__(self, config: TSConfig, dynamics: dict[str, torch.Tensor]):
        contract = control_contract(config.control_thrust_parameterization)
        self.law = contract.law
        # the column's own box, a load-factor box wherever `keep_lift` re-coordinates it
        self.column_lower, self.column_upper = contract.lower[VERTICAL_ACTUATOR], contract.upper[VERTICAL_ACTUATOR]
        if self.law.RESOLVES_LOAD_AT_FLOWN_BANK:
            # built once, on the batch's device: a hook is called at every segment
            self.frame_params = dynamics["frame_params"]
            self.parameters = self.law.parameter_matrix(
                dynamics["max_thrust_n"], dynamics["initial_state"],
                dtype=torch.float64, device=dynamics["max_thrust_n"].device,
            )

    def _condition(self, state: RolloutStateView, dtype: torch.dtype):
        return transport_chart_kinematics(state.chart.to(dtype), self.frame_params.to(dtype)).condition

    def flown_load(self, state: RolloutStateView, actuators: torch.Tensor) -> torch.Tensor:
        """``[B]``: the load factor being flown — the law's resolution of the actuators at this state."""
        if not self.law.RESOLVES_LOAD_AT_FLOWN_BANK:
            return actuators[:, VERTICAL_ACTUATOR]      # the law's load IS its third actuator
        return self.law.resolve_load(
            self._condition(state, actuators.dtype), actuators, self.parameters.to(actuators.dtype)
        )

    def held_load(self, state: RolloutStateView, command: torch.Tensor) -> torch.Tensor:
        """``[B]``: the load factor a stall floor prices ``command`` at, held from this state (`held_load`)."""
        if not self.law.RESOLVES_LOAD_AT_FLOWN_BANK:
            return command[:, VERTICAL_ACTUATOR]        # a load command is held as given
        return self.law.held_load(
            self._condition(state, command.dtype), command, self.parameters.to(command.dtype)
        )

    def keep_lift(self, vertical: torch.Tensor, bank: torch.Tensor, new_bank: torch.Tensor) -> torch.Tensor:
        """The third command column once a module moves the bank from ``bank`` to ``new_bank``: the load
        factor re-coordinated to keep ``n cos μ`` (clamped to its box), or the column itself where the
        law keeps the lift at the flown bank."""
        if self.law.RESOLVES_LOAD_AT_FLOWN_BANK:
            return vertical
        return (vertical * torch.cos(bank) / torch.cos(new_bank)).clamp(min=self.column_lower, max=self.column_upper)
