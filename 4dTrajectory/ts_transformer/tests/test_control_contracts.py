"""The control contract registry (`outputs/envelope.py`) and the config scope table
(`config.CONTROL_PARAMETERIZATION_SCOPES`): one row per `control_thrust_parameterization` on each
side, and every consumer reads a row instead of comparing the value.

What this pins is the STRUCTURE the refactor of 2026-09-16 bought
(`docs/2026-09-16_two_tier_transformer_feasibility.zh.md` §11): the two tables name the same
values, a row is internally consistent (its law flies the columns its names say, the record
carries a command exactly where the newton column is the law's resolution of it), and the
identities a stored checkpoint carries are spelled verbatim. The laws' physics and each
contract's behaviour are pinned in their own files (`test_specific_force.py`,
`test_speed_command.py`, `test_path_angle.py`, `aerodynamic_model/tests/test_torch_lag_*`).
"""

from __future__ import annotations

import pytest

from aerodynamic_model.torch_dynamics import CONTROL_NAMES as NEWTON_CONTROL_NAMES
from aerodynamic_model.torch_lag_dynamics import (
    PathAngleLaw,
    SpecificForceLaw,
    SpeedCommandLaw,
    ThrustFractionLaw,
)
from ts_transformer.config import (
    CONTROL_HOOK_SPEED_FLOOR,
    CONTROL_PARAMETERIZATION_SCOPES,
    CONTROL_SPECIFIC_FORCE,
    CONTROL_SPECIFIC_FORCE_PATH_ANGLE,
    CONTROL_SPEED_COMMAND,
    CONTROL_THRUST_FRACTION,
    CONTROL_THRUST_PARAMETERIZATIONS,
)
from ts_transformer.inference.export import CONTROL_SEGMENT_FIELDS
from ts_transformer.outputs.constraints import speed_floor
from ts_transformer.outputs.control.training.diagnostics import saturation_labels
from ts_transformer.outputs.envelope import control_contract

#: What each value's law is, and which of its columns are NOT the newton record's own quantity.
EXPECTED = {
    CONTROL_THRUST_FRACTION: (ThrustFractionLaw, ()),
    CONTROL_SPECIFIC_FORCE: (SpecificForceLaw, (0,)),
    CONTROL_SPEED_COMMAND: (SpeedCommandLaw, (0,)),
    CONTROL_SPECIFIC_FORCE_PATH_ANGLE: (PathAngleLaw, (0, 2)),
}


def test_the_registry_and_the_scope_table_name_the_same_values():
    assert tuple(CONTROL_PARAMETERIZATION_SCOPES) == CONTROL_THRUST_PARAMETERIZATIONS
    assert set(EXPECTED) == set(CONTROL_THRUST_PARAMETERIZATIONS)
    for value in CONTROL_THRUST_PARAMETERIZATIONS:
        assert control_contract(value).parameterization == value
    with pytest.raises(ValueError, match="no control contract"):
        control_contract("newtons")


@pytest.mark.parametrize("value", CONTROL_THRUST_PARAMETERIZATIONS)
def test_a_row_is_consistent(value):
    contract = control_contract(value)
    law_type, command_columns = EXPECTED[value]
    assert type(contract.law) is law_type
    assert contract.record_command_columns == command_columns
    # the record writes a command under its contract name, never over one of the segment's own fields
    assert all(contract.names[column] not in CONTROL_SEGMENT_FIELDS for column in command_columns)
    # the bank column is the same everywhere; only the first and third may change meaning
    assert contract.names[1] == "bank_rad" and contract.units[1] == "rad"
    # the law's engine floor is the thrust-fraction box's, so every law admits the same thrusts
    if law_type is not ThrustFractionLaw:
        assert contract.law.min_thrust_fraction == control_contract(CONTROL_THRUST_FRACTION).lower[0]
    # the default is never spelled; every other value is, and differently
    scope = CONTROL_PARAMETERIZATION_SCOPES[value]
    assert (scope.slug == "") == (value == CONTROL_THRUST_FRACTION)
    # only the default flies on the point-mass rows or reads the fitted teacher
    assert scope.point_mass == scope.fitted_teacher == (value == CONTROL_THRUST_FRACTION)


def test_slugs_are_distinct():
    slugs = [scope.slug for scope in CONTROL_PARAMETERIZATION_SCOPES.values() if scope.slug]
    assert len(slugs) == len(set(slugs)) == len(CONTROL_THRUST_PARAMETERIZATIONS) - 1


def test_the_stored_checkpoint_identities_are_spelled_verbatim():
    """A changed suffix refuses every stored checkpoint of the contract at load; an added one
    refuses the contracts that never had one (N3's specific force, every thrust-fraction run)."""
    assert control_contract(CONTROL_THRUST_FRACTION).identity_suffix == ""
    assert control_contract(CONTROL_SPECIFIC_FORCE).identity_suffix == ""
    assert control_contract(CONTROL_SPEED_COMMAND).identity_suffix == (
        "+speed-command(tau-v=8s,box=-90..20m/s,neutral=0)-v1"
    )
    assert control_contract(CONTROL_SPECIFIC_FORCE_PATH_ANGLE).identity_suffix == (
        "+specific-force+path-angle(tau-gamma=3s,box=-15..10deg,neutral=-2.9deg)-v1"
    )


def test_the_thrust_fraction_saturation_labels_stay_the_stored_ones():
    assert saturation_labels(CONTROL_THRUST_FRACTION) == NEWTON_CONTROL_NAMES == ("thrust_N", "bank_rad", "load_factor")
    for value in CONTROL_THRUST_PARAMETERIZATIONS:
        if value != CONTROL_THRUST_FRACTION:
            assert saturation_labels(value) == control_contract(value).names


def test_every_contract_the_speed_floor_is_admitted_on_has_an_inversion():
    for value, scope in CONTROL_PARAMETERIZATION_SCOPES.items():
        if CONTROL_HOOK_SPEED_FLOOR in scope.hook_modules:
            assert control_contract(value).longitudinal in speed_floor._FLOORS, value
    assert CONTROL_HOOK_SPEED_FLOOR not in CONTROL_PARAMETERIZATION_SCOPES[CONTROL_SPEED_COMMAND].hook_modules
    # the path-angle contract's modules read the load through its law (two-tier design §10.8)
    assert set(CONTROL_PARAMETERIZATION_SCOPES[CONTROL_SPECIFIC_FORCE_PATH_ANGLE].hook_modules) == set(
        CONTROL_PARAMETERIZATION_SCOPES[CONTROL_SPECIFIC_FORCE].hook_modules)
