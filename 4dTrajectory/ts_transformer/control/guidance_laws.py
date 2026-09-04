"""Fixed tracking laws on the final approach: the nominal controller a hook builds on.

Lateral: L1 guidance (Park, Deyst, How, AIAA GNC 2004) toward the runway centreline —
the desired heading points at a reference point ``L1`` metres ahead on the centreline, and
the lateral acceleration ``2 V² / L1 · sin η`` (η = heading error to that point) is turned
into a bank angle through the level-turn relation ``tan μ = a / g``.

Vertical: a flight-path-angle law toward the glidepath — the desired path angle is the
glidepath angle steepened or shallowed by the height error over a lookahead distance, and
the load factor that produces the needed path-angle rate follows from the point-mass
equation ``γ̇ = g (n cos μ − cos γ) / V``.

Conventions match ``final_approach_geometry`` and the dynamics: cross-track ``xt`` is
positive RIGHT of the inbound course; heading ψ is math-ENU (0 = east, CCW), and a
positive bank turns ψ POSITIVE (``ψ̇ = g n sin μ / (V cos γ)``), so returning from the right
of the course (xt > 0) needs a positive heading error and a positive bank. All functions
are plain torch arithmetic on ``[B]`` tensors and differentiable.
"""

from __future__ import annotations

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap to ``(−π, π]``."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def l1_bank(
    cross_track_m: torch.Tensor,
    heading_error_rad: torch.Tensor,
    ground_speed_mps: torch.Tensor,
    *,
    l1_distance_m: float,
    bank_limit_rad: float,
) -> torch.Tensor:
    """Bank that steers onto the centreline: L1 lateral guidance, saturated at the limit."""
    desired_error = torch.atan(cross_track_m / l1_distance_m)
    eta = wrap_angle(desired_error - heading_error_rad)
    lateral_acceleration = 2.0 * ground_speed_mps.square() / l1_distance_m * torch.sin(eta)
    bank = torch.atan(lateral_acceleration / GRAVITY_MPS2)
    return bank.clamp(min=-bank_limit_rad, max=bank_limit_rad)


def glidepath_load_factor(
    height_error_m: torch.Tensor,
    path_angle_rad: torch.Tensor,
    speed_mps: torch.Tensor,
    bank_rad: torch.Tensor,
    *,
    glidepath_tan: torch.Tensor,
    lookahead_m: float,
    gain_per_s: float | torch.Tensor,
    load_limits: tuple[float, float],
) -> torch.Tensor:
    """Load factor that steers the path angle toward the glidepath, saturated at the box.

    ``height_error_m`` is the height ABOVE the glidepath (positive = high), so the desired
    path angle is the glidepath angle made steeper when high and shallower when low.
    ``gain_per_s`` may be a ``[B]`` tensor (a caller holding the command for Δt caps it at
    ``1/Δt`` so one hold does not overshoot the desired path angle).
    """
    desired = -torch.atan(glidepath_tan) - torch.atan(height_error_m / lookahead_m)
    path_angle_rate = gain_per_s * wrap_angle(desired - path_angle_rad)
    load = (torch.cos(path_angle_rad) + speed_mps * path_angle_rate / GRAVITY_MPS2) / torch.cos(bank_rad)
    return load.clamp(min=load_limits[0], max=load_limits[1])
