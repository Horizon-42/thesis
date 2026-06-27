import math
import numpy as np
from aircraft.aircraft_sets import AIRCRAFT_PRESETS, AircraftSpec, A320
try:
    from .common import State, Atmosphere, rk4_step, Control
except ImportError:  # pragma: no cover - compatibility for flat script imports.
    from common import State, Atmosphere, rk4_step, Control


class Simulator:
    aircraft: AircraftSpec
    S: float # wing area in m^2, set by aircraft config

    # Alpha max
    alpha_max_rad: float = math.radians(18.0) # maximum angle of attack in radians, can be tuned based on aircraft type

    g: float = 9.81 # gravitational acceleration in m/s^2

    # Lift coefficients, these can be tuned or replaced with more complex models
    CL0: float = 0.2 # zero angle of attack lift coefficient
    CL_alpha: float = 5.7 # lift curve slope in per radian, typical for subsonic airfoils, can be tuned based on aircraft type

    # Drag coefficients, these can be tuned or replaced with more complex models
    Cd0: float = 0.02 # zero-lift drag coefficient
    k: float = 0.04 # induced drag factor

    def __init__(self, aircraft: AircraftSpec = A320):
        self.aircraft = aircraft
        self.S = aircraft.wing_area_m2

    def _get_lift_coefficient(self, attack_angle: float) -> float: 
        # need to consider stall behavior
        return self.CL0 + self.CL_alpha * attack_angle

    def _get_drag_coefficient(self, lift_coefficient: float) -> float: 
        return self.Cd0 + self.k * lift_coefficient**2

    def get_aerodynamic_coefficients(self, attack_angle: float) -> tuple:
        Cl = self._get_lift_coefficient(attack_angle)
        Cd = self._get_drag_coefficient(Cl)
        return Cl, Cd
    
    def get_aerodynamic_forces(self, V: float, attack_angle: float, atmosphere: Atmosphere, h: float) -> tuple:
        rho = atmosphere.get_ISA_density(h)
        Cl, Cd = self.get_aerodynamic_coefficients(attack_angle)
        L = 0.5 * rho * V**2 * Cl * self.S
        D = 0.5 * rho * V**2 * Cd * self.S
        return L, D
    
    def get_load_factor(self, L: float, m: float) -> float:
        return L / (m * self.g)

    def dynamics(self, t, state_vec, control: Control, atmosphere: Atmosphere):
        state_vec = np.asarray(state_vec, dtype=float)

        # Unpack state vector
        x, y, h, V, psi, gamma, m = state_vec

        T, phi, alpha = control.thrust, control.bank_rad, control.attack_rad

        L, D = self.get_aerodynamic_forces(V, control.attack_rad, atmosphere, h)

        # Compute state derivatives
        dxdt = V * math.cos(gamma) * math.cos(psi)
        dydt = V * math.cos(gamma) * math.sin(psi)
        dhdt = V * math.sin(gamma)
        dVdt = (T * math.cos(alpha) - D) / m - self.g * math.sin(gamma)
        dpsidt = (L + T * math.sin(alpha)) * math.sin(phi) / (m * V * math.cos(gamma))
        dgamadt = ((L+ T * math.sin(alpha))*math.cos(phi) - m * self.g * math.cos(gamma)) / (m * V)
        dmdt = 0

        return np.array([dxdt, dydt, dhdt, dVdt, dpsidt, dgamadt, dmdt], dtype=float)

    def simulate(
        self,
        initial_state: State,
        control: Control,
        atmosphere: Atmosphere,
        t: float,
        dt: float,
    ) -> State:
        """
        Advance aircraft dynamics by one fixed RK4 step.

        t is the current simulation time passed to the dynamics model. The
        current dynamics are time-invariant, but keeping t in the API makes the
        integrator usable if time-varying winds or controls are added later.
        dt is the fixed integration step in seconds.
        """
        state_vec = np.array([
            initial_state.x,
            initial_state.y,
            initial_state.h,
            initial_state.V,
            initial_state.psi,
            initial_state.gamma,
            initial_state.m,
        ], dtype=float)

        next_state = rk4_step(
            self.dynamics,
            t,
            state_vec,
            control,
            atmosphere,
            dt,
        )

        return State(*next_state.tolist())
