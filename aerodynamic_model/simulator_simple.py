# Simple Aerodynamic Simulator, with load factor, assume alpha = 0
from dataclasses import dataclass
import numpy as np
import math
from aircraft_sets import AIRCRAFT_PRESETS, AircraftSpec, A320
from common import State, Atmosphere, rk4_step

@dataclass
class LoadFactorControl:
    thrust: float
    bank_rad: float # roll angle / bank angle in radians, mu
    load_factor: float # load factor, n

class LoadFactorSimulator:
    aircraft: AircraftSpec

    S: float # wing area in m^2, set by aircraft config

    g: float = 9.81 # gravitational acceleration in m/s^2

    # Lift coefficients
    Cl_max: float = 1.5 # maximum lift coefficient, can be tuned based on aircraft type

    # Drag coefficients, these can be tuned or replaced with more complex models
    Cd0: float = 0.02 # zero-lift drag coefficient
    k: float = 0.04 # induced drag factor

    # Stall parameters
    stall_threshold: float = 0.9 # threshold for stall onset, as a fraction of Cl_max
    K_stall: float = 0.1 # stall drag coefficient factor, can be tuned based on aircraft type

    def __init__(self, aircraft: AircraftSpec = A320):
        self.aircraft = aircraft
        self.S = aircraft.wing_area_m2
    
    def _get_Cl_required(self, load_factor: float, speed: float) -> float:
        return load_factor * self.aircraft.mass_kg * self.g / (0.5 * self.S * speed**2)

    def _get_drag_coefficient(self, Cl: float, Cd_stall: float = 0.0) -> float:
        return self.Cd0 + self.k * Cl**2 + Cd_stall

    def get_aerodynamic_coefficients(self, load_factor: float) -> tuple:
        Cl_required = self._get_Cl_required(load_factor, self.aircraft.cruise_speed_mps)

        # check stall and compute stall drag 
        Cd_stall = 0.0
        r = Cl_required / self.Cl_max
        Cl = min(Cl_required, self.Cl_max)
        stalled = False
        if r > self.stall_threshold:
            stalled = True
            # TODO get reference for stall drag models, this is a simple smooth transition to high drag as we approach stall, can be tuned based on aircraft type
            x = min(max((r - self.stall_threshold) / (1.0 - self.stall_threshold), 0.0), 1.0)
            smooth = x * x * (3 - 2 * x)
            Cd_stall = self.K_stall * smooth # simple linear stall drag model, can be tuned
        Cd = self._get_drag_coefficient(Cl, Cd_stall)
        return Cl, Cd, stalled
    
    def dynamics(self, t, state_vec, control: LoadFactorControl, atmosphere: Atmosphere):
        state_vec = np.asarray(state_vec, dtype=float)

        # Unpack state vector
        x, y, h, V, psi, gamma, m = state_vec

        # unpack control
        T, mu, n_cmd = control.thrust, control.bank_rad, control.load_factor

        # Get aerodynamic coefficients
        Cl, Cd, stalled = self.get_aerodynamic_coefficients(n_cmd)

        # Get atmospheric density
        rho = atmosphere.get_ISA_density(h)

        # n load factor, consider stall
        n = n_cmd
        if stalled:
            n = 0.5 * rho * V**2 * self.Cl_max * self.S / (m * self.g) # compute actual load factor based on current lift, which is reduced due to stall

        # Get aerodynamic forces
        L = n * m * self.g
        D = 0.5 * rho * V**2 * Cd * self.S

        # Compute state derivatives
        dxdt = V * math.cos(psi) * math.cos(gamma)
        dydt = V * math.sin(psi) * math.cos(gamma)
        dhdt = V * math.sin(gamma)
        dVdt = (T - D) / m - self.g * math.sin(gamma)
        dpsidt = self.g*n * math.sin(mu) / (V * math.cos(gamma))
        dgamadt = self.g * (n * math.cos(mu) - math.cos(gamma)) / V
        dmdt = 0.0 # assume mass is constant

        return np.array([dxdt, dydt, dhdt, dVdt, dpsidt, dgamadt, dmdt], dtype=float)

