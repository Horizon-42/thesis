from dataclasses import dataclass
import math
from scipy.integrate import solve_ivp
from aircraft_sets import AIRCRAFT_PRESETS, AircraftSpec, A320

@dataclass
class State:
    x: float
    y: float
    h: float
    V: float
    psi: float
    gamma: float
    m: float

@dataclass
class Control:
    thrust: float
    bank_rad: float # roll angle / bank angle in radians, mu
    attack_rad: float # angle of attack in radians, alpha

class Atmosphere:
    rho0: float = 1.225 # sea level density in kg/m^3
    H: float = 8500.0 # scale height in meters

    def __init__(self, rho0: float=1.225, H: float=8500.0):
        self.rho0 = rho0
        self.H = H
    
    def get_density(self, h: float) -> float:
        return self.rho0 * math.exp(-h / self.H)
    
    def get_ISA_temperature(self, h: float) -> float:
        # Simplified ISA temperature model
        T0 = 288.15 # sea level standard temperature in K
        L = 0.0065 # temperature lapse rate in K/m
        return T0 - L * h
    
    def get_ISA_density(self, h_atm: float) -> float:
        # must use h for atmospheric properties, not altitude above ground level
        T = self.get_ISA_temperature(h_atm)
        return self.rho0 * (T / self.get_ISA_temperature(0))**4.25588

class Simulator:
    aircraft: AircraftSpec
    S: float # wing area in m^2, set by aircraft config

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

    def _get_lift_coefficient(self, attack_angle: float) -> float: return self.CL0 + self.CL_alpha * attack_angle

    def _get_drag_coefficient(self, lift_coefficient: float) -> float: return self.Cd0 + self.k * lift_coefficient**2

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

        return [dxdt, dydt, dhdt, dVdt, dpsidt, dgamadt, dmdt]

    def simulate(self, initial_state: State, control: Control, atmosphere: Atmosphere, t_span: tuple, t_eval: list):
        # Convert initial state to vector form
        state_vec = [initial_state.x, initial_state.y, initial_state.h, initial_state.V, initial_state.psi, initial_state.gamma, initial_state.m]

        # Solve the ODEs using solve_ivp
        sol = solve_ivp(lambda t, X: self.dynamics(t, X, control, atmosphere), t_span, state_vec, t_eval=t_eval)

        return sol
