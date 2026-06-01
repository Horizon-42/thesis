from dataclasses import dataclass
import math
from scipy.integrate import solve_ivp

@dataclass(frozen=True)
class ReferenceArea:
    NarrowBody_S: float = 122.6 # m^2, reference area for narrow-body aircraft
    WideBody_S: float = 300.0 # m^2, reference area for wide-body aircraft
    GeneralAviation_S: float = 16.2 # m^2, reference area for general aviation aircraft

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
    load_factor: float # n

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
    g: float = 9.81 # gravitational acceleration in m/s^2

    S:float = ReferenceArea.NarrowBody_S # reference area for drag calculation, default to narrow-body

    Cd0: float = 0.02 # zero-lift drag coefficient
    k: float = 0.04 # induced drag factor

    def __init__(self, S: float = ReferenceArea.NarrowBody_S):
        self.S = S

    def _get_lift_coefficient(self, n:float, V:float, rho: float, m: float) -> float:
        # Placeholder lift coefficient model, replace with actual lift calculation
        return 2 * m * self.g * n / (rho * V**2 * self.S)

    def _get_drag_coefficient(self, lift_coefficient: float) -> float: return self.Cd0 + self.k * lift_coefficient**2

    def get_aerodynamic_coefficients(self, h: float, V: float, m: float, load_factor: float, rho: float) -> tuple:
        Cl = self._get_lift_coefficient(load_factor, V, rho, m)
        Cd = self._get_drag_coefficient(Cl)
        return Cl, Cd

    def dynamics(self, t, state_vec, control: Control, atmosphere: Atmosphere):
        # Unpack state vector
        x, y, h, V, psi, gamma, m = state_vec

        # Get atmospheric density
        rho = atmosphere.get_ISA_density(h)

        Cl, Cd = self.get_aerodynamic_coefficients(h, V, m, control.load_factor, rho)

        D = 0.5 * rho * V**2 * Cd * self.S 
        # 1.41, bank 45, should be a level turn.

        # Stall, max load factor

        # maxmium lift coefficient,  1.5

        # Compute state derivatives
        dxdt = V * math.cos(gamma) * math.cos(psi)
        dydt = V * math.cos(gamma) * math.sin(psi)
        dhdt = V * math.sin(gamma)
        dVdt = (control.thrust - D) / m - self.g * math.sin(gamma)
        dpsidt = self.g/(V * math.cos(gamma)) * math.sin(control.bank_rad) * control.load_factor
        dgamadt = self.g / V * (control.load_factor * math.cos(control.bank_rad) - math.cos(gamma))
        dmdt = 0

        return [dxdt, dydt, dhdt, dVdt, dpsidt, dgamadt, dmdt]

    def simulate(self, initial_state: State, control: Control, atmosphere: Atmosphere, t_span: tuple, t_eval: list):
        # Convert initial state to vector form
        state_vec = [initial_state.x, initial_state.y, initial_state.h, initial_state.V, initial_state.psi, initial_state.gamma, initial_state.m]

        # Solve the ODEs using solve_ivp
        sol = solve_ivp(lambda t, X: self.dynamics(t, X, control, atmosphere), t_span, state_vec, t_eval=t_eval)

        return sol
