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
    bank: float # roll angle, miu
    load_factor: float # n

class Atmosphere:
    rho0: float = 1.225 # sea level density in kg/m^3
    H: float = 8500.0 # scale height in meters

    def __init__(self, rho0: float, H: float):
        self.rho0 = rho0
        self.H = H
    
    def get_density(self, h: float) -> float:
        # ISA model for density variation with altitude
        # return self.rho0 * (1 - h / self.H) ** 4.2561
        return self.rho0 * math.exp(-h / self.H)

class Simulator:
    g: float = 9.81 # gravitational acceleration in m/s^2

    S:float = ReferenceArea.NarrowBody_S # reference area for drag calculation, default to narrow-body

    def __init__(self, S: float = ReferenceArea.NarrowBody_S):
        self.S = S

    def _get_lift_coefficient(self, n:float, V:float, rho: float, m: float) -> float:
        # Placeholder lift coefficient model, replace with actual lift calculation
        return 2 * m * self.g * n / (rho * V**2 * self.S)

    def _get_drag_coefficient(self, lift_coefficient: float) -> float:
        # Placeholder drag coefficient model, replace with actual drag calculation
        Cd0 = 0.02 # zero-lift drag coefficient
        k = 0.04 # induced drag factor
        return Cd0 + k * lift_coefficient**2

    def dynamics(self, t, state_vec, control: Control, atmosphere: Atmosphere):
        # Unpack state vector
        x, y, h, V, psi, gamma, m = state_vec

        # Get atmospheric density
        rho = atmosphere.get_density(h)

        # Get lift and drag coefficients
        Cl = self._get_lift_coefficient(control.load_factor, V, rho, m)
        Cd = self._get_drag_coefficient(Cl)

        D = 0.5 * rho * V**2 * Cd * self.S 
        # 1.41, bank 45, should be a level turn.

        # Stall, max load factor

        # maxmium lift coefficient,  1.5

        # Compute state derivatives
        dxdt = V * math.cos(gamma) * math.cos(psi)
        dydt = V * math.cos(gamma) * math.sin(psi)
        dhdt = V * math.sin(gamma)
        dVdt = (control.thrust - D) / m - self.g * math.sin(gamma)
        dpsidt = self.g/(V * math.cos(gamma)) * math.sin(control.bank) * control.load_factor
        dgamadt = self.g / V * (control.load_factor * math.cos(control.bank) - math.cos(gamma))
        dmdt = 0

        return [dxdt, dydt, dhdt, dVdt, dpsidt, dgamadt, dmdt]

    def simulate(self, initial_state: State, control: Control, atmosphere: Atmosphere, t_span: tuple, t_eval: list):
        # Convert initial state to vector form
        state_vec = [initial_state.x, initial_state.y, initial_state.h, initial_state.V, initial_state.psi, initial_state.gamma, initial_state.m]

        # Solve the ODEs using solve_ivp
        sol = solve_ivp(lambda t, X: self.dynamics(t, X, control, atmosphere), t_span, state_vec, t_eval=t_eval)

        return sol