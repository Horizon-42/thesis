
from dataclasses import dataclass
import math
import numpy as np

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
class GeodeticState:
    latitude: float
    longitude: float
    altitude: float
    V: float
    psi: float
    gamma: float
    m: float

@dataclass
class Control:
    thrust: float
    bank_rad: float # roll angle / bank angle in radians, mu
    attack_rad: float # angle of attack in radians, alpha

@dataclass
class LoadFactorControl:
    thrust: float
    bank_rad: float # roll angle / bank angle in radians, mu
    load_factor: float # load factor, n

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
    

def rk4_step(func, t, state_vec, control, atmosphere, dt):
    """
    Perform a single RK4 step for the given dynamics function.
    func: the dynamics function, should have signature func(t, state_vec, control, atmosphere)
    t: current time
    state_vec: current state vector as a numpy array
    control: current control input
    atmosphere: current atmospheric conditions
    dt: time step for integration"""
    y = np.asarray(state_vec, dtype=float)

    k1 = func(t, y, control, atmosphere)
    k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1, control, atmosphere)
    k3 = func(t + 0.5 * dt, y + 0.5 * dt * k2, control, atmosphere)
    k4 = func(t + dt, y + dt * k3, control, atmosphere)

    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)