# Simple Aerodynamic Simulator, with load factor, assume alpha = 0
import numpy as np
import math
try:
    from .aircraft_sets import AircraftSpec, A320
    from .common import State, Atmosphere, rk4_step, LoadFactorControl
except ImportError:  # pragma: no cover - compatibility for flat script imports.
    from aircraft_sets import AircraftSpec, A320
    from common import State, Atmosphere, rk4_step, LoadFactorControl

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
    
    def _get_Cl_required(
        self,
        load_factor: float,
        speed: float,
        rho: float,
        mass: float,
    ) -> float:
        return load_factor * mass * self.g / (0.5 * rho * self.S * speed**2)

    def _get_drag_coefficient(self, Cl: float, Cd_stall: float = 0.0) -> float:
        return self.Cd0 + self.k * Cl**2 + Cd_stall

    def get_aerodynamic_coefficients(
        self,
        load_factor: float,
        speed: float,
        atmosphere: Atmosphere,
        h: float,
        mass: float,
    ) -> tuple:
        rho = atmosphere.get_ISA_density(h)
        Cl_required = self._get_Cl_required(load_factor, speed, rho, mass)

        # check stall and compute stall drag
        Cd_stall = 0.0
        r = Cl_required / self.Cl_max
        Cl = min(Cl_required, self.Cl_max)
        stalled = r > 1.0
        if r > self.stall_threshold:
            # TODO get reference for stall drag models, this is a simple smooth transition to high drag as we approach stall, can be tuned based on aircraft type
            # Use the capped ratio only for drag blending so post-stall commands do not push the smoothstep outside its designed [0, 1] range.
            stall_fraction = min(r, 1.0)
            x = min(max((stall_fraction - self.stall_threshold) / (1.0 - self.stall_threshold), 0.0), 1.0)
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

        # Get atmospheric density
        rho = atmosphere.get_ISA_density(h)

        # Get aerodynamic coefficients
        Cl, Cd, stalled = self.get_aerodynamic_coefficients(
            n_cmd,
            V,
            atmosphere,
            h,
            m,
        )

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

    def simulate(
        self,
        initial_state: State,
        control: LoadFactorControl,
        atmosphere: Atmosphere,
        t: float,
        dt: float,
    ) -> State:
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
