from aircraft.aircraft_sets import AircraftSpec
from simulator import Atmosphere, Control, Simulator, State
import math

class NaiveAutoPilot:
    def __init__(self, aircraft: AircraftSpec, atmosphere: Atmosphere):
        self.aircraft = aircraft
        self.simulator = Simulator(aircraft=aircraft)
        self.atmosphere = atmosphere

    def maintain_level_turn(self, thrust: float, bank_angle: float, state: State) -> Control:
        # This is a trim calculator for one instant, not a closed-loop controller.
        # It chooses alpha so the current simulator's gamma equation has zero
        # acceleration for level flight.

        # Step 1: level flight means the target flight-path angle is gamma = 0,
        # so the vertical-plane force balance is:
        #
        #   (L + T sin(alpha)) cos(bank) = m g
        #
        # This includes the current simulator's thrust-normal term T sin(alpha).
        required_vertical_force = state.m * self.simulator.g
        bank_vertical_fraction = math.cos(bank_angle)

        if bank_vertical_fraction <= 0.0:
            raise ValueError("bank_angle must be less than 90 degrees in magnitude")

        # Step 2: lift is q S CL(alpha), with q = 1/2 rho V^2. Density must use
        # the current altitude, not sea-level rho0, because lift changes with h.
        rho = self.atmosphere.get_ISA_density(state.h)
        dynamic_pressure_area = 0.5 * rho * state.V**2 * self.simulator.S

        # Step 3: define the balance error. A positive value means alpha creates
        # too much upward vertical force; a negative value means alpha is too low.
        def vertical_balance_error(attack_angle: float) -> float:
            lift_coefficient = (
                self.simulator.CL0 + self.simulator.CL_alpha * attack_angle
            )
            lift = dynamic_pressure_area * lift_coefficient
            normal_force = lift + thrust * math.sin(attack_angle)
            return normal_force * bank_vertical_fraction - required_vertical_force

        # Step 4: solve for alpha in a practical subsonic range. The equation is
        # slightly nonlinear because of sin(alpha), so use bisection instead of
        # dropping the thrust-normal term with a small-angle shortcut.
        low_attack_angle = math.radians(-10.0)
        high_attack_angle = math.radians(18.0)
        low_error = vertical_balance_error(low_attack_angle)
        high_error = vertical_balance_error(high_attack_angle)

        if low_error * high_error > 0.0:
            raise ValueError("level-turn trim alpha is outside the search range")

        for _ in range(60):
            attack_angle = 0.5 * (low_attack_angle + high_attack_angle)
            error = vertical_balance_error(attack_angle)
            if error == 0.0:
                break
            if low_error * error < 0.0:
                high_attack_angle = attack_angle
                high_error = error
            else:
                low_attack_angle = attack_angle
                low_error = error
        else:
            attack_angle = 0.5 * (low_attack_angle + high_attack_angle)
        
        return Control(
            thrust=thrust,
            bank_rad=bank_angle,
            attack_rad=attack_angle
        )

    
