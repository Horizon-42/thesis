from dataclasses import dataclass
import math
from pathlib import Path
import sys


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from aircraft_sets import AircraftSpec, A320
from coordinates_convertor import CoordinateConverter, GeodeticCoordinate, ENUCoordinate
from simulator import Atmosphere, Control, Simulator, State


@dataclass
class GeodeticState:
    latitude: float
    longitude: float
    altitude: float
    V: float
    psi: float
    gamma: float
    m: float


class GeodeticSimulator:
    def __init__(self, aircraft: AircraftSpec = A320):
        self.simulator = Simulator(aircraft=aircraft)
        self.atmosphere = Atmosphere()

    @staticmethod
    def get_enu_velocity_components(
        V: float,
        gamma: float,
        psi: float,
    ) -> tuple[float, float, float]:
        # Step 1: read the simulator velocity state in the current local ENU frame.
        # psi/gamma are radians; psi=0 points East and psi=pi/2 points North.
        V_east = V * math.cos(gamma) * math.cos(psi)
        V_north = V * math.cos(gamma) * math.sin(psi)
        V_up = V * math.sin(gamma)
        return (V_east, V_north, V_up)

    @staticmethod
    def enu_velocity_to_ecef_velocity(
        enu_velocity: tuple[float, float, float],
        geo_S: GeodeticCoordinate,
    ) -> tuple[float, float, float]:
        # Step 2: expand local ENU components into the global ECEF basis.
        # e_hat/n_hat/u_hat are local ENU axes, but each axis is expressed as
        # an ECEF unit vector. Multiplying and summing gives one ECEF vector.
        V_east, V_north, V_up = enu_velocity
        lat_rad = math.radians(geo_S.latitude)
        lon_rad = math.radians(geo_S.longitude)
        e_hat = (-math.sin(lon_rad), math.cos(lon_rad), 0)
        n_hat = (
            -math.sin(lat_rad) * math.cos(lon_rad),
            -math.sin(lat_rad) * math.sin(lon_rad),
            math.cos(lat_rad),
        )
        u_hat = (
            math.cos(lat_rad) * math.cos(lon_rad),
            math.cos(lat_rad) * math.sin(lon_rad),
            math.sin(lat_rad),
        )

        return (
            V_east * e_hat[0] + V_north * n_hat[0] + V_up * u_hat[0],
            V_east * e_hat[1] + V_north * n_hat[1] + V_up * u_hat[1],
            V_east * e_hat[2] + V_north * n_hat[2] + V_up * u_hat[2],
        )

    @staticmethod
    def ecef_velocity_to_enu_velocity(
        ecef_velocity: tuple[float, float, float],
        geo_S: GeodeticCoordinate,
    ) -> tuple[float, float, float]:
        # Step 3: project the same physical ECEF vector onto a new local ENU
        # frame. This gives the updated local velocity components after moving.
        lat_rad = math.radians(geo_S.latitude)
        lon_rad = math.radians(geo_S.longitude)

        # Project from ECEF onto the destination ENU unit vectors.
        e_hat = (-math.sin(lon_rad), math.cos(lon_rad), 0)
        n_hat = (
            -math.sin(lat_rad) * math.cos(lon_rad),
            -math.sin(lat_rad) * math.sin(lon_rad),
            math.cos(lat_rad),
        )
        u_hat = (
            math.cos(lat_rad) * math.cos(lon_rad),
            math.cos(lat_rad) * math.sin(lon_rad),
            math.sin(lat_rad),
        )
        V_east = (
            e_hat[0] * ecef_velocity[0]
            + e_hat[1] * ecef_velocity[1]
            + e_hat[2] * ecef_velocity[2]
        )
        V_north = (
            n_hat[0] * ecef_velocity[0]
            + n_hat[1] * ecef_velocity[1]
            + n_hat[2] * ecef_velocity[2]
        )
        V_up = (
            u_hat[0] * ecef_velocity[0]
            + u_hat[1] * ecef_velocity[1]
            + u_hat[2] * ecef_velocity[2]
        )

        return (V_east, V_north, V_up)

    def step(
        self,
        state: GeodeticState,
        control: Control,
        dt: float,
    ) -> GeodeticState:
        geo_S = GeodeticCoordinate(state.latitude, state.longitude, 0.0)

        # Re-anchor x/y at the current WGS84 point for this local tangent-plane step.
        state_vec = State(
            0.0,
            0.0,
            state.altitude,
            state.V,
            state.psi,
            state.gamma,
            state.m,
        )

        # Simulate one time step.
        solution = self.simulator.simulate(
            initial_state=state_vec,
            control=control,
            atmosphere=self.atmosphere,
            t_span=(0.0, dt),
            t_eval=[dt],
        )
        if not solution.success:
            raise ValueError(solution.message)

        new_x, new_y, new_h, speed, heading, flight_path, mass = [
            float(value) for value in solution.y[:, -1]
        ]
        if new_h < 0.0:
            raise ValueError("simulation stopped: altitude below 0")

        # Convert back to geodetic coordinates.
        new_enu_P = ENUCoordinate(new_x, new_y, new_h)
        new_geo = CoordinateConverter.enu_to_geodetic(new_enu_P, geo_S)

        # Keep velocity as the same physical vector while the local ENU frame
        # moves: source ENU components -> ECEF vector -> destination ENU components.
        old_enu_velocity = self.get_enu_velocity_components(
            V=speed,
            gamma=flight_path,
            psi=heading,
        )
        new_ecef_velocity = self.enu_velocity_to_ecef_velocity(old_enu_velocity, geo_S)
        new_enu_velocity = self.ecef_velocity_to_enu_velocity(
            new_ecef_velocity,
            new_geo,
        )
        V_east, V_north, V_up = new_enu_velocity
        V = math.sqrt(V_east**2 + V_north**2 + V_up**2)
        horizontal_V = math.hypot(V_east, V_north)

        return GeodeticState(
            latitude=new_geo.latitude,
            longitude=new_geo.longitude,
            altitude=new_geo.altitude,
            V=V,
            psi=math.atan2(V_north, V_east),
            gamma=math.atan2(V_up, horizontal_V),
            m=mass,
        )
