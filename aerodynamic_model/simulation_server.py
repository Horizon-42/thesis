from coordinates_convertor import CoordinateConverter, GeodeticCoordinate, ECEFCoordinate, ENUCoordinate, ENUUnitVectors
from simulator import Simulator, Control, Atmosphere, ReferenceArea
from dataclasses import dataclass
import math

@dataclass
class GeodeticState:
    latitude: float
    longitude: float
    altitude: float
    V: float
    psi: float
    gamma: float
    m: float

class SimulationServer():
    def __init__(self, S: float = ReferenceArea.NarrowBody_S):
        self.simulator = Simulator(S)
        self.atmosphere = Atmosphere()
    
    @staticmethod
    def get_ecef_velocity_vector(V: float, gamma: float, psi: float) -> tuple[float, float, float]:
        # Convert geodetic velocity components to ECEF velocity vector
        # Currently use 0 as East and pi/2 as North, but this can be improved by using the actual heading and flight path angle to compute the velocity components in the ECEF frame.
        gamma_rad = math.radians(gamma)
        psi_rad = math.radians(psi)
        V_east = V * math.cos(gamma_rad) * math.cos(psi_rad)
        V_north = V * math.cos(gamma_rad) * math.sin(psi_rad)
        V_up = V * math.sin(gamma_rad)
        return (V_east, V_north, V_up)
    
    @staticmethod
    def ecef_velocity_to_enu_velocity(ecef_velocity: tuple[float, float, float], geo_S: GeodeticCoordinate) -> tuple[float, float, float]:
        # Convert ECEF velocity vector to ENU velocity components at the reference point geo_S
        lat_rad = math.radians(geo_S.latitude)
        lon_rad = math.radians(geo_S.longitude)

        # Compute the rotation matrix from ECEF to ENU
        e_hat = (-math.sin(lon_rad), math.cos(lon_rad), 0)
        n_hat = (-math.sin(lat_rad) * math.cos(lon_rad), -math.sin(lat_rad) * math.sin(lon_rad), math.cos(lat_rad))
        u_hat = (math.cos(lat_rad) * math.cos(lon_rad), math.cos(lat_rad) * math.sin(lon_rad), math.sin(lat_rad))
        V_east = e_hat[0] * ecef_velocity[0] + e_hat[1] * ecef_velocity[1] + e_hat[2] * ecef_velocity[2]
        V_north = n_hat[0] * ecef_velocity[0] + n_hat[1] * ecef_velocity[1] + n_hat[2] * ecef_velocity[2]
        V_up = u_hat[0] * ecef_velocity[0] + u_hat[1] * ecef_velocity[1] + u_hat[2] * ecef_velocity[2]

        return (V_east, V_north, V_up)

    def step(self, state: GeodeticState, control: Control, dt: float) -> GeodeticState:
        # Convert geodetic state to ECEF for simulation
        geo_P = GeodeticCoordinate(state.latitude, state.longitude, state.altitude)
        geo_S = GeodeticCoordinate(state.latitude, state.longitude, 0.0)  # Reference point for ENU is at the same lat/lon but sea level
        enu_P = CoordinateConverter.geodetic_to_enu(geo_P, geo_S)  # Using the same point as reference

        # Create state vector for simulator (x, y, h, V, psi, gamma, m)
        state_vec = [enu_P.x, enu_P.y, state.altitude, state.V, state.psi, state.gamma, state.m]

        # Simulate one time step
        new_state_vec = self.simulator.dynamics(0, state_vec, control, self.atmosphere)

        # Convert back to geodetic coordinates
        new_enu_P = ENUCoordinate(new_state_vec[0], new_state_vec[1], new_state_vec[2])
        new_geo = CoordinateConverter.enu_to_geodetic(new_enu_P, geo_S)  # Convert back to geodetic using the same reference point

        # convert velocity vector to ECEF and then to ENU to get the new V, psi, gamma
        new_ecef_velocity = self.get_ecef_velocity_vector(
            V=new_state_vec[3],
            gamma=new_state_vec[5],
            psi=new_state_vec[4]
        )
        new_enu_velocity = self.ecef_velocity_to_enu_velocity(new_ecef_velocity, new_geo)

        return GeodeticState(
            latitude=new_geo.latitude,
            longitude=new_geo.longitude,
            altitude=new_geo.altitude,
            V=new_enu_velocity[0],
            psi=new_enu_velocity[1],
            gamma=new_enu_velocity[2],
            m=new_state_vec[6]
        )