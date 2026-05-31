import math
from dataclasses import dataclass


@dataclass
class GeodeticCoordinate:
    # Geodetic coordinates (latitude, longitude, altitude) for Earth, WGS84 ellipsoid, units in degrees and meters
    latitude: float
    longitude: float
    altitude: float

@dataclass
class ECEFCoordinate:
    # Earth-Centered, Earth-Fixed (ECEF) coordinates, units in meters
    x: float
    y: float
    z: float

@dataclass
class ENUCoordinate:
    # East-North-Up (ENU) coordinates relative to a reference point, units in meters
    east: float
    north: float
    up: float

class CoordinateConverter:
    a = 6378137.0  # Semi-major axis of the Earth (WGS84)
    b = 6356752.314245  # Semi-minor axis of the Earth (WGS84)
    f = 1 / 298.257223563  # Flattening of the Earth
    e2 = f * (2 - f)  # Square of eccentricity

    # for closed form approximation of latitude from ECEF
    e_prime2 = e2 / (1 - e2)  # Second eccentricity squared


    @staticmethod
    def geodetic_to_ecef(geo: GeodeticCoordinate) -> ECEFCoordinate:
        lat_rad = math.radians(geo.latitude)
        lon_rad = math.radians(geo.longitude)
        
        N = CoordinateConverter.a / math.sqrt(1 - CoordinateConverter.e2 * math.sin(lat_rad)**2)
        
        x = (N + geo.altitude) * math.cos(lat_rad) * math.cos(lon_rad)
        y = (N + geo.altitude) * math.cos(lat_rad) * math.sin(lon_rad)
        z = (N * (1 - CoordinateConverter.e2) + geo.altitude) * math.sin(lat_rad)
        
        return ECEFCoordinate(x, y, z)
    
    @staticmethod
    def ecef_to_geodetic(ecef: ECEFCoordinate) -> GeodeticCoordinate:
        p = math.sqrt(ecef.x**2 + ecef.y**2)
        lamda = math.atan2(ecef.y, ecef.x)
        # Closed-form approximation for latitude
        q = math.atan2(ecef.z * CoordinateConverter.a, p * CoordinateConverter.b)
        sin_q = math.sin(q)
        cos_q = math.cos(q)
        lat_rad = math.atan2(ecef.z + CoordinateConverter.e_prime2 * CoordinateConverter.b * sin_q**3,
                             p - CoordinateConverter.e2 * CoordinateConverter.a * cos_q**3)
        N = CoordinateConverter.a / math.sqrt(1 - CoordinateConverter.e2 * math.sin(lat_rad)**2)
        alt = p / math.cos(lat_rad) - N
        
        return GeodeticCoordinate(math.degrees(lat_rad), math.degrees(lamda), alt)

    @staticmethod
    def ecef_to_enu(ecef: ECEFCoordinate, ref_geo: GeodeticCoordinate) -> ENUCoordinate:
        pass
    
    @staticmethod
    def enu_to_ecef(enu: ENUCoordinate, ref_geo: GeodeticCoordinate) -> ECEFCoordinate:
        pass