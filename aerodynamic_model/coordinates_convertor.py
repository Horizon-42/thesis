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
class ENUUnitVectors:
    # Unit vectors for the ENU coordinate system at the reference point
    e_hat: tuple[float, float, float]  # East unit vector
    n_hat: tuple[float, float, float]  # North unit vector
    u_hat: tuple[float, float, float]  # Up unit vector

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
        # Handle the special case where p is zero (at the poles) to avoid division by zero in the latitude calculation
        if math.isclose(p, 0.0, abs_tol=1e-12):
            lat = 90.0 if ecef.z >= 0 else -90.0
            alt = abs(ecef.z) - CoordinateConverter.b
            return GeodeticCoordinate(lat, math.degrees(lamda), alt)

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
    def geodetic_to_enu(geo: GeodeticCoordinate, ref_geo: GeodeticCoordinate) -> ENUCoordinate:
        r0 = CoordinateConverter.geodetic_to_ecef(ref_geo)
        r = CoordinateConverter.geodetic_to_ecef(geo)

        phi = math.radians(ref_geo.latitude)
        lamda = math.radians(ref_geo.longitude)

        e_hat = (-math.sin(lamda), math.cos(lamda), 0)
        n_hat = (-math.sin(phi) * math.cos(lamda), -math.sin(phi) * math.sin(lamda), math.cos(phi))
        u_hat = (math.cos(phi) * math.cos(lamda), math.cos(phi) * math.sin(lamda), math.sin(phi))
        dx = r.x - r0.x
        dy = r.y - r0.y
        dz = r.z - r0.z
        east = e_hat[0] * dx + e_hat[1] * dy + e_hat[2] * dz
        north = n_hat[0] * dx + n_hat[1] * dy + n_hat[2] * dz
        up = u_hat[0] * dx + u_hat[1] * dy + u_hat[2] * dz

        return ENUCoordinate(east, north, up)

    @staticmethod
    def enu_to_geodetic(enu: ENUCoordinate, ref_geo: GeodeticCoordinate) -> GeodeticCoordinate:
        r0 = CoordinateConverter.geodetic_to_ecef(ref_geo)
        phi = math.radians(ref_geo.latitude)
        lamda = math.radians(ref_geo.longitude)
        e_hat = (-math.sin(lamda), math.cos(lamda), 0)
        n_hat = (-math.sin(phi) * math.cos(lamda), -math.sin(phi) * math.sin(lamda), math.cos(phi))
        u_hat = (math.cos(phi) * math.cos(lamda), math.cos(phi) * math.sin(lamda), math.sin(phi))

        dx = e_hat[0] * enu.east + n_hat[0] * enu.north + u_hat[0] * enu.up
        dy = e_hat[1] * enu.east + n_hat[1] * enu.north + u_hat[1] * enu.up
        dz = e_hat[2] * enu.east + n_hat[2] * enu.north + u_hat[2] * enu.up

        r = ECEFCoordinate(r0.x + dx, r0.y + dy, r0.z + dz)
        return CoordinateConverter.ecef_to_geodetic(r)
