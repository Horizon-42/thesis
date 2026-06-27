import casadi as ca

# WGS84 ellipsoid constants come from the shared geokit package (single source of
# truth). Re-exported here so existing imports of these names keep working.
from geokit import WGS84_A, WGS84_B, WGS84_E2, WGS84_E_PRIME2, WGS84_F

def radians_expr(degrees):
    return degrees * ca.pi / 180.0


def degrees_expr(radians):
    return radians * 180.0 / ca.pi


def geodetic_deg_to_ecef_expr(lat, lon, alt):
    lat_rad = radians_expr(lat)
    lon_rad = radians_expr(lon)

    N = WGS84_A / ca.sqrt(1 - WGS84_E2 * ca.sin(lat_rad)**2)

    x = (N + alt) * ca.cos(lat_rad) * ca.cos(lon_rad)
    y = (N + alt) * ca.cos(lat_rad) * ca.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * ca.sin(lat_rad)

    return x, y, z


def ecef_to_geodetic_expr(x, y, z):
    p = ca.sqrt(x**2 + y**2)
    lamda = ca.atan2(y, x)

    q = ca.atan2(z * WGS84_A, p * WGS84_B)
    sin_q = ca.sin(q)
    cos_q = ca.cos(q)
    lat_rad = ca.atan2(z + WGS84_E_PRIME2 * WGS84_B * sin_q**3,
                        p - WGS84_E2 * WGS84_A * cos_q**3)
    N = WGS84_A / ca.sqrt(1 - WGS84_E2 * ca.sin(lat_rad)**2)
    alt = p / ca.cos(lat_rad) - N

    lat = degrees_expr(lat_rad)
    lon = degrees_expr(lamda)

    return lat, lon, alt


def ecef_to_enu_rotation_matrix_expr(ref_lat, ref_lon):
    phi = radians_expr(ref_lat)
    lamda = radians_expr(ref_lon)

    e_hat = (-ca.sin(lamda), ca.cos(lamda), 0)
    n_hat = (-ca.sin(phi) * ca.cos(lamda), -ca.sin(phi) * ca.sin(lamda), ca.cos(phi))
    u_hat = (ca.cos(phi) * ca.cos(lamda), ca.cos(phi) * ca.sin(lamda), ca.sin(phi))

    R = ca.vertcat(
        ca.horzcat(e_hat[0], e_hat[1], e_hat[2]),
        ca.horzcat(n_hat[0], n_hat[1], n_hat[2]),
        ca.horzcat(u_hat[0], u_hat[1], u_hat[2]),
    )
    return R


def geodetic_to_enu_expr(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    x, y, z = geodetic_deg_to_ecef_expr(lat, lon, alt)
    ref_x, ref_y, ref_z = geodetic_deg_to_ecef_expr(ref_lat, ref_lon, ref_alt)

    R = ecef_to_enu_rotation_matrix_expr(ref_lat, ref_lon)

    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    enu = R @ ca.vertcat(dx, dy, dz)
    return enu[0], enu[1], enu[2]


def enu_to_geodetic_expr(east, north, up, ref_lat, ref_lon, ref_alt):
    ref_x, ref_y, ref_z = geodetic_deg_to_ecef_expr(ref_lat, ref_lon, ref_alt)

    R = ecef_to_enu_rotation_matrix_expr(ref_lat, ref_lon)
    offset = R.T @ ca.vertcat(east, north, up)
    x = ref_x + offset[0]
    y = ref_y + offset[1]
    z = ref_z + offset[2]

    lat, lon, alt = ecef_to_geodetic_expr(x, y, z)

    return lat, lon, alt
