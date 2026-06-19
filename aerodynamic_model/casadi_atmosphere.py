import casadi as ca


def make_atmosphere_model():
    # Define symbolic variables for altitude and temperature
    h = ca.SX.sym('h')  # Altitude (m)
    T = ca.SX.sym('T')  # Temperature (K)

    # Constants for the International Standard Atmosphere
    T0 = 288.15  # Sea level standard temperature (K)
    L = 0.0065   # Temperature lapse rate (K/m)
    RHO0 = 1.225  # Sea level standard density (kg/m^3)
    R = 287.05   # Specific gas constant for dry air (J/(kg*K))

    # Calculate temperature at altitude h
    T_h = T0 - L * h


    rho_h = RHO0 * (T_h / T0)**(L * R / (R * L))  # Density at altitude h using the barometric formula

    # Create a CasADi function to compute air density given altitude
    atmosphere_model = ca.Function('atmosphere_model', [h], [rho_h])

    return atmosphere_model