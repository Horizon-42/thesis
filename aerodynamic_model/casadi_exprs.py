import casadi as ca

def clamp_expr(x, lower, upper):
    return ca.fmin(ca.fmax(x, lower), upper)

def isa_density_expr(h):
    T0 = 288.15  # Sea level standard temperature (K)
    L = 0.0065   # Temperature lapse rate (K/m)
    RHO0 = 1.225  # Sea level standard density (kg/m^3)

    T_h = T0 - L * h
    rho_h = RHO0 * (T_h / T0)**4.25588
    return rho_h

def aerodynamic_coefficients_expr(n_cmd, V, m, rho, aero_params):
    S, Cl_max, Cd0, k, stall_threshold, k_stall = aero_params[0], aero_params[1], aero_params[2], aero_params[3], aero_params[4], aero_params[5]
    g = 9.81  # gravity (m/s^2)

    Cl_req = n_cmd*m*g/(0.5*rho*S*V**2)
    r = Cl_req / Cl_max
    stalled = r > 1.0
    Cl = ca.if_else(stalled, Cl_max, Cl_req)  # cap Cl at Cl_max to model stall

    stall_fraction = ca.fmin(r, 1.0)
    x_stall = clamp_expr((stall_fraction - stall_threshold) / (1.0 - stall_threshold), 0.0, 1.0)
    smooth = x_stall * x_stall * (3 - 2 * x_stall)  # smooth transition for stall drag
    Cd_stall = ca.if_else(r > stall_threshold, smooth * k_stall, 0.0)  # simple stall drag model
    Cd = Cd0 + k * Cl**2 + Cd_stall

    return {
        'Cl_req': Cl_req,
        'Cl': Cl,
        'Cd': Cd,
        'r': r,
        'stalled': stalled,
    }

# GEO coordinates convert to enu coordinates
def enu_velocity_components_expr(V, gamma, psi):
    Vx = V * ca.cos(gamma) * ca.cos(psi)
    Vy = V * ca.cos(gamma) * ca.sin(psi)
    Vz = V * ca.sin(gamma)
    return Vx, Vy, Vz