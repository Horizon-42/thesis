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