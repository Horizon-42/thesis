Lagrange interpolation is a polynomial interpolation method: it builds a polynomial that passes exactly through known sample points.

If you have points $(x_0,y_0),\dots,(x_n,y_n)$, the classic form is:

$$
P(x)=\sum_{i=0}^{n} y_i\,L_i(x), \quad
L_i(x)=\prod_{j\ne i}\frac{x-x_j}{x_i-x_j}
$$

Linear interpolation is simpler: between each neighboring pair of points, it uses a straight line segment:

$$
y(x)=y_i+\frac{y_{i+1}-y_i}{x_{i+1}-x_i}(x-x_i), \quad x\in[x_i,x_{i+1}]
$$

Key differences:

1. Curve type  
Lagrange: polynomial curve (smooth).  
Linear: piecewise straight segments.

2. Point influence  
Lagrange: more global behavior; one point can affect the curve broadly.  
Linear: local behavior; changing one point mainly affects nearby segments.

3. Overshoot risk  
Lagrange: can overshoot and oscillate between samples, especially with higher degree or noisy data.  
Linear: no oscillation between two endpoints of a segment.

4. Smoothness  
Lagrange: smooth curve shape.  
Linear: has corners at sample points.

5. Robustness for trajectories  
Linear is usually safer when you must preserve sampled shape and avoid fake dips/spikes.  
Lagrange can look smoother, but may introduce non-physical artifacts (like below-ground dips).

Practical note: many rendering engines use local Lagrange windows (for example degree 3), not one huge global polynomial, but the overshoot tendency can still happen.