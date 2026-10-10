# Inputs of the prior

Source: `4dTrajectory/ts_transformer/docs/two_tier/design/prior.md` §2, §9; code `4dTrajectory/ts_transformer/prior/inputs.py`.
LaTeX version: `inputs.tex`.

Paper names (the code keeps its own):

| Paper | Code |
|---|---|
| aircraft kinematic state $\mathbf{s}_t$ | own state — `own`, `OWN_FEATURES` |
| runway vector $\mathbf{r}_{t,k}$ | candidate vector — `candidates`, `CANDIDATE_FEATURES` |

## Table 1 — Aircraft kinematic state $\mathbf{s}_t$ at row $t$

The state describes the aircraft alone: it contains no position, no direction and no runway.

| # | Component | Definition | Scale |
|---|---|---|---|
| 1 | Height above airport | $h_t - E$ | 1 km |
| 2 | Ground speed | $v_t = \sqrt{(e_t-e_t^-)^2+(n_t-n_t^-)^2}\,/\,\delta$ | 100 m/s |
| 3 | Vertical rate | $w_t = (h_t-h_t^-)\,/\,\delta$ | 10 m/s |
| 4 | No-motion flag | $m_t = 1$ at the first row of an aircraft (no state before it), else 0 | — |

$(e_t, n_t, h_t)$: position in the airport frame, east and north in m, height above mean sea level;
$(e_t^-, n_t^-, h_t^-)$: the position $\delta = 2$ s before row $t$; $E$: elevation of the airport. At the first row,
$v_t = w_t = 0$. The height is taken above the airport because the altitude words of the vocabulary are above the airport.

## Table 2 — Runway vector $\mathbf{r}_{t,k}$ of candidate runway $k$ at row $t$

The aircraft in the frame of the runway, its height above the published glidepath, and the recent traffic on the runway.
The prior reads one runway vector for each of the $K$ runway ends of the airport, with shared weights and no runway
constants.

| # | Component | Definition | Scale |
|---|---|---|---|
| 1 | Distance before threshold | $d_{t,k} = -\Delta\mathbf{p}_{t,k}\cdot\mathbf{u}_k$ | asinh(· / 1 km) |
| 2 | Offset right of final | $y_{t,k} = \Delta p_e\cos\psi_k - \Delta p_n\sin\psi_k$ | asinh(· / 1 km) |
| 3 | Height above threshold | $\Delta h_{t,k} = h_t - z_k$ | 1 km |
| 4 | Direction minus course, sine | $\sin(\chi_t - \psi_k)$ | — |
| 5 | Direction minus course, cosine | $\cos(\chi_t - \psi_k)$ | — |
| 6 | Height above glidepath | $\Delta h_{t,k} - g_k(d_{t,k})$, $\;g_k(d) = \mathrm{TCH}_k + d\tan\gamma_k + d^2/2R_k$ | asinh(· / 100 m) |
| 7 | Landings in the last 30 min | $L_{t,k}$ | 10 |

$\Delta\mathbf{p}_{t,k} = (e_t - e_k,\, n_t - n_k)$: offset of the aircraft from the threshold of runway $k$;
$\mathbf{u}_k = (\sin\psi_k, \cos\psi_k)$: unit vector along its course; $\psi_k$, $z_k$: course and threshold elevation;
$\mathrm{TCH}_k$, $\gamma_k$: crossing height and angle of its published glidepath; $R_k$: radius of curvature of the
earth along the course; $\chi_t = \operatorname{atan2}(e_t - e_t^-,\, n_t - n_t^-)$: direction of motion, set to 0 in
both direction terms where there is no motion ($m_t = 1$); $L_{t,k}$: landings on runway $k$ in the 30 min before row
$t$, the own landing of the flight not counted.
