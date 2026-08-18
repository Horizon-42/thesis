# geokit — shared geodesy / units

Pure geodesy + unit constants. Imported by `final_approach`, `flight_scenarios`,
`evaluation`, `trajectory_data_process`, `4dTrajectory`. It is the single source for
every length/angle conversion in the repo — never redefine one downstream.

- **src-layout** (`geokit/src/geokit/`) because a top-level `geokit/` dir on sys.path
  (CWD under pytest) would shadow the installed package. Installed editable
  (`pip install -e`) into the conda `aeroviz` env.
- **`geokit.METRES_PER_DEG_LAT` is derived from `WGS84_A·π/180` (111319.4908…), not the old
  hand-rounded `111_320.0`** — the rounded value put a 4.6 ppm seam (~0.11 m at the 25 km ring)
  between every geokit-derived frame (ts channels, `start_state` velocity fits) and the
  optimizer's NE frame (`approach_constraints.frame` + the NLP's metric-position normalization,
  both `WGS84_A·DEG2RAD`). One definition, bit-identical everywhere;
  `metres_per_deg_lon = METRES_PER_DEG_LAT·cos(lat)` (pure cosine, no ellipsoidal correction).
  Frontend mirror regenerated (`aeroviz-4d/src/generated/geoConstants.json`).
  The `hermiteSimpsonNormalizedFullTransport` scheme's decision state IS these
  threshold-anchored NE metres (`(lat−lat_t)·R`, `(lon−lon_t)·R·cos lat_t`) — an exact affine
  change of variables for Jacobian conditioning; the defect still evaluates the exact geodetic
  full-transport RHS on the reconstructed physical state (only the `localEnu` scheme family
  approximates dynamics in a flat frame).
- `geokit.wgs84_curvature_radii` is the **numeric** single source for the tangent scales
  (`R_M+h`, `(R_N+h)·cosφ`); the casadi RHS in `4dTrajectory` and its mirror comment are the
  symbolic twin. See `flight_scenarios/CLAUDE.md` for the velocity/chart-derivative seam that
  depends on it.
