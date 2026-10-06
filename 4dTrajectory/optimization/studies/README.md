# Optimizer studies

One-off comparison scripts behind the optimizer's design documents. Nothing imports them. Each one writes its own
data file or prints its table; run one from this directory with the thesis environment active, e.g.
`python transport_term_comparison.py --help`.

- `collocation_scheme_comparison.py` — the fitting schemes (trapezoidal, Hermite-Simpson, rk4) on one problem.
- `solver_backend_benchmark.py` — IPOPT against `sqpmethod`.
- `geodetic_vs_reanchored_error.py` — the continuous geodetic dynamics against the per-step re-anchored ENU stepper.
- `transport_term_comparison.py` — the transport terms of the geodetic dynamics (writes `transport_term_comparison_data.json`,
  read by `4dTrajectory/docs/transport_term_comparison.zh.html`).
- `fixed_enu_frame_error.py` — the error of one fixed ENU frame over distance.
