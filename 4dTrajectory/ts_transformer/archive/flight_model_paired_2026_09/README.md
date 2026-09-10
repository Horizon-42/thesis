# flight_model_paired (archived 2026-09-10, review §4.5)

`run_ts_flight_model_paired.py` — the one-shot paired comparison of the two flight models
(`point-mass` vs `first-order-lag`) whose result is the `control_dynamics_model` row of
`CLAUDE.md`'s defaults table ("`first-order-lag` buys smoothness + 3.4 % ADE; τ=2.0 s is
defensible, not CV-selected"). Classified one-shot by the 2026-09-07 audit (T4-27) and
archived at commit `947c907`'s successor; it is off the import path like every archived
campaign, and its imports still name the pre-`outputs/` layout it was run under.
