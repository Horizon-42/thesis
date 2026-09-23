"""The two-tier executor's closed loop, and the gates and diagnostics read off it
(`docs/2026-09-18_two_tier_plan_v3.zh.md`).

The INTENT-CODE layer this package was built for — the learned FSQ tokenizer and its codebook,
the code sequences, the causal prior over codes, the gate-T readout and the code atlas, and the
executor conditioned on a code — is **ARCHIVED 2026-09-20**
(`archive/manoeuvre_codes_2026_09/`, README there): plan v3 §10's audit found both layers
trained on truth and evaluated closed-loop. Its readings stay citable in
`docs/2026-09-18_manoeuvre_token_results.zh.md` §9–§11.

What is here, one module each:

    lockstep       the no-token closed loop: one round = `executed_step_s`, protocol ``none``
    gates          the grid gate (stage A1) and the relative gate (stage A3 / B)
    failure_modes  A2's six non-crossing modes, in the course frame

**Layering** (`tests/test_architecture.py`): every module here imports the control path, the
guidance layer, the data plane and the inference helpers, never the reverse; runners under
`experiments/` are the only consumers of this package. Nothing is re-exported here on purpose
(layout rule L2).
"""
