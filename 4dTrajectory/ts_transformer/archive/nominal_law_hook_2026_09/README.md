# Archived: the nominal-law command hook (2026-09-05 → 2026-09-06)

A **completed** campaign arm, kept in the repository as the record behind published numbers
and taken off the import path. Nothing here runs against the current package.

## What it was

`control_command_hook="nominal-residual"`: inside the rollout, where the on-final gate is
open, the network's command was read as a bounded residual around a fixed tracking law —

- `guidance_laws.py`: L1 lateral guidance to the centreline, a flight-path-angle law to the
  glidepath, and a speed hold that pulls the thrust toward the network's own unhooked
  rollout (`T' = T + k m (V_ref − V)`, the v2 energy fix);
- `nominal_residual.py`: `u' = w · (u_nominal(x) + sat(u − u_nominal(x); ±r_max)) + (1 − w) · u`,
  the planner/tracker split applied per control segment. The first hook to declare
  `needs_reference = True`.

**The reference contract it reads has since changed (L3.f, 2026-09-09).** `nominal_residual.py`
here calls `runway_axes_view(state.reference, ...)`, i.e. `reference` as a lock-step
`RolloutStateView` — one unhooked state per segment. It is now the hook-free schedule WHOLE
(`[B,N+1,7]`, one chart row per segment boundary, the anchor first), because the trombone's
surplus estimator needs the path still AHEAD of the reference and no per-segment state can
give it that. Reviving this law means reading `state.reference[:, segment_index]` and wrapping
it in a view; the file is left exactly as it was taken, not silently ported.

## Why it is archived

`docs/2026-09-06_control_hooks_results.zh.md` §结论: the **barrier** filter was ADOPTED as a
predict-time safety layer (`predict --command-hook barrier --hook-saturation soft`); the
nominal law was kept "as an option" and never adopted. Its second consumer — the P1.d closure
tracker, which reused the `control_nominal_*` gains — was deleted in the same audit (T1-9,
its own BLOCKER unfixed). With the tracker gone, the unadopted hook was the only thing left
holding these two modules up.

## What stayed live

`control/constraints/barrier_filter.py` and `control/constraints/gates.py` — the adopted
predict-time barrier and the on-final gate it shares.

## The documents that cite it

- `docs/2026-09-06_control_hooks_results.zh.md` — the R-arm numbers (KRDU + KSJC).
- `docs/2026-09-05_control_constraint_design.zh.md` — the design (§P1, the v2 thrust law).
- `docs/2026-09-07_control_training_review.zh.md` §5 proposes a closed-loop (DAgger) teacher
  built on these three laws. That is a PROPOSAL against archived code: reviving it means
  taking the laws back out of here deliberately, not importing them from the archive.

## The vocabulary value stays

`control_command_hook="nominal-residual"` is still accepted by `TSConfig`, because six stored
2026-09-06 configs (and their checkpoints) carry it and `load_checkpoint` goes through
`TSConfig.from_dict`. `control/constraints/build_command_hook` refuses it with a pointer
here; the six `control_nominal_*` gain fields are in `RETIRED_SERIALIZED_FIELDS` (no stored
config ever set one away from its default, so no run name moves).

## Taken from

Commit `ed7fccc` (`dev-t2`), package audit T2, 2026-09-07. Its tests lived inside the shared
`tests/test_control_constraints.py` (the barrier's tests are still there) and were deleted
rather than split out — `git show ed7fccc:4dTrajectory/ts_transformer/tests/test_control_constraints.py`
has them.
