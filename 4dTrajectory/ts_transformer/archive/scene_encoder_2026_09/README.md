# Archived: the scene encoder's data plane and its explainability readout (2026-09-05 → 09-07)

A **completed** line of the scene design (`docs/2026-09-05_scene_phase0_results.zh.md`;
`docs/2026-09-07_latent_intent_design.zh.md` §L4), kept in the repository as the record
behind its numbers and taken off the import path. Nothing here runs against the current
package; the files are left exactly as they were taken (archive convention:
`tests/test_architecture.py` refuses a live import of anything under `archive/`).

## What it was

- `scene/features.py` — the tensor side of the scene data plane (§五 P2.c): the
  neighbours' ENTITY rows and the runway-use scalars as arrays in the ego's chart, from
  `flight_scenarios.scene_context`. Its per-neighbour SEQUENCE half was already deleted on
  2026-09-07 (package audit T2).
- `run_ts_scene_explainability.py` — the L4 gate readout: gradient-boosted R² of the truth
  join distance from the scene entity features against the ego-only context
  (`ts_transformer.intent_explainability`, which STAYS live — the Phase 0 diagnostics in
  `docs/phase0_intent_diagnostics.py` read it).
- `tests/test_scene_features.py` — its tests (they used the `scene_fixture` under
  `flight_scenarios/tests`, which stays with `flight_scenarios.scene_context`).

## Why it is archived

The L4 gate FAILED: scene entity features added nothing to the join-distance prediction
(R² 0.37 against 0.38 ego-only), and the observable lead ETA correlated 0.11 with the lead's
true landing time, so the scene encoder was never built (`docs/OPEN_ITEMS.md`, L4 row).
The `intent_conditioning=truth-*` oracles the Phase 0 instrument used are FROZEN in
`config.INTENT_CONDITIONINGS_AVAILABLE` (2026-09-09, package review §5): their stored arms
load, no new oracle arm trains.

Taken from `dev-pkg-review` at the commit that created this directory (2026-09-09).
