# closure_2026_09 — the closure output (scene design P1.c, 2026-09-05 / 06)

The closed-form decision vector regressed on per-flight labels: a comparison arm whose tracker
was deleted on 2026-09-07 and whose training was FROZEN on 2026-09-09. Archived 2026-09-18 with
the manoeuvre-token plan's repository clean-up (`docs/2026-09-18_manoeuvre_token_plan.zh.md`
§5.1). Results: `docs/2026-09-06_closure_p1c_results.zh.md`, `docs/2026-09-06_closure_p1d_tracking_results.zh.md`.
Off the import path on purpose: nothing live imports it.

| file | was |
|---|---|
| `closure/` | `outputs/closure/` — the `closure` strategy, its model, geometry, profile and forecast |
| `p1_closure_oracle.py` | `docs/` — the labels writer (`closure_labels_path`) and the P1 oracle |

Their tests are archived beside them, unmodified (`tests/`); they do not run. The `PREDICTION_CLOSURE` name
survives only in `config.PREDICTION_OUTPUTS_RETIRED` (its six published categories keep their
`predictionOutput`); a stored closure config is refused at load. `config.CLOSURE_TIMING_SCALE_S`
stays only as the value `RETIRED_CONSTANT_FIELDS` drops from stored state/control configs written
while the output existed.
