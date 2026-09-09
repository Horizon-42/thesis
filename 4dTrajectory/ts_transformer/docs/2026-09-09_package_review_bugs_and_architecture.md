# ts_transformer — package review 2026-09-09: bugs, and why it is heavy

A second full read of the package, two days after the 2026-09-07 audit's T0–T3 landed. Five
parallel Opus reviews (data plane / training / control / inference+CLI / config), every finding
below re-verified by grep or by running it (marked VERIFIED, else PLAUSIBLE); the measurements and
the architecture assessment are the main session's. Baseline: `dev-leg-ctrl @ a8a6ef5`, clean tree,
no campaign running on this machine (a `dev-l3f` pytest was running in `thesis-l3f`).

The audit fixed what was *dead*. This review is about what is *wrong* (§2) and about why the
package stays heavy after the audit (§3): the price of adding one experiment axis is ten touch
points, and the 2026-09-09 plan-and-guidance design is the next axis.

## 〇 Status

| item | state |
|---|---|
| bug findings | 4 number-changing on live paths, 4 crash paths, ~20 contract holes — §2, all with `file:line` |
| architecture | diagnosis §3, target §4, retirement candidates with census evidence §5, order §6 |
| **resolution (2026-09-09/10, `dev-pkg-review`)** | **§7**: A-1 A-2 A-3 A-4 B-1 B-2 B-3 B-4 fixed; C-1 C-2 C-3 C-4(gate) C-5 C-6 C-7(dead checks) C-8 C-10 C-11 C-13 C-14 C-15 C-16 C-17 C-18 C-19 C-20 fixed; C-12 decided (grader stays 0.5); C-4 (floor: stays a field, see §4.3's resolution), C-7 (directory binding), C-9 stay open. §6 steps 1 (the package), 2 (§5), 3 (§4.3), 4 (§4.2) and 5 (§4.4, see their resolution paragraphs) done in the same branch |
| decisions needed from the user | §5's freeze/delete list; whether the config defaults move to the current recipe (§4.3); the five §7 decisions |

## 1. Measured

| what | number |
|---|---|
| source (no tests / docs / vendor / archive) | 27,597 lines, 85 modules |
| tests | 20,000 lines, 46 files, 778 tests; `test_ts_transformer.py` 5,745 |
| root runners `run_ts_*.py` | 23 files, 11,072 lines |
| `docs/*.py` | 13 scripts, 3 imported by package code (audit T5, not started) |
| `TSConfig` | 116 fields; validators 775 lines; `run_naming` reads 95 fields; `cli/common.add_training_args` 369 lines |
| fan-in (module-level importers) | `config` 36, `dataset` 29, `channels` 18, `prediction_outputs` 11 |
| import cycles | `dataset → closure_output → {approach_difficulty, closure_geometry, intent_conditioning}` (lazy) and `dataset → control.basis_fit → prediction_outputs → dataset` |
| `sys.path` manipulation sites | 66 (no `__init__.py`; flat absolute imports) |
| functions ≥ 120 lines | 31 (`fit_model` 506, `predict.run_cli` 467, `add_training_args` 369, `train` 270, `_validate_output_contract` 259, `run_ts_pipeline.main` 250, …) |
| signatures ≥ 9 parameters | 37 (`TrainingPlan.__init__` 37, `_forecast_control_batch` 17, `forecast_approaches` 14, …) |
| branch sites on `prediction_output` / feature flags | forecast 13, objective 11, `cli/predict` 8, models 7, dataset 5, batching 5, validation 3, export 2, `run_naming` 10, `run_ts_pipeline` 15 |
| stored runs on disk (`history.json['config']`) | 219 (82 in 2026-07, 52 in 2026-08, 85 in 2026-09); 112 load under today's `from_dict` |
| fields never non-default in any stored run | 51 of 116 (20 backbone knobs, 9 scale/τ/gain, 7 loss weights, …) |
| fields pinned to one value by all four named recipes | 64 of 116; 20 of those differ from the dataclass default |

## 2. Bugs

### A — change numbers on a live path (all VERIFIED)

**A-1 `run_ts_pipeline.py:663-702` and `:737-786` — `control_dynamics_model` is missing from both
rebuilt override dicts.** `_recipe_args:479-482` emits `--control-dynamics-model` to the training
subprocess, but `_expected_cv_base_config()` and `resolved_train_config()` rebuild the recipe
without it, i.e. as `point-mass`. Consequences: a first-order-lag run's published category label
reads `control · iTransformer · point-mass · custom` (`PredictionPlan.label:886`);
`checkpoint_reuse_error:596` compares `point-mass` against the metadata's `first-order-lag`, so
`--skip-train` never reuses a lag checkpoint and retrains it; `cv_reuse_error:634` never reuses a
lag CV artifact and `steps():797` silently falls back to base hyperparameters. **A-1b
`:377-405`, `:857-879`**: no directory or category tag reads the dynamics *model* (the tags read
the *backend*), so a point-mass cell and a lag cell with otherwise equal flags share `train_dir`,
`pred_dir` and `category` — the second run overwrites the first. Verified by running the module
(`resolved dynamics model pm/lag: point-mass / point-mass`, `train_dir equal: True`).

**A-2 `fixed_anchor_validation.py:95` (also `:109`, `:161`) and `train.py:405`, `:451` — the
common-grid truth and the cohort future floor are anchored at `seq_len − 1`, whatever
`minimum_anchor_index` the run trains at.** `FixedAnchorTrajectoryWindows` anchors at
`max(seq_len − 1, minimum_anchor_index)`; `fixed_anchor_common_truth` ignores the floor;
`validation.build_validation_batch_plan:548` builds the plan's truth this way six lines after
`:498` computed the bucketing durations from the dataset's real anchors. Live caller:
`run_ts_history_ablation.py:277,285` passes `minimum_anchor_index=maximum_l − 1` to
`fit_model` and `evaluate_split` — its whole purpose is identical anchors across candidate
`seq_len`. Measured on the test fixture at `minimum_anchor_index=89`: dataset-anchor durations
`[93.4, 69.0]` s against common-truth durations `[213.4, 189.0]` s at `seq_len=30` (`[153.4,
129.0]` at 60; equal only at 90). Under the default `fixed-anchor-common-grid-ade` selection that
corrupts which epoch is kept, the LR-plateau signal and every `metrics` block that runner
publishes, with the error growing as `(maximum_l − seq_len) · dt_s`. Note: code-health item #26
("`config.seq_len − 1` → `default_anchor(config)`, a pure rename") lists these lines and would
preserve the defect.

**A-3 `dataset.py:1019-1028` vs `:1060-1065` — the threshold crossing is supervised under two
contracts, chosen by ADS-B coverage.** When `_observed_threshold_crossing` fires,
`_build_supervision` returns early with the crossing row at `1/6` on all six channels (velocities
included, no terminal emphasis); otherwise the fitted tail's last row carries
`fitted_tail_position_weight/3 + fitted_terminal_position_weight/3 = 0.4167` per position channel
and 0 on velocities. An observed crossing's terminal position is weighted 2.5× less than a fitted
one, and its velocity is supervised where the fitted one's is masked. Verified on the first 60
KRDU arrivals: `Counter({0.4167: 59, 0.1667: 1})`. Small population, systematic, feeds FDE.

**A-4 `control/constraints/barrier_filter.py:167` + `control/constraints/trombone.py:263-273` —
under `hook_saturation=soft` the barrier and the trombone rewrite the same step.** The composite
(`composite.py:20`, `config.py:526`, `constraints/__init__.py:13`) rests on complementary gates:
barrier only where `on_final`, trombone only where `~hard_aligned` (> 30° off course). But the
barrier's blend weight under soft is `soft_on_final = lateral · sigmoid((cos_align − cos 30°)/0.02)`,
non-zero for 30–40° of misalignment — exactly where the trombone is admitted. Re-measured here:
d = 12 km, xt = 500 m, heading error −33°, hold 6 s: soft barrier weight 0.148, barrier bank
**+0.112 rad** (6.4° toward the centreline, load 1.0063); the trombone then returns **−0.249 rad**
(soft) against **−0.262 rad** (hard, where the barrier is inert) — the barrier's correction is
discarded and the trombone's own command moves 0.7° because it reads the barrier's coordinated
load. At xt = 700 m the barrier weight is 0.0003 and the effect vanishes; at xt = 300 m the
barrier moves 0.00008 rad. So: the stated invariant is false, the order does change the answer,
and the affected band is the soft gate's shoulder. `soft` is the adopted delivery form and what
`docs/experiments/l3e_path_stretch_arms.json` pre-registers. Magnitude on the L3.e/L3.f readouts:
not measured; the gate-4 readout (lateral RMS *inside* the hard gate) is outside the band. Also:
`composite.py:23` and `config.py:527` cite a "25 deg turn cap"; `trombone.py:185` is 15°.

### B — crash on a reachable path (all VERIFIED)

**B-1 `dataset.py:551-579` `probe_dynamics` + `batching.py:117`** — the auto-batch probe omits
`reference_controls` / `reference_control_weight` / `reference_heading_rate_dps` /
`reference_heading_rate_weight`, which `objective.control_imitation_mse:427` and
`control_heading_rate_mse:484` index unconditionally after the weight check.
`resolve_batch_size:205` catches only `RuntimeError`, so `--batch-size auto` dies with a bare
`KeyError` after the dataset build, for any run with a non-zero imitation or heading-rate weight.
The named recipes are shielded (`cli/common.py:669` refuses `auto` for a frozen recipe); every
`custom` arm is not — that is every latent, L1.b and quantile arm — and
`run_ts_history_ablation.py:498` *defaults* to `auto` and calls `resolve_batch_size` directly.
`probe_dynamics`' docstring promises "a key added to the real batch appears here in the same commit".

**B-2 `batching.py:65-69`** — `_heterogeneous_control_probe_prediction` rebuilds
`ControlPrediction` without `duration_quantiles_s`; `objective.py:690,702` then call
`pinball_duration_loss(None, …)`. `--batch-size auto` + `duration_head ∈ {quantile, two-head}`
→ `AttributeError`, again post-build and uncaught.

**B-3 `export.py:537-548`** — `accuracy_block` raises on any non-finite `ade_m` / `fde_m` /
`arrival_endpoint_error_m` *after* every record file was written (`:473-495`), leaving a record
directory with no `summary.json`; the null branch in `_json_optional_metrics:394-403` for those
three is dead. `run_ts_anytime_curve --write-records` stages into `.partial-*` for this reason;
`predict` does not.

**B-4 `dataset.py:344-357`** — `_smoothed_heading_rate_dps` returns NaN (0/0) when the
post-anchor series has one sample, and the target rides with weight 1.0 into
`control_heading_rate_mse` — a NaN batch, not an error. Reachable with
`random_train_anchor_min_future_s=0` plus a heading-rate weight; its sibling
`reference_control_supervision` raises loudly on the same anchor.

### C — silent contract holes

| # | where | what | status |
|---|---|---|---|
| C-1 | `config.py:1895`, `control/loss/components.py:88,165` | `control_imitation_loss_weight` and `control_velocity_loss_weight` are accepted under `control_state_objective=normalized-mse`, where neither term is built — while the two later siblings (`heading_rate`, `bank_tv`) are refused for exactly that reason. The run is *named* with the dose (`custom(vel=0.003, imit=64)`), `dataset.py:1518` solves inverse dynamics per sample for it, and `loss_component_names` never lists it. No stored artifact affected (census). | VERIFIED |
| C-2 | `config.py:1853` | the "a barrier gain needs a hook with a barrier" refusal sits under `elif hook_modules:`, so `control_command_hook=off` skips it: `train --control-recipe-name simple-v3 --control-barrier-alpha 0.3` trains bit-identically to the arm without it under a different name **and slug**; `predict` refuses the same thing. `control_hook_saturation=hard` under `off` likewise. | VERIFIED |
| C-3 | `run_naming.py:155` | five CLI-settable fields are in no naming list: `validation_common_grid_points`, `lr_plateau_patience`, `lr_plateau_factor`, `random_train_anchor_min_future_s`, `device`. Two `custom` arms differing only in `--validation-common-grid-points 64\|128` (which changes the kept epoch) get an identical name and slug; the import-time guard checks only that named fields exist, never the reverse. | VERIFIED |
| C-4 | `config.py:1089`, `control/heads.py:250-258`; `config.py:943`, `dataset.py:422` | `control_duration_uniform_floor` is read only by the `factorized` head, yet accepted under `uniform` (what every recipe pins) and renames the run (`duration-floor=0.4`); `corridor_gate` is read only under `corridor-bounded`, yet accepted elsewhere and renames (`gate=faf`). The package's own rule ("a knob that cannot change an answer is refused") inverted. | VERIFIED |
| C-5 | `cli/predict.py:322-325` | `--aircraft-type X` builds the series under X (target Vref, crossing height) but is never `replace`d into `config`; `summary.json` and the run name record the checkpoint's type. In `CLI_CONFIG_FIELDS` but not `PREDICT_CONFIG_FLAGS`, so neither assertion covers it. | VERIFIED |
| C-6 | `cli/predict.py:665-698` | the `modes/`, `random/`, `quantiles/`, `shuffled/` `write_batch` calls omit `skipped`, so under `--cta-offset-s` (which drops flights at `:461-470`) those directories publish `skipped: {}` over a subset — the thing `write_batch`'s docstring says `skipped` exists to prevent. | VERIFIED |
| C-7 | `evaluation_protocol.py:30-32`, `:109-112`, `:142` | the one-shot test-release ledger is `checkpoint.with_name("test_release.json")` — bound to the *directory*, not the checkpoint digest: copy `checkpoint.pt` elsewhere and the frozen test flights can be released again. Two of the three "bound to a different split / data" checks recompute from the payload whose sha256 was just verified equal, so they cannot fire. | VERIFIED |
| C-8 | `run_ts_frame_ablation.py:242-266`, `:168` | resume = "`checkpoint.pt` exists"; `arm_config` rewrites `config.json` *before* the skip test, so editing an arm's overrides and re-running keeps the old checkpoint beside the new config and reports the campaign complete. And `train.py:1069` writes `checkpoint.pt` before `checkpoint_metadata.json` / `history.json`, so a tail crash leaves an arm that is permanently skipped, has no metadata, and stays `failed` in its manifest; the runner never runs `evaluate-fit`. | VERIFIED |
| C-9 | `cross_validation.py:503-521` | `mean_/std_/best_val_macro_loss` hold the **selection metric** (ADE, metres), not the macro loss; the resume validator (`:317`) and every reader of `cv_results.json` say "loss". | VERIFIED |
| C-10 | `fixed_anchor_validation.py:634`; `validation.py:370` | `"invalid_flights": 0` is a literal published in every per-airport block; `prediction_horizon_cap_rate` is structurally 0 on control, closure and normalized-horizon runs; `active_segments` can never be False so `raw_kinematic_metrics`' masking is inert outside tests. | VERIFIED |
| C-11 | `final_approach_geometry.py:80-109`, `dataset.py:449-468`, `forecast.py:994-1012` | the corridor geometry is chart-origin-relative (docstring: "because `target_chart` is the origin") and `final_approach_arrays` never carries `target_chart`; `config.py:1620` guards the training consumers with the ENU rule but not `--project-final`, which reads `state_position_reference=absolute`. On an `airport-enu` state checkpoint the projection clamps about the airport reference point, 1.3–1.9 km from the threshold (KRDU 05L/23R, KSJC 12L). `cut_at_threshold_crossing:1220` does subtract it — two consumers disagree. | VERIFIED (frozen arm) |
| C-12 | `control/envelope.py:55` vs `flyability.py:94,277` | the head's load-factor box floor is 0.2; the grader's hard floor is 0.5 and `load_factor_low` is a HARD violation. Any segment commanded at n ∈ [0.2, 0.5) is unflyable by construction on the metric `cli/predict.py:56` publishes for every control arm; observed tracks (n ∈ [0.92, 1.18]) can never trip it, so "read as a delta" does not absorb it. How often trained heads emit n < 0.5: not measured (`control/training/diagnostics.py` saturation counts would say). | PLAUSIBLE |
| C-13 | `config.py:1955-1972` | the actuator-τ rule refuses `τ < control_rollout_integrator_dt_s` citing explicit-RK4 stability on `y' = −y/τ`; the criterion it cites is `h/τ ≤ 2.785` and the substep is `h ≤ dt`, so the rule is ~2.8× stricter than the instability it names (refuses `τ_load = 0.45/0.3/0.2` s, all stable). Not binding at the defaults. | VERIFIED |
| C-14 | `config.py:861` | `REQUIRED_SERIALIZED_CONTROL_FIELDS` lacks `control_state_supervision_clock`, which `objective.target_contract:133` keys the stored contract on; a checkpoint missing only that key would fail as an opaque "target contract does not match" instead of the curated message. 28 stored configs lack it (all also lack other required fields). | VERIFIED |
| C-15 | `calibration.py:470-500`, `:391-395` | `interval_stratum` picks from the UNION of calibrated strata across α, `conformal_intervals` looks it up under EVERY α → `KeyError` the day a stratum calibrates at one α only (today union == intersection because `_stratum_block:266` refuses on counts alone). The covariate guard checks presence, not non-null: a present-null `established_at_anchor` reads False in `strata_masks:83` and moves the flight out of `vectored`. | VERIFIED (latent) |
| C-16 | `approach_clustering/model.py:106-111` | the silhouette that selects K is computed on a silent 2,000-row subsample; neither artifact says so. | VERIFIED |
| C-17 | `dataset.py:302-310` + `objective.py:434` | `reference_control_supervision` can return an all-zero weight vector (anchor at/after the last measured sample with a fitted tail); `control_imitation_mse` divides by `weight.sum().clamp(min=1)` → exactly 0 = "perfect imitation". Neither docstring says what an empty mask means. | PLAUSIBLE |
| C-18 | `dataset.py:1459` | `fixed_anchor_fraction` is identically 0 under any `minimum_anchor_index` floor (compares to `seq_len − 1`); the `l1_share` docs read it as an upper bound on `l1_share_drawn`. | PLAUSIBLE |
| C-19 | `cli/predict.py:357` + `cli/common.py:789` + `flight_scenarios/build.py:82` | `predict` accepts repeated `--data` with `--airport`, which train/cross-validate refuse; the flag wins over `arr_airport`, so a pooled predict with `--airport KRDU` re-homes every KSJC flight onto KRDU's thresholds. No live caller does it. | VERIFIED (latent) |
| C-20 | small, each verified by reading | `train.py:1251` `payload.get("input_channels", …)` → `None` for a present-null key then `list(None)`; `time_grids.py:47` `offsets[~active] = capped_duration` can never change a value; `channels.py:177-181` zero-ground-speed fallback returns `V·sin γ ≠ udot`; `closure_geometry.py:373` `strictly_increasing` yields *equal* neighbours on a decreasing input and `closure_output.py:398` divides by the spacing (no live caller reaches it); `dataset.py:1517` fills `cta_s` on `== given` while `control/heads.py:51` builds the token on `!= off` (safe only while `self-q` is kept out of training); `terminal_state_loss.py:14` names a physical argument `normalized_anchor_state`; `run_ts_pipeline.py:324,365` `control_bank_time_constant_s` is stored and never emitted; `build_multiflight_capacity_report.py:379` hard-codes `generated_at: 2026-08-01`; `intent_explainability.py:101,103` clip fallback indistinguishable from a clipped value; `CLAUDE.md` Contracts says `overlap` is a REQUIRED arg to `write_batch` — no such parameter exists. | — |

**Checked and clean** (worth recording so nobody re-reads them): the torch point-mass and
transport-chart RHS against the casadi models (equations, RK4 step, ISA density, stall-limited
load, ω×v sign, transport rate); `inverse.actual_controls` as the algebraic inverse of that RHS;
the CBF barrier derivation, the trombone geometry and termination argument, the speed floor's
thrust inversion; the latent KL / free-bits / mixture estimator; `closure_profile` quadrature and
the slowness→duration round trip; all four Dubins CSC branches; the evaluation record contract in
`export.py` (boundary-sample control assignment matches `aerodynamic_model/rollout.py:95` and
`evaluation/arrival.py:204`; `final_time_s` read off the serialized array; MSL conversion once, at
`build_series`); no split leak, no non-determinism, `.detach()` placement, train/eval mode; every
vocabulary against its dispatch table; `from_dict`/`to_dict` round trip; `CLI_CONFIG_FIELDS` in
both directions; no field with zero readers (the τ constants reach the backend through
`control_time_constants_s`; the 20 backbone knobs are read by the vendored `configs`).

## 3. Why the package is heavy — the price of one experiment axis

Every campaign since July added an *axis* (a config value the run name has to say). Measured on
the current tree, one axis costs touch points in **ten places**:

| touch point | where | evidence on the tree |
|---|---|---|
| a `TSConfig` field + a rule in one of five validators + `REQUIRED_SERIALIZED_*` / `RETIRED_*` bookkeeping | `config.py` | 116 fields, 775 validator lines, 51 fields never non-default in any of the 219 stored runs |
| a CLI flag | `cli/common.add_training_args` | 369 lines, one function; `CLI_CONFIG_FIELDS` import-time assertion |
| a run-name token | `run_naming.py` | reads 95 of the 116 fields; C-3 is a token that was not added |
| a branch in the data plane | `dataset.py` | imports `control.basis_fit`, `control.conditioning`, `control.dynamics.inverse`, `control.envelope`, `intent_conditioning`, `target_conditioning`, `closure_output` (lazy); two import cycles |
| a branch in the objective, the forecast and the export | `objective.py`, `forecast.py`, `export.py` | branch sites: forecast 13, objective 11, `cli/predict` 8, models 7, dataset 5, batching 5, validation 3, export 2 |
| an `EpochResult` field + a print block | `train.fit_model` | 20 record fields; `fit_model` is 506 lines, 61 of them `print` |
| a `Forecast` field | `forecast.py` | 33 fields, 20 of them "control only" / "closure only" / "latent only" / "CTA only" |
| a runner | repo root `run_ts_*.py` | 23 runners, 11,072 lines; `run_ts_pipeline.TrainingPlan.__init__` takes 37 arguments, re-declares 48 config fields, and carries 13 directory-name tag functions beside `run_naming` (A-1 is the axis that was not added to two of its five copies) |
| a test file | `tests/` | 46 files, 20,000 lines; `test_ts_transformer.py` alone 5,745 |
| a `docs/*.py` script | `docs/` | 13 scripts, three imported by package code (audit T5, not started) |

That is the whole diagnosis. Five of the bugs in §2 (A-1, B-1, B-2, C-1, C-3) are the same
event: an axis was added and one of its ten touch points was missed. The 2026-09-07 audit removed
dead axes and split the largest files, which is why `train.py` is 1,271 lines and not 3,246 —
but it did not lower the price of the *next* axis, and the 2026-09-09 plan-and-guidance design is
exactly that: a fourth output strategy, which on the current structure means edits in all ten
places again.

Three secondary causes, all measurable:

1. **It is not a package.** No `__init__.py`; the modules are found because `__main__.py`,
   `tests/conftest.py`, `batch_benchmark.py`, every root runner and every `docs/*.py` script
   put the directory on `sys.path` (66 manipulation sites). So the module names are global:
   `config`, `dataset`, `train`, `models`, `metrics`, `validation`, `export`, `calibration` —
   none of them says which project it belongs to, and a same-named module anywhere earlier on the
   path shadows it silently. The CLAUDE.md "flat listing" complaint about `control_*.py` was this
   problem one level down.
2. **The three output paths are dispatched by string in ten modules but by type in three.**
   `models.OUTPUT_MODEL_BUILDERS`, `objective.PREDICTION_LOSS_HANDLERS` and
   `anchor_eligibility._POLICIES` are already registries; `forecast.forecast_approaches`,
   `dataset.TrajectoryWindows.batch`, `validation._prediction_batch_replay`, `export`,
   `batching._probe_training_step` and `cli/predict` still write `if prediction_output ==`.
   The registries prove the interface exists; the branches prove it was never written down.
3. **One config object is passed everywhere, so every module reads every field, and the
   defaults are a configuration nothing runs.** `train` reads 35 distinct fields, `objective`
   32, `dataset` 28, `run_ts_pipeline` 48. The vendored backbones receive the whole `TSConfig` as
   their upstream `configs` namespace. And the dataclass defaults are the 2026-07 state-era
   settings: 20 of the 64 fields every named recipe pins differ from them (`d_model` 512 vs 256,
   `batch_size` 512 vs 2048, `learning_rate` 3e-5 vs 5e-4, `control_state_objective`
   true-time-position vs normalized-mse, …). Because `run_naming` defines "special" as "differs
   from the default", every current `custom` control run carries six architecture items, `+N
   more` and a diff hash in its slug
   (`…_d-model-512_d-ff-1024_layers-4_n-64_batch-512_lr-3e-05_4-more_afa072c5`, on eight KRDU
   arms) — which is what makes C-3's collision reachable.

## 4. Target architecture

The goal is one number: the touch points per axis, from ten to **three** (the strategy
sub-package, its sub-config, its test file). Everything below serves that.

### 4.1 Make it a package (mechanical, behaviour-free, first)

```
4dTrajectory/pyproject.toml          packages = ["ts_transformer", "ts_transformer.*"]  (editable, like geokit)
ts_transformer/__init__.py
from ts_transformer.config import TSConfig      # instead of `from config import TSConfig`
```

Delete the `sys.path` bootstrap from `__main__.py`, `conftest.py`, `batch_benchmark.py`, the 23
runners and the `docs/*.py` scripts. `python -m ts_transformer train …` then works from any
directory (today the docs use the script-path form because `-m` only works inside
`4dTrajectory/`). `tests/test_import_boundaries.py` keeps working (it builds its own path list).
Verification: the suite, plus the T3 equivalence harness (bit-identical histories on the six CPU
fixtures) — this step must move no number and no run name.

### 4.2 One `OutputStrategy` per prediction path

The interface is what the spine already calls, collected from the branch sites:

```python
class OutputStrategy(Protocol):
    name: str                                   # "state" | "control" | "closure" | "plan"
    config_type: type                           # its sub-config (4.3)
    def build_model(self, backbone, cfg, normalizer) -> nn.Module
    def batch_context(self, series, anchor, cfg) -> dict[str, np.ndarray]   # dataset.batch's `context` slot
    def anchor_eligibility(self, series, anchors, cfg) -> list[int]           # anchor_eligibility._POLICIES
    def loss(self, prediction, batch, cfg, epoch) -> LossComponents           # objective.PREDICTION_LOSS_HANDLERS
    def forecast(self, model, batch, cfg, options) -> list[Forecast]          # forecast._forecast_{control,closure}_batch / _forecast_state
    def record_fields(self, forecast) -> dict                                 # export.build_prediction_record's per-output blocks
    def epoch_record(self, diagnostics, cfg, epoch) -> dict                   # EpochResult.{procedure,command_hook,latent,control_training_diagnostics}
    def validation_replay(self, ...)                                          # validation.py:107,181
```

Layout:

```
ts_transformer/
  data/        series.py (FlightSeries, build_series)  windows.py (TrajectoryWindows + the 4 anchor policies)
               splits.py  provenance.py  channels.py  frames.py  time_grids.py  anchors/{eligibility,grid,strata}.py
  backbone/    adapters.py (ITransformerAdapter, PatchTSTAdapter)  vendor/
  outputs/     base.py (the protocol + registry; the procedure penalty, which acts on BOTH paths)
               state/    config.py  model.py  loss.py  forecast.py  record.py
               control/  config.py  model.py (heads, latent)  loss/  forecast.py  record.py  dynamics/  constraints/  basis_fit.py
               closure/  config.py  model.py  loss.py  forecast.py  record.py  geometry.py  profile.py
  training/    session.py (prepare)  epoch.py (train_epoch, validate_epoch)  selection.py  checkpoint.py  cross_validation.py
  inference/   predict.py  calibration.py  export.py
  cli/         (as today, one module per subcommand)
  experiments/ (the runners, 4.5)
  tests/       one file per module, `support.py` for fixtures
```

What this removes, concretely:

- `dataset.py` (2,057) loses `reference_control_supervision`, `reference_heading_rate_supervision`,
  `dynamics_arrays`, `probe_dynamics`, `_dynamics_arrays`, `_fixed_dt_supervision`, the closure
  label loading and the final-approach context (≈ 450 lines) to `outputs/control` and
  `outputs/closure`; its imports of `control.*` and `closure_output` go with them and both
  import cycles disappear. `__getitem__` — unused by any live path (`iter_batches` calls
  `batch`), and already disagreeing with `batch` about the final-approach context — is deleted.
  B-1 becomes unrepresentable: the strategy that adds a batch key is the strategy that builds the
  probe batch.
- `forecast.py` (1,271) keeps `Forecast`, `forecast_approaches`, the anchor/history helpers and
  `cut_at_threshold_crossing` (≈ 400 lines); `_forecast_control_batch` (124 lines, 17
  parameters), the four `*_latent_forecasts`, the CTA/quantile plumbing and the closure batch go
  to their strategies.
- `objective.py` (1,126) keeps the dispatch, `masked_mse`, the state objective and the procedure
  multipliers (≈ 350 lines); the control terms and the two adapters go to `outputs/control/loss`.
  `LossComponents.kinematic`, structurally zero on the state and control paths and used only by
  the closure (as "height"), moves into `extras`.
- `Forecast` keeps its 12 common fields plus `extras: Mapping[str, Any]` owned by the strategy.
- `EpochResult` keeps its 10 common fields plus `extras` from `epoch_record`.

The plan-and-guidance design then becomes `outputs/plan/` with the eight methods and its own
sub-config — no edit in `dataset`, `objective`, `forecast`, `export`, `train`, `validation` or
`batching`.

**Resolution (2026-09-10, `dev-pkg-review`).** Built as `outputs/`: the lazy registry
(`outputs.strategy(config)` / `strategy_for(name)`), the `OutputStrategy` base
(`outputs/base.py` — the methods the branch sites asked for, one per site, tabulated in its
docstring), and one package per path — `outputs/state/{strategy,model,loss,forecast}`,
`outputs/closure/{strategy,model,geometry,profile,forecast}`,
`outputs/control/{strategy,supervision,forecast,loss/objective,…}` (the `control/` package
moved whole). The spine's branch sites are gone: `models.build_model`; `dataset`
(`bind_windows` → one `WindowContext` per window set, `batch()` asks it for the context row
and the dense supervision; `__getitem__` deleted; `anchor_eligibility.py` deleted — the policy
is `eligible_anchors`); `batching` (the probe's context / dense / prediction hooks and the
margin flag); `objective` (`target_contract`, `loss_component_names`,
`prediction_loss_components` dispatch by config, not by prediction type);
`forecast.forecast_approaches` (one `ForecastOptions` value instead of seven keyword
arguments — §4.4's `PredictOptions`, forecast half); `validation._prediction_batch_replay`
(→ `Replay`); `export.build_prediction_record` (`record_fields`); `train.fit_model`
(`check_trainable`, `training_teacher`, `epoch_config`, `training_diagnostics`,
`epoch_record`, `checkpoint_metadata`). Out of the spine: ≈ 380 lines of `dataset` (the
control supervision and dynamics context — `dataset` imports nothing under `outputs/` any
more and both import cycles are gone), ≈ 700 of `forecast`, ≈ 850 of `objective`;
`prediction_outputs.py` split into `outputs/state/model`, `outputs/control/heads` and the
shared `outputs/duration_heads`. **Not adopted from the sketch:** `Forecast.extras` /
`EpochResult.extras` — both records keep their fields (the prediction record and
`history.json` schemas are unchanged; the strategy fills them through `record_fields` /
`epoch_record`); `dataset.build_series` and `TrajectoryWindows.__init__` are not split (§4.7,
later). B-1 is unrepresentable in the sketch's sense: `ControlStrategy.probe_context` and
`ControlContext._build_row` are two methods of one path, pinned equal by
`tests/test_supervision_terms.py`. `tests/test_architecture.py` now enforces the layering:
nothing under `outputs/` imports the loop, only the strategy seam reaches the spine, the
registry is lazy, `dataset` reaches only the registry. Acceptance: full suite 972 passed (the 12 known `test_ts_pipeline.py` fixtures red before and after); 219 stored
configs, 112 load, 0 names moved.

### 4.3 Split `TSConfig` by owner — sum types where the validators say "belongs to X"

Reading the five validators (775 lines) as a specification, most rules are of two kinds: "field
F belongs to output X" (11 rules) and "field F means nothing without feature Y" (the seven
`latent_*`, the hook gains, the duration-head weights, the objective's four extras). Both kinds
disappear if the field lives *inside* the thing it belongs to. The config reviewer's partition,
one owner per rule:

| new type | fields | rules it absorbs |
|---|---|---|
| `CohortSpec` | `channels`, `dt_s`, `seq_len`, `aircraft_type`, `aircraft_filter`, `coordinate_frame`, `reference_velocity_source`, `val_fraction`, `test_fraction`, `split_seed`, `training_cohort_min_future_s`, `target_conditioning`, `intent_conditioning`, `anchor: AnchorPolicy` | the three random-anchor cross-rules, via `AnchorPolicy = Fixed \| RandomUniform(min_future_s) \| RandomPathUniform(min_future_s, l1_share)` |
| `BackboneSpec = ITransformer(…) \| PatchTST(…)` | the 24 architecture fields; this, not `TSConfig`, is what the adapters hand the vendored `configs` (`enc_in`, `pred_len`, `seq_len` as derived properties) | the two "requires iTransformer" rules become type-level; a PatchTST run stops serialising 12 inert iTransformer knobs |
| `TrainingSpec` | `batch_size`, `epochs`, `learning_rate`, `weight_decay`, `lr_plateau_*`, `patience`, `seed`, `device`, `checkpoint_selection_metric`, `validation_common_grid_points` | the `lr_plateau_metric` pair-rules |
| `OutputSpec = StateOutput \| ControlOutput \| ClosureOutput` | `StateOutput(horizon, n_segments, position_reference, corridor_gate, five weights, procedure: ProcedureSpec \| None)`; `ClosureOutput(labels_path, knots, three weights)` | every "belongs to the X output" rule (11); C-4's `corridor_gate` lives inside `CorridorBounded(gate)` |
| inside `ControlOutput` | `DurationSpec = Point(w) \| Quantile(w) \| TwoHead(w, w)` — the floor lives only in `Factorized(floor)`; `DynamicsSpec` = one variant per `backends._BACKENDS` row (`PointMassReanchored \| PointMassChart \| LaggedChart(taus, integrator_dt_s)`); `ControlObjective = NormalizedMSE(terminal_w) \| TrueTimePosition(velocity, imitation: None \| InverseDynamics(w) \| Fitted(w, path), heading_rate, bank_tv, endpoint_w)`; `HookSpec = None \| Hook(modules, saturation, barrier: Gains \| None, floor_margin)`; `LatentSpec = None \| Latent(dim, components, beta, free_bits, init_std, warmup_epochs, aux_w)` | the duration-weight refusals; the (model, backend) rules and the τ/RK4 bound (C-13, restated correctly); the six objective/grid/clock pins; the four hook rules; the "seven fields mean nothing without a latent" rule — and C-1, C-2 and C-4 by construction |

Two rules survive as genuine cross-object constraints and are spelled once at the top:
`uses_final_approach_context ⇒ coordinate_frame == "enu"`, and the three "this metric reads the
future" refusals. `_validate_recipe` survives as what it is, a comparison against a frozen record.

**The serialized form stays the flat dict**: `to_flat_dict` flattens, `from_dict` applies
`RETIRED_*` and nests, keyed on `prediction_output` — one adapter, so the 112 stored configs that
load today still load, `run_naming` keeps reading the flat dict and no run name moves. The
audit's T3 census (0 names moved) is the acceptance test. With a flat dict in hand, C-3's missing
direction is one assertion: `set(flat) − set(named) == KNOWN_UNNAMED`.

**The defaults have to move too** (§3 cause 3). Either the defaults become the current recipe — a
change that moves stored names, so it needs the census as the proof — or the grammar names a run
against its nearest recipe *for every field*, not only the loss fields. This is a user decision.

**Resolution (2026-09-10, `dev-pkg-review`).** Built as views over the flat dataclass rather
than as nested storage: `TSConfig` stays the flat schema every checkpoint serialises and every
module reads, and `__post_init__` builds `CohortSpec`, `BackboneSpec`, `TrainingSpec` and one
`OutputSpec` per path (`StateOutput`, `ClosureOutput`, `ControlOutput` with its `DurationSpec`,
`DynamicsSpec`, `ControlObjective`, `HookSpec`, `LatentSpec`), exposed as
`config.cohort/.backbone/.training/.output`; each view validates its own fields, and the six
rules that read two views sit on `_validate_cross`. The eleven "belongs to output X" checks and
the twelve "supported only by prediction_output='control'" checks are ONE rule
(`_validate_ownership`): a field owned by another output's view is refused off its default —
which also closes the whole class C-1 / C-2 / C-4 came from. `from_dict` normalises a stored
config's foreign-output fields to their defaults (unread by definition; measured: none moves),
so the serialized form is unchanged. Sum types (one variant per backend, `Factorized(floor)`)
were NOT adopted: the flat schema is what `run_naming`, the CLI and 219 stored configs read, and
the views give the same ownership without a second schema. C-4's floor therefore stays a field
(the recipes pin `0.0` under `uniform`, so refusing it there would refuse every recipe).
Defaults unchanged (user decision); the partition is checked at import
(`_check_view_partition`). Census: 219 stored configs, 112 load, 0 names moved; full suite 970
passed.

### 4.4 The training loop and the predict command

`fit_model` (506 lines) → `TrainingSession.prepare()` (window sets, validation plans, model,
optimizer, teacher — the first 120 lines), `train_epoch(session)`, `validate_epoch(session)`,
`select(session, epoch_result)`, `describe(session)` (the 61 print lines, in one place). The
per-axis diagnostics (procedure λ, hook shares, latent KL, control gradients) leave the loop
through `OutputStrategy.epoch_record`. `SplitPredictionReplay.truth` / `.mask` — write-only,
costing a decode and a device→host copy of `[B, 64, 6]` per batch per airport per epoch — go.

`cli/predict.run_cli` (467 lines) → `parse_predict_options(args, config) -> PredictOptions` (the
flag-combination rules, 24 `parser.error` calls today, as a table), `load`, `forecast`, `export`
(one `_emit_directory` for the seven `write_batch` calls, which is where C-6 came from). The
predict-time overrides (`--command-hook`, gains, `--cta-offset-s`, `--cta-from-quantiles`,
`--closure-from-labels`, `--truncate-at-threshold`, and `--aircraft-type`, C-5) become one
`PredictOptions` value the strategy's `forecast` receives and the record stamps, instead of six
keyword arguments threaded through `forecast_approaches` → `_forecast_control_batch` (14 and 17
parameters).

**Resolution (2026-09-10, `dev-pkg-review`).** Pure extraction, statement for statement, so
the epoch record and the RNG order are untouched. `train.fit_model` (506 lines) →
`prepare_session` (the window sets, the plans, the model, the optimizer, the teacher — into a
`TrainingSession`), `describe_session` (the header lines), `train_epoch` (→ `TrainEpoch`),
`validate_epoch` (→ `ValidationEpoch`, the selection metric included), `procedure_update` (the
dual step), `describe_epoch`, and a `fit_model` of 124 lines that is the loop and the
selection. `cli/predict.run_cli` (467 lines) → `load_predict_checkpoint`, `load_predict_series`,
`parse_predict_options` (every flag-combination rule in one function; returns a frozen
`PredictOptions` — the `ForecastOptions` every forecast is asked, the arms decoded beside the
top-1 records, the cohort the CTA offset skipped — plus the restamped config), `predict_sets`
(→ `PredictionSets`), `write_prediction_sets` (the one emitter) and `report_predictions`; the
`run_cli` left is 22 lines. The predict-time overrides reach the strategy as ONE value:
`ForecastOptions` from §4.2 (`_fan_forecast` derives a leaf's from it with `replace`). `SplitPredictionReplay.truth` / `.mask` are removed: write-only, a decode and a device→host copy of the batch targets per batch per airport per epoch that no metric read (the metrics compare against the validation plan's common-grid truth); one test asserted their shape and now does not. Acceptance: full suite 972 passed (the 12 known `test_ts_pipeline.py` fixtures red before and after).

### 4.5 Runners → `ts_transformer/experiments/`

The 23 root `run_ts_*.py` (11,072 lines) move under the package as importable modules with one
entry point (`python -m ts_transformer.experiments <name>`), a shared `experiments/support.py`
(the audit's T4-25 list: `parse_airports` ×5, `write_json_atomic` ×5, `series_digest` ×3,
`write_reports` ×3, `file_sha256` ×7 — 16+ byte-identical copies, several in modules whose own
docstring cites `io_utils` as the reason it exists), and tests beside them. The one-shot ones the
audit already classified (T4-27) go to `archive/`. `run_ts_pipeline.TrainingPlan` holds a
`TSConfig` plus an overrides dict instead of 37 constructor arguments and two hand-written
20-line override copies (A-1's cause), and its 13 `_*_tag` functions become
`run_naming.run_slug` (today the *label* is from `run_naming` and the *directory* from the tags —
two grammars, and the directory one cannot be recomputed from a stored config). The resume rule
in `run_ts_frame_ablation` (C-8) becomes "the arm's serialized config equals the checkpoint's"
plus "history.json exists", not "checkpoint.pt exists".

### 4.6 Tests

`tests/support.py` (T4-24), `test_ts_transformer.py` split along the audit's 19 segments
(T4-26), the six `test_ts_*.py` files that live in `trajectory_data_process/tests/` moved here
and their 13 red fixtures fixed (T4-23 — the suite's exit code carries no information until
then). After 4.2, each strategy's tests live beside it and build only its sub-config.

### 4.7 Per-slice structure findings from the five reviews (the rest, one line each)

- `dataset.build_series:797-954` mixes admission, datum conversion, projection, resampling, truncation and supervision assembly, with three identical `report.skip` exits → split at the resample boundary.
- `TrajectoryWindows.__init__:1180-1346` (167 lines, eleven concerns) → `_bind_labels` / `_bind_teacher` methods; three copies of "series rows → state matrix + aero params" (`:226`, `:281`, `:517`); `_dynamics_arrays` writes the supervised-span expressions twice verbatim.
- `anchor_strata.py` exists only because `anchor_grid.py:228` calls one attribute read of `dataset`; make `truth_duration_s` a `FlightSeries` property and the split collapses.
- Three implementations of the "sorted, newline-joined, sha256" identity idiom (`data_provenance:42` declared THE one; `dataset:1424`; `development_cohorts:276`).
- Two implementations of the headline common-true-time ADE/FDE (`metrics.common_physical_time_flight_metrics`, predict side; `fixed_anchor_validation.fixed_anchor_common_grid_report_metrics`, train side) with no seam test pinning them equal — `history.json` and `summary.json` ADEs are routinely compared.
- `validation.py:232-275` vs `:589-656` duplicate the unpack → move → forward → replay sequence; `:928-992` restates 35 `arc_length_*` keys as a dict literal; `:747-757` recomputes `remaining_path_profiles` once per bin every epoch against its own docstring.
- `train.evaluate_fixed_anchor_series:325` builds windows with `control_supervision=True` for a replay that never reads it → a per-flight inverse-dynamics solve for train+val after every fit.
- `cross_validation.py:469-479` computes `best_epoch` and `best_selection` by two different rules.
- `control/constraints/{barrier_filter,speed_floor,trombone}` share ~30 lines of counter boilerplate and the lag-compensated bank inversion (code-health #28); `Trombone.__call__` is 138 lines → lift the mode machine.
- `control/basis_fit.py:384-537` holds the fitted-teacher TABLE contract inside the fit module, which is why `dataset` imports the optimiser → `control/fitted_teacher.py`.
- `closure_geometry.py:406-593`: the scipy label fits sit in the module the inference path imports → `closure_fits.py`.
- Dead parameters: `control/heads._initialize_control_head(bank_rad=, feature_std=)`, `ControlOutputHead(bounds=)` and its `None` buffers, `run_ts_frame_ablation.arm_steps(label, config)`; `RolloutStateView.reference` machinery with no live `True` producer (already in code-health "T2 leftovers").
- `flyability.py:169-192` and `control/dynamics/inverse.py:102-143` are two transport-correction implementations agreeing only to O(cos²γ); the load-factor inversion is spelled a third time at `flyability.py:260` (code-health #20).
- `batch_benchmark.benchmark_candidate:232` calls `model(x)` directly, so it cannot benchmark a corridor-bounded or control config and does not say so.
- `experiment_index.begin_run` runs after the dataset build (`cli/common.prepare_training_run:810 → :867`), so a dirty-tree refusal costs the full build.
- `run_naming.meta_items:528-541` and `dropped_meta_diffs:555-563` are the same five lines with different tail slices.

## 5. Retirement candidates — the census says what is live

219 stored runs on disk (`history.json['config']`), 112 loadable. What the vocabularies were
actually set to:

| axis | values seen (runs) | reading |
|---|---|---|
| `prediction_output` | control 153 · state 32 · closure 2 · control-mixture 1 (retired) | closure is a comparison arm (2 runs, 2026-09-05/06) |
| `intent_conditioning` | none 41 · truth-join-duration 2 · truth-join 1 · truth-join-lead 1 | Phase 0 only; the L4 gate failed and the scene encoder was not built |
| `state_position_reference` | absolute 63 · corridor-bounded 4 · anchor-relative 4 (vetoed) | corridor-bounded + `corridor_gate` (never non-default) + `project_final` is one 2026-08 arm family |
| `target_conditioning` | none 79 · channels 6 | 2026-08 |
| `cta_conditioning` | off 35 · given 2 | live line (L3), keep |
| `duration_head` | point 16 · quantile 2 · two-head 1 | live line (B1), keep |
| `control_command_hook` (trained) | off 45 · barrier 3 · nominal-residual 3 | hooks are predict-time overrides; the trained-with-hook arms are 2026-09-06 |
| `control_state_loss_grid` | native 118 · fixed-dt 26 | fixed-dt is 2026-08 only; it drags `fixed_dt_supervision.py` + `control/loss/fixed_dt.py` + a `dense_supervision` slot through every signature |
| `control_state_supervision_clock` | observed 113 · predicted 45 | `observed` is the OLD default (2026-07/08); every 2026-09 run is `predicted` |
| `control_dynamics_backend` | scaled-tcv 92 · reanchored-rk4 32 · tcv 13 (retired) | `scaled-transport-chart-velocity` is the 2026-07/08 backend; every current recipe is `reanchored-rk4` |
| `control_duration_parameterization` | uniform 88 · factorized 63 · direct 1 | both live |
| `coordinate_frame` | enu 208 · airport-enu 8 · runway-aligned 3 | the 2026-09-03 frame ablation; its results doc keeps enu |
| `checkpoint_selection_metric` | common-grid-ade 124 · arc-length-geometry 15 · criteria 8 · anchor-grid 5 · objective 3 · terminal-state 1 | the three 2026-07/08 values are already out of the vocabulary (their 24 runs are among the 107 that no longer load) |
| `procedure_loss_*` | never non-default in a stored run (the diverged primal-dual arm the hard-constraints plan cites left no `history.json` under `outputs/`) | **keep**: the 2026-09-08 hard-constraints plan names `objective.procedure_loss`, `ProcedureMultipliers` and `PROCEDURE_LOSS_FIELDS` as the terms it extends (its line 702). Under 4.2 it is the one objective term that acts on BOTH the state and the control path, so it lives in `outputs/base`, not in either strategy |
| `random_train_anchor_sampling` | uniform 16 · remaining-path-uniform 3 | live (A0.b) |

51 fields have never been non-default in any stored run; 20 of them are backbone hyperparameters
(no run ever changed `n_heads`, `dropout`, `activation`, `patch_len`, `stride`, `revin`, …), 9
are scale / time-constant / gain fields and 7 are loss weights that every recipe pins. A field
nobody has ever set is not automatically dead — the three time constants feed the first-order-lag
backend through `control_time_constants_s` — but it is a field the CLI, the run grammar and the
validators carry for nothing. Two consequences the user decides, not this review:

- **Freeze, don't delete, what has a published number**: closure, corridor-bounded, intent
  truth-join, fixed-dt, the observed clock, scaled-tcv, airport-enu / runway-aligned. Under 4.2 a
  frozen strategy costs nothing in the spine — it is a sub-package nobody imports unless its
  checkpoint is loaded — and a frozen *value* is one entry in a stored-vocabulary tuple (the
  `CONTROL_HOOKS` / `CONTROL_HOOKS_AVAILABLE` split the audit already introduced).
- **Delete or archive what has no run and no number**: `corridor_gate=faf` (never set),
  `scene/features.py` together with `run_ts_scene_explainability.py` and the scene half of
  `intent_explainability.py` (L4 gate failed, encoder not built — archive as one campaign, the
  way the oracle teacher was), `train_only_diagnostics.py` (one unreferenced helper, already in
  code-health-followups), the dead parameters listed in 4.7.

**Resolution (2026-09-10, `dev-pkg-review`).** Frozen — a stored checkpoint loads, predicts
and publishes, a new run cannot select the value (`*_AVAILABLE` tuples in `config.py`,
refused at both CLI doors through `cli.common._NEW_RUN_VOCABULARIES`): `closure`,
`intent_conditioning=truth-*`, `control_state_loss_grid=fixed-dt`, `coordinate_frame` ∈
{`airport-enu`, `runway-aligned`}. **NOT frozen, correcting the table above:**
`control_state_supervision_clock=observed` and
`control_dynamics_backend=scaled-transport-chart-velocity` are what `control_simple_v1_overrides`
PINS — every simple-v1/v1-lag/v2/v3 recipe trains under them (the lag model needs the
transport-chart backend), so freezing either would refuse the recipes; and `corridor-bounded`
is the adopted candidate default of the state path (`CLAUDE.md`). Deleted: the `faf` gate
with its whole read path (`final_approach_fix_distance`, the `final_approach_fix_m` context
key, `soft/hard_inside_faf`, the `d_faf` argument) and the `corridor_gate` field itself
(`RETIRED_CONSTANT_FIELDS`: 67 stored configs carry `on-final` and drop it on load; 0 run
names moved); `train_only_diagnostics.py`; the `_initialize_control_head(bank_rad=,
feature_std=)` and `arm_steps(label, config)` dead parameters. Archived:
`archive/scene_encoder_2026_09/` (`scene/features.py`, `run_ts_scene_explainability.py`, its
test; `intent_explainability.py` stays live for the Phase 0 diagnostics). Kept, correcting
§4.7: `RolloutStateView.reference` has a live producer since L3.f (the trombone's
reference-rollout estimator declares `needs_reference`), so it is not dead.

## 6. Order and proof

Every step is one worktree, one review, one commit, merged between campaigns (a formal run imports
the main tree). The acceptance test is the one T3 used: the six CPU fixtures give bit-identical
histories, the 112 loadable stored configs still load, and zero run names move.

0. **The §2 A/B bugs first, as separate small commits, before any restructuring** — A-1 and A-2
   change published numbers of two runners, B-1/B-2 break `--batch-size auto` on every custom
   arm; each is a ten-line fix with a test, and none needs the refactor. A-4 is a decision for the
   L3.f pre-registration (the band is the soft gate's shoulder; either narrow the barrier's soft
   gate to the hard one under a trombone, or state the overlap in the readout).
1. **4.1 package** — one day, zero behaviour. Unblocks every later grep.
2. **§5 deletions** the user approves — reduces what the next steps carry.
3. **4.3 config split** with the flat-dict adapter — the validators shrink first, so the strategies
   in the next step get small constructors. C-1…C-4, C-13, C-14 fall out here.
4. **4.2 strategies**, one at a time: `state` (smallest), then `closure` (already closed-form),
   then `control` (largest; its sub-modules already exist under `control/`). `dataset` loses its
   `control.*` imports at the third one; B-1 becomes unrepresentable.
5. **4.4 loop and predict** — pure extraction once the strategies own their diagnostics; C-5, C-6
   fall out here.
6. **4.5 runners and 4.6 tests** — the audit's T4/T5, mechanical after 4.1; A-1, C-8, C-9 fall
   out here.

Expected size after 1–6: ≈ 19k source lines from 27.6k (the §5 deletions and the runner copies
account for most of it), `config.py` ≈ 900 lines across seven small dataclasses, no function
over 150 lines, no signature over eight parameters, no import cycle, and a fourth output
strategy that touches three files.


## 7. Resolution (2026-09-09, branch `dev-pkg-review`)

Every §2 finding was re-verified on the tree before it was touched (A-3 and A-4 were already
fixed at `c544db0`). The rule for the rest: a fix that could refuse a stored artifact was
first counted against the 219 stored `history.json` configs, and a fix that would move a
stored run's name or directory was not made without saying so. Tests: one per finding, in
the module's own test file.

| finding | state | where |
|---|---|---|
| A-1 | **fixed** — `TrainingPlan._plan_overrides` is the ONE dict both reuse checks and the label read; `control_dynamics_model` in it; `_lag` tag on `train_dir` / `pred_dir` / category (point-mass tag empty, and no pipeline-shaped directory on disk held a lag run, so nothing is orphaned); the never-emitted `control_bank_time_constant_s` plan field deleted | `run_ts_pipeline.py`, `trajectory_data_process/tests/test_ts_pipeline.py` |
| A-2 | **fixed** — `dataset.fixed_anchor_index` + `FixedAnchorTrajectoryWindows.anchor` / `.anchor_indices`; every `fixed_anchor_*` metric takes `anchor=` (required), the validation plan's truth is built at `dataset.anchor_indices`, the cohort floor and `fit_model`'s check read the floor. **Every number `run_ts_history_ablation.py` published before this is stale** (the runner itself is unchanged: it already passed the floor) | `dataset.py`, `fixed_anchor_validation.py`, `validation.py`, `train.py`; `tests/test_anchor_grid_selection.py` |
| A-3, A-4 | fixed at `c544db0` | — |
| B-1 | **fixed** — `probe_dynamics(batch_size, device, config)` adds the imitation / heading-rate / CTA keys under the dataset's conditions; a test pins the key set equal to a real batch's | `dataset.py`, `batching.py`; `tests/test_supervision_terms.py` |
| B-2 | **fixed** — `replace(prediction, segment_durations=…)` keeps `duration_quantiles_s` (and a latent prediction's fields) | `batching.py`; `tests/test_ts_transformer.py` |
| B-3 | **fixed** — the accuracy block is built before the first record is written; the dead null branch for ADE/FDE/endpoint rows removed | `export.py`; `tests/test_ts_transformer.py` |
| B-4 | **fixed** — a one-sample remainder is refused with its numbers, like the imitation sibling | `dataset.py`; `tests/test_supervision_terms.py` |
| C-1 | **fixed** — `control_velocity_loss_weight` / `control_imitation_loss_weight` join the two later siblings under the "built by `true-time-position` only" rule (census: no stored run carries either under another objective) | `config.py`; `tests/test_supervision_terms.py` |
| C-2 | **fixed** — `off` is held to the barrier-gain rule, and `control_hook_saturation≠soft` under `off` is refused; `nominal-residual` (stored-only) stays exempt (census: 45 stored `off` runs all at the defaults) | `config.py`; `tests/test_command_hook.py` |
| C-3 | **fixed (decided the same evening)** — `validation_common_grid_points` names the run (`grid-points=`; stored runs all at 64, 0 renamed) and the reverse guard (`KNOWN_UNNAMED_FIELDS`, asserted at import) lists every unnamed field with its reason. The three identity-bearing fields are named too: `lr-patience=` / `lr-factor=` / `anchor-min-future=`. Measured over the 219 stored configs: 146 display names / slugs moved (75 spelled, 71 folded-count-and-hash only), 0 loadability changes, no directory or key moves; 132 published frontend categories carry the old label (109 publisher-managed → `--refresh-labels-only`, 23 hand-published `ts_*` → `docs/relabel_published_categories.py`). `device` stays excused | `run_naming.py`; `tests/test_run_naming.py` |
| C-4 | **gate fixed, floor DECISION** — `corridor_gate≠on-final` is refused off `corridor-bounded` (0 stored affected). `control_duration_uniform_floor` cannot be refused under `uniform`: its default is 0.8 and the recipes pin 0.0, so 88 stored uniform runs carry a non-default inert value — it moves into `Factorized(floor)` in the §4.3 split | `config.py`; `tests/test_final_constraint.py` |
| C-5 | **fixed** — `--aircraft-type` is replaced into the config predict writes; it joins `PREDICT_CONFIG_FLAGS`, and `HOOK_TUNING_FIELDS` is spelled rather than derived | `cli/predict.py`; `tests/test_cta_conditioning.py` |
| C-6 | **fixed** — one `emit` closure writes every directory with the same `skipped` (pinned on the source: one `write_batch(` call in `run_cli`) | `cli/predict.py`; `tests/test_cta_conditioning.py` |
| C-7 | **dead checks removed; directory binding is a DECISION** — the split and provenance digests stay in the ledger as audit fields but are no longer re-checked (they cannot fire once the file digest matched). Binding the ledger to the checkpoint DIGEST rather than its directory needs a registry outside the run directory (where; how tests isolate it) — no `test_release.json` exists on disk today, so the change is free of artifact impact whenever it is made | `evaluation_protocol.py` |
| C-8 | **fixed** — resume = `history.json` exists AND every DECLARED field agrees with the trained config; a checkpoint without a history is refused by name (move it aside as `<key>.aborted-<UTC>`), and the override file is written only after that check | `run_ts_frame_ablation.py`; `tests/test_frame_ablation_runner.py` |
| C-9 | **DECISION** — renaming `mean_/std_val_macro_loss` to what they hold (the selection metric) means a `cv_results.json` schema bump; the two stored files (2026-08-16 POOLED PatchTST, 2026-08-18 KSJC τ-bank CV) would then no longer be reusable by `--skip-cv`. The writer is documented in place; nothing renamed |
| C-10 | **fixed** — `invalid_flights: 0` and the `valid_segments` mask that could never be False are gone; `prediction_horizon_cap_rate` stays (it is legitimately 0 on those outputs) | `fixed_anchor_validation.py`, `validation.py`, `metrics.py` |
| C-11 | **fixed** — `project_onto_final` translates by `series.target_chart` in and out, like `cut_at_threshold_crossing`; a test shows the two frames agree to the metre | `forecast.py`; `tests/test_final_constraint.py` |
| C-12 | **DECIDED 2026-09-09: the grader stays at 0.5; neither number moves.** The head's box floor (0.2) is the old path's search space, the grader's floor (0.5) is the flyability claim every published number was read against. The plan-and-guidance design's guidance layer commands the load factor inside `flyability`'s envelope (its §11), so the new path cannot inherit the gap. How often a trained head emits n < 0.5 stays unmeasured (`control/training/diagnostics.py` saturation counts would say) and matters only for reading the old arms |
| C-13 | **fixed** — the rule is the bound it cites (`RK4_REAL_AXIS_STABILITY_LIMIT = 2.785`); loosening refuses nothing stored | `config.py`; `tests/test_control_inverse_dynamics.py` |
| C-14 | **fixed** — `control_state_supervision_clock` in `REQUIRED_SERIALIZED_CONTROL_FIELDS` | `config.py` |
| C-15 | **fixed** — `interval_stratum` picks from the INTERSECTION of calibrated strata across α; `strata_masks` refuses a present-null covariate | `calibration.py`, `approach_difficulty.py`; `tests/test_eta_calibration.py` |
| C-16 | **fixed** — every candidate row carries `silhouette_rows` beside `rows` (`SILHOUETTE_MAX_ROWS = 2000`) | `approach_clustering/model.py` |
| C-17 | **documented** — what an all-zero imitation mask means, on both sides | `dataset.py`, `objective.py` |
| C-18 | **fixed** — `fixed_anchor_fraction` compares to `fixed_anchor_index`, not `seq_len − 1` | `dataset.py` |
| C-19 | **fixed** — one `refuse_airport_override_for_pooled_data` for train, cross-validate and predict | `cli/common.py`, `cli/predict.py` |
| C-20 | **fixed** — `input_channels` present-null; the inert `offsets[~active]` line; the zero-ground-speed `gamma` fallback (`atan2` handles it, and V·sin γ = udot again); `strictly_increasing` accumulates first and ramps after; the dataset refuses a CTA mode it cannot fill instead of letting the head build an empty token; `anchor_state` renamed; `generated_at` is `utc_now()`; the plan's dead τ field deleted; the `CLAUDE.md` `overlap` claim corrected. Left: `intent_explainability`'s clip fallback (the scene half is a §5 archive candidate) | various |

**§6 step 1 (the package)** was done on the same branch after the fixes — see the changelog
entry. Steps 2–6 (§5 deletions, the config split, the strategies, the loop/predict extraction,
the runners) are the user decisions this document already names, unchanged.
