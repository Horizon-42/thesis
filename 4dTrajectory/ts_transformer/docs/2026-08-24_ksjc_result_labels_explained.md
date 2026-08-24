# KSJC result labels, expanded — every Prediction / Experiments entry explained

The Observe panel's **Prediction** and **Experiments** pickers name every published
result with one grammar (single source: `4dTrajectory/ts_transformer/run_naming.py`):

```
<split prefix> — <kind>: <output> · <backbone> · <dynamics> · <loss design> · <meta, …>
```

Every part is derived mechanically from the run's stored config
(`history.json['config']`, also serialized into the checkpoint), so a label can always
be recomputed without touching artifacts (`publish_ts_experiment_trajectories.py
--refresh-labels-only`, `docs/relabel_published_categories.py`). This document expands
each part of the grammar, then walks through **all 27 KSJC prediction/experiment
categories**, family by family. ADE figures quoted below are the batch **mean ADE on
that category's split** as published in `categories.json` (`accuracy.adeM.mean`, the
same numbers the "Sort results" control ranks by).

---

## 1. Reading a label, part by part

Worked example (a real KSJC entry):

```
Validation split (model selection) — Predicted: control · iTransformer ·
first-order-lag @scaled-transport-chart-velocity · simple-v2+(vel=0.05) ·
d-model=512, d-ff=1024, layers=4, N=64, batch=512, lr=3e-05, +4 more,
wiggle_loss_design/velocity_32x
```

### 1.1 Split prefix — which data the numbers describe

| Prefix | Meaning |
| --- | --- |
| `Training split (in-sample)` | Flights the checkpoint was trained on. Error here measures *fit*, not skill. |
| `Validation split (model selection)` | Held-back flights used to pick the best epoch. The number to compare models on. |
| `Test split (held-out)` | The outer test set; only released deliberately (`--test-release`). |

The split is by **flight** (never by window) and is persisted inside the checkpoint —
`config.py` (`val_fraction` / `test_fraction`, split rules in `dataset.py`). Defined in
`run_naming.SPLIT_DISPLAY`.

### 1.2 Kind — `Predicted:` vs `Experiment:`

- **`Predicted:`** — published under the *Prediction* result source: a primary result
  meant to stand on its own.
- **`Experiment:`** — published under the *Experiments* source with checkpoint
  metadata for the grouped experiment picker: an arm of a sweep, meant to be compared
  against its siblings.

The distinction is the publisher's `--result-source` flag
(`publish_ts_experiment_trajectories.py`), not a property of the model. KSJC's legacy
`ts_ksjc_*` categories predate the flag and appear under *Prediction* by the `ts_`
key-prefix rule (`aeroviz-4d/src/utils/trajectoryResultSources.ts`,
`categoryResultSource`).

### 1.3 Slot 1 — output: what the network emits

| Value | Meaning |
| --- | --- |
| `state` | The purely **kinematic baseline**: chart channels in, future chart channels out. No dynamics anywhere — deliberately, so the learned component is measured on its own. Its predictions carry no flyability guarantee. |
| `control` | The model emits a **bounded control schedule** (thrust fraction, bank, load factor) and a differentiable RK4 rollout of the shared point-mass equations turns it into the trajectory — every prediction is dynamically admissible by construction. |

Defined: `config.py` `PREDICTION_STATE` / `PREDICTION_CONTROL`; the design rationale is
the "`prediction_output` decides whether dynamics is connected at all" entry in
`4dTrajectory/ts_transformer/CLAUDE.md`. The control box is
`control/envelope.py` (thrust ∈ [−0.2, 1.0] of installed, bank ∈ ±45°, load ∈ [0.2, 2.0]).

### 1.4 Slot 2 — backbone: the sequence model

| Value | Meaning |
| --- | --- |
| `iTransformer` | Attention across *channels* (variate tokens). Vendored byte-identical in `vendor/itransformer/`. |
| `PatchTST` | Channel-independent patched attention along *time*. Vendored in `vendor/patchtst/`. Channel independence cannot represent a turn's east/north coupling — one reason it trails at long horizons. |

Defined: `config.py` `MODELS`; instantiation in `models.py`.

### 1.5 Slot 3 — dynamics: the flight model behind the rollout

| Value | Meaning |
| --- | --- |
| `kinematic` | State output only: **no dynamics attached** (see slot 1). |
| `point-mass` | Each commanded control is applied **instantly** — controls have no derivative. |
| `first-order-lag` | The three controls become actuator states driven toward the commands through first-order ODEs (**controls have a derivative**). Time constants τ = 1.5 / 2.0 / 0.8 s (thrust / bank / load); a non-default τ would appear inline, e.g. `first-order-lag(τ-bank=4s)`. It *wraps* the same force equations as point-mass — one flight model plus three actuators, not a second model. |

The `@…` suffix is the **rollout backend** — the state representation the long rollout
integrates — shown only when it differs from the default `reanchored-rk4`:

| Backend | Meaning |
| --- | --- |
| `reanchored-rk4` (default, unmarked) | Local-ENU RK4 step, re-anchored into geodetic state every sub-step. |
| `@transport-chart-velocity` | Integrates threshold-chart position + moving-local-ENU physical velocity with the full WGS84 transport rate. |
| `@scaled-transport-chart-velocity` | Same, with the chart's normalization scaling — **what every KSJC control run uses**. |

Defined: `config.py` `CONTROL_DYNAMICS_MODELS` / `CONTROL_DYNAMICS_BACKENDS`; the
registry keyed by the *(model, backend)* pair is `control/dynamics/backends.py`. The
two axes are orthogonal — the "Two orthogonal dynamics axes" entry in the package
`CLAUDE.md`.

### 1.6 Slot 4 — loss design: what the training objective scored

**Named recipes** (frozen field bundles, `config.py` `CONTROL_RECIPE_*` +
`control_recipe_overrides()`; a run constructed with a recipe name may not silently
change a frozen field — `TSConfig.__post_init__` raises):

| Recipe | What it scores | Relationship |
| --- | --- | --- |
| `simple-v1` | True-time **position** at the 64 segment endpoints + final time + endpoint term. Point-mass dynamics. | The minimal control objective. Its measured failure: position-only supervision let 71 % of predicted bank energy collapse into one shared wiggle profile. |
| `simple-v1-lag` | Same objective; **first-order-lag** dynamics. | v1 plus the actuator lag. The τ_bank CV sweep ran on this. |
| `simple-v2` | v1-lag **+ velocity term** (`control_velocity_loss_weight = 0.003` ≈ 2× the position term at convergence). | The velocity dose fixed the shared-wiggle profile (17.3 % shared vs 71 %). |
| `simple-v3` | v2 **+ direct control imitation** (`control_imitation_loss_weight = 64` ≈ 47× position) against the inverse-dynamics teacher (`control/dynamics/inverse.py`). | The current recommended recipe: bank skill 0.124 → 0.735 on KRDU. On KSJC the *mechanism* replicates but 64 overshoots — recalibrate per airport. |

**`custom` runs are named against their *nearest* recipe** — the recipe whose frozen
loss fields need the fewest edits to reproduce the run — with the edits in
parentheses:

- `simple-v2+(vel=0.05)` = exactly simple-v2's loss with
  `control_velocity_loss_weight` set to 0.05 instead of 0.003.
- A run matching a recipe's loss fields *exactly* is shown as that recipe bare, even
  though its `control_recipe_name` says `custom` (CV/dev runs are always stamped
  `custom` — "a named recipe cannot be cross-validated as itself").
- More than 4 edits collapse to a stable content hash: `custom-3f2a91bc`.

Edit abbreviations seen in loss slots (full list: `run_naming._ABBREV`):
`vel` = `control_velocity_loss_weight`, `imit` = `control_imitation_loss_weight`,
`obj` = `control_state_objective`, `grid` = `control_state_loss_grid`,
`clock` = `control_state_supervision_clock`.

For **state** runs the loss slot is `state-v1`: the formal direct-state objective
(normalized position MSE + duration + endpoint term) at its frozen coefficients;
deviations would appear the same way (`state-v1(endpoint=0.5)`).

### 1.7 Slot 5 — meta: everything else worth knowing, possibly empty

Three kinds of items, in order:

1. **Horizon**, only when not the default `normalized`: `full horizon` (one 600 s
   pass) or `recursive window` (chained 60 s windows).
2. **Config deviations from *today's* dataclass defaults** (`TSConfig` in
   `config.py`), seed first, at most 6 spelled out, the rest folded into `+N more`.
   Fields frozen by the run's *named* recipe are excluded — which is why a
   `simple-v1-lag` run shows a clean tail while a `custom` run at the *same* capacity
   spells it out (the capacity is a deviation from the dataclass defaults, and nothing
   froze it).
3. **Caller extras**: the run's identity — `campaign/arm` for relabelled legacy runs,
   the run id for publisher-managed ones, `pooled cohort` for pooled trainings.

Meta abbreviations appearing at KSJC (default in parentheses — a shown value *is* a
deviation from it):

| Item | Field (`config.py`) | Meaning |
| --- | --- | --- |
| `d-model=512` | `d_model` (256) | Transformer embedding width. |
| `d-ff=1024` | `d_ff` (512) | Feed-forward width. |
| `layers=4` / `layers=6` | `e_layers` (3) | Encoder depth. |
| `N=64` / `N=32` | `n_segments` (16 for iTransformer, **256 for PatchTST**) | Normalized-progress segments the horizon is split into — for control runs, also the number of control segments. The default is per-model (`DEFAULT_N_SEGMENTS_BY_MODEL`), so `N=64` reads as a deviation on a PatchTST run but an iTransformer run at 16 shows nothing; on named-recipe control runs the 64 is recipe-frozen and hidden, on `custom` runs it is spelled out. |
| `batch=512` / `batch=128` | `batch_size` (2048) | Training batch size (128 = 4× the optimizer updates per epoch). |
| `lr=3e-05` / `lr=0.0001` | `learning_rate` (5e-4) | Adam learning rate. |
| `seed=…` / `split-seed=…` | `seed` (1337) / `split_seed` (None = seed) | Init seed / flight-split seed. `split-seed` shows only when it *differs* from `seed` (same split, different init — a seed ablation). |
| `fleet=openap-direct` | `aircraft_filter` ("all") | Trains only flights whose type OpenAP models directly. |
| `duration=uniform` | `control_duration_parameterization` ("factorized") | One total time + equal segment durations, vs learned softmax partition. |
| `duration-floor=0` | `control_duration_uniform_floor` (0.8) | Uniform-duration reserve fraction. |
| `grad-clip=20` | `control_gradient_clip_norm` (0 = off) | Global gradient-norm cap. |

**The `+4 more` on every legacy KSJC control arm is always the same four items**:
`fleet=openap-direct, duration=uniform, duration-floor=0, grad-clip=20` — the
campaign-wide dev settings, identical across arms, so they never distinguish two KSJC
entries from each other.

---

## 2. The KSJC catalog

27 prediction/experiment categories. (`observed`, `fitted_adsb`, `runway`,
`runway_cons` are the observed baseline and optimizer categories — different result
sources, out of scope here.)

### 2.1 Pooled state baselines — 4 entries under *Prediction*

```
state · iTransformer · kinematic · state-v1 · pooled cohort            (train / val)
state · PatchTST · kinematic · state-v1 · N=64, pooled cohort          (train / val)
```

The two backbones trained as pure kinematic predictors on the **pooled five-airport
cohort** (hence `pooled cohort`), normalized-time horizon, default state objective.
They are the "statistically plausible but unflyable" reference point the control
models are measured against. `N=64` appears only on PatchTST because the default is
**per-model**: the iTransformer run trained at its default N=16 (no deviation, so no
item), while the PatchTST run trained at 64 against its default of 256
(`DEFAULT_N_SEGMENTS_BY_MODEL` in `config.py`).
Checkpoints: `4dTrajectory/outputs/POOLED/ts_{itransformer,patchtst}_normalized_time/`.
Val ADE (KSJC slice): iTransformer 925 m, PatchTST 1361 m.

### 2.2 Pooled control model — 2 entries under *Prediction*

```
control · iTransformer · point-mass @scaled-transport-chart-velocity · simple-v1 ·
ts_itr_control_simple_v1_teacher_all_airports_seed1337                 (train / val)
```

The all-airports control checkpoint: bounded controls through the point-mass rollout,
`simple-v1` objective, trained **with the oracle teacher** (`control/oracle/*`). The
teacher is a training-time curriculum, *not* a `TSConfig` field — two runs differing
only in teacher use have byte-identical configs, which is exactly why the run id stays
in the label as the meta tail. Checkpoint:
`4dTrajectory/outputs/POOLED/ts_itr_control_simple_v1_teacher_all_airports_seed1337/`.
Val ADE (KSJC slice): 814 m.

### 2.3 Teacher ablation — 4 entries under *Experiments* (campaign `control_simple_v1_20260816`)

```
Experiment: control · iTransformer · point-mass @scaled-transport-chart-velocity ·
simple-v1 · development_teacher_current_manifest_seed1337              (train / val)
Experiment: … · simple-v1 · development_no_teacher_current_manifest_seed1337 (train / val)
```

The controlled pair behind §2.2's design choice: same recipe, same seed, same data
manifest — **only the oracle teacher differs**, visible only in the run-id tail
(`…_teacher_…` vs `…_no_teacher_…`). Measured on KSJC: teacher 602/696 m ADE
(train/val) vs no-teacher 776/864 m — the teacher arm wins both splits. Runs:
`4dTrajectory/outputs/KSJC/experiments/control_simple_v1_20260816/`.

### 2.4 Flight-model pair — 2 entries (campaign `flight_model_paired`)

```
… · point-mass @scaled-transport-chart-velocity · simple-v1 · flight_model_paired/point_mass
… · first-order-lag @scaled-transport-chart-velocity · simple-v1-lag · flight_model_paired/first_order_lag
```

The paired point-mass vs first-order-lag comparison (same 1083 validation flights):
the lag arm is smoother (jerk p95 −28 %) *and* slightly more accurate (val ADE 776 m
vs 815 m, better on 67.4 % of flights) — "artifact removed", not "more realistic";
gates unchanged 0/1083 on both. These arms ran as true named recipes, so their meta
tail is clean: every capacity/budget field is recipe-frozen. Full write-up: the lagged
flight-model entries in the package `CLAUDE.md`; runs under
`4dTrajectory/outputs/KSJC/experiments/flight_model_paired/`.

### 2.5 Imitation replication — 3 entries (campaigns `imitation_replication`, `imitation_v5`)

```
… · simple-v2 · …, imitation_replication/ksjc_baseline     (val ADE 640 m)
… · simple-v3 · …, imitation_replication/ksjc_v3           (val ADE 664 m)
… · simple-v2 · …, imitation_v5/v5_baseline                (val ADE 656 m)
```

Does KRDU's `simple-v3` result replicate on KSJC? The loss slot answers directly:
`ksjc_baseline` is exactly simple-v2's loss content, `ksjc_v3` exactly simple-v3's
(both trained as `custom`, hence the spelled-out capacity tail). Result: the
*mechanism* replicates and is stronger (bank skill 0.197 → 0.678, past the 0.543
same-runway twin ceiling) — but the imitation weight 64 **overshoots on KSJC**
(straight-in bank 0.18° vs flown 0.53°; FDE degrades on 68 % of flights), because bank
carries only 18 % of the box-normalized imitation term here vs 41 % at KRDU. Hence the
standing rule: recalibrate the weight per airport, never inherit 64. `imitation_v5`
re-ran the v2 baseline against a later manifest generation.

### 2.6 The bank-wiggle diagnosis — 9 entries (campaigns `wiggle_*`)

Background: under `simple-v1(-lag)`, position-only supervision let the model share one
bank profile across flights (71 % of bank energy; per-flight skill ≈ 0). These
campaigns tested four candidate causes on identical data. Full write-up with figures:
`docs/2026-08-19_control_bank_wiggle_diagnosis.zh.md`; conclusions frozen in the
package `CLAUDE.md` ("three plausible causes … are NOT it").

**`wiggle_loss_design` — the velocity-dose ladder (5 arms), the axis that WAS it:**

| Label (loss slot) | `control_velocity_loss_weight` | ≈ × position term | Val ADE |
| --- | ---: | ---: | ---: |
| `simple-v2+(vel=0.0008)` (`velocity_half`) | 0.0008 | 0.5× | 696 m |
| `simple-v2` (`velocity_double`) | 0.003 | 2× | 640 m |
| `simple-v2+(vel=0.0125)` (`velocity_8x`) | 0.0125 | 8× | 665 m |
| `simple-v2+(vel=0.05)` (`velocity_32x`) | 0.05 | 32× | 671 m |
| `simple-v2+(vel=0.2)` (`velocity_128x`) | 0.2 | 128× | 688 m |

Note what the mechanical naming exposed: the old "velocity 2x" arm **is exactly
simple-v2** — 0.003 was frozen into the recipe precisely because this ladder showed
the 2× dose is the sweet spot (past 8× the bank drops *below* the flown tracks' own
level while FDE climbs — over-constraining looks like the blandness trap).

**`wiggle_conditioning` — capacity (2 arms), not the cause:**
`deep` (`layers=6`) and `wide` (`d-model=1024, d-ff=2048`). Width moved the shared
share (70.7 → 48.8 %) but per-flight skill stayed exactly 0.000; with the velocity
term present, width adds nothing and costs ADE.

**`wiggle_segment_count` — N (2 arms), not the cause:** `n16` and `n32` (vs the
campaign's 64). Halving N moves the shared share 12 pp and is not even monotone.
(`n16`'s label shows no `N=` item — 16 *is* today's iTransformer default, so it reads
as no deviation; `n32` shows `N=32`.)

**`wiggle_training_budget` — optimization budget (2 arms), not the cause:**
`batch128` (4× the updates) and `lr3x` (`lr=0.0001`). The two ways of adding optimizer
steps *disagree* — more optimization just fits the shared profile harder.

**`wiggle_combination` — 1 arm:** `velocity2x_wide` = the v2 velocity dose **and**
`d-model=1024` together (loss slot reads `simple-v2`, capacity in meta). Confirmed the
velocity term does the work: with it present, width adds nothing (val ADE 689 m vs
plain v2's 640 m).

---

## 3. Where everything is defined

| Concern | Definition |
| --- | --- |
| The grammar, abbreviations, nearest-recipe rule, meta cap | `4dTrajectory/ts_transformer/run_naming.py` (tests: `tests/test_run_naming.py`) |
| Recipes and every config field/default | `4dTrajectory/ts_transformer/config.py` (`CONTROL_RECIPE_*`, `control_recipe_overrides`, `TSConfig`) |
| Dynamics models/backends registry | `config.py` `CONTROL_DYNAMICS_*`; `control/dynamics/backends.py` |
| Control box | `control/envelope.py` |
| Label producers | `publish_ts_experiment_trajectories.py` (+ `--refresh-labels-only`), `run_ts_pipeline.py` (pipeline categories), `docs/relabel_published_categories.py` (the legacy one-off) |
| Picker classification & ranking | `aeroviz-4d/src/utils/trajectoryResultSources.ts` (`categoryResultSource`, `sortCategoriesByAccuracy`, `experimentOptions`) |
| Per-category ADE/FDE aggregates | `categories.json` `accuracy` block, stamped by `build_scenario_comparison_czml.category_accuracy_summary` |
| Run directories behind each KSJC arm | `4dTrajectory/outputs/KSJC/experiments/<campaign>/<arm>/` (index: `…/experiments/INDEX.md`) |
