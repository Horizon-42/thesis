# ts_transformer reference — layout

Full text behind the index lines of `4dTrajectory/ts_transformer/CLAUDE.md`, moved here
VERBATIM on 2026-09-16 when that file had grown to 118 KB (1,202 lines) and was loaded into
every ts session. Only the `### <ID>` headings are new, plus two corrected references: L1 pointed at a "Conventions" section that never existed (it is L22), and L22's example imported `ts_transformer.control.envelope`, which is `ts_transformer.outputs.envelope` since the §4.2 layout. Every heading has exactly one
index line in CLAUDE.md that ends in its ID; find one with `grep -n '^### C7 ·' docs/reference/*.md`.

**Maintenance: when a fact here changes, edit it HERE and keep its index line true; a new fact
gets a new ID here and ONE new line in the index.** Measurements and campaign evidence still
belong in `docs/ENGINEERING_NOTES.md` / the design documents, status in `docs/OPEN_ITEMS.md`.

**2026-09-18:** the main line's 2026-09-14…09-17 additions to that CLAUDE.md (the control
contracts, the two-tier axes and traps) were placed here the same way when this branch was
rebased — same rule, new IDs.

## Layout

### L1 · a regular package under `4dTrajectory/`

A regular package under `4dTrajectory/` (`ts_transformer/__init__.py`, `4dTrajectory/pyproject.toml`);
modules are imported by their qualified names — see the package rule, L22.

### L2 · grouped by plane

**The package is grouped by plane** (2026-09-10, the review's §4.2 layout, done last as pure
moves — every module keeps its name and content, only its directory changed):

| directory | what lives there |
|---|---|
| `data/` | the data plane: `dataset`, `splits`, `data_provenance`, `channels`, `coordinate_frames`, `time_grids`, `anchor_grid` / `anchor_strata`, `lateral_eligibility`, `reference_velocity`, `target_conditioning` / `intent_conditioning`, `synthetic`, `development_cohorts`, `approach_difficulty`, `fixed_dt_supervision`, `batch_contract` |
| `geometry/` | final-approach geometry and the metrics read off a trajectory: `final_approach_geometry`, `arc_length_geometry`, `geometric_metrics`, `metrics`, `flyability`, `physical_criteria`, `terminal_state_loss` |
| `backbone/` | `adapters` (the one interface over the two vendored networks, formerly `models.py`) and `vendor/` |
| `outputs/` | one package per prediction path behind one strategy (§4.2): `state/`, `closure/`, `control/`, `plan/`; `base`, the lazy registry, `duration_heads` |
| `training/` | `train`, `validation`, `fixed_anchor_validation`, `objective`, `batching`, `cross_validation`, `training_performance`, `experiment_index` |
| `inference/` | `forecast`, `calibration`, `export`, `evaluation_protocol`, `intent_explainability`, `build_multiflight_capacity_report` |
| `cli/` | one module per subcommand, `common`, and `benchmark_batch` (formerly `batch_benchmark.py`) |
| `experiments/` | the runners, behind `run_ts.py <name>` |
| top level | `config`, `run_naming`, `io_utils`, `repo_layout`, `__main__` — what every plane reads |

A group's `__init__.py` re-exports nothing: a group is a directory, not a namespace, and
the torch-free boundaries (`data.data_provenance`, `data.splits`, `inference.evaluation_protocol`,
`data.anchor_strata`) survive only because importing one module of a group imports nothing else
of it (`tests/test_import_boundaries.py`).

### L3 · the runners behind `run_ts.py <name>`; `repo_layout.py`

**The experiment runners are `experiments/<name>.py` behind ONE door** (2026-09-10, review §4.5):
`python run_ts.py <name> [args]` at the repository root (`--list` prints the names with their
first docstring line), or `python -m ts_transformer.experiments <name>` with `4dTrajectory/` on
the path. They used to be 22 `run_ts_*.py` files at the root, each inserting `4dTrajectory/`
into `sys.path` by hand. A runner is a CONSUMER of the package — it may import anything in it,
and nothing in the package imports a runner (`tests/test_architecture.py`; `batch_benchmark`
used to import the pipeline runner for the harvest paths). **`repo_layout.py` is the one
definition of where this repository keeps things** — `REPO_ROOT`, `HARVEST_ROOT`,
`OPT_OUTPUTS_ROOT`, `COMPARISON_AIRPORTS_ROOT`, `TS_SCRIPT`, `discover_k_airports` /
`arrival_manifest_path` — read by the CLI, the benchmark and the runners (`experiments/support.py`
re-exports them and holds the runners' own `series_digest`; `parse_airports` is `cli.common`'s).
The pipeline's `HARVEST_ROOT` / `OPT_OUTPUTS_ROOT` are module globals on purpose: the tests
point them at a fixture harvest with `monkeypatch`.

### L4 · the training plane's modules and the two edges not to reverse

**Five modules carry the training plane, and each answers one question** (T3, 2026-09-07 —
`training/train.py` was 3,027 lines and `__main__.main` 714):

| module | the question | lines |
|---|---|---|
| `training/objective.py` | what is a prediction scored against | 1,079 |
| `training/validation.py` | how is a fitted model replayed on a split, and which epoch is kept | 1,162 |
| `training/train.py` | the epoch, the cohort, the checkpoint — `fit_model` is `prepare_session` (a `TrainingSession`) → per epoch `train_epoch` / `validate_epoch` / `procedure_update` → the selection, with `describe_session` / `describe_epoch` holding every print (2026-09-10, review §4.4) | 1,407 |
| `data/dataset.py` | observed arrivals → model windows (`FixedAnchor` = one common anchor, `ExplicitAnchor` = one CALLER-SUPPLIED anchor per flight, `RandomAnchor` / `RemainingPathUniformAnchor` = one per flight per epoch, drawn uniformly over samples or over the flight's remaining-path span; `training_window_class` is the one place the two anchor axes pick a class) | 1,941 |
| `data/anchor_grid.py` | which remaining-path anchors a re-anchored reading is taken at (no torch) | 159 |
| `data/anchor_strata.py` | the remaining-path VALUES both sides of that edge read: the grid's km, the strata they cut, the train-anchor draw law (leaf — no `dataset`, no torch) | 105 |
| `data/data_provenance.py` | which arrival rosters produced this run (pure hashing, **no torch**) | 503 |
| `data/splits.py` | which split a flight belongs to | 197 |
| `cli/` | one module per subcommand (`common` 909, `predict` 915 — `run_cli` is `load_predict_checkpoint` → `load_predict_series` → `parse_predict_options` (every flag rule, one frozen `PredictOptions`) → `predict_sets` → `write_prediction_sets` (the one emitter) → `report_predictions`, `evaluate_fit` 119, `cross_validate` 78, `train` 49, `freeze` 42, `__init__` 15) | 2,127 |
| `__main__.py` | the bootstrap and the `COMMANDS` table it dispatches from | 131 |

Two edges that a change must not reverse: `evaluation_protocol` reaches
`data_provenance`, never `dataset` (a boundary test bans torch from that path), and
`batch_contract` sits BELOW `objective` — the closure model and `outputs/control/latent`
return `LossComponents`, so it cannot move up.

### L5 · one strategy per prediction path under `outputs/`

**Each prediction path is a package under `outputs/` behind ONE strategy** (2026-09-10,
review §4.2): `outputs/state/`, `outputs/closure/`, `outputs/control/`, each with a
`strategy.py` whose class answers the spine — `build_model`, the anchor policy,
`bind_windows` (the batch context), `target_contract` / `loss_component_names` / `loss`,
the batch-size probe, `check_trainable` / `training_teacher` / `epoch_config` /
`training_diagnostics` / `epoch_record` / `checkpoint_metadata`, `forecast` with one
`ForecastOptions` value, `replay`, `record_fields` — and the spine (`dataset`, `batching`,
`objective`, `forecast`, `validation`, `export`, `train`, `models`) calls
`outputs.strategy(config).<method>` where it used to branch on `prediction_output`. The
registry is LAZY (`outputs/__init__.py` imports a strategy module on first use), which is
what lets `dataset`/`forecast`/`objective` import `outputs` while the strategies import
them back. `outputs/duration_heads.py` holds the point and quantile heads both paths build.
A fourth path (the plan-and-guidance design) is one package here and no edit in the spine.
`dataset` no longer carries any path's data-side code: a window set holds
`self.context = strategy.bind_windows(self)` and `batch()` asks it for the context row and
the dense supervision (the control path's `ControlContext` is where `dynamics_arrays`, the
imitation and heading-rate references and the fixed-dt rows are built, cached on a
fixed-anchor set); `__getitem__` is gone — `batch([i])` is the one door.

### L6 · `outputs/control/` by role; what every rollout needs is shared

The control path's own code lives in **`outputs/control/`**, by role rather than behind a
`control_` prefix: `strategy`, `supervision` (the per-flight physical context and the two
supervision references), `forecast` (the dense rollout under the hook, the CTA and interval
plumbing, the latent decodes), `heads` (the prediction contract and the heads), `latent`,
`basis_fit`, `loss/{objective,components,fixed_dt}`, `training/diagnostics`. **What the
rollout needs is not one path's** (2026-09-10, the plan-and-guidance path's first commit):
`outputs/dynamics/{backends,rollout,inverse,hooks}`,
`outputs/constraints/{barrier_filter,speed_floor,trombone,composite,gates,saturation}`,
`outputs/envelope` (the dimensionless command box and its newton conversion) and
`outputs/conditioning` (the condition vector) sit beside `outputs/base` and
`outputs/duration_heads`, imported by the control strategy and by the plan guidance alike;
`tests/test_architecture.py` refuses a shared part that imports any path.

### L7 · `outputs/plan/` — skeleton and extractors

**The
plan-and-guidance path is `outputs/plan/`** (2026-09-10,
`docs/2026-09-09_plan_and_guidance_design.md`, built in its §9 order): `skeleton` (the coded
approach in the flight's chart — `flight_scenarios.procedure_final.procedure_skeleton` read
through `runway_skeleton`: the fixes in runway axes about `target_chart`, the FAF, the
transitions' legs, the floor and ceiling coded at the next fix ahead, the optimizer's 150 m
threshold check), `extractors` (the eight plan parameters read off an observed track with
their ranges, `PlanLabels`; the join is `truth_final_gate`'s first row, `h_capture` the height
at that LATERAL join, `V_mid` the anchor speed for a flight anchored inside the 10–20 km
band); `run_ts.py plan_extractors` measures them over a checkpoint's cohort with the median
baseline the design's veto reads (the KRDU numbers are its §12.1).

### L8 · `outputs/plan/guidance/` — the route and its six measured rules

**The guidance layer is
`outputs/plan/guidance/`** (step 2): `route` lays a plan's route in the chart — the held
heading to a turn point, then the turns onto the join (`geometry/dubins`), then the final
leg — and reads the time the speed schedule needs for it (the time closure). **Six rules
the route obeys, each measured in (2026-09-10, the step-2 review)**: the join's intercept is
any angle within the on-final gate's 30° — the ALIGNED join where the plan's length affords
it, else the smallest intercept whose path fits (a downwind flight's turns onto the final
lay ~130 m per degree; five discrete steps left 5 of 48 flights 2–6 km short); every turn
is sized at the speed the schedule has WHERE it is flown (`build_route(speed_at=…)`: the
anchor's for the first, the schedule's at the turn point, the via and the join — at the
anchor's radius the base turn's lengths jump by a circumference exactly where the real
path lies); a turn-straight-turn whose two arcs sweep over 300° together is a LOOP or a
teardrop, not a route (`LOOP_SWEEP_RAD`; a pose beside the centreline heading in takes
the straight chord onto the join instead); inside the RNP box a plan whose join lies
closer ahead than the converging leg the tracker needs is flown as the converge-then-final
route a flight established at the anchor gets (29 of 73 routed straight-in flights were
routed through 12–24 km of teardrop for 200–1600 m of plan — the tracker cut through
them, so the flown path was right and the route's length and time were not); the hold and the
dog-leg offset are searched coarse-then-fine (`_closest_length`: the laid length is
piecewise smooth in either, a bisection lands on the wrong branch — measured 10–23 km MORE
than the plan on 3 of 48); and a hold or dog-leg is taken only when it lays the plan's
length CLOSER than the shortest path and never longer than it by over `HOLD_TOLERANCE_M`
— a shortfall is reported (`Route.shortfall_m`), an extra is never flown.

### L9 · `controller.PlanGuidance` — the one plan hook; the hold is at most ~3 s

`controller.PlanGuidance` is the one `CommandHook` that flies it on the lagged rollout: an
L1 look-ahead tracker with the route's curvature fed forward (bank), the height profile to
the capture height and the glidepath (load factor), the corridor barrier composed on the
final BEFORE the thrust is priced (the floor and the drag read the load factor actually
flown), then the speed schedule — `V_mid` held, a 0.5 m/s² deceleration to `V_final`, a
ground-speed law converted to the airspeed the dynamics carry — through the speed floor's
own thrust inversion, every command inside `flyability`'s envelope; the thrust clamped at
idle is counted beside the thrust over the maximum (`hook_plan_thrust_idle_steps`: a
deceleration the airframe's drag cannot fly). `outputs/plan/forecast.py`'s `fly_plans` is
the batch entry point (the shape `forecast_control_batch` has; a plan's capture height is
clamped into the glidepath window at the join and the clamp counted). **The hold is at
most ~3 s** (`n_segments_for`, one segment count per batch, so a shorter flight's hold is
shorter — `planHoldS` on the record says which): at 7 s the tracker oscillated ±300 m on
the final.

### L10 · fixed-K fly-by waypoints, each with the truth's speed at its turn

**The fixed-K fly-by waypoints are the route representation that reproduces a
vectored path** (step 2b, 2026-09-11, §12.3): `PlanLabels.waypoints` — up to `MAX_WAYPOINTS`
fixes, each the intersection of the two legs a turn joins (`extract_waypoints`; a reversal
is two fixes about its mid-tangent; the turn onto the final is read past the join and its
fix, on the centreline, becomes the join; a fix past the join and off the course is
dropped) — and `build_route(waypoints=…)` lays the polyline through them with each corner
rounded at the schedule's radius (`KIND_WAYPOINTS`; the anchor's heading joins the first
leg by a Dubins path, the last corner onto the course is a fly-by, else the aligned join).
**Each fix carries the speed the truth had at its turn** (`waypoint_speeds`): the corner is
rounded at that speed's radius and the schedule runs through the fix speeds
(`speed_schedule_mps(points=…)`, `PlanToFly.speed_points`) — without them the fixes were
rounded at `V_mid`, which the truth had long left by its base turn, and the tracker cut
the corners (31 % of vectored flights out of the corridor after the join, chamfer 296 m).
With the speeds, on KRDU val: vectored chamfer 636 → 268 m and arrival-time MAE
27 → 16 s from the truth's own fixes — bimodal: half the vectored flights within
~85 m, the other half ~900 m where the corners cannot be flown at the fix's speed (the bank
cap on 15.3 % of steps; corridor left after the join by 19.9 %). Two representations were
measured out before the fixes: (start, heading change) drifts every later leg at another
radius; a turn's end point is ambiguous about the leg heading.

### L11 · `run_ts.py plan_oracle` — every flight's own plan, graded twice

`run_ts.py plan_oracle`
flies every flight's OWN plan (the
design's step 2 ceiling; §12.2) and grades it twice, as a prediction and as a reference —
`--route waypoints` through the fixes, `--anchor-s N` from the one row N seconds after the
slice starts (the anchor a shorter window gives; `fly_plans` takes one anchor per flight),
`--anchor-km` from the anytime grid's remaining-path bin (LATER than L−1 on a vectored
track: remaining path is arc length).

### L12 · what the guidance and the control path both need moved up

What the guidance
needs and the control path also needs moved up with it: `outputs/dynamics/context.py`
(`dynamics_arrays`, `anchor_controls`), the dense query grid in `outputs/dynamics/rollout.py`,
the per-flight hook counts' record surface `per_flight_hook_diagnostics` in
`outputs/dynamics/hooks.py` (beside the protocol it reads), the Dubins primitives in
`geometry/dubins.py` (`dubins_csc` takes an `end_radius` and a `max_sweep_rad`), and the
time-free reading of one forecast against its truth, `experiments/support.forecast_geometry`
(the anytime curve's and the oracle's, once).

### L13 · the plan head: strategy, labels, model, the rolled flight

The
strategy (`outputs/plan/strategy.py`), the `PlanOutput` view (two loss weights) and
`PREDICTION_PLAN` ARE the plan head (step 3, 2026-09-11): `labels.py` turns a label set into
the target vector, the validity mask and the no-fix flag (`targets_from_labels`) and a
prediction back into the `PlanOrder` the guidance flies (`order_from_prediction`, every
clamp recorded); `model.py` the head and `plan_loss_components`; the rolled flight is
`forecast.fly_legs` under a `LegOrders` callable — the oracle's queue of the truth's
instructions (`fly_rolling`) or the head's order at each anchor (`fly_rolling_orders`,
`rolled_history` builds its window from the observed track continued by the flown rows).
`run_ts.py plan_oracle --route next [--policy model]` is the rolled oracle / the rolled
prediction under one instrument, `run_ts.py plan_next_readout` the next-instruction
statistics and, on a plan checkpoint, the single-step reading (HEAD columns).

### L14 · the oracle's vertical verdict

**The
oracle's vertical verdict binds the glidepath window inside the FAF only and the coded
floor before it, and grades the truth's own rows beside every flight** (the truth fails
the pre-FAF floor on 29 % of smoke flights — vectored aircraft are assigned altitudes
below the coded IF floor — so no share there is a gate).

### L15 · the closure and state path packages

The closure
path is `outputs/closure/{strategy,model,geometry,profile,forecast}`; the state path is
`outputs/state/{strategy,model,loss,forecast}` (the fixed-time postprocessors and the
corridor projection live in its `forecast`).

### L16 · `saturation` and `composite`; barrier and trombone gates are complementary

`saturation` holds the ONE definition of what `hook_saturation=soft` means (a scaled softplus)
AND of the bank softness/active-change thresholds two modules now share; `composite` is how
several modules become the one hook the rollout takes, and it refuses two modules that report a
diagnostic under the same name. Barrier and trombone both write bank and load factor and still
compose, because their gates are MADE complementary by the builder: with a trombone in the
value the barrier is built `confine_to_hard_gate=True` (its soft blend times the hard gate —
inside the gate unchanged, outside it silent), and the trombone engages only where the hard
gate has not opened, so no step is ever rewritten by both. The soft gate alone was NOT the
complement: its 30–40° alignment shoulder is the trombone's admission band (2026-09-09
review A-4, measured 0.112 rad of discarded barrier bank there).

### L17 · `RolloutStateView.reference` is the hook-free schedule whole

**`RolloutStateView.reference` is the hook-free schedule WHOLE** (`[B,N+1,7]`, one row per
segment boundary, the anchor first) — not a lock-step state, because the questions that need
it are look-ahead ones; it is the same tensor at every call, so a hook derives its table from
it once at `segment_index == 0`. It exists only where a member declares `needs_reference`.

### L18 · a dynamics backend is a row keyed by the pair

**A dynamics backend is a ROW, not a class**: `outputs/dynamics/backends.py` maps the
`(control_dynamics_model, control_dynamics_backend)` PAIR to
`(endpoint_fn, dense_fn, post_fn, runs_hooks)`. `post_fn` turns whatever state the
integrator carries into the one public `(channels, geodetic)` pair; `runs_hooks` is why the
point-mass rows refuse a command hook. A new pair supplies three functions and a flag.

### L19 · `archive/` is not the package

**`archive/` is not the package.** Completed campaigns are kept there (README each, naming
the result documents that cite them, the commit they were taken from, and anything vendored
in to keep them self-contained) and are OFF the import path:

- `archive/oracle_teacher_2026_08/` — the 2026-08 inverse-dynamics teacher: eight modules,
  five runners (four of them `run_ts_*`), two test files. Superseded by `simple-v3`'s
  in-training imitation term.
- `archive/nominal_law_hook_2026_09/` — the never-adopted nominal tracking law +
  bounded-residual command hook (`nominal_residual.py`, `guidance_laws.py`). The adopted hook
  is the predict-time barrier, which stays live.
- `archive/scene_encoder_2026_09/` — the scene data plane's tensor side (`scene/features.py`),
  its L4-gate readout (`run_ts_scene_explainability.py`) and its test. The L4 gate failed and
  the encoder was never built; `inference/intent_explainability.py` stays live for the Phase 0
  diagnostics.

`tests/test_architecture.py` asserts nothing live imports the archive — package modules,
`tests/`, the runners and `run_ts.py` alike — and that no `__init__.py` makes it importable.
A finished one-off driver belongs there, not beside the live runners.

### L20 · measurement code is code; `docs/` holds documents

**Measurement code is CODE.** Reusable logic goes in the package with tests
(`outputs/control/basis_fit.py`, `geometry/geometric_metrics.py`, `approach_difficulty.strata_masks`);
a runnable experiment goes in `experiments/<name>.py` behind `python run_ts.py <name>`;
**`docs/` holds documents**. The `docs/*.py` scripts predate this rule and are a layout
defect, not a pattern to copy (`docs/code-health-followups.md`) — do not add to them, and
move what you touch. `tests/conftest.py` already puts the package's PARENT on `sys.path`,
so a new test file needs no path preamble.

### L21 · `tests/` one file per topic, `tests/support.py`

**`tests/` is one file per topic, and `tests/support.py` holds what more than one file
shares** (2026-09-10, review §4.6): `test_ts_transformer.py` — 5,855 lines, 203 tests — is 23
single-topic files (`test_windows`, `test_anchor_policies`, `test_validation_replay`,
`test_control_objective`, `test_end_to_end`, …), each carrying only the helpers it uses.
`support.py` has the fixtures that were copied byte for byte across files —
`fake_data_provenance(*airports)`, `dynamics_context(batch, cta_s=None)`,
`terminal_contexts()` — imported as `from ts_transformer.tests.support import …` (`tests/`
is a namespace package under the package). A `_config` / `_series` recipe with its own
defaults stays in its file: two helpers with the same name and different defaults are two
helpers, not one copied.

### L22 · `ts_transformer` is a package; every import is qualified

**`ts_transformer` is a PACKAGE (2026-09-09, review §4.1), and every import is qualified:**
`from ts_transformer.config import TSConfig`, `import ts_transformer.data.channels as ch`,
`from ts_transformer.outputs.envelope import …`. What goes on `sys.path` is `4dTrajectory/`
(the package's parent — `__main__.py`, `tests/conftest.py` and every runner do it; `pip
install -e 4dTrajectory` makes it unnecessary, like `geokit`), NEVER `ts_transformer/`
itself: with the directory on the path the flat names resolve again and load a SECOND copy
of every module beside the qualified one — its own `TSConfig`, its own registries — and
every identity check between the two fails silently. `tests/test_architecture.py` refuses
both a flat import and a bootstrap that inserts the package directory.

### L23 · membership rule for `outputs/control/`

**Membership rule**: a module belongs under `outputs/control/` only if EVERY consumer of it
is control-specific. `terminal_state_loss`, `arc_length_geometry`, `fixed_dt_supervision` and
`flyability` therefore stay at the top level — `fixed_anchor_validation` and `dataset` share
them with the state path, and filing them under the control path would claim an ownership
that does not exist.

### L28 · direction BETWEEN paths (the control path may read the plan path)

**Direction between paths**: the control path may read the plan path (two-tier T1's plan token, `outputs/control/plan_token.py`, fuses the plan head's label), never the reverse (`test_the_plan_path_never_imports_the_control_path`).

### L29 · the manoeuvre package: two leaves BELOW the control path, everything else above it

**ARCHIVED 2026-09-20** — code under `archive/manoeuvre_codes_2026_09/` (its README says what moved and why; plan v3 §10's audit). The text below describes the archived code and is kept as its record. **Live now**: `manoeuvre/` has no leaf — `lockstep`, `gates` and `failure_modes` all sit above the control path — and **nothing under `outputs/` reaches `manoeuvre` at all**: only the runners under `experiments/` do (`test_only_the_runners_reach_the_manoeuvre_package`). The reverse edge the leaf rule existed for (the executor holding the tokenizer as a submodule) went with the archive; a new edge from `outputs/` is a layering decision, not an import. `manoeuvre/` (the intent-token model, plan §5.3; 2026-09-18) is a regular group beside `outputs/`. **`manoeuvre.segments` and `manoeuvre.tokenizer` are LEAVES**: they import the data plane, `config`, `io_utils` and torch only — an ALLOW-list, `tests/test_architecture.py::test_the_manoeuvre_leaves_import_no_layer_above_the_data_plane` — because the control path builds the truth segment rows into its context rows (`outputs/control/plan_token.py`) and holds the tokenizer as a submodule of the executor (`outputs/control/heads.py`). Only `outputs/control/*` and `experiments/*` may import `manoeuvre` at all, and the control path only those two leaves (`test_the_control_path_reaches_only_the_manoeuvre_leaves_and_nothing_else_reaches_manoeuvre`). Every other manoeuvre module (`sequences`, `prior`, `lockstep`, `readout`, `gates`, `scene`, `graph`, `decode`) imports the control path, the guidance layer, the data plane and the inference helpers, never the reverse; nothing in `manoeuvre/` imports `experiments` or `cli` (runners are consumers, L3). The decision: the plan's "manoeuvre may import outputs.control, reverse forbidden" could not hold literally once the tokenizer became the executor's submodule; the two leaves are the one reverse edge, and they cannot form a cycle because they reach no `outputs` module.

### L24 · import direction

**Direction**: the loop, the replay, the export and the CLI import `outputs/`, never the
reverse. Only a path's STRATEGY SEAM (`strategy`, `forecast`, `supervision`, `loss` /
`loss/objective`) reaches the spine's shared modules (`objective`, `forecast`, `models`,
`dataset`); the path's inner modules (heads, dynamics, constraints, the loss terms) may
import `dataset` (`Normalizer` and the window types are data-plane values they genuinely
consume) but not `objective`/`forecast`/`models`, and nothing under `outputs/` imports
`train`/`validation`/`batching`/`export`/`cli`. `dataset` imports the registry (`outputs`)
and nothing under it, and the registry and `outputs/base.py` import no spine module at
runtime — that is the whole reason the strategies can import `dataset`, `forecast` and
`objective` back without a cycle. `data/batch_contract.py` holds
`unpack_batch`/`model_forward`/`anchor_state` and the `LossComponents` contract so a loss
module can read a batch and return an objective without importing `objective`, which imports
it. `tests/test_architecture.py` enforces all of it.

`outputs/control/__init__.py` re-exports nothing on purpose — flattening forty names into
one namespace would restore the undifferentiated listing the package exists to remove.

### L25 · every CLI flag is named after its `TSConfig` field

**Every CLI flag is named after the `TSConfig` field it sets** (`--dt-s`,
`--control-rollout-integrator-dt-s`, `--control-state-supervision-clock`, …; fifteen were
renamed on 2026-09-07). `cli/common.CLI_CONFIG_FIELDS` is therefore a list of field names,
asserted against `fields(TSConfig)` at import; `tests/test_architecture.py` checks both
remaining directions — each listed field has a flag spelled as itself, and every parser dest
is either a listed field or one of the frozen `NON_CONFIG_DESTS` (so DELETING a name from
the list fails instead of leaving its flag silently ignored). **Every parser passes
`allow_abbrev=False`**: without it argparse accepts any unambiguous prefix and four of the
renamed flags kept working under their old spellings. Exceptions, all recorded where they
occur — on train: `batch_size` (its flag also takes `"auto"`), `use_norm`/`revin` (one
`--instance-norm`), `control_recipe_name` (resolved against `--config-overrides` first); on
predict, whose config comes from the CHECKPOINT so its flags are overrides rather than
settings (`cli/predict.PREDICT_CONFIG_FLAGS`, asserted the same way):
`--command-hook` / `--hook-saturation` keep their short names because `CLAUDE.md` names them
as the adopted delivery form and two arm files spell them in re-runnable `predict_args`.
(`--aircraft-type` was in that table until 2026-09-24, when the A320 fallback it chose was
retired; review C-5's concern went with it.) Predict refuses
repeated `--data` with `--airport` exactly as train does
(`cli/common.refuse_airport_override_for_pooled_data`, review C-19).

### L26 · run and category naming; never rename on-disk directories

**Run and category naming**: `run_naming.py` is the single source for one grammar —
`output · backbone · dynamics · loss · meta` — rendered from the run's serialized config by
every surface that names a trained run. A default change deliberately shifts old runs' names.
**Every `TSConfig` field is in a naming list or excused by name in
`run_naming.KNOWN_UNNAMED_FIELDS`**, asserted at import in both directions (review C-3: five
CLI-settable fields named nothing, so two runs differing only in `--validation-common-grid-points`
shared a name and a slug; it names the run now, `grid-points=`, and no stored run moves).
**The three identity-bearing fields that were unnamed are named since 2026-09-09 (decided):**
`lr_plateau_patience` / `lr_plateau_factor` (`lr-patience=` / `lr-factor=`; the named recipes
pin both, so a recipe run is unchanged) and `random_train_anchor_min_future_s`
(`anchor-min-future=`). Measured before landing it: 146 of the 219 stored runs' display names
and slugs moved (75 gain a spelled token, 71 only move their folded `+N more` count and slug
hash), 0 changed loadability, and 132 published frontend categories carried the old label:
109 publisher-managed ones (`publish_ts_experiment_trajectories.py --refresh-labels-only`,
run once per publication root) and 23 hand-published `ts_*` ones
(`docs/relabel_published_categories.py`) — labels only; no CZML, records, keys or directories
move. Only `device` and the never-set backbone knobs stay excused.
**On-disk run/category directories are historical record — never rename them.** Grammar,
fallbacks and the relabel tooling: `docs/ENGINEERING_NOTES.md`.

### L27 · `run_parameter_rows` and `experiment.intent`

**The grammar has a structured form, `run_parameter_rows(config)`** (2026-09-12) —
`{section, name, value[, field]}` rows: `Model` (output, backbone, dynamics, loss design, horizon,
seed — always), `Loss edits vs <base>` (every edit from `loss_design_parts`' base: the NAME hashes
a long loss design, the rows never do), then every non-default META field, recipe-frozen ones
included, under `SETTING_SECTIONS` — asserted at import to place every META field but `seed`
exactly once, so **a new META field needs a section** or the import fails. `loss_design_name` is
rendered from `loss_design_parts`; the split was measured name-preserving on all 513 stored
configs. The publisher stamps the rows as `experiment.parameters` and **`experiment.intent` from
`docs/experiments/intents.json`**: `campaigns[<picker group>]` title + intent,
`campaigns[<training campaign>].runs[<run id>]`, optional `variants[<run>@<variant id>]` (an
anytime bin's group is its RECORD campaign, its run intent still the training campaign's). **No
entry ⇒ the publication is blocked** before any predict/CZML work; `--refresh-labels-only
--output-root <root>` restamps a published root (metadata only, all-or-nothing on intents).

### L30 · `instructions/`: the second layer's language, below every model

2026-09-23. The instruction vocabulary (`spec`, `words`), the per-step signals in the airport frame
(`signals`, `airport`), the envelopes (`envelope`, the one implementation the labeller, the executor
and the display share), the piecewise fit, the labeller (`labeller/`: `records`, `lateral`,
`vertical`, `speed`, `sentence`, `read`), the measurements (`measure`), the artefact (`artefact`),
the readout and the eye-check figures. Torch-free; inside the package it imports only
`data.channels`, `data.coordinate_frames` and `io_utils` (outside it: `flight_scenarios`,
`aerodynamic_model.common`, `geokit`, numpy)
(`tests/test_architecture.py::test_the_instructions_package_sits_below_the_models`), and until the
prior exists only the runners and the executor (`autopilot/`, L31) consume it
(`test_only_the_runners_and_the_executor_reach_the_instructions_package`). The planned groups above
it — `prior/`, `closed_loop/`, `constraints/` — are in the framework document §2.

**Note (2026-09-25):** `instructions/training_files.py` holds the frontend's Training files — the index, the sets,
the overlays manifest, their schemas (mirrored by `aeroviz-4d/src/data/training*.ts`) and the checks every writer and
reader shares (`check_readback`, `open_base_set`, `require_stored_sentence`, `words_in_force`, `band_payload`,
`require_index_unchanged` / `require_overlays_unchanged`). It was the runner `instruction_training_export`, which the other
exporters and the backend imported; now nothing imports a runner for it, and everything in it raises `ValueError` (a
runner lets it propagate with its traceback; the backend answers it), never `SystemExit`, which a server thread would
let escape its handler.

### L31 · `autopilot/`: the executor, flying the words through the shared dynamics

2026-09-24 (`docs/2026-09-23_executor_design.zh.md`). Stage 3 of the two-tier framework: `frame`
(the dynamics' geodetic rows read the way the words read a flight — airport frame, compass track,
geometric MSL height; a positive bank turns LEFT), `sentence` (the word in force per column per
control cycle, each column's delay after its step), `flights` (a labelled flight rebuilt from the
recorded manifest with `build_series`, refused unless it reproduces the stored signals row for row;
its physical context is `outputs.dynamics.context.rollout_context` at row 0), `plant` (one cycle of
the control path's point-mass scaled-chart dynamics through `rollout_control_endpoints` — that
backend runs no command hooks, so the executor steps it cycle by cycle, the same computation),
`inverse` (wanted rates → bank, load factor, thrust; limits in the design's order, each recorded),
`lateral` / `vertical` / `speed` (the three laws), `params` (the executor's parameters and the design's
constraints on them), `executor` (the cycle loop, `fly`), `judge` (the three-layer verdict, with the
labeller's own checks), `replay` (who is flown — own dynamics or a stand-in's, the performance index's
substitute; a flight without aircraft dynamics is counted, never flown, C31 — drawing, flying and reading
a batch), `derive` (method A: τ_ψ and p from the vocabulary), `measure` (the data values the vocabulary does not
settle yet — the speed changes' pace and the landing aim — torch-free for the runner's workers) and `spec`
(`ts-executor-spec-v4` since 2026-09-24: the parameters written once with their sha and the executor's source hash,
`executor_source_files`; a replay refuses a spec measured by other code, `replay.open_executor`). Method B (`observe`,
the word delays) and method A's flown checks are archived (`archive/executor_vocabulary_only_2026_09/`: the executor
takes no information beyond the vocabulary, the user's rule of 2026-09-24). It may import the data plane
(`data.dataset`), the shared dynamics and geometry, and `instructions/`; never
`training`, `experiments`, `cli`, `backbone`, `inference`, `manoeuvre`, `outputs.control`,
`outputs.guidance`, `outputs.state`
(`tests/test_architecture.py::test_the_executor_flies_through_the_shared_dynamics_only`); only the
runners consume it (`test_only_the_runners_reach_the_executor_for_now`).
