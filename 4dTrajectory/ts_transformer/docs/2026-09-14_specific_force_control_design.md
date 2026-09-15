# Specific-force control parameterisation — design (2026-09-14)

The control head's thrust channel is today normalised by the ACTUATOR, `δ = T/T_max`
(`outputs/envelope.py`, since 2026-08-18). This design adds an axis that normalises it by the
EFFECT: the head predicts the specific force along the path, `n_x = (T − D)/W`. With it,
all three controls `(n_x, φ, n)` move every airframe identically, and the aircraft enters
only through feasibility (thrust and stall limits) and the exported thrust. The question,
the literature (36 sources) and the measurements behind the choice are in
`docs/literature/control_normalization/README.md` §3 (commit `683e604`).

## 0. Status

| step | what | state | where |
|---|---|---|---|
| N0 | fleet spread, per-class readout, literature | **done** 2026-09-14 | `683e604` on `dev-leg-ctrl`; `docs/literature/control_normalization/` |
| D | this design, plus the teacher-distribution measurement (§2.4) | **done** 2026-09-14 | branch `specific-force-control`, worktree `.claude/worktrees/specific-force` |
| M1 | core: the lag model's specific-force law, config axis, contract box, context, inverse, heads, objective, export, naming, tests | **done** 2026-09-14, reviewed (opus; 1 blocker + 3 should-fix, all fixed and re-verified, §9) | `8ce4568` |
| M2 | speed-floor hook under the new law, CLI flag, name/load census, docs (CLAUDE.md, CHANGELOG, OPEN_ITEMS), smoke train → predict → evaluate on real data | **done** 2026-09-14, review §10 | the commit after `8ce4568` |
| N3 | KRDU arms (§7): `B1_point_matched` recipe, thrust-fraction vs specific-force, 2 seeds each | **ran** 2026-09-15 00:18–02:33 UTC at `47b4b40`, campaign `4dTrajectory/outputs/KRDU/experiments/sf_n3`, **published** (KRDU picker, group `sf_n3`). **Final, both seeds against same-code twins (§7.3):** gate 1 passes (0.0042 / 0.0044 vs 0.0125 / 0.0118 g), gate 2 fails, the straight-in FDE veto trips (+240 / +222 m). Pooled ADE −57 / −73 m. Not adopted; the evidence for N6 | §7.2, §7.3 |
| N4 | the condition vector as ratios (own axis, §11): the alternative-hypothesis control for N3; plus the same-code δ twins N3 and N4 are both read against (§11.6: the stored twins drifted) | built 2026-09-15 on branch `sf-n4` (`608f14a`, review §11.5). **Running** since 2026-09-15 ~02:40 UTC from the `specific-force` worktree fast-forwarded to `f1b19c5`: campaign `4dTrajectory/outputs/KRDU/experiments/sf_n4`, log `…/experiments/sf_n4.log`, 4 arms (twins first). **Ran** 02:40–06:01 UTC, all four arms `completed` at `f1b19c5`. **Final (§11.7): the alternative hypothesis is rejected.** Ratios move the δ head's class bias only 0.0125 → 0.0119 and 0.0118 → 0.0105 g (N3: 0.0042 / 0.0044); `ratios` not adopted. **Published** (KRDU picker, group `sf_n4`, 4 categories; validator: picker loads) | §11.7 |
| N5 | pooled five-airport arm | not started; only if the mechanism holds on all gates (§11.4) — N3's provisional reading does not | — |
| N6 | a speed command (`speed-command`, Δv relative to the anchor speed through a τ_V speed loop): the invariant WITH a restoring force (§12) | built and reviewed 2026-09-15 on branch `sf-n6` (`94f822e`; review §12.8). §12.6's condition met (§7.3). **Running** since 06:01 UTC: the queue script fast-forwarded the runner worktree to `eaf409a` and launched campaign `4dTrajectory/outputs/KRDU/experiments/sf_n6` (log `…/experiments/sf_n6.log`) | §12 |

**Decision state.** The user approved on 2026-09-14: design, then implement on a new branch
the user merges. Every code milestone gets a subagent review before its commit
(`review-per-milestone`); docs are self-checked. Nothing on `dev-leg-ctrl` moves until the
user merges.

**Resume conventions.**
- Two worktrees since N3's launch. `.claude/worktrees/specific-force` (branch
  `specific-force-control`) is the RUNNER: a campaign is launched from it, and it is never
  edited while one runs (a dirty tree refuses the next arm's `begin_run`).
  `.claude/worktrees/sf-n4` (branch `sf-n4`, a descendant) is where development continues.
  Once a campaign has exited, `git -C .claude/worktrees/specific-force merge --ff-only sf-n4`
  brings the runner up to date. The user merges `specific-force-control` into `dev-leg-ctrl`.
- Work only in a worktree. Its `data`, `trajectory_data_process/outputs`,
  `4dTrajectory/outputs` and `aeroviz-4d/public/data/airports` are SYMLINKS to live data, so a
  test must write only to `tmp_path`.
- Stage explicit paths; check `git diff --cached --stat` before every commit.
- A recursive `grep` in this shell skips `ts_transformer/outputs/` (it honours `.gitignore`);
  use `command grep -r` for a census.
- Run the full ts suite in the foreground with a 600 s timeout.

## 1. The answer in one paragraph

A thrust fraction already removes the thrust-in-newtons problem, and OpenAP sizes engines to
airframes: T_max/W at landing mass is 0.31–0.42 over the cohort's 28 types. What is left is
measured, and it is structural. On KRDU val (`B1_point_matched`, 2 seeds, established
stratum) the head commands nearly the same δ on every class: the class spread is 0.003–0.005,
while the truth needs 0.028–0.041. That produces a class-dependent specific-force bias spanning
0.010–0.012 g, against the truth's own class spread of 0.0045 g, and heavies fly 4.7–4.9 m/s
fast relative to the 737 family. Predicting `n_x` makes the head's natural invariance the
data's. Mass and thrust are not separately identifiable from tracks (Alligier 2015, Sun 2018,
Pepper 2024), while `(T − D)/m` is. The same pattern is standard elsewhere: quadrotors use
mass-normalised collective thrust (QuadBenchmark, Swift, XAdapt), and Khatib's unit-mass
command and BADA's `(T − D)/(mg)` have the same form.

## 2. The contract

### 2.1 The law

```
controls  (n_x, φ, n),   n_x = (T − D)/W         [specific-force]
          (δ,   φ, n),   δ   = T / T_max          [thrust-fraction, unchanged]

first-order lag (both laws): actuator a = (a_x, φ_a, n_a),  ȧ = (command − a)/τ

specific-force thrust inside EVERY RHS evaluation (every RK4 stage):
    T = clamp( W·a_x + D(V, h, m, n_a),  T_lo,  T_max ),   T_lo = MIN_THRUST_FRACTION·T_max = −0.2·T_max
    then the unchanged transport_chart_rhs(chart, (T, φ_a, n_a))
    ⇒ V̇ = (T − D)/m − g·sin γ = g·(a_x − sin γ)   wherever T is not clamped
```

- **Drag cancels exactly where thrust does not saturate.** The same `aerodynamic_coefficients`
  is evaluated on the same inputs.
- **The airframe (m, T_max, S, CLmax, polar) enters in three places only:** the thrust clamp,
  the existing stall clamp on `n`, and the exported thrust.
- **The clamp keeps today's thrust RANGE, not today's dynamics.** At every state the clamp
  admits exactly the thrusts δ ∈ [−0.2, 1] does, so no command becomes feasible or infeasible
  by the change. Dropping the lower clamp (the floor exists because the clean polar lacks flap
  and gear drag) would admit the 0.87 % of truth segments the floor cuts (§2.4); that is a
  separate axis, not built. But the same range does NOT give the same trajectories for the
  same piecewise-constant schedule — see the next point.
- **With the drag cancelled, the speed has no drag feedback.** Under thrust-fraction
  `∂V̇/∂V = −(∂D/∂V)/m`: a biased command settles where drag balances it (stabilising above
  the minimum-drag speed, destabilising on the back side below it). Under specific-force,
  unclamped, `∂V̇/∂V = 0`: the speed integrates the command, and a bias drifts linearly until
  the T_max clamp binds, where `∂V̇/∂a_x = 0`. Measured by the M1 review with untrained heads
  at level-trim neutrals on four synthetic KRDU descents (2.5–3°, 416 s):
  - thrust-fraction peaked at 209–213 m/s;
  - specific-force at n_x = 0 peaked at 311–328 m/s, stopped only by the clamp.

  So an n_x head must supply the speed regulation that drag partly supplied for a δ head.
  The imitation teacher and the trajectory loss train it, and N3 measures the
  parameterisation INCLUDING this difference — it is not a pure change of coordinates. The
  specific-force neutral is chosen against it (§2.3).
- **The thrust is recomputed at every RK4 stage.** A thrust held over a segment is not a held
  `n_x`, which is why the law lives inside the lag RHS and not in a pre-rollout conversion.
- **`control_thrust_time_constant_s` (1.5 s) lags `a_x` under this law.** That approximates
  spool lag, since the drag term moves with V and n during the lag; it is stated, not hidden.

### 2.2 Scope

- **The new law is implemented on the `first-order-lag` flight model only,** the row every
  control recipe since `simple-v1-lag` flies and the one that runs hooks. `point-mass` rows
  hold newton controls across a segment and would need their own RK4 step with drag inside;
  `TSConfig` refuses the pair (a scope statement, lifted by adding the rows).
- **The plan path keeps thrust-fraction.** Its guidance commands δ through the speed floor's
  thrust inversion. The field belongs to the `control` output; `_validate_ownership` refuses it
  off its default elsewhere.
- **`control_imitation_target=fitted` is refused with specific-force.** A fitted table carries
  no parameterisation in its schema, so an n_x run could read a δ table. It is lifted by
  stamping the schema.
- **The speed-floor hook** writes the thrust channel and is refused under specific-force in M1,
  then rebuilt in M2 (§3, M2). The barrier and the trombone pass the thrust column through
  untouched (`command[:, 0]`) and read only `actuators[:, 1:]`, so they work under both laws
  unchanged.

### 2.3 The head's box

| contract | names | lower | upper | half width | neutral |
|---|---|---|---|---|---|
| thrust-fraction | `thrust_fraction, bank_rad, load_factor` | −0.2, −π/4, 0.2 | 1.0, π/4, 2.0 | 0.6, π/4, 0.9 | 0.2, 0, 1 |
| **specific-force** | `specific_force, bank_rad, load_factor` | **−0.20**, −π/4, 0.2 | **0.23**, π/4, 2.0 | **0.215**, π/4, 0.9 | **−0.05**, 0, 1 |

- **Half width 0.215 g** is today's δ half width (0.6) times the fleet-median T_max/W (0.358).
  So a given physical error costs the same in the imitation MSE for the median airframe, and
  `SIMPLE_V3_IMITATION_LOSS_WEIGHT` (64) stays comparable across the two arms.
- **Lower −0.20** sits below the truth's p0.1 (−0.179) and below the feasible floor of 99.7 %
  of states (§2.4). The upper bound follows from the width.
- **Neutral −0.05** holds speed on a ~2.9° descent, on every airframe (`n_x = sin γ` is the
  speed hold). That is an approach's dominant condition, and it sits beside the truth
  teacher's median, −0.059 g (§2.4).
  - It is deliberately NOT level trim, which is what δ's neutral 0.2 means for the median
    airframe (0.2·0.358 − 0.076 ≈ 0).
  - Level trim is benign under thrust-fraction only because drag bounds the descent
    overspeed; under specific-force nothing does (§2.1).
  - Measured on the same four synthetic descents: at −0.05 the untrained heads stay within
    103–144 m/s, from 118–136 m/s.
  - It was 0 until the M1 review measured the overspeed.
- **The box is the head's search space, as today's is, not a flyability claim;** the clamp in
  §2.1 is the feasibility.

### 2.4 Measured: where the truth lives in each coordinate

KRDU val, 1404 flights × 32 midpoints, from the L−1 anchor. Source: the observed tracks in
`4dTrajectory/outputs/KRDU/experiments/b1_quantile_20260907/B1_point_matched_pred_val`, run
through the package's lagged-command inversion (τ = 1.5 / 2.0 / 0.8) exactly as
`segment_controls` samples it. Script: `docs/specific_force_teacher_distribution.py`.

| percentile | 0.1 | 1 | 5 | 50 | 95 | 99.9 |
|---|---:|---:|---:|---:|---:|---:|
| δ teacher | −0.321 | −0.191 | −0.124 | 0.039 | 0.181 | 0.382 |
| n_x teacher (g) | −0.179 | −0.135 | −0.112 | −0.059 | −0.0085 | 0.068 |
| feasible floor `(−0.2·T_max − D)/W` | −0.220 | −0.179 | −0.165 | −0.143 | −0.127 | −0.121 |
| feasible ceiling `(T_max − D)/W` | 0.220 | 0.244 | 0.259 | 0.287 | 0.329 | 0.351 |

- **The two coordinates have the same admissible set.** The δ teacher leaves `[−0.2, 1.0]` on
  0.82 % of segments (0.80 % below the floor); the n_x teacher leaves its own feasible interval
  on 0.87 %.
- **The specific-force box `[−0.20, 0.23]` cuts 0.10 % of the n_x teacher (0.08 % below).**
  Only 0.29 % of states have a feasible floor below −0.20.
- **Nothing on an approach uses the thrust ceiling.** The n_x teacher's p99.9 (0.068 g) is a
  quarter of the lowest feasible ceiling.

## 3. The code plan

M1 is "core" and M2 is "the rest". Every file is listed with what changes.

**`aerodynamic_model/` (additive; defaults bit-identical; the CasADi model untouched).**
The torch modules have no importer outside `ts_transformer` and the package's own tests
(census 2026-09-14).
- `torch_dynamics.py`: `specific_force_thrust_n(n_x, load, speed, altitude, mass, aero_params,
  min_thrust_n, max_thrust_n)`. It computes `clamp(m·g·n_x + D, T_lo, T_max)`, with D from
  `aerodynamic_coefficients`, the one polar.
- `torch_transport_chart_dynamics.py`: `transport_chart_specific_force_thrust_n(state_chart,
  n_x, load, aero_params, frame_params, min_thrust_n, max_thrust_n)`. It reads speed, altitude
  and mass off the chart through the module's own `_wgs84_geometry`, the same geometry the RHS
  reads.
- `torch_lag_dynamics.py`:
  - A control-law value: `THRUST_FRACTION_LAW`, or `SpecificForceLaw(min_thrust_fraction)`.
  - `lag_rhs(..., specific_force: bool)`, a static Python flag, with the thrust converted per
    law.
  - Separate module-level step functions and compiled-CUDA caches per law (torch.compile caches
    per code object).
  - The specific-force step's context gains `min_thrust_n`.
  - The three rollout functions take a keyword `control_law=THRUST_FRACTION_LAW`.

**`ts_transformer/` — M1**
- `config.py`:
  - `CONTROL_THRUST_FRACTION` / `CONTROL_SPECIFIC_FORCE`, `CONTROL_THRUST_PARAMETERIZATIONS`.
  - The field `control_thrust_parameterization = "thrust-fraction"`, on `DynamicsSpec`, which
    refuses specific-force off `first-order-lag`.
  - `ControlOutput` refuses it with `fitted` and with a hook containing `speed-floor` (the
    latter until M2).
  - `control_simple_v1_overrides` pins thrust-fraction, so every named recipe means today's
    law and an n_x arm is `custom`.
  - Not in `REQUIRED_SERIALIZED_CONTROL_FIELDS`: the default reproduces every stored checkpoint.
- `outputs/envelope.py`:
  - A frozen `ControlContract` (names, lower, upper, half width, neutral) and
    `control_contract(parameterization)`.
  - `CONTROL_LOWER` / `CONTROL_UPPER` / `CONTROL_HALF_WIDTH` / `CONTROL_NAMES` stay as the
    thrust-fraction contract's arrays, which are what the plan path and the speed floor read.
  - `physical_controls` / `fraction_controls` stay thrust-fraction only.
- `outputs/dynamics/context.py`: `dynamics_arrays(series, anchor, *, parameterization)` and
  `anchor_controls(..., parameterization)`.
  - **Required, never defaulted.** A control call site that forgot it would silently hand an
    n_x rollout a δ box and a δ initial actuator. At δ ≈ 0.04 read as 0.04 g, that is
    +0.4 m/s² on every flight.
  - The plan path passes thrust-fraction explicitly.
- `outputs/dynamics/inverse.py`: `actual_controls` / `commanded_controls` take `parameterization`
  (required).
  - Under specific-force the first column is `(tangential + g sin γ)/g`: the same tangential
    term, transport included, with no drag, mass or thrust.
  - `reference_controls` reads `config.control_thrust_parameterization`, so the teacher is the
    inverse of the configured forward model.
- `outputs/dynamics/backends.py`: the lag rows pass `control_law` from the config.
  `RolloutInputs.newton_controls` is read only by the point-mass rows, which stay
  thrust-fraction by the scope rule.
- `outputs/control/supervision.py`: the imitation target is clipped to the contract's box; the
  probe batch reads the contract's box and neutral.
- `outputs/control/heads.py`: `_initialize_control_head(head, contract)`, so the neutral bias
  comes from the contract. The saturation diagnostics' names also come from the contract; they
  keep today's `thrust_N` label under thrust-fraction so stored histories stay comparable.
- `outputs/control/loss/objective.py`:
  - The imitation MSE divides by the contract's half width.
  - The heading-rate term passes a ZERO thrust column under both laws. The ψ row of the RHS
    reads only bank and load, so this is bit-identical under thrust-fraction, and a test pins
    the independence. Converting the column would need the state's drag under specific-force,
    for a value the row never reads.
- `outputs/control/forecast.py` + `inference/export.py`:
  - Under specific-force, a segment's record thrust is the command's thrust at its START state:
    `clamp(W·n_x_cmd + D(state, n_cmd), T_lo, T_max)`. Per-state controls stay zero-order-hold
    by segment, so the record contract (newtons, 1:1 with states) holds.
  - `control_segments[*].specific_force` carries the command itself.
  - `source.controlThrustParameterization` is written only under specific-force. Absent means
    thrust-fraction, so every stored-config record reproduces to the bit.
- `run_naming.py`: the dynamics word gains `+specific-force`
  (`first-order-lag+specific-force @scaled-transport-chart-velocity`) and the slug gains `-sf`.
  The field is directly named. No stored run is renamed; the census in M2 proves it.
- Callers updated to pass the parameterisation: `outputs/control/strategy.py`,
  `outputs/control/forecast.py`, `outputs/plan/forecast.py`,
  `experiments/predictability_report.py`, `outputs/control/basis_fit.py`, and the tests.

**`ts_transformer/` — M2**
- `outputs/constraints/speed_floor.py` under specific-force: the demand is drag-free,
  `n_x ≥ sin γ + (V_floor − V)/(g·hold)`, with the lag credit on `a_x`.
  - "Saturated" means the demand reached the state's feasible ceiling `(T_max − D)/W`, which
    is still "full thrust is not enough".
  - The soft-max width scales with the half width: 0.02 × 0.215/0.6 = 0.0072 g.
  - Lift M1's refusal.
- CLI: `--control-thrust-parameterization` on train (`cli/common.py`) with its `AVAILABLE`
  tuple. `predict` reads the checkpoint's own value.
- Census over every stored `history.json`: display name, slug and `from_dict` before and after
  must be identical.
- Docs: this file's status table, `ts_transformer/CLAUDE.md` (the controls contract line, the
  current-defaults table), `docs/OPEN_ITEMS.md`, the root `docs/CHANGELOG.md`.
- Smoke: a short real-data train (a few epochs, KRDU) → predict → `python -m evaluation`, under
  both laws, to prove the chain end to end. The numbers are not quotable.

## 4. What does NOT change

- **Stored runs:** every stored checkpoint, config, name, slug and record. The default is
  today's law, the field is absent from every stored config, and the record key is absent under
  it.
- **The evaluation record contract:** newtons, `thrust / bank_rad / load_factor`, 1:1 with
  states.
- **The CasADi model and the optimizer's envelope.**
- **The plan path and its guidance.**
- **`flyability`:** it recomputes the required thrust from the states, under the same clean
  polar.

## 5. Tests (M1 unless marked)

1. **Drag cancels:** under specific-force with the clamp inactive, `V̇ = g·(a_x − sin γ)` to
   1e-9 relative, across types and states; with the clamp active, V̇ equals the δ law's at
   `δ = 1` / `δ = −0.2`.
2. **The δ law is untouched:** its lag rollout is bit-identical to the pre-change output on a
   fixed fixture (`control_law` defaulted and passed explicitly).
3. **Round trip:** a known n_x command schedule rolled forward, then run through
   `reference_controls` under a specific-force config, returns the schedule (the existing δ
   round-trip test, parameterised over both laws).
4. **Same admissible set:** for one track, the specific-force feasible interval maps onto δ's
   box `[−0.2, 1]` through `n_x = (δ·T_max − D)/W`.
5. **Config:** specific-force is refused on `point-mass`, with `fitted`, and with a `speed-floor`
   hook (M1); ownership refuses it on the plan output; every named recipe refuses it; `custom`
   accepts it.
6. **Naming:** default names and slugs unchanged; specific-force shows `+specific-force` / `-sf`.
7. **Context:** `dynamics_arrays` requires the argument; the box and the initial actuator are in
   contract units; the probe's key set still equals a real batch's
   (`test_supervision_terms.py`).
8. **Heads:** the neutral init reproduces the contract's neutral under both laws.
9. **Heading rate:** the ψ row is independent of the thrust column (random thrust, same ψ̇).
10. **Export:** a specific-force record carries newton thrust equal to the segment-start
    formula, `specific_force` per segment, and the source key; a thrust-fraction record has no
    key.
11. (M2) **Speed floor:** under specific-force the demand holds `V(hold) = V_floor` on the
    fixture, the soft form is inert at a non-binding demand, and "saturated" fires at the
    feasible ceiling.

## 6. Milestones and review gates

| milestone | gate before commit |
|---|---|
| M1 | the ts suite and `aerodynamic_model/tests` green (the known numpy failure in the optimizer suite excepted); an opus subagent review over the diff with this doc's §2–§5 as the checklist; findings verified and fixed; one re-verify |
| M2 | the same, plus the naming/load census identical before and after, and the smoke chain run under both laws |
| N3 launch | intent entries in `docs/experiments/intents.json` and the arm file committed first; launched from a clean tree; an opus queue agent owns the runs |

## 7. N3 — the experiment this axis exists for (planned, not launched)

- **Arms:** the `B1_point_matched` configuration (control, iTransformer, first-order-lag @
  scaled chart, simple-v3 + final-time 26, N = 32), as `custom` with only
  `control_thrust_parameterization` moved; seeds 1337 and 2024; KRDU, the same split. The base
  arm is the stored `B1_point_matched` / `_s2024` pair.
- **Primary gates (mechanism):**
  - the per-class `n_x` bias range (0.0104 / 0.0116 g today) falls toward the truth's own
    0.0045 g;
  - the heavy − 737 speed-bias gap (4.7 / 4.9 m/s) shrinks;
  - `docs/literature/control_normalization/measurements/per_class_readout.py` is the
    instrument.
- **Veto gates:**
  - pooled ADE within the control path's ~125 m seed line of the base;
  - straight-in FDE not worse beyond its seed spread;
  - flyability delta against observed not worse.
- **Expected (reading):** in-distribution ties are the norm in the literature (Villar Exp. 1,
  QuadBenchmark Table V). A pooled-ADE win is not the claim; the class structure is.

## 7.2 N3 results — PROVISIONAL (against the stored twins, which drifted; §11.6)

**Ran:** 2026-09-15 00:18–02:33 UTC, campaign `4dTrajectory/outputs/KRDU/experiments/sf_n3`, commit `47b4b40`, both
arms `completed`. Both ran all 180 epochs (selection ADE 1268.2 / 1221.7 m). The stored twins: 1247.9 (180
epochs) / 1373.1 m (143 epochs, early stop).

**This reading is NOT binding.** The twins were trained before `c544db0`, so every delta below mixes the law's
effect with that fix's. The binding reading is against `N4_twin` / `N4_twin_s2024` (§11.4), same instrument.
Instrument: `per_class_readout.flight_row` on each arm's `pred_val` (the established stratum for the class
rows) plus the package strata on `summary.json`. Script in the session scratchpad (`sf_readout.py`), which
reproduces the twins' published numbers exactly (1248 / 1373 m ADE, 0.0104 / 0.0117 g, 4.70 / 4.90 m/s).

| KRDU val, 1404 flights | seed 1337: arm / stored twin | seed 2024: arm / stored twin |
|---|---|---|
| **gate 1** per-class n_x bias range (truth's own class spread 0.0046 g) | **0.0042** / 0.0104 g | **0.0044** / 0.0117 g |
| **gate 2** heavy − B737 speed bias | **+6.17** / +4.70 m/s | **+6.31** / +4.90 m/s |
| ADE pooled / straight-in / vectored (m) | 1268 / 437 / 2737 vs 1248 / 413 / 2721 | 1222 / 412 / 2649 vs 1373 / 451 / 3004 |
| **FDE p50** pooled / straight-in (m) | 1103 / **887** vs 877 / 582 | 1110 / **885** vs 937 / 659 |
| duration MAE pooled / straight-in (s) | 26.2 / 13.9 vs 25.1 / 13.1 | 25.7 / 13.6 vs 26.2 / 13.9 |
| paired ADE, arm better | 47.5 % (median +13 m) | 61.9 % (median −65 m) |
| flyability Δ vs observed: fully / samples | −0.634 / −0.050 vs −0.981 / −0.070 | −0.724 / −0.051 vs −0.984 / −0.072 |

Per class, established stratum (n_x bias = time-mean predicted − observed n_x; δ = the time-mean thrust fraction
the exported thrust implies; the truth's inverted δ is 0.036 / 0.042 / 0.031 / 0.028 for B737 / A320 / regional /
heavy):

| class (n) | n_x bias 1337 / 2024 (g) | stored twins (g) | speed bias 1337 / 2024 (m/s) | stored twins (m/s) | δ flown 1337 / 2024 | stored twins' δ |
|---|---|---|---|---|---|---|
| B737 fam (314) | −0.0028 / −0.0024 | −0.0056 / −0.0083 | −3.93 / −4.04 | −1.41 / −1.86 | 0.063 / 0.068 | 0.051 / 0.045 |
| A320 fam (233) | +0.0015 / +0.0017 | −0.0043 / −0.0072 | −0.63 / −0.98 | +0.90 / +0.58 | 0.065 / 0.068 | 0.050 / 0.043 |
| regional (250) | +0.0005 / +0.0012 | +0.0010 / −0.0006 | −1.62 / −2.21 | +1.15 / +1.17 | 0.052 / 0.060 | 0.053 / 0.048 |
| heavy (44) | +0.0013 / +0.0020 | +0.0048 / +0.0033 | +2.28 / +2.28 | +3.30 / +3.04 | 0.033 / 0.038 | 0.052 / 0.047 |

**Where the straight-in FDE goes** (seed 1337, `run_ts.py straight_in_residual_readout`, straight-in stratum,
p50 by remaining distance):
- The arm tracks the truth's speed to 5 km, then flies **4.4 m/s SLOWER** in the last 5 km, where the twin flew
  1.8 m/s fast. The along-track error there is −207 m against the twin's −119 m.
- The deceleration POINT is unchanged (offset p50 0.0 km in both).
- The vertical error falls: RMS p50 79 → 53 m.

**Reading (provisional; the twins drifted):**
- **Gate 1 passes on both seeds.** The head's specific-force bias is class-invariant, at the truth's own class
  spread. The class-dependent δ it flies (heavies 0.033–0.038 against 0.063–0.068 for the 737s) is what the
  thrust-fraction head never learned.
- **Gate 2 fails on both seeds.** The heavy − 737 speed gap grows by about 1.4 m/s.
  - Since n_x = V̇/g + sin γ, a class-invariant n_x bias with a class-dependent speed bias puts the class
    structure into the path angle.
  - That is a reading. The vertical channel's per-class bias has not been measured.
- **The straight-in FDE veto trips on both seeds**, identically: 887 / 885 m against 582 / 659.
  - The late-final speed deficit is the missing drag feedback (§2.1): a slightly low n_x over the last
    segments accumulates into lost speed instead of settling.
- **Pooled ADE is inside the seed line. Flyability improves by a third.**
- **Nothing here is final until `N4_twin` reads the same-code baseline.** `c544db0`'s terminal emphasis
  may have moved the twins' endpoint error too.

### 7.3 N3 against the SAME-CODE twins — BINDING on both seeds (2026-09-15)

`sf_n4/N4_twin` is `B1_point_matched` re-trained at the N3/N4 code (seed 1337, 180 epochs). N3 seed 1337
against it, same instrument:

| KRDU val, 1404 flights | N3_specific_force | N4_twin | stored B1_point_matched |
|---|---|---|---|
| **gate 1** per-class n_x bias range (truth 0.0045 g) | **0.0042** | 0.0125 | 0.0104 |
| **gate 2** heavy − B737 speed bias (m/s) | **+6.17** | +5.16 | +4.70 |
| ADE pooled / straight-in / vectored (m) | 1268 / 437 / 2737 | 1325 / 444 / 2880 | 1248 / 413 / 2721 |
| **FDE p50** pooled / straight-in (m) | 1103 / **887** | 873 / 647 | 877 / 582 |
| duration MAE pooled / straight-in (s) | 26.2 / 13.9 | 25.5 / 13.3 | 25.1 / 13.1 |
| paired ADE vs N4_twin, arm better | 54.4 % (median −22 m) | — | — |
| flyability Δ vs observed: fully / samples | −0.634 / −0.050 | −0.980 / −0.067 | −0.981 / −0.070 |

- **Gate 1 passes, gate 2 fails, the straight-in FDE veto trips: +240 m against the same-code twin.** That
  is three times the twins' straight-in seed spread (77 m).
- **The provisional reading stands, and the FDE penalty is the law's, not the drift's.**
- **The drift, measured:** `N4_twin` against the stored twin is +77 m pooled ADE and +65 m straight-in
  FDE p50, inside the seed line. `c544db0` moved epoch 1 by −11 % but not the converged model.
- **N3's pooled ADE is 57 m BETTER than its same-code twin** (not worse by 20, as against the stored
  one). It stays inside the seed line either way.
- **§12.6's condition is met:** `N4_twin`'s straight-in FDE (647 m) sits 240 m below N3's. N6 is queued
  after `sf_n4`.

**Seed 2024** (`N4_twin_s2024`: 168 epochs, early stop; selection ADE 1294.6 m):

| KRDU val, 1404 flights | N3_specific_force_s2024 | N4_twin_s2024 | stored B1_point_matched_s2024 |
|---|---|---|---|
| **gate 1** per-class n_x bias range (truth 0.0045 g) | **0.0044** | 0.0118 | 0.0117 |
| **gate 2** heavy − B737 speed bias (m/s) | **+6.31** | +4.89 | +4.90 |
| ADE pooled / straight-in / vectored (m) | 1222 / 412 / 2649 | 1295 / 446 / 2793 | 1373 / 451 / 3004 |
| **FDE p50** pooled / straight-in (m) | 1110 / **885** | 911 / 663 | 937 / 659 |
| duration MAE pooled / straight-in (s) | 25.7 / 13.6 | 25.4 / 13.3 | 26.2 / 13.9 |
| paired ADE vs N4_twin_s2024, arm better | 59.3 % (median −36 m) | — | — |
| flyability Δ vs observed: fully / samples | −0.724 / −0.051 | −0.982 / −0.069 | −0.984 / −0.072 |

**N3's verdict, both seeds against same-code twins (final):**
- **Gate 1 passes on both.** The per-class n_x bias range is 0.0042 / 0.0044 g against 0.0125 / 0.0118,
  at the truth's own class spread (0.0045). The specific force IS the invariant: under it the head stops
  commanding one δ for every class.
- **Gate 2 fails on both.** The heavy − 737 speed gap grows by 1.0 / 1.4 m/s.
- **The straight-in FDE veto trips on both,** by +240 / +222 m. The mechanism is the late-final speed
  deficit (§7.2): no drag feedback, `∂V̇/∂V = 0`.
- **Pooled ADE is better by 57 / 73 m and vectored ADE by 143 / 144 m, on both seeds.** Paired, the arm
  is better on 54 / 59 % of flights. Flyability improves by about a third on both.
- **The seed line needs re-reading.** The same-code δ twins differ by only 30 m of pooled ADE (1325 /
  1295) and 16 m of straight-in FDE (647 / 663), against the stored pair's 125 m. The stored pair's
  spread came with an early stop at 143 epochs; this pair ran 180 / 168. So N3's consistent 57–73 m is
  below the documented 125 m line but twice this pair's spread: suggestive, not established.
- **N3 is not adopted:** it trips a veto. It is the evidence N6 is built on: the invariant is right, and
  the law lacks the restoring force.

## 7.1 Smoke run (M2) — the chain works; its numbers are NOT results

**Run:** KRDU, `B1_point_matched`'s config with only the law moved, 2 epochs, one seed, train → predict val →
`python -m evaluation`. Artifacts are in the session scratchpad and are not kept.

**What ran:**
- Every stage completed.
- 1404 records, all stamped `controlThrustParameterization: specific-force`, with `specific_force` per segment
  and newton thrust.
- The per-class readout (the N3 instrument) runs on them.

**Sanity reads — not results:**
- Training is not pathological. Validation ADE was 3447 / 2750 m at epochs 1 / 2, against the thrust-fraction
  arm's 6176 / 3535 m at the same epochs. **Correction (2026-09-15, §11.6):** 6176 / 3535 are the STORED
  twin's, trained 140 commits earlier. The thrust-fraction configuration at this code reads 5482 / 3754.
- The control-head gradient at epoch 1 was mean 177 / max 1014, against 741 / 1619.
- The head's n_x was already class-invariant after 2 epochs (−0.0621 … −0.0629 g on the established stratum), so
  its class bias range, 0.0049 g, sits at the truth's own 0.0045 g — the mechanism the design predicts.
- The model is 2 epochs old: duration bias +125 s, 0.6 % fully flyable. N3 is the measurement.

**Speed floor under specific-force (M2):**
- The demand is drag-free: `a_c = (a_req·dt − a_0·τ_eff)/(dt − τ_eff)`,
  `a_req = (V_floor − V)/(g·dt) + sin γ`.
- The lag credit is the actuator capped at the engine ceiling.
- It is capped at the ENGINE's `(T_max − D)/W` only; "saturated" means full thrust, as under thrust-fraction.
- **It is NOT held inside the head's 0.23 g box.** On this fleet that box is below the engine ceiling
  (0.24–0.33 g at 1.1·V_s, 1 g). The M2 review found the box-capped version saturating on the box in 11 of 11
  steps, with less authority than the thrust-fraction floor.
- Its soft form is exactly inert by construction: an unused demand is parked one softness past softplus's linear
  threshold, and the command is returned there. The thrust-fraction path relies on a rounding accident for the
  same property; it is left bit-identical and logged (`docs/code-health-followups.md` §34).
- The M1 config refusal is lifted.

## 8. Traps to know before touching it

- **The speed has no drag feedback under specific-force** (§2.1). A constant n_x bias drifts
  the speed without bound until the thrust clamp binds, so do not read a specific-force
  rollout's speed drift as a bug in the law, and do not give it a level-trim neutral.

- **`dynamics_arrays` without the parameterisation is a TypeError on purpose.** A δ box under
  an n_x rollout is a silent +0.4 m/s² bias.
- **The specific-force thrust must be computed at every RK4 stage.** Computing it once per
  segment or once per substep holds T, not n_x.
- **The heading-rate term's zero thrust column is exact, not a shortcut.** If a later RHS makes
  ψ̇ read thrust (it must not), test 9 fails.
- **A hook that writes the thrust channel must know the law.** Today that is only the speed
  floor. The plan controller writes δ and is plan-only.

## 9. M1 review (2026-09-14, opus subagent) and what changed

**Verified by the reviewer:**
- **Default law bit-identical to `775b59e`:** the lag fixture 7/7; CUDA compiled path 9/9;
  interleaving both laws gives 4 graphs and 0 recompiles; the full objective + backward and the
  forecast→record JSON 44/44.
- **Census:** 0 of 245 stored `history.json` differ in name, slug, parameter rows or
  loadability.
- **Every consumer of the longitudinal column** passes the parameterisation (AST scan: 35
  calls).
- **The specific-force thrust of every exported segment** matches its own start state.

**Fixed after it:**
1. **Blocker — the campaign runner refused to resume every recipe arm on disk.**
   - Cause: the recipe pin is a field no stored `history.json` carries, and
     `frame_ablation.stale_arm_error` read the absent field as `None`.
   - Fix: it now reads a stored config the way `TSConfig.from_dict` does, through
     `config.absent_field_defaults`: the default, except for REQUIRED fields.
   - Checked on the real campaigns: B1 4/4, L1 3/3, B1b 1/1, A2b 2/2 resumable.
   - Test: `test_frame_ablation_runner.py`.
2. **The record export restated the law's thrust floor.** It now reads it from
   `backends.lag_control_law(config)`.
3. **The same-admissible-set claim was wrong**, and the neutral 0 overspeeds (§2.1, §2.3).
   The claim is reworded, and the neutral is now −0.05.
4. **Test gaps:**
   - The default-law test was vacuous; it is now a golden pinned from `775b59e`.
   - The export test now checks every segment.
   - A real barrier rollout now asserts column 0 passes through.
5. **Smaller fixes:**
   - The drag expression now exists once (`torch_dynamics.drag_force_n`); every RHS is still
     bit-identical.
   - Stale comments corrected.
   - The predictability report labels columns by contract, fixing the old `thrust_N`/`N`
     mislabel of δ.
   - Commands are exported at the head's precision under both laws.

## 10. M2 review (2026-09-14, opus subagent)

No blocker. Three should-fix issues, all fixed:
- The floor saturated on the head's box, not the engine.
- The engine-ceiling test was vacuous: all six floor tests passed with the engine term disabled. Now 55 m/s
  binds below the box and 40 m/s goes past it.
- The soft form's inertness was still on the switch boundary.

One nit fixed: the lag credit is now capped at the engine.

**Verified by the reviewer:**
- The thrust-fraction floor is bit-identical to `775b59e`: 18/18, on CPU and CUDA, float32 and float64, soft
  and hard, outputs, gradients and diagnostics.
- The N3 arm file differs from its twins only in the law.
- The intents key resolves.

**Re-review, all five fixes pass:**
- **Downstream accepts past-box commands:** a hooked command beyond the box is accepted by the export, by
  `record_from_dict`, by a hooked training epoch and by the dense predict.
- **Inertness is exact:** across 20,000 random boxes, 0 misses (it was 25–29 %).
- **Sharper tests added:**
  - engine-versus-box saturation, at T/W 0.40 and 52.5 m/s;
  - the soft form's engine cap;
  - an unsaturated credit case that lands exactly on the floor.

## 11. N4 — the condition vector as ratios (the alternative-hypothesis control)

### 11.1 The question

N3 tests one explanation of the thrust-fraction head's class structure: a δ command means a
different speed rate on every airframe, and the head cannot separate them. There is a second
explanation, and N3 alone cannot rule it out. **The head may fail because of how the airframe
is written into its condition vector.** Under `raw` the vector holds the mass, the installed
thrust and the wing area as three separately scaled numbers. The δ that flies a given speed
rate is `δ = (n_x + D/W) / (T_max/W)`, so the head needs the RATIO of two of its inputs, and
its first layer is linear in them.

N4 hands the δ head that ratio and changes nothing else. If N4 closes the class bias as far as
N3 does, the presentation explains the failure and the specific-force law is not needed for it.
If N4 leaves the bias while N3 closes it, the parameterisation is the mechanism.

### 11.2 The feature set (measured on KRDU, 26 OpenAP-direct types, 9,729 arrivals)

| channel | `raw` | `ratios` | fleet range under `ratios` |
|---|---|---|---|
| 1 | `mass_100t` | `mass_100t` | 0.068–2.51 (37×) |
| 2 | `max_thrust_1MN` (46×) | `thrust_to_weight` = T_max/(m·g) | 0.310–0.417 (1.3×) |
| 3 | `wing_area_500m2` (14×) | `stall_speed_100mps` = √(2mg/(ρ₀·S·CLmax))/100 | 0.394–0.620 (1.6×) |
| 4–8 | `cl_max_3`, `cd0_0p1`, `induced_k_0p1`, `stall_threshold`, `stall_k_0p2` | the same | — |

- **Four of the eight channels are constant on this fleet.** Cd0 = 0.02, k = 0.04,
  stall_threshold = 0.9 and k_stall = 0.1 for every type (the package's one clean polar).
  Only the mass, T_max, S and CLmax vary. With Cd0 and k fixed, D/W at a fixed multiple of
  the stall speed depends on CLmax alone. So under the δ law the per-class δ the head needs
  is a function of T_max/W and CLmax, and the ratios set gives it both directly.
- **The same information, and the same width.**
  - Given CLmax, `(m, T_max/W, V_s)` and `(m, T_max, S)` determine each other; the test
    recovers T_max and S from the ratios to float32 precision on three airframes.
  - The mass stays in both. It carries what the rollout cannot see (wake category, the speeds
    controllers give heavies), and without it the ratios set would carry less information
    than the raw set.
  - Both sets have eight channels, so the condition encoder has one shape. Under one seed a
    ratios head starts from its raw twin's weights, value for value (tested). The arm
    differs from its twin only in what it reads.
- **The stall speed is the package's one definition** (`aircraft.aero_params.stall_speed_ms`,
  also read by the optimizer's velocity floor and evaluation's threshold speed gate), at sea
  level and 1 g. It is a size-free proxy for the wing loading; it is not the speed floor.
- **Not a dimensionless set in the strict sense.** The stall speed is scaled by 100 m/s and
  the mass by 100 t. The value is named `ratios`, not `dimensionless`, for that reason.

### 11.3 The code (branch `sf-n4`)

- `config.py`:
  - `CONTROL_CONDITION_FEATURES = ("raw", "ratios")`.
  - The field `control_condition_features = "raw"` on `ControlOutput`, so ownership refuses
    it off the default on every other output.
  - Every named recipe pins `raw`, so a ratios run is `custom`.
  - Not required-serialized: absence reads `raw`, which every stored checkpoint ran.
  - `control_recipe()` carries `condition_features` only off the default.
- `outputs/conditioning.py`:
  - `CONDITION_FEATURE_SETS[features]`, `condition_names(features)` and `CONDITION_WIDTH`.
    The import asserts that every set has one width.
  - `condition_vector(..., features=)` takes the set as a REQUIRED keyword.
    `DYNAMICS_CONDITION_NAMES` and `CONDITION_CHANNELS` are gone.
- `outputs/dynamics/context.dynamics_arrays(..., condition_features=)` is REQUIRED, for the
  same reason as `parameterization`: one width means a forgotten argument is a silent
  mislabel. The plan path passes `raw` explicitly, since no plan head reads the row.
- `outputs/control/heads.py`: the encoder's input width is read from the configured set.
- `outputs/control/supervision.probe_dynamics`: the probe's condition row is now written by
  `condition_vector` from the probe's own airframe (the literal is gone). Under `raw` it is
  bit-identical to the old literal (checked).
- `run_naming.py`: META field `airframe=ratios`, placed ahead of the backbone knobs that fold,
  in the `Conditioning` section of the parameter rows.
- CLI: `train --control-condition-features`. `predict` reads the checkpoint's own value and
  offers no override: a head trained on one set would read the other silently.
- Tests: `tests/test_condition_features.py` (14 tests); the architecture one-source test covers
  both sets; the frame-ablation resume test drops both recipe-pinned fields.

**Verified before review:**
- **Default path bit-identical to `47b4b40`**, measured on a lagged simple-v3 training step
  (4 synthetic flights): loss, all 32 parameter gradients, the dynamics batch, and 4 forecast
  records.
- **Census over the 245 stored `history.json`**: display name, slug, parameter rows and
  `from_dict` are identical before and after (107 unloadable in both: the stale generations).
- **Campaign resume**: across 105 stored arms (38 arm files, 37 campaign directories),
  `stale_arm_error` is identical before and after (79 resumable in both).
- **The N4 arms** differ from the stored `B1_point_matched` / `_s2024` configs only in
  `control_condition_features`. The other two differences are artefacts of the comparison,
  not of the arms: `split_seed` is passed on the command line, and `corridor_gate` is a
  retired field. The dry run constructs both arms.
- ts suite: 1181 passed.

### 11.4 The run order

1. N3's runner exits. Its arms are read provisionally against the stored twins, marked as such.
2. `git -C .claude/worktrees/specific-force merge --ff-only sf-n4`, then launch the `sf_n4`
   campaign from that worktree, the same way N3 was launched:
   `run_ts.py frame_ablation --arms 4dTrajectory/ts_transformer/docs/experiments/sf_n4_arms.json
   --campaign 4dTrajectory/outputs/KRDU/experiments/sf_n4 --airport KRDU --split-seed 1337`.
   Four arms, in this order: `N4_twin`, `N4_twin_s2024` (the thrust-fraction twins re-trained at
   this code, §11.6), then `N4_ratios`, `N4_ratios_s2024`. That is about 4.5 h of GPU.
3. **Both N3 and N4 are read against `N4_twin` / `N4_twin_s2024`**, with the same instrument
   and the same gates. `N4_twin` against the stored `B1_point_matched` measures what 140
   commits moved.
4. **N5 (pooled five airports) only if N3's mechanism holds on both seeds.** There is no
  stored pooled thrust-fraction twin, so N5 is four arms (δ and n_x, two seeds), not two.

### 11.7 N4 results (2026-09-15, both seeds — final)

`sf_n4/N4_ratios` (180 epochs, selection ADE 1346.0 m) against its same-code twin `N4_twin`, with N3's
instrument:

| KRDU val, 1404 flights | N4_ratios | N4_twin |
|---|---|---|
| per-class n_x bias range (truth 0.0045 g) | 0.0119 | 0.0125 |
| heavy − B737 speed bias (m/s) | +5.19 | +5.16 |
| δ flown B737 / A320 / regional / heavy (the truth's: 0.036 / 0.042 / 0.031 / 0.028) | 0.044 / 0.042 / 0.047 / 0.047 | 0.044 / 0.043 / 0.047 / 0.047 |
| ADE pooled / straight-in / vectored (m) | 1346 / 451 / 2927 | 1325 / 444 / 2880 |
| FDE p50 pooled / straight-in (m) | 909 / 639 | 873 / 647 |
| paired ADE, arm better | 47.6 % (median +5 m) | — |

**Seed 2024:** `N4_ratios_s2024` (94 epochs, early stop; selection ADE 1361.8 m) against `N4_twin_s2024`:

| KRDU val, 1404 flights | N4_ratios_s2024 | N4_twin_s2024 |
|---|---|---|
| per-class n_x bias range (truth 0.0045 g) | 0.0105 | 0.0118 |
| heavy − B737 speed bias (m/s) | +4.50 | +4.89 |
| δ flown B737 / A320 / regional / heavy | 0.046 / 0.046 / 0.048 / 0.048 | 0.044 / 0.042 / 0.048 / 0.046 |
| ADE pooled / straight-in / vectored (m) | 1362 / 456 / 2964 | 1295 / 446 / 2793 |
| FDE p50 pooled / straight-in (m) | 883 / 642 | 911 / 663 |
| paired ADE, arm better | 43.2 % (median +20 m) | — |

**N4's verdict, both seeds (final): the alternative hypothesis is rejected.**
- Handed T_max/W and the stall speed directly, the δ head still commands a near-constant δ on every class
  (0.042–0.048 against the truth's 0.028–0.042).
- The class bias range moves 0.0125 → 0.0119 and 0.0118 → 0.0105. It moves under the specific-force law:
  0.0042 / 0.0044, at the truth's own 0.0045.
- **So the class structure is the parameterisation's.** The conditioning's presentation does not explain
  it. Making T_max/W an explicit input was not enough for a δ head to scale its command by it.
- The ratios arm also costs pooled ADE (+21 / +67 m; seed 2024 stopped early at 94 epochs) and nothing
  improves beyond the twins' spread. **`ratios` is not adopted.**

### 11.5 N4 review (2026-09-15, opus subagent, code only)

No blocker. The reviewer mutated the code in a /tmp copy: 11 of 13 mutations failed the new tests. The two
that passed were the predictability report's call sites, now tested. Checked and passing:
- every builder and reader of the condition row;
- a ratios checkpoint's save → load → forecast round trip;
- ownership, the recipe pins and the pipeline's reuse rule;
- the channels' units and the mass they read;
- the default path on CPU and CUDA;
- the arm file against its twins.

**Should-fix S1: the twins were trained 140 commits earlier.** Their manifests record `c0f2b9e` (2026-09-07).
Any drift in the default training path since then would land in BOTH N3's and N4's delta. Checked by
training the twin's configuration for 2 epochs with the current code, same seed and split, and comparing its
epoch records with the twin's `history.json` (§11.6).

**Fixed:**
- N1: the predictability report's model batch had no test. Its rows are now pinned to the forecast's
  (followups §36: it duplicates the forecast's context batch).
- N2: a test called "measured" that only read three hand-picked airframes is dropped.
- N3: the plan-output refusal now matches its message.
- N4: a ratios checkpoint round trip is added.
- N5: the stall-speed comment is corrected. The control path's anchor gate still restates the formula
  (followups §35).
- The pipeline's CV-reuse note is added to followups §33.

### 11.6 The stored twins do not reproduce at this code (2026-09-15)

**Measured** (the S1 check). `B1_point_matched`'s exact configuration, re-trained for 2 epochs with the
same seed and split. The code is `sf-n4`, whose default path is bit-identical to N3's `47b4b40` (§11.3).
Output in the session scratchpad.

| epoch | stored twin (`c0f2b9e`): train / val / selection ADE | this code, run 1 | this code, run 2 |
|---|---|---|---|
| 1 | 12.8653 / 8.3809 / 6176.0 m | 12.7541 / 7.7371 / 5481.6 m | bit-identical to run 1 |
| 2 | 6.7472 / 5.0437 / 3535.2 m | 6.3738 / 4.9609 / 3753.7 m | bit-identical to run 1 |

- **The same data.** The splits, the eligible set, the data provenance, the cohort, the windows and
  the parameter count are all equal.
- **The same initialisation and first batch.** At the first update the duration head's gradient
  agrees to 12 digits.
- **Only the control head's gradient differs**, by 0.25 % at its first-update maximum. The
  validation components that moved most are `state` and `terminal`.
- **The environment did not change.** torch 2.11.0 and numpy 2.5.1 were installed on 2026-07-19,
  before the twins were trained.
- **So the default control training path changed in code** between `c0f2b9e` (2026-09-07) and
  `47b4b40`.

**Consequence.** A stored-twin delta mixes the arm's effect with 140 commits of drift. `sf_n4`
therefore re-trains the twins at this code, and N3 and N4 are both read against them (§11.4).

**Cause: `c544db0` (2026-09-09, package review A-3), an intended and documented fix.** Bisected with
1-epoch real-data trainings (opus subagent; artifacts in `/tmp/claude-1000/bisect/`):

| code | epoch-1 train / val / selection ADE |
|---|---|
| `c0f2b9e` (the twins' commit) | 12.8653 / 8.3809 / 6176.0 m, bit-identical to the stored twin |
| `731ee58` = `c544db0^` | the same, bit for bit |
| `c544db0` | 12.7541 / 7.7371 / 5481.6 m, bit-identical to this code |
| `c544db0` with only its `dataset.py` hunk reversed | back to the twin's row, bit for bit |

- **The change** is in `dataset._build_supervision`. A flight whose OBSERVED track reaches the threshold
  used to get a flat 1/6 on all six channels of its last supervision row. It now gets
  1/6 + `fitted_terminal_position_weight`/3 = 0.5 on its three position channels, the terminal emphasis a
  fitted-tail flight already had.
- **Who it touches:** 132 of 6,851 KRDU train flights (1.9 %) and 22 of 1,404 val flights.
- **Why it moves training this much:** under `true-time-position` such a flight's terminal endpoint goes
  from about 3 % to about 9 % of its position term. That is the 0.25 % in the control head's
  first-update gradient; the duration head does not see it.
- **The measured effect:** epoch-1 selection ADE −694 m (−11 %). The commit stated the population
  (1 of the first 60 arrivals) but never measured the effect on training.
- **No later commit moves epoch 1**, and A-4 (the other half of the commit) does not: the hook is off.
- **`val_loss` is not comparable across `c544db0`, even for an identical model**, because the val `state`
  term reads the same weights on those 22 flights. The selection ADE reads no supervision weights and is
  comparable.
- **It is a data-side contract, not a config field.** A retrain of any pre-09-09 config picks it up with
  nothing in its config, name or `history.json` to show it — which is exactly how the twins stopped being
  twins.

## 12. N6 — a speed command (BUILT on `sf-n6`; launched only if the binding N3 reading confirms §7.2)

### 12.1 Why

N3 (provisional, §7.2) splits the question in two.
- **The specific force is the right invariant.** Its per-class bias collapses to the truth's own class
  spread on both seeds.
- **The law has lost the restoring force.** Under δ, drag gives `∂V̇/∂V = −(∂D/∂V)/m`. Under n_x it gives
  `0`, so a command bias integrates. That is the late-final speed deficit (−4.4 m/s in the last 5 km) and
  the straight-in FDE veto.
- The fix is not to take the drag back; its strength differs per airframe, which is the problem N3
  removed. The fix is a restoring force that is the SAME on every airframe.

The aviation answer is the autothrottle's SPEED mode:
- The command is a target airspeed.
- A speed loop turns it into thrust through the airframe's own weight and drag.

That is pattern P1 of the literature assessment (a platform-free command plus a per-platform inverse model):
the speed means the same on every airframe, and the airframe enters only through the thrust clamp, the stall
clamp and the exported thrust, exactly as under n_x. It is also what the plan path's guidance already does
(a speed schedule flown through the speed floor's thrust inversion).

### 12.2 The law (first-order-lag only)

```
controls (Δv, φ, n)   Δv = target airspeed − the anchor's airspeed V₀   [m/s]
actuator a_Δ lags Δv with τ_T (unchanged)
inside EVERY RK4 stage:
    n_x = sin γ + (V₀ + a_Δ − V) / (g · τ_V)
    T   = clamp(W · n_x + D, T_lo, T_max)            (the specific-force clamp, unchanged)
⇒   V̇ = (V₀ + a_Δ − V) / τ_V                         wherever T is not clamped
```

- **`∂V̇/∂V = −1/τ_V` on every airframe.** Where the clamp binds, it is the δ law at its bound, as under n_x.
  At idle, the deceleration is the airframe's own drag, which is what a real idle descent is.
- **τ_V = 8 s is a constant of the CONTRACT** (`envelope.SPEED_LOOP_TIME_CONSTANT_S`), like the
  specific-force box. It is not a config field. The plan was a field; the build showed why not:
  - the inverse needs τ_V at every call site (the anchor actuator, the teacher, basis_fit) — a third
    required argument through `dynamics_arrays`;
  - the teacher barely moves with it (§12.3);
  - autothrottle speed loops are 5–10 s;
  - a run that wants another value is another contract.
  RK4 at h = 0.5 s is far inside its limit.
- **The command is RELATIVE to the anchor speed**, so the neutral `Δv = 0` is "hold the speed you have".
  An untrained head then flies no transient on any airframe. An absolute neutral (one speed for all)
  would drive every flight toward it from step 0.
- Implemented as a third value of `control_thrust_parameterization`: `speed-command`.
  - It is a third `LagControlLaw` (`SpeedCommandLaw(min_thrust_fraction, speed_time_constant_s)`).
    Its step context also carries V₀ (each flight's initial-state airspeed) and τ_V.
  - The thrust goes through the specific-force law's own helper. The loop's specific force is one
    expression, `torch_dynamics.speed_loop_specific_force`, read by the chart RHS and by the exported
    record thrust alike.
  - The δ and n_x laws stay bit-identical: goldens, and the identity check in §12.7.

### 12.3 Measured: where the teacher lives (KRDU val, 1404 × 32, the same tracks as §2.4)

Inverse on the observed track: `a_Δ = V − V₀ + τ_V·V̇`, `command = a_Δ + τ_T·ȧ_Δ`. Averaged over each of the
32 uniform segments. Script: session scratchpad `speed_command_teacher.py`.

| τ_V | command − V₀: p0.1 / p1 / p5 / p50 / p95 / p99 / p99.9 (m/s) | segment-to-segment step p50 / p99 |
|---|---|---|
| 5 s | −84.8 / −73.9 / −60.1 / −16.2 / −0.3 / +3.2 / +15.2 | 1.2 / 9.2 |
| 8 s | −85.1 / −74.1 / −60.4 / −16.5 / −0.3 / +3.6 / +15.2 | 1.5 / 10.0 |
| 12 s | −86.0 / −74.6 / −60.7 / −17.0 / −0.3 / +4.1 / +15.3 | 1.9 / 11.6 |

- **Box: Δv ∈ [−90, +20] m/s**, half width 55, neutral 0. It cuts none of the 0.1–99.9 % range.
  - The absolute target lives in 52–150 m/s. The box is a search space, as the others are; the stall
    floor is the speed floor's job.
- **The imitation weight does not transfer from δ.** A speed command is a target, not a rate. N6 keeps
  64 and says so, like the per-airport rule for that weight.

### 12.4 What else changes (the M1/M2 checklist, applied to the third law)

- **Inverse and teacher:**
  - `actual_controls` under `speed-command` returns the loop's ABSOLUTE target `V + τ_V·tangential`, from
    the kinematics alone. It does not know the anchor.
  - `inverse.anchor_relative(controls, anchor_speed, parameterization=)` subtracts the anchor's speed.
    Only the callers that know the anchor call it:
    - `reference_controls` (the teacher; its `states[0]` is the anchor);
    - `context.anchor_controls` (the actuator's initial condition; the window's last row is the anchor).
  - The lag inverse commutes with that constant shift.
- **Context:** the initial actuator is the inverted `a_Δ` at the anchor, clipped to the box. It equals
  `g·τ_V·(n_x − sin γ)` of the specific-force inversion on the same track (tested).
- **Speed floor:**
  - The demand becomes a speed. Holding `V(hold) ≥ V_floor` under the loop is
    `V₀ + Δv ≥ (V_floor − V·e^{−dt/τ_V})/(1 − e^{−dt/τ_V})`, with the actuator's lag credited as before.
  - Saturation means the engine's `(T_max − D)/W`.
  - Refused in the first milestone, like M1 did.
- **Barrier and trombone:** pass column 0 through (unchanged, already tested under n_x).
- **Export:**
  - Each segment's record thrust is the law's thrust at its start state.
  - Each segment carries its command under the contract's name, `speed_command_delta` (relative, m/s;
    the anchor speed is the record's `initial_state.V`).
  - `source.controlThrustParameterization` is `speed-command`.
  - `Forecast.specific_force_commands` became `longitudinal_commands` + `longitudinal_parameterization`.
    The specific-force records keep their `specific_force` key, bit for bit.
- **Naming, CLI, recipes:**
  - Named `first-order-lag+speed-command` / slug `-sc`.
  - Every recipe pins thrust-fraction; `control_recipe()` names the law off the default.
  - The config refuses `speed-command` off `first-order-lag`, with the fitted teacher, and with any hook
    that contains the speed floor (not built yet).
- **Tests:**
  - Under the loop, `V̇ = (V₀ + a_Δ − V)/τ_V` exactly where the clamp does not bind.
  - At the clamp, the law equals δ's at its bound.
  - Round trip through the inverse.
  - The neutral holds the anchor speed on three airframes.
  - Export, naming, config, and the two old laws' goldens.

### 12.5 The arms (after the build and its review)

`N6_speed_command` / `_s2024` are `B1_point_matched` with only the law moved. They are read against
`N4_twin` / `_s2024` with N3's instrument.
- **Gates:** the per-class n_x bias range (N3's, kept); the heavy − 737 speed gap closing (N3 failed it);
  the straight-in FDE back inside the twins' seed spread (N3 failed it).
- **Veto:** pooled ADE worse than the twin by more than ~125 m.
- **Reading:** if N6 passes all three where N3 passed one, the invariant needed the restoring force.

### 12.6 Order and condition

1. Wait for `N4_twin` (the first `sf_n4` arm).
2. If its straight-in FDE p50 stays below N3's 885–887 m by more than the seed spread (the stored twins
   582 / 659), the veto is confirmed and N6 is queued after `sf_n4`.
   - The runner worktree is fast-forwarded to `sf-n6`, which descends from `sf-n4`, and the campaign is
     launched from it.
3. If not, N3's FDE penalty was the stored twins' drift. N6 is dropped and `sf-n6` is not merged.

**Outcome (2026-09-15, §7.3):** met. `N4_twin`'s straight-in FDE p50 is 647 m against N3's 887 m, and the
seed spread is 77 m. N6 is queued after `sf_n4`.

### 12.7 Verified before review (2026-09-15)

- **The law:** `aerodynamic_model/tests/test_torch_lag_speed_command.py`, 22 tests:
  - `V' = (V₀ + a_Δ − V)/τ_V` to 1e-9 on three airframes and four commands;
  - at the clamp, the law equals δ's at its bound;
  - a constant step follows the two-pole response `V₀ + Δv(1 − (τ_V e^{−t/τ_V} − τ_T e^{−t/τ_T})/(τ_V −
    τ_T))` to 1e-3 m/s and settles on the command;
  - each flight's reference speed is its own anchor speed;
  - the CUDA compile path and its backward pass.
- **The ts side:** `tests/test_speed_command.py`, 11 tests:
  - the round trip through the inverse (Δv to 0.1 m/s);
  - the anchor actuator's tie to the specific-force column;
  - the neutral head, the probe, the export's thrust at every segment start, one training step;
  - the barrier's pass-through, the config refusals, the recipes, the name, the CLI.
- **The two stored laws are bit-identical to `e959bc0`:** thrust-fraction and specific-force, a lagged
  simple-v3 step (loss, all 32 gradients, the batch, 4 records).
- **Stored runs:** over 247 stored runs, names, slugs, parameter rows and loading are identical to
  `e959bc0`'s. Resume is identical over 107 arms (81 resumable).
- **The arm file:** the dry run constructs both arms. Their base is N3's with only the law moved.
- **Suites:** ts 1193 passed; aerodynamic_model 124 passed.

### 12.8 N6 review (2026-09-15, opus subagent, code only)

No blocker. The reviewer made 38 mutations in a /tmp copy: 31 were killed, 7 survived. Of the survivors,
one was equivalent and the rest were fixed below. What it checked:
- the law's algebra and context layout;
- the CUDA caches: 6 code objects, 0 recompiles when the three laws interleave;
- every branch on the law;
- `anchor_relative` applied exactly once per path, with the teacher's `states[0].V` equal to the
  rollout's V₀ bit for bit on all 1404 KRDU val flights;
- the export, the config refusals, the arm file;
- the δ and n_x identity, regenerated independently.

**Measured by the reviewer: the teacher flown open-loop.** Each law's own inverse-dynamics teacher, flown
through its own rollout, on 300 KRDU val flights:

| teacher | ADE mean | ADE p50 |
|---|---|---|
| speed-command | **375 m** | **36 m** |
| specific-force | 2606 m | 158 m |
| thrust-fraction | 2700 m | 321 m |

The speed loop's teacher reproduces the truth ~7× better. Its restoring force keeps an imperfect command
from integrating away, which is §12.1's argument, measured on the teacher. On the same set the thrust clamp
binds at 0.2 % / 1.3 % of segment ends (above the maximum / below the floor). The box clips 0.037 % /
0.069 % of the teacher, and 0.07 % of the anchor actuators.

**Fixed:**
- **S1:** the hooked rollout's physics was untested. Dropping the law from the hooked path flew Δv as a
  thrust fraction and passed every test. Now the hooked states must equal a plain rollout of the flown
  schedule, and the per-flight reference speed is tested in all three rollouts.
- **S2:** τ_V and the box are module constants that no checkpoint recorded.
  `envelope.speed_command_identity()` is now spelled into the target contract under this law, so a moved
  constant is refused at load. The specific-force box has the same gap (followups §37).
- **S3:** the "holds the speed" test checked only the neutral's value. It now flies the untrained head
  through the forecast: the excursion is bounded by the anchor actuator, and the speed ends on V₀.
- **N1:** under the loop, the record's per-segment thrust is the hold's extreme, not its mean (the error
  decays across the segment). Documented in `record_newton_controls`. A readout of the record's thrust
  fraction must read it so; the n_x and speed read off the states are exact.
- **N2:** stale docstrings corrected.
- **N3:** one chart read (`_chart_speed_altitude_mass`) shared by the two thrust helpers.
- **N4:** tests for the anchor-actuator clip and the saturation labels.

After the fixes: δ and n_x are still bit-identical to `e959bc0`. ts + aerodynamic_model: 1323 passed.
